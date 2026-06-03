/**
 * @file my_postprocess.cc
 * @brief YOLO11 模型推理结果的后处理实现
 *
 * 本文件实现了 YOLO11 目标检测模型在 RKNPU 上推理完成后，对原始输出张量
 * 进行解码、过滤、非极大值抑制（NMS）和坐标映射的全流程后处理。
 *
 * 主要流程：
 * 1. 从模型输出中读取三个尺度的特征图（大/中/小目标各一个分支）
 * 2. 对每个特征图上的每个网格点，解析出边界框偏移量和类别得分
 * 3. 使用 DFL（Distribution Focal Loss）将分布回归值解码为边界框坐标
 * 4. 根据置信度阈值过滤低置信度检测
 * 5. 按类别分别执行 NMS，消除重复检测
 * 6. 将坐标从特征图空间映射回原始图像空间（含 letterbox 补偿）
 *
 * 数据类型支持：int8 量化 / uint8 量化 / fp32 浮点
 * 平台支持：RKNPU1 / RKNPU2（含 RV1106/RV1103 特殊 NHWC 布局）
 */
// Copyright (c) 2024 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "my_yolo11.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include <set>
#include <vector>

/**
 * COCO 80 类标签文件路径
 * 该文件每行一个类别名称，共 80 行，对应 COCO 数据集的 80 个目标类别。
 * 用于将模型输出的类别 ID（0~79）映射为可读的类别名称字符串。
 */
#define LABEL_NALE_TXT_PATH "./model/coco_80_labels_list.txt"

/** 全局类别名称数组，索引对应类别 ID，通过 loadLabelName 从文件加载 */
static char *labels[OBJ_CLASS_NUM];

/**
 * @brief 将浮点数值限幅（clamp）到指定整数范围 [min, max]
 *
 * 实现原理：三个条件表达式嵌套，依次判断是否低于下限、是否高于上限。
 * 用于在后处理中将浮点坐标裁剪到图像边界范围内，防止越界访问。
 *
 * @param val 输入的浮点数值
 * @param min 下限值（包含）
 * @param max 上限值（包含）
 * @return int 限幅后的整数值
 */
inline static int clamp(float val, int min, int max) { return val > min ? (val < max ? val : max) : min; }

/**
 * @brief 从文件中读取一行文本（动态内存分配）
 *
 * 算法原理：
 * 逐字符从文件流中读取，直至遇到换行符 '\n' 或文件结束符 EOF。
 * 使用 malloc + realloc 动态扩展缓冲区，可处理任意长度的行。
 * 调用者负责释放返回的字符串内存。
 *
 * @param fp 已打开的文件指针（FILE*）
 * @param buffer 输出参数，函数内部会 malloc 分配内存并填充行内容
 * @param len 输出参数，记录读取到的行长度（不含 null 终止符）
 * @return char* 成功返回指向行字符串的指针，失败或到达文件末尾返回 NULL
 */
static char *readLine(FILE *fp, char *buffer, int *len)
{
    int ch;
    int i = 0;
    size_t buff_len = 0;

    // 初始分配 1 字节（仅 null 终止符）
    buffer = (char *)malloc(buff_len + 1);
    if (!buffer)
        return NULL; // 内存分配失败

    // 逐字符读取，直到遇到换行符或 EOF
    while ((ch = fgetc(fp)) != '\n' && ch != EOF)
    {
        buff_len++;
        void *tmp = realloc(buffer, buff_len + 1); // 扩展缓冲区
        if (tmp == NULL)
        {
            free(buffer);
            return NULL; // 内存重分配失败
        }
        buffer = (char *)tmp;

        buffer[i] = (char)ch; // 存储当前字符
        i++;
    }
    buffer[i] = '\0'; // null 终止

    *len = buff_len;

    // 检测是否提前到达文件末尾（空行或读取错误）
    if (ch == EOF && (i == 0 || ferror(fp)))
    {
        free(buffer);
        return NULL;
    }
    return buffer;
}

/**
 * @brief 从文件中读取最多 max_line 行文本，存入字符串指针数组
 *
 * 算法原理：
 * 循环调用 readLine() 逐行读取，将每行的字符串指针存入预设数组中。
 * 用于加载类别标签文件（如 coco_80_labels_list.txt）。
 * 数组中的每个元素需在 deinit_post_process 中释放。
 *
 * @param fileName 标签文件的路径字符串
 * @param lines 输出参数，字符串指针数组，存放每行文本
 * @param max_line 最大读取行数（不能超过数组容量）
 * @return int 成功返回实际读取的行数，文件打开失败返回 -1
 */
static int readLines(const char *fileName, char *lines[], int max_line)
{
    FILE *file = fopen(fileName, "r");
    char *s;
    int i = 0;
    int n = 0;

    if (file == NULL)
    {
        printf("Open %s fail!\n", fileName);
        return -1;
    }

    // 逐行读取，直到文件末尾或达到最大行数
    while ((s = readLine(file, s, &n)) != NULL)
    {
        lines[i++] = s;
        if (i >= max_line)
            break;
    }
    fclose(file);
    return i;
}

/**
 * @brief 加载类别标签名称列表
 *
 * 算法原理：
 * 从指定路径的文本文件中读取类别名称，每行一个类别名，
 * 存入全局 label 数组中供 coco_cls_to_name() 查询使用。
 * 文件行数应与 OBJ_CLASS_NUM（即 COCO 80 类）一致。
 *
 * @param locationFilename 标签文件的路径
 * @param label 输出参数，字符串指针数组，存放类别名称
 * @return int 成功返回 0，失败返回 -1
 */
static int loadLabelName(const char *locationFilename, char *label[])
{
    printf("load lable %s\n", locationFilename);
    readLines(locationFilename, label, OBJ_CLASS_NUM);
    return 0;
}

/**
 * @brief 计算两个边界框的交并比（IoU, Intersection over Union）
 *
 * 算法原理：
 * IoU 衡量两个矩形框的重叠程度，是 NMS 的核心判定指标。
 *
 *       交集面积
 *   IoU = -------  其中 并集面积 = 框1面积 + 框2面积 - 交集面积
 *       并集面积
 *
 *   交集宽度 w = max(0, min(xmax0, xmax1) - max(xmin0, xmin1) + 1)
 *   交集高度 h = max(0, min(ymax0, ymax1) - max(ymin0, ymin1) + 1)
 *   交集面积 = w * h
 *
 * IoU 取值范围 [0.0, 1.0]，值越大表示两个框重叠越严重。
 *
 * @param xmin0,ymin0,xmax0,ymax0 第一个框的左上角和右下角坐标
 * @param xmin1,ymin1,xmax1,ymax1 第二个框的左上角和右下角坐标
 * @return float 两个框的 IoU 值
 */
static float CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0, float xmin1, float ymin1, float xmax1,
                              float ymax1)
{
    // 计算交集区域的宽度和高度（+1.0 是像素坐标修正，使面积计算准确）
    float w = fmax(0.f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1) + 1.0);
    float h = fmax(0.f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1) + 1.0);
    float i = w * h;       // 交集面积
    float u = (xmax0 - xmin0 + 1.0) * (ymax0 - ymin0 + 1.0) + (xmax1 - xmin1 + 1.0) * (ymax1 - ymin1 + 1.0) - i; // 并集面积
    return u <= 0.f ? 0.f : (i / u);
}

/**
 * @brief 非极大值抑制（NMS, Non-Maximum Suppression）
 *
 * 算法原理：
 * NMS 用于消除同一个目标上的多个重复检测框，只保留最优的那个。
 * 操作步骤（按类别独立处理）：
 * 1. 遍历按置信度降序排列的候选框（由 quick_sort_indice_inverse 预先排序）
 * 2. 对于当前类别 filterId 的最高分框 n：
 *    a. 保留该框（不修改 order[n]）
 *    b. 遍历其后所有同类别框 m
 *    c. 计算框 n 与框 m 的 IoU
 *    d. 若 IoU > threshold，则将 order[m] 置为 -1（标记为抑制）
 * 3. 遍历下一个未被抑制的框，重复步骤 2
 *
 * 坐标存储格式：outputLocations 中每 4 个值为 (x, y, w, h)
 * 计算 IoU 时需要先转换为 (xmin, ymin, xmax, ymax) 格式。
 *
 * @param validCount 候选框总数
 * @param outputLocations 所有候选框坐标（每 4 个一组：x, y, w, h）
 * @param classIds 每个候选框对应的类别 ID
 * @param order 置信度降序索引数组（排序后，被抑制的框置为 -1）
 * @param filterId 当前处理的类别 ID
 * @param threshold NMS 的 IoU 阈值（通常 0.45 ~ 0.7）
 * @return int 成功返回 0
 */
static int nms(int validCount, std::vector<float> &outputLocations, std::vector<int> classIds, std::vector<int> &order,
               int filterId, float threshold)
{
    for (int i = 0; i < validCount; ++i)
    {
        int n = order[i];
        // 跳过已被抑制的框或非当前类别的框
        if (n == -1 || classIds[n] != filterId)
        {
            continue;
        }
        for (int j = i + 1; j < validCount; ++j)
        {
            int m = order[j];
            // 跳过已被抑制的框或非当前类别的框
            if (m == -1 || classIds[m] != filterId)
            {
                continue;
            }
            // 将第 n 个框从 (x, y, w, h) 转换为 (xmin, ymin, xmax, ymax)
            float xmin0 = outputLocations[n * 4 + 0];
            float ymin0 = outputLocations[n * 4 + 1];
            float xmax0 = outputLocations[n * 4 + 0] + outputLocations[n * 4 + 2];
            float ymax0 = outputLocations[n * 4 + 1] + outputLocations[n * 4 + 3];

            // 将第 m 个框从 (x, y, w, h) 转换为 (xmin, ymin, xmax, ymax)
            float xmin1 = outputLocations[m * 4 + 0];
            float ymin1 = outputLocations[m * 4 + 1];
            float xmax1 = outputLocations[m * 4 + 0] + outputLocations[m * 4 + 2];
            float ymax1 = outputLocations[m * 4 + 1] + outputLocations[m * 4 + 3];

            // 计算两个框的 IoU
            float iou = CalculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1);

            if (iou > threshold)
            {
                order[j] = -1; // 标记为抑制（丢弃）
            }
        }
    }
    return 0;
}

/**
 * @brief 快速排序（降序），同时同步维护原始索引数组
 *
 * 算法原理：
 * 标准快速排序（Hoare 分区方案）的变体，取区间第一个元素为基准值（pivot）。
 * 将 input 数组按值降序排列，同时 indices 数组同步交换，
 * 确保 indices[i] 保持对应 input[i] 在原始数组中的位置索引。
 *
 * 用途：
 * 对检测框的置信度进行降序排序后，仍能通过 indices 数组追溯
 * 每个置信度对应的原始检测框在 filterBoxes/classId 中的位置。
 *
 * @param input 待排序的浮点数组（会被原地修改为降序排列）
 * @param left 本次排序区间的左边界索引（包含）
 * @param right 本次排序区间的右边界索引（包含）
 * @param indices 索引数组，与 input 同步排序交换
 * @return int 返回基准值最终位置
 */
static int quick_sort_indice_inverse(std::vector<float> &input, int left, int right, std::vector<int> &indices)
{
    float key;
    int key_index;
    int low = left;
    int high = right;
    if (left < right)
    {
        key_index = indices[left]; // 记录基准值对应的原始索引
        key = input[left];         // 基准值
        while (low < high)
        {
            // 从右向左找第一个大于基准值的元素
            while (low < high && input[high] <= key)
            {
                high--;
            }
            // 将右侧大值移到左侧空位，同时同步移动索引
            input[low] = input[high];
            indices[low] = indices[high];
            // 从左向右找第一个小于基准值的元素
            while (low < high && input[low] >= key)
            {
                low++;
            }
            // 将左侧小值移到右侧空位，同时同步移动索引
            input[high] = input[low];
            indices[high] = indices[low];
        }
        // 将基准值放到最终位置
        input[low] = key;
        indices[low] = key_index;
        // 递归对左右子区间排序
        quick_sort_indice_inverse(input, left, low - 1, indices);
        quick_sort_indice_inverse(input, low + 1, right, indices);
    }
    return low;
}

/**
 * @brief Sigmoid 激活函数
 *
 * 公式：sigmoid(x) = 1 / (1 + exp(-x))
 *
 * 将任意实数映射到 (0, 1) 开区间，在 YOLO 中用于将网络输出的逻辑值
 * 转换为概率值（如置信度、类别得分）。具有单调递增和 S 形曲线特性。
 */
static float sigmoid(float x) { return 1.0 / (1.0 + expf(-x)); }

/**
 * @brief Sigmoid 反函数（对数几率函数）
 *
 * 公式：unsigmoid(y) = -ln((1/y) - 1)
 *
 * sigmoid 的逆运算，将概率值 y 还原为逻辑值。
 * 在量化感知训练场景中，有时需要将概率阈值反变换到逻辑空间
 * 进行等价的比较操作。
 */
static float unsigmoid(float y) { return -1.0 * logf((1.0 / y) - 1.0); }

/**
 * @brief 浮点数限幅（inline 内联版本）
 *
 * 将浮点值 val 限制在 [min, max] 闭区间内。
 * 主要用于量化操作中的值域裁剪，防止量化后的值超出目标数据类型的表示范围。
 *
 * @param val 输入的浮点值
 * @param min 下限（包含）
 * @param max 上限（包含）
 * @return int32_t 限幅后的整数值
 */
inline static int32_t __clip(float val, float min, float max)
{
    float f = val <= min ? min : (val >= max ? max : val);
    return f;
}

/**
 * @brief 将 float32 浮点数量化为 int8（仿射量化）
 *
 * 仿射量化公式：q = round(f / scale) + zp
 *
 * 仿射量化（Affine Quantization）使用两个参数：
 * - scale（缩放因子）：控制浮点值到整数的映射步长
 * - zp（zero-point，零点）：浮点值 0.0 对应的整数值
 *
 * 量化后的值会裁剪到 int8 范围 [-128, 127]。
 * 用于将浮点阈值转换为量化域的阈值，以便在量化域直接比较。
 *
 * @param f32 输入的 float32 浮点值
 * @param zp zero-point（零点偏移）
 * @param scale 缩放因子
 * @return int8_t 量化后的 int8 值
 */
static int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale)
{
    float dst_val = (f32 / scale) + zp;
    int8_t res = (int8_t)__clip(dst_val, -128, 127); // 限幅到 int8 表示范围
    return res;
}

/**
 * @brief 将 float32 浮点数量化为 uint8（仿射量化）
 *
 * 与 qnt_f32_to_affine 类似，但输出类型为 uint8（无符号 8 位整数），
 * 值域范围为 [0, 255]。用于 RKNPU1 平台的 uint8 量化模型。
 */
static uint8_t qnt_f32_to_affine_u8(float f32, int32_t zp, float scale)
{
    float dst_val = (f32 / scale) + zp;
    uint8_t res = (uint8_t)__clip(dst_val, 0, 255); // 限幅到 uint8 表示范围
    return res;
}

/**
 * @brief 将 int8 量化值反量化为 float32 浮点数
 *
 * 反量化公式：f = (q - zp) * scale
 *
 * 是量化的逆过程。将模型输出的 int8 量化张量中的每个值，
 * 结合对应的 zero-point 和 scale 参数，恢复为近似的浮点值。
 * 用于解码量化模型的输出结果。
 *
 * @param qnt int8 量化值
 * @param zp zero-point（零点偏移）
 * @param scale 缩放因子
 * @return float 反量化后的浮点值
 */
static float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) { return ((float)qnt - (float)zp) * scale; }

/**
 * @brief 将 uint8 量化值反量化为 float32 浮点数
 *
 * 与 deqnt_affine_to_f32 功能相同，但输入为 uint8 类型。
 * 用于 RKNPU1 平台的 uint8 量化模型输出解码。
 */
static float deqnt_affine_u8_to_f32(uint8_t qnt, int32_t zp, float scale) { return ((float)qnt - (float)zp) * scale; }

/**
 * @brief DFL（Distribution Focal Loss）分布解码为边界框坐标偏移量
 *
 * 算法原理：
 * YOLO11 使用分布回归（Distribution-based Regression）取代传统的直接边界框回归。
 * 对于边界框的每一条边（共 4 条边：左、上、右、下），
 * 网络不是直接回归偏移值，而是预测一个长度为 dfl_len 的概率分布。
 *
 * 解码步骤：
 * 1. 对第 b 条边的 dfl_len 个原始值做指数运算（exp），得到非负值
 *    exp_t[i] = exp(tensor[i + b * dfl_len])
 * 2. 计算 softmax 归一化：p_i = exp_t[i] / sum(exp_t)
 * 3. 以归一化概率为权重，对位置索引做加权平均：
 *    box[b] = sum(p_i * i)  for i in [0, dfl_len-1]
 *
 * 结果 box[b] 是第 b 条边相对于网格中心的归一化偏移量（范围约 [0, dfl_len-1]）。
 * 这种分布回归的优势在于能建模边界的不确定性，提升定位精度。
 *
 * @param tensor 输入的 DFL 特征数据指针，布局为 [4, dfl_len]（4 条边 × 分布长度）
 * @param dfl_len 每条边的分布长度（即离散化的 bin 数量）
 * @param box 输出数组 [4]，解码后的四条边偏移量 [left, top, right, bottom]
 */
static void compute_dfl(float* tensor, int dfl_len, float* box){
    for (int b=0; b<4; b++){
        float exp_t[dfl_len];  // 存储 exp 后的值
        float exp_sum=0;       // exp 值总和，用于 softmax 归一化
        float acc_sum=0;       // 加权求和累加器
        // 第 1 步：对第 b 条边的所有 bin 做 exp 指数运算
        for (int i=0; i< dfl_len; i++){
            exp_t[i] = exp(tensor[i+b*dfl_len]);
            exp_sum += exp_t[i];
        }
        // 第 2-3 步：softmax 归一化后加权求和
        for (int i=0; i< dfl_len; i++){
            acc_sum += exp_t[i]/exp_sum *i; // p_i = exp_t[i] / exp_sum, 权重为 i
        }
        box[b] = acc_sum; // 第 b 条边的解码偏移量
    }
}

/**
 * @brief 处理 uint8 量化模型的单个特征图尺度，提取检测结果
 *
 * YOLO11 的后处理核心函数之一，处理一个尺度的特征图（如 80×80、40×40、20×20）。
 * 该尺度对应特定大小的目标（大 stride 检测大目标，小 stride 检测小目标）。
 *
 * 算法原理（每个网格点的处理流程）：
 * 1. 【快速过滤】如果提供了 score_sum 张量，先判断该网格的总体得分是否超过阈值
 *    ——这是一种加速策略，避免在所有类别上做完整遍历
 * 2. 【找最佳类别】遍历网格在所有 OBJ_CLASS_NUM 个类别上的得分，找出得分最高的类别
 *    - 得分张量在 CHW 布局下按 channel 维度排列，访问需跨 grid_len 步长
 * 3. 【DFL 解码】读取该网格的 4×dfl_len 个 box 特征值，反量化后执行 DFL 解码
 * 4. 【坐标转换】将 DFL 输出的偏移量转换为原图坐标系中的 (x1, y1, w, h)
 *    - 网格中心原图坐标：(j + 0.5) × stride, (i + 0.5) × stride
 *    - 左边界 x1 = (center_x - box[0]) × stride = (-box[0] + j + 0.5) × stride
 *    - 上边界 y1 = (center_y - box[1]) × stride = (-box[1] + i + 0.5) × stride
 *    - 宽 w = box[2] + box[0]... 实际计算为 (x2 - x1)
 * 5. 【存储结果】将有效检测框的坐标、置信度、类别 ID 存入输出向量
 *
 * @param box_tensor 边界框特征张量（uint8 量化），形状 [4*dfl_len, grid_h, grid_w]
 * @param box_zp box_tensor 的 zero-point
 * @param box_scale box_tensor 的 scale
 * @param score_tensor 类别得分特征张量（uint8 量化），形状 [OBJ_CLASS_NUM, grid_h, grid_w]
 * @param score_zp,score_scale score_tensor 的量化参数
 * @param score_sum_tensor 可选：类别得分总和张量（已合并的总体置信度，用于加速过滤）
 * @param score_sum_zp,score_sum_scale score_sum 的量化参数
 * @param grid_h 特征图高度（如 80、40、20）
 * @param grid_w 特征图宽度（如 80、40、20）
 * @param stride 该尺度对应的下采样倍数（如 8、16、32）
 * @param dfl_len DFL 分布长度
 * @param boxes 输出：所有检测框坐标向量（每 4 个一组：x1, y1, w, h）
 * @param objProbs 输出：所有检测框的置信度
 * @param classId 输出：所有检测框的类别 ID
 * @param threshold 置信度阈值（浮点值，会被量化到 uint8 域进行比较）
 * @return int 该尺度上检测到的有效目标数量
 */
static int process_u8(uint8_t *box_tensor, int32_t box_zp, float box_scale,
                      uint8_t *score_tensor, int32_t score_zp, float score_scale,
                      uint8_t *score_sum_tensor, int32_t score_sum_zp, float score_sum_scale,
                      int grid_h, int grid_w, int stride, int dfl_len,
                      std::vector<float> &boxes,
                      std::vector<float> &objProbs,
                      std::vector<int> &classId,
                      float threshold)
{
    int validCount = 0;         // 该尺度的有效检测计数
    int grid_len = grid_h * grid_w; // 单通道的网格总数
    // 将浮点阈值预量化为 uint8 域，避免在循环内重复量化
    uint8_t score_thres_u8 = qnt_f32_to_affine_u8(threshold, score_zp, score_scale);
    uint8_t score_sum_thres_u8 = qnt_f32_to_affine_u8(threshold, score_sum_zp, score_sum_scale);

    // 遍历特征图的每个网格点（行优先）
    for (int i = 0; i < grid_h; i++)
    {
        for (int j = 0; j < grid_w; j++)
        {
            int offset = i * grid_w + j; // 该网格在 CHW 布局中第一个 channel 的索引
            int max_class_id = -1;       // 最佳类别 ID，初始化为 -1 表示无效

            // 如果提供了 score_sum 张量，先做快速过滤
            // score_sum 是预先合并的总体置信度，比遍历所有类别更快
            if (score_sum_tensor != nullptr)
            {
                if (score_sum_tensor[offset] < score_sum_thres_u8)
                {
                    continue; // 该网格总体得分低，跳过
                }
            }

            uint8_t max_score = -score_zp; // 初始化最高得分为最小可表示值
            // 遍历所有类别，找出得分最高的类别
            // 注意：score_tensor 以 CHW 布局存储 (C, H, W)，同一网格不同类别跨 grid_len 步长
            for (int c = 0; c < OBJ_CLASS_NUM; c++)
            {
                if ((score_tensor[offset] > score_thres_u8) && (score_tensor[offset] > max_score))
                {
                    max_score = score_tensor[offset];
                    max_class_id = c;
                }
                offset += grid_len; // 前进到下一个类别的同一网格位置
            }

            // 如果最高分超过阈值，解码边界框
            if (max_score > score_thres_u8)
            {
                offset = i * grid_w + j; // 回到 box 张量的起始偏移
                float box[4];           // 存储 DFL 解码后的 4 条边偏移量
                float before_dfl[dfl_len * 4]; // DFL 解码前的反量化值
                // 读取 4*dfl_len 个 box 特征值并反量化
                for (int k = 0; k < dfl_len * 4; k++)
                {
                    before_dfl[k] = deqnt_affine_u8_to_f32(box_tensor[offset], box_zp, box_scale);
                    offset += grid_len; // box 张量也以 CHW 布局存储
                }
                compute_dfl(before_dfl, dfl_len, box); // DFL 解码

                // 将 DFL 偏移量转换为原图坐标系中的边界框坐标
                float x1, y1, x2, y2, w, h;
                x1 = (-box[0] + j + 0.5) * stride; // 左边界（相对网格中心向左偏移）
                y1 = (-box[1] + i + 0.5) * stride; // 上边界（相对网格中心向上偏移）
                x2 = (box[2] + j + 0.5) * stride;  // 右边界（相对网格中心向右偏移）
                y2 = (box[3] + i + 0.5) * stride;  // 下边界（相对网格中心向下偏移）
                w = x2 - x1;
                h = y2 - y1;
                // 以 (x1, y1, w, h) 格式存储
                boxes.push_back(x1);
                boxes.push_back(y1);
                boxes.push_back(w);
                boxes.push_back(h);

                // 存储置信度和类别 ID
                objProbs.push_back(deqnt_affine_u8_to_f32(max_score, score_zp, score_scale));
                classId.push_back(max_class_id);
                validCount++;
            }
        }
    }
    return validCount;
}

/**
 * @brief 处理 int8 量化模型的单个特征图尺度，提取检测结果
 *
 * 功能与 process_u8 完全相同，但处理的是 int8（有符号 8 位）量化张量。
 * 对应 RKNPU2 平台的 int8 量化模型输出。
 *
 * 算法原理与 process_u8 一致，区别仅在于：
 * - 输入张量类型为 int8_t（而非 uint8_t）
 * - 使用 qnt_f32_to_affine / deqnt_affine_to_f32 进行量化/反量化
 * - int8 的取值范围是 [-128, 127]
 *
 * @param box_tensor 边界框特征张量（int8 量化），形状 [4*dfl_len, grid_h, grid_w]
 * @param box_zp box_tensor 的 zero-point
 * @param box_scale box_tensor 的 scale
 * @param score_tensor 类别得分特征张量（int8 量化），形状 [OBJ_CLASS_NUM, grid_h, grid_w]
 * @param score_zp,score_scale score_tensor 的量化参数
 * @param score_sum_tensor 可选：得分总和张量（int8，用于快速过滤）
 * @param score_sum_zp,score_sum_scale score_sum 的量化参数
 * @param grid_h 特征图高度
 * @param grid_w 特征图宽度
 * @param stride 该尺度的下采样倍数
 * @param dfl_len DFL 分布长度
 * @param boxes 输出：检测框坐标向量（每 4 个一组：x1, y1, w, h）
 * @param objProbs 输出：检测框置信度
 * @param classId 输出：检测框类别 ID
 * @param threshold 置信度阈值
 * @return int 该尺度的有效检测框数量
 */
static int process_i8(int8_t *box_tensor, int32_t box_zp, float box_scale,
                      int8_t *score_tensor, int32_t score_zp, float score_scale,
                      int8_t *score_sum_tensor, int32_t score_sum_zp, float score_sum_scale,
                      int grid_h, int grid_w, int stride, int dfl_len,
                      std::vector<float> &boxes,
                      std::vector<float> &objProbs,
                      std::vector<int> &classId,
                      float threshold)
{
    int validCount = 0;
    int grid_len = grid_h * grid_w;
    int8_t score_thres_i8 = qnt_f32_to_affine(threshold, score_zp, score_scale);
    int8_t score_sum_thres_i8 = qnt_f32_to_affine(threshold, score_sum_zp, score_sum_scale);

    for (int i = 0; i < grid_h; i++)
    {
        for (int j = 0; j < grid_w; j++)
        {
            int offset = i* grid_w + j;
            int max_class_id = -1;

            // 通过 score sum 起到快速过滤的作用
            if (score_sum_tensor != nullptr){
                if (score_sum_tensor[offset] < score_sum_thres_i8){
                    continue;
                }
            }

            int8_t max_score = -score_zp;
            for (int c= 0; c< OBJ_CLASS_NUM; c++){
                if ((score_tensor[offset] > score_thres_i8) && (score_tensor[offset] > max_score))
                {
                    max_score = score_tensor[offset];
                    max_class_id = c;
                }
                offset += grid_len;
            }

            // compute box
            if (max_score> score_thres_i8){
                offset = i* grid_w + j;
                float box[4];
                float before_dfl[dfl_len*4];
                for (int k=0; k< dfl_len*4; k++){
                    before_dfl[k] = deqnt_affine_to_f32(box_tensor[offset], box_zp, box_scale);
                    offset += grid_len;
                }
                compute_dfl(before_dfl, dfl_len, box);

                float x1,y1,x2,y2,w,h;
                x1 = (-box[0] + j + 0.5)*stride;
                y1 = (-box[1] + i + 0.5)*stride;
                x2 = (box[2] + j + 0.5)*stride;
                y2 = (box[3] + i + 0.5)*stride;
                w = x2 - x1;
                h = y2 - y1;
                boxes.push_back(x1);
                boxes.push_back(y1);
                boxes.push_back(w);
                boxes.push_back(h);

                objProbs.push_back(deqnt_affine_to_f32(max_score, score_zp, score_scale));
                classId.push_back(max_class_id);
                validCount ++;
            }
        }
    }
    return validCount;
}

/**
 * @brief 处理 fp32 浮点模型的单个特征图尺度，提取检测结果
 *
 * 功能与 process_u8/process_i8 相同，但处理的是未量化的 float32 浮点张量。
 * 浮点模型不需要量化/反量化操作，可直接比较和计算。
 *
 * 算法原理：
 * 与量化版本流程一致，但：
 * - 不需要预先将阈值量化（直接在浮点域比较）
 * - box 特征值直接使用（不需反量化）
 * - 类别得分直接使用（不需反量化）
 *
 * @param box_tensor 边界框特征张量（float32），形状 [4*dfl_len, grid_h, grid_w]
 * @param score_tensor 类别得分特征张量（float32），形状 [OBJ_CLASS_NUM, grid_h, grid_w]
 * @param score_sum_tensor 可选：得分总和张量（float32，用于快速过滤）
 * @param grid_h 特征图高度
 * @param grid_w 特征图宽度
 * @param stride 该尺度的下采样倍数
 * @param dfl_len DFL 分布长度
 * @param boxes 输出：检测框坐标向量（每 4 个一组：x1, y1, w, h）
 * @param objProbs 输出：检测框置信度
 * @param classId 输出：检测框类别 ID
 * @param threshold 置信度阈值
 * @return int 该尺度的有效检测框数量
 */
static int process_fp32(float *box_tensor, float *score_tensor, float *score_sum_tensor,
                        int grid_h, int grid_w, int stride, int dfl_len,
                        std::vector<float> &boxes,
                        std::vector<float> &objProbs,
                        std::vector<int> &classId,
                        float threshold)
{
    int validCount = 0;
    int grid_len = grid_h * grid_w;
    for (int i = 0; i < grid_h; i++)
    {
        for (int j = 0; j < grid_w; j++)
        {
            int offset = i* grid_w + j;
            int max_class_id = -1;

            // 通过 score sum 起到快速过滤的作用
            if (score_sum_tensor != nullptr){
                if (score_sum_tensor[offset] < threshold){
                    continue;
                }
            }

            float max_score = 0;
            for (int c= 0; c< OBJ_CLASS_NUM; c++){
                if ((score_tensor[offset] > threshold) && (score_tensor[offset] > max_score))
                {
                    max_score = score_tensor[offset];
                    max_class_id = c;
                }
                offset += grid_len;
            }

            // compute box
            if (max_score> threshold){
                offset = i* grid_w + j;
                float box[4];
                float before_dfl[dfl_len*4];
                for (int k=0; k< dfl_len*4; k++){
                    before_dfl[k] = box_tensor[offset];
                    offset += grid_len;
                }
                compute_dfl(before_dfl, dfl_len, box);

                float x1,y1,x2,y2,w,h;
                x1 = (-box[0] + j + 0.5)*stride;
                y1 = (-box[1] + i + 0.5)*stride;
                x2 = (box[2] + j + 0.5)*stride;
                y2 = (box[3] + i + 0.5)*stride;
                w = x2 - x1;
                h = y2 - y1;
                boxes.push_back(x1);
                boxes.push_back(y1);
                boxes.push_back(w);
                boxes.push_back(h);

                objProbs.push_back(max_score);
                classId.push_back(max_class_id);
                validCount ++;
            }
        }
    }
    return validCount;
}


#if defined(RV1106_1103)
/**
 * @brief 处理 RV1106/RV1103 平台的 int8 量化模型输出（特殊 NHWC 张量布局）
 *
 * RV1106/RV1103 平台的 NPU 输出张量使用 NHWC（N, H, W, C）内存布局，
 * 与通用 RKNPU2 的 NCHW 布局不同。因此元素访问的 offset 计算方式不同：
 *
 * 通用 RKNPU2（NCHW 布局）：
 *   - box 张量: [1, 4*dfl_len, grid_h, grid_w]
 *     访问 (c, h, w) 处的元素: offset = c * grid_h * grid_w + h * grid_w + w
 *   - score 张量: [1, OBJ_CLASS_NUM, grid_h, grid_w]
 *     访问类别 c 在网格 (h,w) 处的得分: offset = c * grid_h * grid_w + h * grid_w + w
 *
 * RV1106（NHWC 布局）：
 *   - box 张量: [1, grid_h, grid_w, 4*dfl_len]
 *     访问网格 (h,w) 的第 c 个 box 通道: offset = (h * grid_w + w) * (4*dfl_len) + c
 *   - score 张量: [1, grid_h, grid_w, OBJ_CLASS_NUM]
 *     访问网格 (h,w) 的类别 c 得分: offset = (h * grid_w + w) * OBJ_CLASS_NUM + c
 *
 * 因此 RV1106 版本中：
 * - 类别查找直接使用 offset = (i * grid_w + j) * OBJ_CLASS_NUM（不用跨 grid_len 步长）
 * - box 读取使用 offset = (i * grid_w + j) * 4 * dfl_len（连续读取，不用步进）
 *
 * @param box_tensor 边界框特征张量（int8），NHWC 布局 [1, grid_h, grid_w, 4*dfl_len]
 * @param score_tensor 类别得分特征张量（int8），NHWC 布局 [1, grid_h, grid_w, OBJ_CLASS_NUM]
 * @param score_sum_tensor 可选得分总和张量（int8）
 * @param grid_h 特征图高度
 * @param grid_w 特征图宽度
 * @param stride 下采样倍数
 * @param dfl_len DFL 分布长度
 * @param boxes 输出：检测框坐标向量
 * @param objProbs 输出：检测框置信度
 * @param classId 输出：检测框类别 ID
 * @param threshold 置信度阈值
 * @return int 该尺度的有效检测框数量
 */
static int process_i8_rv1106(int8_t *box_tensor, int32_t box_zp, float box_scale,
                             int8_t *score_tensor, int32_t score_zp, float score_scale,
                             int8_t *score_sum_tensor, int32_t score_sum_zp, float score_sum_scale,
                             int grid_h, int grid_w, int stride, int dfl_len,
                             std::vector<float> &boxes,
                             std::vector<float> &objProbs,
                             std::vector<int> &classId,
                             float threshold) {
    int validCount = 0;
    int grid_len = grid_h * grid_w;
    int8_t score_thres_i8 = qnt_f32_to_affine(threshold, score_zp, score_scale);
    int8_t score_sum_thres_i8 = qnt_f32_to_affine(threshold, score_sum_zp, score_sum_scale);

    for (int i = 0; i < grid_h; i++) {
        for (int j = 0; j < grid_w; j++) {
            int offset = i * grid_w + j;
            int max_class_id = -1;

            // 通过 score sum 起到快速过滤的作用
            if (score_sum_tensor != nullptr) {
                //score_sum_tensor [1, 1, 80, 80]  NHWC 布局下 shape 含义
                if (score_sum_tensor[offset] < score_sum_thres_i8) {
                    continue;
                }
            }

            int8_t max_score = -score_zp;
            // RV1106: score 张量是 NHWC 布局，所有类别的得分在连续内存中
            // shape: [1, grid_h, grid_w, OBJ_CLASS_NUM]
            offset = offset * OBJ_CLASS_NUM;
            for (int c = 0; c < OBJ_CLASS_NUM; c++) {
                if ((score_tensor[offset + c] > score_thres_i8) && (score_tensor[offset + c] > max_score)) {
                    max_score = score_tensor[offset + c]; //80类 [1, 80, 80, 80] 3588NCHW 1106NHWC
                    max_class_id = c;
                }
            }

            // compute box
            if (max_score > score_thres_i8) {
                // RV1106: box 张量是 NHWC 布局，4*dfl_len 个通道在连续内存中
                // shape: [1, grid_h, grid_w, 4*dfl_len]
                offset = (i * grid_w + j) * 4 * dfl_len;
                float box[4];
                float before_dfl[dfl_len*4];
                for (int k=0; k< dfl_len*4; k++){
                    before_dfl[k] = deqnt_affine_to_f32(box_tensor[offset + k], box_zp, box_scale);
                }
                compute_dfl(before_dfl, dfl_len, box);

                float x1, y1, x2, y2, w, h;
                x1 = (-box[0] + j + 0.5) * stride;
                y1 = (-box[1] + i + 0.5) * stride;
                x2 = (box[2] + j + 0.5) * stride;
                y2 = (box[3] + i + 0.5) * stride;
                w = x2 - x1;
                h = y2 - y1;
                boxes.push_back(x1);
                boxes.push_back(y1);
                boxes.push_back(w);
                boxes.push_back(h);

                objProbs.push_back(deqnt_affine_to_f32(max_score, score_zp, score_scale));
                classId.push_back(max_class_id);
                validCount ++;
            }
        }
    }
    printf("validCount=%d\n", validCount);
    printf("grid h-%d, w-%d, stride %d\n", grid_h, grid_w, stride);
    return validCount;
}
#endif

/**
 * @brief YOLO11 后处理主入口函数
 *
 * 整合三个尺度的特征图处理结果，执行排序、NMS、坐标映射，最终输出检测结果。
 *
 * 整体处理流程：
 * 1. 遍历 YOLO11 的三个输出分支（大/中/小目标各对应一个下采样尺度）
 * 2. 对每个分支，根据数据类型（量化/浮点）和平台（RKNPU1/RKNPU2/RV1106）选择对应的处理函数
 * 3. 将所有尺度的检测结果汇总到 filterBoxes/objProbs/classId 向量中
 * 4. 如果有效检测数为 0，提前返回
 * 5. 对置信度降序排序（quick_sort_indice_inverse）
 * 6. 按类别分别执行 NMS，去除重复检测
 * 7. 将检测框坐标从模型输入空间映射回原始图像空间（去除 letterbox 填充并缩放）
 *
 * 坐标映射公式：
 *   原始图坐标 = clamp(模型坐标 - pad偏移, 0, 模型尺寸) / 缩放系数
 *   其中 pad 偏移和缩放系数由 letterbox 预处理时计算得出，记录在 letter_box 中
 *
 * @param app_ctx RKNN 应用上下文，包含模型输入尺寸、输出张量属性等信息
 * @param outputs 模型推理输出的张量数据指针（rknn_output* 或 rknn_tensor_mem**）
 * @param letter_box letterbox 预处理参数（pad 偏移和缩放系数）
 * @param conf_threshold 置信度阈值，低于此值的检测将被过滤
 * @param nms_threshold NMS 的 IoU 阈值
 * @param od_results 输出：最终的目标检测结果列表（坐标映射回原图空间）
 * @return int 成功返回 0，RV1106 非量化模式返回 -1
 */
int post_process(rknn_app_context_t *app_ctx, void *outputs, letterbox_t *letter_box, float conf_threshold, float nms_threshold, object_detect_result_list *od_results)
{
#if defined(RV1106_1103)
    // RV1106/RV1103 平台使用 rknn_tensor_mem**（零拷贝模式）
    rknn_tensor_mem **_outputs = (rknn_tensor_mem **)outputs;
#else
    // 通用平台使用 rknn_output*（普通模式）
    rknn_output *_outputs = (rknn_output *)outputs;
#endif
    std::vector<float> filterBoxes; // 汇聚所有尺度的检测框坐标 (x1, y1, w, h)
    std::vector<float> objProbs;    // 汇聚所有尺度的置信度
    std::vector<int> classId;       // 汇聚所有尺度的类别 ID
    int validCount = 0;             // 跨三个尺度的总有效检测数
    int stride = 0;                 // 当前分支的下采样步长
    int grid_h = 0;                 // 当前分支的特征图高度
    int grid_w = 0;                 // 当前分支的特征图宽度
    int model_in_w = app_ctx->model_width;  // 模型输入宽度
    int model_in_h = app_ctx->model_height; // 模型输入高度

    memset(od_results, 0, sizeof(object_detect_result_list));

    // 默认 3 个输出分支（大/中/小目标各一个），但每个分支可能包含 2~3 个输出张量
#ifdef RKNPU1
    int dfl_len = app_ctx->output_attrs[0].dims[2] / 4;
#else
    int dfl_len = app_ctx->output_attrs[0].dims[1] /4;
#endif
    int output_per_branch = app_ctx->io_num.n_output / 3; // 每个分支的输出张量数（2 或 3）
    for (int i = 0; i < 3; i++)
    {
#if defined(RV1106_1103)
        dfl_len = app_ctx->output_attrs[0].dims[3] /4;
        void *score_sum = nullptr;
        int32_t score_sum_zp = 0;
        float score_sum_scale = 1.0;
        // 如果每个分支有 3 个输出（含 score_sum），则读取 score_sum 张量
        if (output_per_branch == 3) {
            score_sum = _outputs[i * output_per_branch + 2]->virt_addr;
            score_sum_zp = app_ctx->output_attrs[i * output_per_branch + 2].zp;
            score_sum_scale = app_ctx->output_attrs[i * output_per_branch + 2].scale;
        }
        int box_idx = i * output_per_branch;      // box 输出索引
        int score_idx = i * output_per_branch + 1; // score 输出索引
        grid_h = app_ctx->output_attrs[box_idx].dims[1];
        grid_w = app_ctx->output_attrs[box_idx].dims[2];
        stride = model_in_h / grid_h;

        if (app_ctx->is_quant) {
            validCount += process_i8_rv1106((int8_t *)_outputs[box_idx]->virt_addr, app_ctx->output_attrs[box_idx].zp, app_ctx->output_attrs[box_idx].scale,
                                (int8_t *)_outputs[score_idx]->virt_addr, app_ctx->output_attrs[score_idx].zp,
                                app_ctx->output_attrs[score_idx].scale, (int8_t *)score_sum, score_sum_zp, score_sum_scale,
                                grid_h, grid_w, stride, dfl_len, filterBoxes, objProbs, classId, conf_threshold);
        }
        else
        {
            printf("RV1106/1103 only support quantization mode\n", LABEL_NALE_TXT_PATH);
            return -1;
        }

#else
        void *score_sum = nullptr;
        int32_t score_sum_zp = 0;
        float score_sum_scale = 1.0;
        if (output_per_branch == 3){
            score_sum = _outputs[i*output_per_branch + 2].buf;
            score_sum_zp = app_ctx->output_attrs[i*output_per_branch + 2].zp;
            score_sum_scale = app_ctx->output_attrs[i*output_per_branch + 2].scale;
        }
        int box_idx = i*output_per_branch;
        int score_idx = i*output_per_branch + 1;

#ifdef RKNPU1
        grid_h = app_ctx->output_attrs[box_idx].dims[1];
        grid_w = app_ctx->output_attrs[box_idx].dims[0];
#else
        grid_h = app_ctx->output_attrs[box_idx].dims[2];
        grid_w = app_ctx->output_attrs[box_idx].dims[3];
#endif
        stride = model_in_h / grid_h; // stride = 输入尺寸 / 特征图尺寸

        if (app_ctx->is_quant)
        {
#ifdef RKNPU1
            // RKNPU1 使用 uint8 量化
            validCount += process_u8((uint8_t *)_outputs[box_idx].buf, app_ctx->output_attrs[box_idx].zp, app_ctx->output_attrs[box_idx].scale,
                                     (uint8_t *)_outputs[score_idx].buf, app_ctx->output_attrs[score_idx].zp, app_ctx->output_attrs[score_idx].scale,
                                     (uint8_t *)score_sum, score_sum_zp, score_sum_scale,
                                     grid_h, grid_w, stride, dfl_len,
                                     filterBoxes, objProbs, classId, conf_threshold);
#else
            // RKNPU2 使用 int8 量化
            validCount += process_i8((int8_t *)_outputs[box_idx].buf, app_ctx->output_attrs[box_idx].zp, app_ctx->output_attrs[box_idx].scale,
                                     (int8_t *)_outputs[score_idx].buf, app_ctx->output_attrs[score_idx].zp, app_ctx->output_attrs[score_idx].scale,
                                     (int8_t *)score_sum, score_sum_zp, score_sum_scale,
                                     grid_h, grid_w, stride, dfl_len,
                                     filterBoxes, objProbs, classId, conf_threshold);
#endif
        }
        else
        {
            // 非量化（fp32）模式
            validCount += process_fp32((float *)_outputs[box_idx].buf, (float *)_outputs[score_idx].buf, (float *)score_sum,
                                       grid_h, grid_w, stride, dfl_len,
                                       filterBoxes, objProbs, classId, conf_threshold);
        }
#endif
    }

    // 如果三个尺度都没有有效检测，提前返回
    if (validCount <= 0)
    {
        return 0;
    }

    // 创建索引数组并按照置信度降序排序
    // 排序后 indexArray 的前面是高置信度检测的原始索引
    std::vector<int> indexArray;
    for (int i = 0; i < validCount; ++i)
    {
        indexArray.push_back(i);
    }
    quick_sort_indice_inverse(objProbs, 0, validCount - 1, indexArray);

    // 收集所有检测中出现的类别 ID（去重），为每类分别执行 NMS
    std::set<int> class_set(std::begin(classId), std::end(classId));

    // 按类别分别执行非极大值抑制
    for (auto c : class_set)
    {
        nms(validCount, filterBoxes, classId, indexArray, c, nms_threshold);
    }

    int last_count = 0;
    od_results->count = 0;

    /* 遍历排序后的检测结果，将有效的边界框坐标映射回原始图像空间 */
    for (int i = 0; i < validCount; ++i)
    {
        // 跳过被 NMS 抑制的框（indexArray[i] == -1）或结果已满
        if (indexArray[i] == -1 || last_count >= OBJ_NUMB_MAX_SIZE)
        {
            continue;
        }
        int n = indexArray[i];

        // 去除 letterbox 填充偏移（x_pad, y_pad），得到模型输入空间中的坐标
        float x1 = filterBoxes[n * 4 + 0] - letter_box->x_pad;
        float y1 = filterBoxes[n * 4 + 1] - letter_box->y_pad;
        float x2 = x1 + filterBoxes[n * 4 + 2];
        float y2 = y1 + filterBoxes[n * 4 + 3];
        int id = classId[n];
        float obj_conf = objProbs[i];

        // 将坐标从模型输入空间映射回原始图像空间
        // clamp 确保坐标不越界，除以 scale 还原 letterbox 的缩放
        od_results->results[last_count].box.left = (int)(clamp(x1, 0, model_in_w) / letter_box->scale);
        od_results->results[last_count].box.top = (int)(clamp(y1, 0, model_in_h) / letter_box->scale);
        od_results->results[last_count].box.right = (int)(clamp(x2, 0, model_in_w) / letter_box->scale);
        od_results->results[last_count].box.bottom = (int)(clamp(y2, 0, model_in_h) / letter_box->scale);
        od_results->results[last_count].prop = obj_conf;
        od_results->results[last_count].cls_id = id;
        last_count++;
    }
    od_results->count = last_count;
    return 0;
}

/**
 * @brief 初始化后处理模块（加载 COCO 类别标签）
 *
 * 从指定路径的标签文件中读取所有类别名称，存入全局 labels 数组。
 * 必须在调用 coco_cls_to_name() 和 post_process() 之前调用此函数。
 *
 * @return int 成功返回 0，标签文件加载失败返回 -1
 */
int init_post_process()
{
    int ret = 0;
    ret = loadLabelName(LABEL_NALE_TXT_PATH, labels);
    if (ret < 0)
    {
        printf("Load %s failed!\n", LABEL_NALE_TXT_PATH);
        return -1;
    }
    return 0;
}

/**
 * @brief 将 COCO 类别 ID 转换为可读的类别名称字符串
 *
 * 通过类别 ID 查询全局 labels 数组，返回对应的类别名。
 * 如果 ID 超出范围或对应标签未加载，返回 "null"。
 *
 * @param cls_id 类别 ID（COCO 数据集范围为 0~79）
 * @return char* 类别名称字符串，失败返回 "null"
 */
char *coco_cls_to_name(int cls_id)
{

    if (cls_id >= OBJ_CLASS_NUM)
    {
        return "null";
    }

    if (labels[cls_id])
    {
        return labels[cls_id];
    }

    return "null";
}

/**
 * @brief 反初始化后处理模块（释放标签内存）
 *
 * 释放所有通过 readLine/readLines 动态分配的标签字符串内存，
 * 并将指针置空，防止野指针。
 * 在程序退出或不再需要后处理功能时调用。
 */
void deinit_post_process()
{
    for (int i = 0; i < OBJ_CLASS_NUM; i++)
    {
        if (labels[i] != nullptr)
        {
            free(labels[i]);
            labels[i] = nullptr;
        }
    }
}
