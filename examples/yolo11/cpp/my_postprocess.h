// ============================================================================
// my_postprocess.h — YOLO11 后处理声明（带中文注释）
//
// 【本文件作用】
//   声明后处理相关的数据结构、常量和函数。
//   后处理负责将 NPU 推理的原始输出转换为有意义的检测结果：
//     NPU 输出 tensor（浮点或量化 int8）
//       → 解码框坐标（DFL 解码）
//       → 找每个网格最高分的类别
//       → NMS 非极大值抑制去重
//       → letterbox 坐标反算到原图
//       → object_detect_result_list
//
// 【数据结构关系】
//   post_process() 接收 rknn_app_context_t + 原始输出
//       ↓
//   内部调用 process_i8 / process_fp32 / process_u8 处理每个分支
//       ↓
//   NMS 去重
//       ↓
//   object_detect_result_list（最终结果，给 main.cc 画框用）
// ============================================================================

#ifndef _RKNN_MY_YOLO11_DEMO_POSTPROCESS_H_
#define _RKNN_MY_YOLO11_DEMO_POSTPROCESS_H_

#include <stdint.h>
#include <vector>
#include "rknn_api.h"
#include "common.h"
#include "image_utils.h"  // 定义 image_buffer_t、image_rect_t、letterbox_t 等

// ============================================================================
// 常量（可按需调整）
// ============================================================================

#define OBJ_NAME_MAX_SIZE 64    // 类别名称字符串最大长度
#define OBJ_NUMB_MAX_SIZE 128   // 单张图最多检测的目标数
#define OBJ_CLASS_NUM 80        // COCO 数据集类别数（YOLO11 预训练模型默认 80 类）

#define NMS_THRESH 0.45         // NMS 阈值：两个框 IoU 超过此值认为是同一个目标
#define BOX_THRESH 0.25         // 置信度阈值：低于此值的检测结果被过滤

// ============================================================================
// 数据结构
// ============================================================================

/**
 * 单个检测结果
 * box 为原图坐标系下的矩形框（已从 letterbox 缩放宽高反算回原图）
 */
typedef struct {
    image_rect_t box;  // 矩形框 {left, top, right, bottom}，原点在输入图片的左上角
    float prop;        // 置信度（0~1）
    int cls_id;        // 类别 ID（0~79，对应 COCO 80 类）
} object_detect_result;

/**
 * 单张图片的所有检测结果
 * results 数组按置信度从高到低排列（在 post_process 中快排的）
 */
typedef struct {
    int id;                                     // 保留字段（当前未使用）
    int count;                                  // 实际检测到的目标数量（<= OBJ_NUMB_MAX_SIZE）
    object_detect_result results[OBJ_NUMB_MAX_SIZE];  // 检测结果数组
} object_detect_result_list;

// ============================================================================
// 函数声明
// ============================================================================

/**
 * @brief 初始化后处理（加载 COCO 类别标签文件）
 * 从 ./model/coco_80_labels_list.txt 读取 80 个类别名称
 * 如果标签文件不存在，后处理仍然可以运行但类别名显示为 "null"
 *
 * @return 0 成功，-1 失败（标签文件不存在会提示但继续执行）
 */
int init_post_process();

/**
 * @brief 释放后处理资源（释放 labels 数组中 malloc 的字符串）
 */
void deinit_post_process();

/**
 * @brief 根据类别 ID 获取类别名称
 * @param cls_id 类别 ID（0~79）
 * @return 类别名称字符串指针（如 "person", "car"），无效 ID 返回 "null"
 */
char *coco_cls_to_name(int cls_id);

/**
 * @brief 【核心后处理函数】将 NPU 输出解码为检测结果
 *
 * 处理流程：
 *   1. 遍历 3 个检测分支（大/中/小目标各一个）
 *   2. 对每个分支，遍历 grid_h x grid_w 网格
 *   3. 对每个网格，找 80 类中最高分（超过 BOX_THRESH）
 *   4. 用 DFL（Distribution Focal Loss）解码框坐标
 *   5. 将框坐标从网格坐标转为像素坐标（乘以 stride）
 *   6. 收集所有有效检测，按置信度快排
 *   7. 逐类做 NMS 去重
 *   8. 将坐标从 letterbox 坐标系反算回原图坐标系
 *
 * @param app_ctx          模型上下文（含维度、量化参数）
 * @param outputs          原始输出指针
 *                         标准模式：rknn_output* 类型（非 RV1106/1103）
 *                         零拷贝模式：rknn_tensor_mem** 类型（RV1106/1103）
 *                         函数内部通过 #if defined(RV1106_1103) 区分
 * @param letter_box       letterbox 信息（偏移量 + 缩放比例）
 * @param conf_threshold   置信度阈值（通常 BOX_THRESH = 0.25）
 * @param nms_threshold    NMS 阈值（通常 NMS_THRESH = 0.45）
 * @param od_results       [out] 最终检测结果
 * @return 0 成功
 */
int post_process(rknn_app_context_t *app_ctx, void *outputs, letterbox_t *letter_box,
                 float conf_threshold, float nms_threshold,
                 object_detect_result_list *od_results);

#endif //_RKNN_MY_YOLO11_DEMO_POSTPROCESS_H_
