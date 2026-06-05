#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO11 图片检测脚本（支持 PyTorch / ONNX / RKNN）

功能说明：
    对输入图片进行目标检测，识别图片中的物体并标注位置和类别。
    支持三种模型格式：PyTorch(.pt)、ONNX(.onnx)、RKNN(.rknn)

使用方法（在终端中运行）：
    cd /home/ztl/dltt/code/rknn/rknn_model_zoo-v2.3.0/examples/yolo11/python

    # PyTorch 模型推理（显示结果）
    python yolo11.py --model_path ../model/yolo11n.pt --img_show

    # ONNX 模型推理（保存结果）
    python yolo11.py --model_path ../model/yolo11n.onnx --img_save

    # 指定图片目录
    python yolo11.py --model_path ../model/yolo11n.pt --img_folder ../model --img_save

    # COCO mAP 评测（测试模型准确度）
    python yolo11.py --model_path ../model/yolo11n.pt --coco_map_test

参数说明:
    --model_path      模型路径 (.pt / .onnx / .rknn)，必填
    --target          目标平台 (default: rk3588)
    --img_show        显示检测结果图像
    --img_save        保存结果到 ./result/
    --img_folder      图片目录 (default: ../model)
    --coco_map_test   启用 COCO mAP 评测
"""

import os
import cv2
import sys
import argparse

# ============================================
# 【路径设置】让 Python 能找到项目中的工具库
# ============================================
realpath = os.path.abspath(__file__)
_sep = os.path.sep
realpath = realpath.split(_sep)
# 在路径中查找 rknn_model_zoo-v2.3.0，如果找不到则使用父目录
try:
    zoo_idx = realpath.index('rknn_model_zoo-v2.3.0')
    sys.path.append(os.path.join(realpath[0]+_sep, *realpath[1:zoo_idx+1]))
except ValueError:
    # 如果未找到，添加父目录
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(realpath))))

from py_utils.coco_utils import COCO_test_helper
import numpy as np


# ============================================
# 【全局参数】检测的"门槛"设置
# ============================================

# 目标检测置信度阈值：低于此分数的检测框会被过滤
OBJ_THRESH = 0.25

# 非极大值抑制（NMS）的 IoU 阈值：重叠度高于此值的框会被抑制
NMS_THRESH = 0.45

# 下面两个参数用于 mAP 测试（更低的阈值可以获取更多检测结果）
# OBJ_THRESH = 0.001
# NMS_THRESH = 0.65

# 输入图像尺寸（宽度，高度）
IMG_SIZE = (640, 640)  # (width, height), such as (1280, 736)

# ============================================
# 【类别定义】模型能识别的 80 种物体
# ============================================
# 这是 COCO 数据集的标准类别
CLASSES = ("person", "bicycle", "car","motorbike ","aeroplane ","bus ","train","truck ","boat","traffic light",
           "fire hydrant","stop sign ","parking meter","bench","bird","cat","dog ","horse ","sheep","cow","elephant",
           "bear","zebra ","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
           "baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife ",
           "spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza ","donut","cake","chair","sofa",
           "pottedplant","bed","diningtable","toilet ","tvmonitor","laptop	","mouse	","remote ","keyboard ","cell phone","microwave ",
           "oven ","toaster","sink","refrigerator ","book","clock","vase","scissors ","teddy bear ","hair drier", "toothbrush ")

# COCO 数据集类别 ID 列表（与 CLASSES 对应，COCO 的 ID 不是连续的）
coco_id_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
         35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
         64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]


# ============================================
# 【函数1】过滤低置信度的检测框
# ============================================
def filter_boxes(boxes, box_confidences, box_class_probs):
    """
    根据目标置信度阈值过滤检测框
    
    参数:
        boxes: 检测框坐标数组，每个框有4个值 [x1, y1, x2, y2]
        box_confidences: 目标置信度分数
        box_class_probs: 各类别概率
    
    返回:
        boxes: 过滤后的检测框
        classes: 过滤后的类别索引
        scores: 过滤后的最终分数（类别概率 × 目标置信度）
    """
    # 将置信度展平为一维数组
    box_confidences = box_confidences.reshape(-1)
    candidate, class_num = box_class_probs.shape

    # 找出每个框的最高类别概率及其对应的类别索引
    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)

    # 筛选出 类别概率 × 目标置信度 超过阈值的框
    _class_pos = np.where(class_max_score * box_confidences >= OBJ_THRESH)
    scores = (class_max_score * box_confidences)[_class_pos]

    # 提取符合条件的框、类别和分数
    boxes = boxes[_class_pos]
    classes = classes[_class_pos]

    return boxes, classes, scores


# ============================================
# 【函数2】非极大值抑制（NMS）- 去除重复框
# ============================================
def nms_boxes(boxes, scores):
    """
    非极大值抑制（Non-Maximum Suppression）：去除重叠的检测框

    原理：保留分数最高的框，抑制与其重叠度（IoU）过高的其他框。
    
    参数:
        boxes: 检测框数组
        scores: 对应的分数

    返回:
        keep: 保留的框的索引数组
    """
    # 提取框的坐标和宽高
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    # 计算每个框的面积
    areas = w * h
    # 按分数降序排序，优先处理高分框
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        # 取出当前分数最高的框
        i = order[0]
        keep.append(i)

        # 计算当前框与其他框的交集区域坐标
        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        # 计算交集区域的宽高和面积（最小为0）
        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        # 计算 IoU（交并比）= 交集 / 并集
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        # 保留 IoU 小于阈值的框
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]

    keep = np.array(keep)
    return keep


# ============================================
# 【函数3】DFL解码 - YOLOv8/v11的核心技术
# ============================================
def dfl(position):
    """
    分布焦点损失（Distribution Focal Loss, DFL）解码

    YOLOv8/v11 使用 DFL 来表示边界框的位置分布，将分布预测转换为具体的坐标偏移量。
    
    参数:
        position: 位置预测张量，形状为 (n, c, h, w)，其中 c 是分布通道数

    返回:
        解码后的位置偏移量数组
    """
    import torch
    x = torch.tensor(position)
    n, c, h, w = x.shape
    p_num = 4  # 四个方向：左、上、右、下
    mc = c // p_num  # 每个方向的分布通道数

    # 重塑为 (n, 4, mc, h, w)
    y = x.reshape(n, p_num, mc, h, w)
    # 对分布维度进行 softmax，得到概率分布
    y = y.softmax(2)
    # 创建权重矩阵 [0, 1, 2, ..., mc-1]
    acc_metrix = torch.tensor(range(mc)).float().reshape(1, 1, mc, 1, 1)
    # 加权求和，得到期望值
    y = (y * acc_metrix).sum(2)
    return y.numpy()


# ============================================
# 【函数4】边界框后处理 - 把模型输出变成实际坐标
# ============================================
def box_process(position):
    """
    边界框后处理：将模型输出的位置分布转换为实际的边界框坐标

    参数:
        position: 模型输出的位置预测，形状为 (1, c, h, w)

    返回:
        xyxy: 转换后的边界框坐标，格式为 (x1, y1, x2, y2)
              即 [左上角x, 左上角y, 右下角x, 右下角y]
    """
    # 获取特征图的高度和宽度
    grid_h, grid_w = position.shape[2:4]
    # 生成网格坐标：col 表示 x 方向，row 表示 y 方向
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    # 合并为网格坐标 (1, 2, h, w)，其中 [0] 是 x，[1] 是 y
    grid = np.concatenate((col, row), axis=1)
    # 计算步长（原图尺寸 / 特征图尺寸）
    stride = np.array([IMG_SIZE[1]//grid_h, IMG_SIZE[0]//grid_w]).reshape(1, 2, 1, 1)

    # 通过 DFL 解码位置分布
    position = dfl(position)
    # 计算边界框的中心点偏移：grid + 0.5 - position[0:2]
    box_xy = grid + 0.5 - position[:, 0:2, :, :]
    # 计算边界框的对角点偏移：grid + 0.5 + position[2:4]
    box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
    # 乘以步长，映射回原图尺寸
    xyxy = np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)

    return xyxy


# ============================================
# 【函数5】完整的后处理流程 - 串联所有步骤
# ============================================
def post_process(input_data):
    """
    YOLO11 模型输出后处理：解码检测框、过滤、NMS

    YOLO11 有三个检测头（大、中、小目标），每个检测头输出：
    - 位置分布（用于 DFL 解码）
    - 类别概率
    - 目标置信度（这里用全 1 替代）

    参数:
        input_data: 模型输出列表，包含 9 个元素（3 个检测头 × 3 种输出）

    返回:
        boxes: 最终的检测框数组
        classes: 对应的类别索引
        scores: 对应的分数
    """
    boxes, scores, classes_conf = [], [], []
    defualt_branch = 3  # 三个检测头
    pair_per_branch = len(input_data) // defualt_branch  # 每个检测头的输出数量

    # 对每个检测头的位置输出进行解码
    for i in range(defualt_branch):
        boxes.append(box_process(input_data[pair_per_branch * i]))
        # 提取类别概率
        classes_conf.append(input_data[pair_per_branch * i + 1])
        # 目标置信度用全 1 替代（优化后的模型已移除该分支）
        scores.append(np.ones_like(input_data[pair_per_branch * i + 1][:, :1, :, :], dtype=np.float32))

    # 定义空间展平函数：将 (n, c, h, w) 转为 (-1, c)
    def sp_flatten(_in):
        ch = _in.shape[1]
        _in = _in.transpose(0, 2, 3, 1)  # 转为 (n, h, w, c)
        return _in.reshape(-1, ch)  # 展平为 (-1, c)

    # 对所有检测头的输出进行展平
    boxes = [sp_flatten(_v) for _v in boxes]
    classes_conf = [sp_flatten(_v) for _v in classes_conf]
    scores = [sp_flatten(_v) for _v in scores]

    # 拼接所有检测头的结果
    boxes = np.concatenate(boxes)
    classes_conf = np.concatenate(classes_conf)
    scores = np.concatenate(scores)

    # 根据阈值过滤低分检测框
    boxes, classes, scores = filter_boxes(boxes, scores, classes_conf)

    # 按类别分别进行 NMS（去重）
    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        # 找出当前类别的所有框
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]
        # 对当前类别的框进行 NMS
        keep = nms_boxes(b, s)

        if len(keep) != 0:
            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])

    # 如果没有检测到任何目标，返回 None
    if not nclasses and not nscores:
        return None, None, None

    # 合并所有类别的检测结果
    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    return boxes, classes, scores


# ============================================
# 【函数6】绘制检测结果 - 在图片上画框和标签
# ============================================
def draw(image, boxes, scores, classes):
    """
    在图像上绘制检测结果（边界框、类别标签、分数）
    
    参数:
        image: 原始图像
        boxes: 检测框数组
        scores: 分数数组
        classes: 类别索引数组
    """
    for box, score, cl in zip(boxes, scores, classes):
        # 将浮点数坐标转换为整数
        top, left, right, bottom = [int(_b) for _b in box]
        # 打印检测结果到终端
        print("%s @ (%d %d %d %d) %.3f" % (CLASSES[cl], top, left, right, bottom, score))
        # 绘制矩形框（蓝色，线宽 2）
        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        # 绘制类别标签和分数（红色文字）
        cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                    (top, left - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


# ============================================
# 【函数7】模型加载器 - 根据文件类型选择加载方式
# ============================================
def setup_model(args):
    """
    根据模型文件类型初始化模型容器

    支持三种模型格式：
    - PyTorch (.pt/.torchscript) - 原始训练好的模型
    - RKNN (.rknn) - 为 Rockchip NPU 优化的模型
    - ONNX (.onnx) - 通用格式的模型

    参数:
        args: 命令行参数

    返回:
        model: 模型容器对象
        platform: 平台名称字符串（'pytorch'、'rknn' 或 'onnx'）
    """
    model_path = args.model_path
    if model_path.endswith('.pt') or model_path.endswith('.torchscript'):
        platform = 'pytorch'
        from py_utils.pytorch_executor import Torch_model_container
        model = Torch_model_container(args.model_path)
    elif model_path.endswith('.rknn'):
        platform = 'rknn'
        from py_utils.rknn_executor import RKNN_model_container
        model = RKNN_model_container(args.model_path, args.target, args.device_id)
    elif model_path.endswith('onnx'):
        platform = 'onnx'
        from py_utils.onnx_executor import ONNX_model_container
        model = ONNX_model_container(args.model_path)
    else:
        print('Model-{} is not rknn/pytorch/onnx model'.format(model_path))
        assert False, "{} is not rknn/pytorch/onnx model".format(model_path)
    print('Model-{} is {} model, starting val'.format(model_path, platform))
    return model, platform


# ============================================
# 【函数8】图片格式检查
# ============================================
def img_check(path):
    """
    检查文件是否为支持的图片格式

    支持的格式：.jpg, .jpeg, .png, .bmp（不区分大小写）

    参数:
        path: 文件路径

    返回:
        True: 是图片文件
        False: 不是图片文件
    """
    img_type = ['.jpg', '.jpeg', '.png', '.bmp']
    for _type in img_type:
        if path.endswith(_type) or path.endswith(_type.upper()):
            return True
    return False


# ============================================
# 【主程序】从这里开始执行！
# ============================================
if __name__ == '__main__':
    # ============================================
    # 步骤1：解析命令行参数
    # ============================================
    parser = argparse.ArgumentParser(description='Process some integers.')

    # ===== 基本参数 =====
    parser.add_argument('--model_path', type=str, required=True, help='模型路径，可以是 .pt、.rknn 或 .onnx 文件')
    parser.add_argument('--target', type=str, default='rk3588', help='目标 RKNPU 平台，如 rk3566、rk3588 等')
    parser.add_argument('--device_id', type=str, default=None, help='设备 ID（多设备时指定）')

    parser.add_argument('--img_show', action='store_true', default=False, help='显示检测结果图像')
    parser.add_argument('--img_save', action='store_true', default=False, help='保存检测结果图像到 ./result 目录')

    # ===== 数据参数 =====
    parser.add_argument('--anno_json', type=str, default='../../../datasets/COCO/annotations/instances_val2017.json', help='COCO 标注文件路径')
    # COCO 验证集文件夹路径：'../../../datasets/COCO//val2017'
    parser.add_argument('--img_folder', type=str, default='../model', help='待检测图片文件夹路径')
    parser.add_argument('--coco_map_test', action='store_true', help='启用 COCO mAP 测试模式')

    args = parser.parse_args()

    # ============================================
    # 步骤2：加载模型
    # ============================================
    model, platform = setup_model(args)

    # ============================================
    # 步骤3：获取待检测的图片列表
    # ============================================
    file_list = sorted(os.listdir(args.img_folder))
    img_list = []
    for path in file_list:
        if img_check(path):
            img_list.append(path)

    # 创建 COCO 测试助手（用于 letterbox 处理和 mAP 评估）
    co_helper = COCO_test_helper(enable_letter_box=True)

    # ============================================
    # 步骤4：遍历所有图片进行推理
    # ============================================
    for i in range(len(img_list)):
        print('infer {}/{}'.format(i+1, len(img_list)), end='\r')

        img_name = img_list[i]
        img_path = os.path.join(args.img_folder, img_name)
        if not os.path.exists(img_path):
            print("{} is not found", img_name)
            continue

        # 读取图像
        img_src = cv2.imread(img_path)
        if img_src is None:
            continue

        '''
        # 用于测试 C Demo 导出的输入数据
        img_src = np.fromfile('./input_b/demo_c_input_hwc_rgb.txt', dtype=np.uint8).reshape(640,640,3)
        img_src = cv2.cvtColor(img_src, cv2.COLOR_RGB2BGR)
        '''

        # ============================================
        # 步骤4.1：预处理 - Letterbox 缩放
        # ============================================
        # 保持宽高比地将图片缩放到 640x640，不足部分用黑色填充
        pad_color = (0, 0, 0)
        img = co_helper.letter_box(im=img_src.copy(), new_shape=(IMG_SIZE[1], IMG_SIZE[0]), pad_color=(0, 0, 0))
        # BGR 转 RGB（OpenCV 使用 BGR 顺序，模型使用 RGB 顺序）
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # ============================================
        # 步骤4.2：预处理 - 格式转换和归一化
        # ============================================
        if platform in ['pytorch', 'onnx']:
            # 转为 CHW 格式并添加 batch 维度
            input_data = img.transpose((2, 0, 1))
            input_data = input_data.reshape(1, *input_data.shape).astype(np.float32)
            # 归一化到 [0, 1]
            input_data = input_data / 255.
        else:
            # RKNN 模型已在内部处理归一化（通过 mean_values 和 std_values）
            input_data = img

        # ============================================
        # 步骤4.3：执行推理
        # ============================================
        outputs = model.run([input_data])
        
        # ============================================
        # 步骤4.4：后处理
        # ============================================
        boxes, classes, scores = post_process(outputs)

        # ============================================
        # 步骤5：显示或保存结果
        # ============================================
        if args.img_show or args.img_save:
            print('\n\nIMG: {}'.format(img_name))
            img_p = img_src.copy()
            if boxes is not None:
                # 将检测框从 letterbox 坐标还原到原图坐标
                draw(img_p, co_helper.get_real_box(boxes), scores, classes)

            # 保存结果图像
            if args.img_save:
                if not os.path.exists('./result'):
                    os.mkdir('./result')
                result_path = os.path.join('./result', img_name)
                cv2.imwrite(result_path, img_p)
                print('Detection result save to {}'.format(result_path))

            # 显示结果图像
            if args.img_show:
                cv2.imshow("full post process result", img_p)
                print("按任意键继续下一张，按 'q' 退出...")
                key = cv2.waitKey(0) & 0xFF
                if key == ord('q'):
                    cv2.destroyAllWindows()
                    model.release()
                    print("用户退出")
                    exit(0)

        # ============================================
        # 步骤6：记录 mAP 测试结果（可选）
        # ============================================
        if args.coco_map_test is True:
            if boxes is not None:
                 # COCO mAP 测试要求图片文件名为数字（如 000001.jpg）
                try:
                    image_id = int(img_name.split('.')[0])
                except ValueError:
                    print(f'\n警告: 文件名 "{img_name}" 不是数字格式，跳过 COCO mAP 记录')
                    continue
                for i in range(boxes.shape[0]):
                    co_helper.add_single_record(
                        image_id=image_id,
                        category_id=coco_id_list[int(classes[i])],
                        bbox=boxes[i],
                        score=round(scores[i], 5).item()
                    )

    # ============================================
    # 步骤7：计算并输出 mAP（如果启用了评测模式）
    # ============================================
    if args.coco_map_test is True:
        pred_json = args.model_path.split('.')[-2] + '_{}'.format(platform) + '.json'
        pred_json = pred_json.split('/')[-1]
        pred_json = os.path.join('./', pred_json)
        co_helper.export_to_json(pred_json)

        from py_utils.coco_utils import coco_eval_with_json
        coco_eval_with_json(args.anno_json, pred_json)

    # ============================================
    # 步骤8：释放资源
    # ============================================
    cv2.destroyAllWindows()
    model.release()
