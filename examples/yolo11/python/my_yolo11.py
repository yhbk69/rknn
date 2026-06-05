#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO11 视频检测脚本（支持 ONNX / RKNN）- Construction-PPE 安全帽检测

注意：不支持PT文件，因为post_process函数与PyTorch模型输出格式不兼容

使用方法:
    cd /home/ztl/dltt/code/rknn/rknn_model_zoo-v2.3.0/examples/yolo11/python

    # ONNX 模型视频检测（安全帽检测）
    python my_yolo11.py --model_path ../model/helmet_13.onnx --video_path ../003.avi

    # RKNN 模型视频检测（指定目标平台）
    python my_yolo11.py --model_path ../model/helmet_13.rknn --video_path ../003.avi --target rk3588

    # 保存检测结果视频
    python my_yolo11.py --model_path ../model/helmet_13.onnx --video_path ../003.avi --video_save

参数说明:
    --model_path      模型路径 (.onnx / .rknn)，必填
    --video_path      视频路径，必填
    --target          目标平台 (default: rk3588)
    --video_save      保存结果视频到 ./result/
"""

import os
import cv2
import sys
import argparse
import time

# 添加路径：将 rknn_model_zoo 根目录添加到系统路径中
realpath = os.path.abspath(__file__)
_sep = os.path.sep
realpath = realpath.split(_sep)
try:
    zoo_idx = realpath.index('rknn_model_zoo-v2.3.0')
    sys.path.append(os.path.join(realpath[0]+_sep, *realpath[1:zoo_idx+1]))
except ValueError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(realpath))))

from py_utils.coco_utils import COCO_test_helper
import numpy as np


# 目标检测置信度阈值
OBJ_THRESH = 0.25
# 非极大值抑制（NMS）的 IoU 阈值
NMS_THRESH = 0.45

# 输入图像尺寸
IMG_SIZE = (640, 640)

# Construction-PPE 数据集的 11 个类别名称
CLASSES = ("helmet", "gloves", "vest", "boots", "goggles",
           "none", "Person", "no_helmet", "no_goggle", "no_gloves", "no_boots")


def filter_boxes(boxes, box_confidences, box_class_probs):
    """根据目标置信度阈值过滤检测框"""
    box_confidences = box_confidences.reshape(-1)
    candidate, class_num = box_class_probs.shape

    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)

    _class_pos = np.where(class_max_score * box_confidences >= OBJ_THRESH)
    scores = (class_max_score * box_confidences)[_class_pos]

    boxes = boxes[_class_pos]
    classes = classes[_class_pos]

    return boxes, classes, scores


def nms_boxes(boxes, scores):
    """非极大值抑制（NMS）：去除重叠的检测框"""
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]

    keep = np.array(keep)
    return keep


def dfl(position):
    """分布焦点损失（DFL）解码"""
    import torch
    x = torch.tensor(position)
    n, c, h, w = x.shape
    p_num = 4
    mc = c // p_num

    y = x.reshape(n, p_num, mc, h, w)
    y = y.softmax(2)
    acc_metrix = torch.tensor(range(mc)).float().reshape(1, 1, mc, 1, 1)
    y = (y * acc_metrix).sum(2)
    return y.numpy()


def box_process(position):
    """边界框后处理：将模型输出转换为实际的边界框坐标"""
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([IMG_SIZE[1]//grid_h, IMG_SIZE[0]//grid_w]).reshape(1, 2, 1, 1)

    position = dfl(position)
    box_xy = grid + 0.5 - position[:, 0:2, :, :]
    box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
    xyxy = np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)

    return xyxy


def post_process(input_data):
    """YOLO11 模型输出后处理：解码检测框、过滤、NMS

    支持两种输出格式：
    1. 单张量格式 [1, 4+num_classes, 8400]（如 helmet_13.onnx）
    2. 多分支格式（如 yolo11n.onnx）
    """
    # 检测输出格式
    if len(input_data) == 1 and len(input_data[0].shape) == 3:
        # 单张量格式 [1, 4+num_classes, 8400]
        output = input_data[0]
        num_classes = output.shape[1] - 4
        boxes_xywh = output[:, :4, :].transpose(0, 2, 1)  # [1, 8400, 4]
        class_probs = output[:, 4:, :].transpose(0, 2, 1)  # [1, 8400, num_classes]

        # 将 cx,cy,w,h 转为 x1,y1,x2,y2
        cx = boxes_xywh[0, :, 0]
        cy = boxes_xywh[0, :, 1]
        w = boxes_xywh[0, :, 2]
        h = boxes_xywh[0, :, 3]

        boxes = np.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], axis=1)
        scores = np.max(class_probs[0], axis=1)
        classes = np.argmax(class_probs[0], axis=1)

        # 过滤低置信度
        mask = scores >= OBJ_THRESH
        boxes = boxes[mask]
        scores = scores[mask]
        classes = classes[mask]

        if len(boxes) == 0:
            return None, None, None

        # NMS
        keep = nms_boxes(boxes, scores)
        boxes = boxes[keep]
        classes = classes[keep]
        scores = scores[keep]

        return boxes, classes, scores

    # 多分支格式（原始逻辑）
    boxes, scores, classes_conf = [], [], []
    defualt_branch = 3
    pair_per_branch = len(input_data) // defualt_branch

    for i in range(defualt_branch):
        boxes.append(box_process(input_data[pair_per_branch * i]))
        classes_conf.append(input_data[pair_per_branch * i + 1])
        scores.append(np.ones_like(input_data[pair_per_branch * i + 1][:, :1, :, :], dtype=np.float32))

    def sp_flatten(_in):
        ch = _in.shape[1]
        _in = _in.transpose(0, 2, 3, 1)
        return _in.reshape(-1, ch)

    boxes = [sp_flatten(_v) for _v in boxes]
    classes_conf = [sp_flatten(_v) for _v in classes_conf]
    scores = [sp_flatten(_v) for _v in scores]

    boxes = np.concatenate(boxes)
    classes_conf = np.concatenate(classes_conf)
    scores = np.concatenate(scores)

    boxes, classes, scores = filter_boxes(boxes, scores, classes_conf)

    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]
        keep = nms_boxes(b, s)

        if len(keep) != 0:
            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    return boxes, classes, scores


def draw(image, boxes, scores, classes):
    """在图像上绘制检测结果"""
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = [int(_b) for _b in box]
        print("%s @ (%d %d %d %d) %.3f" % (CLASSES[cl], top, left, right, bottom, score))
        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                    (top, left - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


def setup_model(args):
    """根据模型文件类型初始化模型容器（仅支持ONNX和RKNN）"""
    model_path = args.model_path
    if model_path.endswith('.rknn'):
        platform = 'rknn'
        from py_utils.rknn_executor import RKNN_model_container
        model = RKNN_model_container(args.model_path, args.target, args.device_id)
    elif model_path.endswith('.onnx'):
        platform = 'onnx'
        from py_utils.onnx_executor import ONNX_model_container
        model = ONNX_model_container(args.model_path)
    else:
        raise ValueError(f"YOLO11 视频检测不支持 .pt 文件，仅支持 .onnx 和 .rknn 模型\n请检查模型路径: {model_path}")
    print(f'Model-{model_path} is {platform} model, starting detection')
    return model, platform


if __name__ == '__main__':
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='YOLO11 安全帽检测视频脚本')

    # ===== 基本参数 =====
    parser.add_argument('--model_path', type=str, required=True, help='模型路径（.onnx 或 .rknn），不支持 .pt 文件')
    parser.add_argument('--video_path', type=str, required=True, help='视频文件路径')
    parser.add_argument('--target', type=str, default='rk3588', help='目标 RKNPU 平台')
    parser.add_argument('--device_id', type=str, default=None, help='设备 ID（多设备时指定）')
    parser.add_argument('--video_save', action='store_true', default=False, help='保存结果视频到 ./result 目录')

    args = parser.parse_args()

    # 初始化模型
    model, platform = setup_model(args)

    # 打开视频文件
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"错误：无法打开视频文件 {args.video_path}")
        model.release()
        exit(1)

    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"视频信息: {width}x{height}, {fps:.2f} FPS, {total_frames} 帧")

    # 创建视频写入器（如果需要保存）
    video_writer = None
    if args.video_save:
        if not os.path.exists('./result'):
            os.mkdir('./result')
        result_path = os.path.join('./result', 'result_' + os.path.basename(args.video_path))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(result_path, fourcc, fps, (width, height))
        print(f"结果视频将保存到: {result_path}")

    # 创建 COCO 测试助手（用于 letterbox 处理）
    co_helper = COCO_test_helper(enable_letter_box=True)

    frame_count = 0
    start_time = time.time()

    print("按 'q' 退出视频检测...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("视频播放完毕")
            break

        frame_count += 1
        print(f'处理帧 {frame_count}/{total_frames}', end='\r')

        # Letterbox 预处理
        img = co_helper.letter_box(im=frame.copy(), new_shape=(IMG_SIZE[1], IMG_SIZE[0]), pad_color=(0, 0, 0))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 预处理
        if platform in ['onnx']:
            input_data = img.transpose((2, 0, 1))
            input_data = input_data.reshape(1, *input_data.shape).astype(np.float32)
            input_data = input_data / 255.
        else:  # rknn
            input_data = img

        # 执行推理
        outputs = model.run([input_data])

        # 后处理
        boxes, classes, scores = post_process(outputs)

        # 绘制检测结果
        img_p = frame.copy()
        if boxes is not None:
            draw(img_p, co_helper.get_real_box(boxes), scores, classes)

        # 显示帧率
        elapsed = time.time() - start_time
        current_fps = frame_count / elapsed if elapsed > 0 else 0
        cv2.putText(img_p, f'FPS: {current_fps:.2f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 显示结果
        cv2.imshow("YOLO11 Construction-PPE Detection", img_p)

        # 保存结果视频
        if args.video_save and video_writer is not None:
            video_writer.write(img_p)

        # 按键检测
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("用户退出")
            break

    # 清理资源
    elapsed = time.time() - start_time
    print(f"\n检测完成！共处理 {frame_count} 帧，用时 {elapsed:.2f} 秒，平均 FPS: {frame_count/elapsed:.2f}")

    cap.release()
    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()
    model.release()
