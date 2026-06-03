#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO11 实时视频检测脚本
读取 003.avi 视频文件，逐帧推理并实时显示检测结果

使用方法:
    cd /home/ztl/dltt/code/rknn/rknn_model_zoo-v2.3.0
    python examples/yolo11/python/video_detect.py

模型自动搜索优先级: yolo11n.pt > yolo11n.onnx
视频路径: examples/yolo11/003.avi

按键操作:
    q   - 退出
    空格 - 暂停/继续

显示信息:
    [PT]   FPS: xx.x | Frame: xxx/xxxx (xx%)   - PyTorch 后端
    [ONNX] FPS: xx.x | Frame: xxx/xxxx (xx%)   - ONNX 后端
"""

import os
import sys
import time
import cv2
import numpy as np

# 添加 rknn_model_zoo 根目录到路径
realpath = os.path.abspath(__file__)
_sep = os.path.sep
realpath = realpath.split(_sep)
try:
    zoo_idx = realpath.index('rknn_model_zoo-v2.3.0')
    sys.path.append(os.path.join(realpath[0]+_sep, *realpath[1:zoo_idx+1]))
except ValueError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# ======== YOLO11 后处理参数 ========
SUPPORTED_EXTS = ('.pt', '.torchscript', '.onnx')  # 只用 PT/ONNX，不用 RKNN
OBJ_THRESH = 0.25      # 置信度阈值
NMS_THRESH = 0.45      # NMS IoU 阈值
IMG_SIZE = (640, 640)  # 模型输入尺寸

CLASSES = ("person", "bicycle", "car","motorbike ","aeroplane ","bus ","train","truck ","boat","traffic light",
           "fire hydrant","stop sign ","parking meter","bench","bird","cat","dog ","horse ","sheep","cow","elephant",
           "bear","zebra ","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
           "baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife ",
           "spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza ","donut","cake","chair","sofa",
           "pottedplant","bed","diningtable","toilet ","tvmonitor","laptop","mouse","remote ","keyboard ","cell phone","microwave ",
           "oven ","toaster","sink","refrigerator ","book","clock","vase","scissors ","teddy bear ","hair drier", "toothbrush ")

# ======== 后处理函数（复用 yolo11.py 逻辑） ========

def filter_boxes(boxes, box_confidences, box_class_probs):
    box_confidences = box_confidences.reshape(-1)
    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)
    _class_pos = np.where(class_max_score * box_confidences >= OBJ_THRESH)
    scores = (class_max_score * box_confidences)[_class_pos]
    boxes = boxes[_class_pos]
    classes = classes[_class_pos]
    return boxes, classes, scores

def nms_boxes(boxes, scores):
    x = boxes[:, 0]; y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]; h = boxes[:, 3] - boxes[:, 1]
    areas = w * h
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i]+w[i], x[order[1:]]+w[order[1:]])
        yy2 = np.minimum(y[i]+h[i], y[order[1:]]+h[order[1:]])
        w1 = np.maximum(0.0, xx2-xx1+0.00001)
        h1 = np.maximum(0.0, yy2-yy1+0.00001)
        inter = w1 * h1
        ovr = inter / (areas[i]+areas[order[1:]]-inter)
        order = order[np.where(ovr <= NMS_THRESH)[0]+1]
    return np.array(keep)

def dfl(position):
    import torch
    x = torch.tensor(position)
    n, c, h, w = x.shape
    p_num = 4; mc = c // p_num
    y = x.reshape(n, p_num, mc, h, w).softmax(2)
    acc_metrix = torch.tensor(range(mc)).float().reshape(1,1,mc,1,1)
    y = (y*acc_metrix).sum(2)
    return y.numpy()

def box_process(position):
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0,grid_w), np.arange(0,grid_h))
    col = col.reshape(1,1,grid_h,grid_w); row = row.reshape(1,1,grid_h,grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([IMG_SIZE[1]//grid_h, IMG_SIZE[0]//grid_w]).reshape(1,2,1,1)
    position = dfl(position)
    box_xy  = grid + 0.5 - position[:,0:2,:,:]
    box_xy2 = grid + 0.5 + position[:,2:4,:,:]
    return np.concatenate((box_xy*stride, box_xy2*stride), axis=1)

def post_process(input_data):
    boxes, scores, classes_conf = [], [], []
    default_branch = 3
    pair_per_branch = len(input_data) // default_branch

    for i in range(default_branch):
        boxes.append(box_process(input_data[pair_per_branch*i]))
        classes_conf.append(input_data[pair_per_branch*i+1])
        scores.append(np.ones_like(input_data[pair_per_branch*i+1][:,:1,:,:], dtype=np.float32))

    def sp_flatten(_in):
        ch = _in.shape[1]; _in = _in.transpose(0,2,3,1)
        return _in.reshape(-1, ch)

    boxes = np.concatenate([sp_flatten(v) for v in boxes])
    classes_conf = np.concatenate([sp_flatten(v) for v in classes_conf])
    scores = np.concatenate([sp_flatten(v) for v in scores])

    boxes, classes, scores = filter_boxes(boxes, scores, classes_conf)

    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b, cl, s = boxes[inds], classes[inds], scores[inds]
        keep = nms_boxes(b, s)
        if len(keep) != 0:
            nboxes.append(b[keep]); nclasses.append(cl[keep]); nscores.append(s[keep])

    if not nclasses: return None, None, None
    return np.concatenate(nboxes), np.concatenate(nclasses), np.concatenate(nscores)

def letter_box(img, new_shape, pad_color=(0,0,0)):
    """保持宽高比的缩放 + 填充"""
    shape = img.shape[:2]
    r = min(new_shape[0]/shape[0], new_shape[1]/shape[1])
    new_unpad = (int(round(shape[1]*r)), int(round(shape[0]*r)))
    dw, dh = new_shape[1]-new_unpad[0], new_shape[0]-new_unpad[1]
    dw, dh = dw//2, dh//2
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    img = cv2.copyMakeBorder(img, dh, dh, dw, dw, cv2.BORDER_CONSTANT, value=pad_color)
    return img, (r, dw, dh)

def scale_boxes(boxes, scale_info, orig_shape):
    """将 letterbox 坐标还原到原图坐标"""
    r, dw, dh = scale_info
    boxes[:, [0,2]] -= dw
    boxes[:, [1,3]] -= dh
    boxes /= r
    boxes[:, [0,2]] = np.clip(boxes[:, [0,2]], 0, orig_shape[1])
    boxes[:, [1,3]] = np.clip(boxes[:, [1,3]], 0, orig_shape[0])
    return boxes

# ======== 模型加载 ========

def load_pt_model(model_path):
    """加载 ultralytics PyTorch 模型"""
    from ultralytics import YOLO
    print(f"Loading PyTorch model: {model_path}")
    return YOLO(model_path)

def load_onnx_model(model_path):
    """加载 ONNX 模型"""
    from py_utils.onnx_executor import ONNX_model_container
    print(f"Loading ONNX model: {model_path}")
    return ONNX_model_container(model_path)

def load_model(model_path):
    """根据文件扩展名自动选择模型容器"""
    ext = os.path.splitext(model_path)[1].lower()
    if ext in ('.pt', '.torchscript'):
        return load_pt_model(model_path), 'pytorch'
    elif ext == '.onnx':
        return load_onnx_model(model_path), 'onnx'
    else:
        raise ValueError(f"不支持的模型格式: {ext}")

def find_model(model_dir):
    """自动搜索模型目录下的模型文件，优先级: *.pt > *.onnx"""
    priority = ['yolo11n.pt', 'yolo11n.onnx']
    for name in priority:
        p = os.path.join(model_dir, name)
        if os.path.exists(p):
            return p
    # fallback: 搜索目录下第一个匹配的模型
    for f in sorted(os.listdir(model_dir)):
        if f.lower().endswith(SUPPORTED_EXTS):
            return os.path.join(model_dir, f)
    return None

def preprocess(frame, platform):
    """预处理：letterbox + 格式转换，PyTorch/ONNX 需要额外归一化"""
    img, scale_info = letter_box(frame.copy(), (IMG_SIZE[1], IMG_SIZE[0]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if platform in ('pytorch', 'onnx'):
        # CHW + batch + normalize
        img = img.transpose((2, 0, 1))
        img = img.reshape(1, *img.shape).astype(np.float32)
        img = img / 255.0
    return img

# ======== 主程序 ========

def main():
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'model')
    video_path = os.path.join(os.path.dirname(__file__), '..', '003.avi')

    # 自动搜索模型
    model_path = find_model(model_dir)
    if model_path is None:
        print(f"ERROR: 模型目录下未找到模型文件 (支持: {SUPPORTED_EXTS})")
        print(f"  目录: {model_dir}")
        return

    # 检查视频
    if not os.path.exists(video_path):
        print(f"ERROR: 视频文件不存在: {video_path}")
        return

    print("=" * 60)
    print(f"模型: {model_path}")
    print(f"后端: {os.path.splitext(model_path)[1].strip('.').upper()}")
    print(f"视频: {video_path}")
    print("=" * 60)

    # 加载模型
    model, platform = load_model(model_path)

    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("ERROR: 无法打开视频")
        return

    fps_video = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"视频 FPS: {fps_video:.1f}, 总帧数: {total_frames}")
    print("按 'q' 退出, 按 '空格' 暂停/继续, 按 's' 切换后端")
    print("-" * 60)

    paused = False
    frame_count = 0
    fps_timer = time.time()
    fps_counter = 0
    fps_display = 0.0

    cv2.namedWindow("YOLO11 Video Detection", cv2.WINDOW_NORMAL)

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("\n视频播放完毕")
                break
            frame_count += 1

            display_frame = frame.copy()

            if platform == 'pytorch':
                # ultralytics YOLO predict（自带预处理和后处理）
                results = model(frame, conf=OBJ_THRESH, iou=NMS_THRESH,
                                imgsz=IMG_SIZE, verbose=False)
                boxes_data = results[0].boxes
                if boxes_data is not None and len(boxes_data) > 0:
                    for box in boxes_data:
                        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        label = f"{CLASSES[cls_id]} {conf:.2f}"
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(display_frame, label, (x1, y1-6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            else:  # onnx
                img_input = preprocess(frame, platform)
                _, scale_info = letter_box(frame.copy(), (IMG_SIZE[1], IMG_SIZE[0]))

                outputs = model.run([img_input])
                boxes, classes, scores = post_process(outputs)

                if boxes is not None:
                    boxes = scale_boxes(boxes, scale_info, frame.shape[:2])
                    for box, score, cl in zip(boxes, scores, classes):
                        x1, y1, x2, y2 = [int(v) for v in box]
                        label = f"{CLASSES[int(cl)]} {score:.2f}"
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(display_frame, label, (x1, y1-6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 显示帧率、进度、后端类型
            progress = frame_count / total_frames * 100
            platform_label = {'pytorch': 'PT', 'onnx': 'ONNX'}.get(platform, '???')
            info = f"[{platform_label}] FPS: {fps_display:.1f} | Frame: {frame_count}/{total_frames} ({progress:.0f}%)"
            cv2.putText(display_frame, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2)

            cv2.imshow("YOLO11 Video Detection", display_frame)

        # 按键处理
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n用户退出")
            break
        elif key == ord(' '):
            paused = not paused
            print(f"\r{'[暂停]' if paused else '[播放]'} Frame: {frame_count}/{total_frames}", end='')

        # 计算 FPS
        if not paused:
            fps_counter += 1
            if fps_counter >= 10:
                elapsed = time.time() - fps_timer
                fps_display = fps_counter / elapsed
                fps_timer = time.time()
                fps_counter = 0

    # 清理
    if platform == 'onnx':
        model.release()
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n处理完成, 共 {frame_count} 帧")

if __name__ == "__main__":
    main()
