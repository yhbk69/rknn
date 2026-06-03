"""
===========================================================================
my_yolo11.py —— YOLO11 多模型对比推理脚本
===========================================================================
功能:
    同时加载 PyTorch (.pt)、ONNX (.onnx)、RKNN (.rknn) 三种格式的模型，
    在同一批测试图片上执行推理，对比各模型的检测结果和推理耗时。

用法:
    # 自动查找当前目录下的模型文件和测试图片
    python3 my_yolo11.py

    # 指定测试图片文件夹
    python3 my_yolo11.py --img_folder ../model

    # 指定具体模型文件（只对比指定模型）
    python3 my_yolo11.py --pt_model best.pt --onnx_model best.onnx --rknn_model best.rknn

    # 显示/保存检测结果图像
    python3 my_yolo11.py --img_show --img_save

输出说明:
    - 控制台打印各模型的检测结果对比表和耗时对比表
    - 可选保存带检测框的结果图像到 ./result/ 目录
    - 对比表中 ✓ 表示该目标被检测到，✗ 表示未检测到
===========================================================================
"""

"""
================================================================================

    [统计] 各模型检测到的目标数量:
      PyTorch        : 2264 个目标
      ONNX           : 2293 个目标
      RKNN           : 1906 个目标


======================================================================
  耗时对比汇总
======================================================================
    --------------------------------------------------------------------------------
    模型             平均(ms)       最快(ms)       最慢(ms)       标准差(ms)      中位数(ms)      加速比            
    --------------------------------------------------------------------------------
    PyTorch        8492.30        7446.91        12648.66       585.25         8406.43        1.00x          
    ONNX           2837.52        2639.29        3878.71        156.31         2807.82        2.99x          
    RKNN           255.19         245.11         270.16         4.72           254.25         33.28x         
    --------------------------------------------------------------------------------

======================================================================
  释放模型资源...
======================================================================
  [释放] PyTorch ...
  [释放] ONNX ...
  [释放] RKNN ...
  所有模型已释放。

======================================================================
  对比结论
======================================================================
  测试图片: 15 张
  测试模型: PyTorch, ONNX, RKNN
    PyTorch   : 平均 8492.30 ms/帧
    ONNX      : 平均 2837.52 ms/帧
    RKNN      : 平均 255.19 ms/帧
======================================================================
"""





import os
import cv2
import sys
import time
import argparse
import numpy as np

# =========================================================================
# 尝试导入 tabulate（用于格式化表格输出），非必需，不存在则回退纯文本
# =========================================================================
try:
    from tabulate import tabulate
    _HAS_TABULATE = True
except ImportError:
    _HAS_TABULATE = False


# =========================================================================
# 路径处理：将 rknn_model_zoo 根目录添加到系统路径，确保能导入 py_utils 模块
# =========================================================================
realpath = os.path.abspath(__file__)
_sep = os.path.sep
realpath = realpath.split(_sep)
try:
    zoo_idx = realpath.index('rknn_model_zoo-v2.3.0')
    sys.path.append(os.path.join(realpath[0] + _sep, *realpath[1:zoo_idx + 1]))
except ValueError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(realpath))))

from py_utils.coco_utils import COCO_test_helper


# =========================================================================
# 检测参数常量
# =========================================================================
# 目标置信度阈值：低于此分数的检测框会被过滤掉
OBJ_THRESH = 0.35
# NMS 的 IoU 阈值：重叠度高于此值的框会被抑制
NMS_THRESH = 0.5
# 输入图像尺寸（模型要求的宽高）
IMG_SIZE = (640, 640)  # (width, height)

# 推理预热次数：模型刚加载时前几次推理较慢，预热后可得到稳定耗时
WARMUP_COUNT = 5
# 正式测试的推理轮数（取平均耗时）
TEST_COUNT = 20

# =========================================================================
# 从 data.yaml 读取类别定义（与当前目录下的 data.yaml 保持一致）
# 该模型是 Construction-PPE 数据集，用于检测工地安全装备
# =========================================================================
CLASSES = (
    "helmet",      # 0: 安全帽
    "gloves",      # 1: 手套
    "vest",        # 2: 反光背心
    "boots",       # 3: 安全靴
    "goggles",     # 4: 护目镜
    "none",        # 5: 无防护（背景/其他）
    "Person",      # 6: 人员
    "no_helmet",   # 7: 未戴安全帽
    "no_goggle",   # 8: 未戴护目镜
    "no_gloves",   # 9: 未戴手套
    "no_boots",    # 10: 未穿安全靴
)


# =========================================================================
# 模型加载工厂函数
# =========================================================================

def load_model(model_path: str, target: str = None, device_id: str = None):
    """
    根据模型文件后缀自动选择合适的执行器加载模型。

    支持的格式:
        .pt / .torchscript  -> PyTorch (TorchScript)
        .onnx               -> ONNX (ONNX Runtime)
        .rknn               -> RKNN (RKNPU)

    参数:
        model_path - 模型文件路径
        target     - (仅 RKNN) 目标平台，如 rk3588
        device_id  - (仅 RKNN) 设备 ID

    返回:
        model    - 模型容器对象（统一有 .run() 和 .release() 接口）
        platform - 模型类型字符串: 'pytorch' / 'onnx' / 'rknn'
    """
    model_path_lower = model_path.lower()

    if model_path_lower.endswith('.pt') or model_path_lower.endswith('.torchscript'):
        # ---------- PyTorch 模型 ----------
        from py_utils.pytorch_executor import Torch_model_container
        model = Torch_model_container(model_path)
        platform = 'pytorch'
        print(f"  [加载] PyTorch 模型: {model_path}")

    elif model_path_lower.endswith('.onnx'):
        # ---------- ONNX 模型 ----------
        from py_utils.onnx_executor import ONNX_model_container
        model = ONNX_model_container(model_path)
        platform = 'onnx'
        print(f"  [加载] ONNX 模型: {model_path}")

    elif model_path_lower.endswith('.rknn'):
        # ---------- RKNN 模型 ----------
        from py_utils.rknn_executor import RKNN_model_container
        model = RKNN_model_container(model_path, target, device_id)
        platform = 'rknn'
        print(f"  [加载] RKNN 模型: {model_path}  (目标: {target})")

    else:
        raise ValueError(f"不支持的模型格式: {model_path} (支持: .pt, .torchscript, .onnx, .rknn)")

    return model, platform


# =========================================================================
# 图像预处理
# =========================================================================

def preprocess_image(img_src: np.ndarray, platform: str, co_helper):
    """
    对输入图像进行预处理，使其符合模型输入要求。

    预处理步骤:
        1. Letterbox 缩放（保持宽高比，填充黑边）
        2. BGR -> RGB 颜色空间转换
        3. (仅 PT/ONNX) HWC -> CHW + 归一化到 [0,1]

    参数:
        img_src   - 原始 BGR 图像 (numpy array)
        platform  - 模型类型: 'pytorch' / 'onnx' / 'rknn'
        co_helper - COCO_test_helper 实例（提供 letter_box 方法）

    返回:
        input_data - 预处理后的输入数据
    """
    # Step 1: Letterbox 缩放 —— 保持宽高比，不足部分用黑色填充
    pad_color = (0, 0, 0)  # 填充颜色（黑色，与 RGA 初始化一致）
    img = co_helper.letter_box(
        im=img_src.copy(),
        new_shape=(IMG_SIZE[1], IMG_SIZE[0]),
        pad_color=pad_color
    )

    # Step 2: BGR -> RGB（OpenCV 默认 BGR，模型需要 RGB）
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Step 3: PT/ONNX 需要额外预处理（RKNN 的量化已在模型构建时配置好）
    if platform in ['pytorch', 'onnx']:
        # HWC (Height, Width, Channel) -> CHW (Channel, Height, Width)
        input_data = img.transpose((2, 0, 1))
        # 添加 batch 维度: (1, C, H, W)，转为 float32
        input_data = input_data.reshape(1, *input_data.shape).astype(np.float32)
        # 归一化到 [0, 1] 区间
        input_data = input_data / 255.0
    else:
        # RKNN 模型的归一化已在 rknn.config() 的 mean_values/std_values 中配置
        input_data = img

    return input_data


# =========================================================================
# 推理耗时测量
# =========================================================================

def benchmark_inference(model, input_data: list, platform: str,
                        warmup: int = WARMUP_COUNT, test_count: int = TEST_COUNT):
    """
    测量模型推理耗时。

    流程:
        1. 预热推理（warmup 次）—— 排除首次加载/缓存的影响
        2. 正式测试（test_count 次）—— 收集耗时
        3. 计算统计值: 平均耗时、最快、最慢、标准差

    参数:
        model      - 模型容器对象
        input_data - 预处理后的输入数据列表
        platform   - 模型类型（仅用于日志）
        warmup     - 预热次数
        test_count - 正式测试次数

    返回:
        avg_time  - 平均推理耗时 (ms)
        times     - 所有测试的耗时列表 (ms)
    """
    # Step 1: 预热推理 —— 让模型达到稳定状态
    print(f"    [预热] 进行 {warmup} 次预热推理...")
    for i in range(warmup):
        _ = model.run(input_data)
        # 打印进度
        print(f"      预热 {i + 1}/{warmup}", end='\r')
    print(f"      预热完成{' ' * 10}")

    # Step 2: 正式测试，记录每次推理耗时
    print(f"    [测试] 进行 {test_count} 次推理耗时测试...")
    times = []
    for i in range(test_count):
        # 使用 time.perf_counter() 进行高精度计时（纳秒级）
        start = time.perf_counter()
        outputs = model.run(input_data)
        elapsed = time.perf_counter() - start

        # 转换为毫秒
        elapsed_ms = elapsed * 1000
        times.append(elapsed_ms)
        print(f"      第 {i + 1:2d} 次: {elapsed_ms:8.2f} ms", end='\r')

    print(f"      测试完成{' ' * 20}")

    # Step 3: 计算统计值
    times = np.array(times)
    avg_time = np.mean(times)

    return avg_time, times, outputs


# =========================================================================
# 目标检测一致性对比
# =========================================================================

def compare_detections(results: dict, img_name: str):
    """
    对比不同模型在同一张图片上的检测结果。

    对比策略:
        1. 统计每个模型检测到的目标数量
        2. 找出所有模型检测到的目标类别并集
        3. 对每个类别，标记哪些模型检测到了

    参数:
        results  - 字典，格式: {模型名称: {'boxes': ..., 'classes': ..., 'scores': ...}}
        img_name - 当前图片名称
    """
    model_names = list(results.keys())

    # ================================================================
    # 收集所有模型检测到的目标信息
    # ================================================================
    all_detections = {}  # {类别名: {模型名: [分数列表]}}
    for model_name in model_names:
        res = results[model_name]
        if res['boxes'] is None:
            continue
        for cls_id, score in zip(res['classes'], res['scores']):
            cls_name = CLASSES[int(cls_id)]
            if cls_name not in all_detections:
                all_detections[cls_name] = {}
            if model_name not in all_detections[cls_name]:
                all_detections[cls_name][model_name] = []
            all_detections[cls_name][model_name].append(score)

    # ================================================================
    # 如果没有检测到任何目标，直接返回
    # ================================================================
    if not all_detections:
        print(f"    [对比] {img_name}: 所有模型均未检测到目标")
        return

    # ================================================================
    # 构建对比表格
    # ================================================================
    table_data = []
    # 按类别名排序，方便查看
    for cls_name in sorted(all_detections.keys()):
        row = [cls_name]
        det_dict = all_detections[cls_name]
        for model_name in model_names:
            if model_name in det_dict:
                scores = det_dict[model_name]
                # 显示最高分和检测数量
                max_score = max(scores)
                row.append(f"✓ ({max_score:.3f})")
            else:
                row.append("✗")
        table_data.append(row)

    # ================================================================
    # 打印对比结果
    # ================================================================
    headers = ["类别名称"] + model_names
    print(f"\n    [对比] {img_name} 检测结果对比:")
    if _HAS_TABULATE:
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
    else:
        # 如果没有安装 tabulate，使用简单的分隔线格式
        print(f"    {'=' * 80}")
        print(f"    {'类别名称':<20}", end="")
        for name in model_names:
            print(f"{name:<25}", end="")
        print()
        print(f"    {'-' * 80}")
        for row in table_data:
            print(f"    {row[0]:<20}", end="")
            for val in row[1:]:
                print(f"{val:<25}", end="")
            print()
        print(f"    {'=' * 80}")

    # 统计每个模型检测到的目标总数
    print(f"\n    [统计] 各模型检测到的目标数量:")
    for model_name in model_names:
        total = 0
        for cls_name, det_dict in all_detections.items():
            if model_name in det_dict:
                total += len(det_dict[model_name])
        print(f"      {model_name:<15}: {total} 个目标")


# =========================================================================
# 图像绘制工具
# =========================================================================

def draw_detections(image: np.ndarray, boxes: np.ndarray, scores: np.ndarray,
                    classes: np.ndarray, color=(255, 0, 0)):
    """
    在图像上绘制检测结果。

    参数:
        image   - 图像数组（会被直接修改）
        boxes   - 检测框数组，每行为 [x1, y1, x2, y2]
        scores  - 分数数组
        classes - 类别索引数组
        color   - 绘制框的颜色 (BGR)
    """
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = [int(b) for b in box]
        # 绘制矩形框
        cv2.rectangle(image, (top, left), (right, bottom), color, 2)
        # 绘制标签文字（类别 + 分数）
        label = f"{CLASSES[int(cl)]} {score:.2f}"
        cv2.putText(image, label, (top, left - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


# =========================================================================
# 图片文件过滤
# =========================================================================

def is_image_file(path: str) -> bool:
    """
    检查文件是否为支持的图片格式。
    支持: .jpg, .jpeg, .png, .bmp（不区分大小写）
    """
    img_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    for ext in img_extensions:
        if path.lower().endswith(ext):
            return True
    return False


# =========================================================================
# 后处理函数（直接复用 yolo11.py 中的逻辑）
# =========================================================================

def nms_boxes(boxes, scores):
    """非极大值抑制 (NMS)。"""
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


def post_process(input_data):
    """
    YOLO11 模型输出的完整后处理流程。

    支持的输出格式：
    - 单输出格式 [1, 15, 8400]（ultralytics 导出的 ONNX / PyTorch / RKNN）
      15 通道: 4(bbox cx,cy,w,h) + 1(objectness) + 10(class probs)
    - 多输出格式，查找 shape 中包含 [1, 15, 8400] 的张量
    - 9 输出格式（原始检测头，自动检测）

    返回:
        boxes   - 检测框坐标 (N, 4) 格式 [x1, y1, x2, y2]
        classes - 类别索引 (N,)
        scores  - 置信度分数 (N,)
    """
    # ---------- 自动检测单输出格式 ----------
    # 检查是否有任何输出是 [1, 15, *] 格式（ultralytics 合并输出）
    for data in input_data:
        if len(data.shape) == 3 and data.shape[1] == 15:
            return _post_process_single(data)

    # 否则尝试原始的 9 输出格式（3 检测头 × 3 分支）
    return _post_process_multi(input_data)


def _post_process_single(pred):
    """
    处理 ultralytics 导出的单输出格式。
    pred shape: (1, 15, 8400) 或 (1, 8400, 15)
    """
    # 确保 shape 为 (1, 15, 8400)
    if pred.shape[1] != 15 and pred.shape[2] == 15:
        pred = pred.transpose(0, 2, 1)
    if pred.shape[1] != 15:
        raise ValueError(f"Unexpected shape: {pred.shape}")

    # (1, 15, 8400) -> (8400, 15)
    pred = pred[0].T  # (8400, 15)

    # 提取各分量
    bbox = pred[:, :4]          # cx, cy, w, h (在 640x640 坐标系下)
    objectness = pred[:, 4:5]   # 目标置信度
    class_probs = pred[:, 5:]   # 类别概率 (10 classes)

    # Sigmoid 激活
    objectness = 1.0 / (1.0 + np.exp(-objectness))
    class_probs = 1.0 / (1.0 + np.exp(-class_probs))

    # 综合得分 = objectness * max(class_prob)
    class_scores = class_probs * objectness
    classes = np.argmax(class_scores, axis=1)
    max_scores = np.max(class_scores, axis=1)

    # 低分过滤
    keep_mask = max_scores >= OBJ_THRESH
    if not np.any(keep_mask):
        return None, None, None

    boxes = bbox[keep_mask]
    classes = classes[keep_mask]
    scores = max_scores[keep_mask]

    # cx, cy, w, h -> x1, y1, x2, y2
    x1 = boxes[:, 0] - boxes[:, 2] / 2.0
    y1 = boxes[:, 1] - boxes[:, 3] / 2.0
    x2 = boxes[:, 0] + boxes[:, 2] / 2.0
    y2 = boxes[:, 1] + boxes[:, 3] / 2.0
    boxes = np.stack([x1, y1, x2, y2], axis=1)

    # 按类别分别做 NMS
    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        cl = classes[inds]
        s = scores[inds]
        keep = nms_boxes(b, s)
        if len(keep) != 0:
            nboxes.append(b[keep])
            nclasses.append(cl[keep])
            nscores.append(s[keep])

    if not nclasses:
        return None, None, None

    return (np.concatenate(nboxes),
            np.concatenate(nclasses),
            np.concatenate(nscores))


def _post_process_multi(input_data):
    """
    处理原始 YOLO11 9 输出格式（3 检测头 × 3 分支）。
    后备方案，兼容旧模型格式。
    """
    import torch

    def _dfl(position):
        x = torch.tensor(position)
        n, c, h, w = x.shape
        p_num = 4
        mc = c // p_num
        y = x.reshape(n, p_num, mc, h, w)
        y = y.softmax(2)
        acc_metrix = torch.tensor(range(mc)).float().reshape(1, 1, mc, 1, 1)
        y = (y * acc_metrix).sum(2)
        return y.numpy()

    def _box_process(position):
        grid_h, grid_w = position.shape[2:4]
        col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
        col = col.reshape(1, 1, grid_h, grid_w)
        row = row.reshape(1, 1, grid_h, grid_w)
        grid = np.concatenate((col, row), axis=1)
        stride = np.array([IMG_SIZE[1] // grid_h, IMG_SIZE[0] // grid_w]).reshape(1, 2, 1, 1)
        position = _dfl(position)
        box_xy = grid + 0.5 - position[:, 0:2, :, :]
        box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
        xyxy = np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)
        return xyxy

    boxes, scores, classes_conf = [], [], []
    default_branch = 3
    pair_per_branch = len(input_data) // default_branch

    for i in range(default_branch):
        boxes.append(_box_process(input_data[pair_per_branch * i]))
        classes_conf.append(input_data[pair_per_branch * i + 1])
        scores.append(np.ones_like(
            input_data[pair_per_branch * i + 1][:, :1, :, :], dtype=np.float32))

    def sp_flatten(_in):
        ch = _in.shape[1]
        _in = _in.transpose(0, 2, 3, 1)
        return _in.reshape(-1, ch)

    boxes = np.concatenate([sp_flatten(v) for v in boxes])
    classes_conf = np.concatenate([sp_flatten(v) for v in classes_conf])
    scores = np.concatenate([sp_flatten(v) for v in scores])

    # filter
    box_conf = scores.reshape(-1)
    class_max = np.max(classes_conf, axis=-1)
    classes = np.argmax(classes_conf, axis=-1)
    pos = np.where(class_max * box_conf >= OBJ_THRESH)
    scores = (class_max * box_conf)[pos]
    boxes = boxes[pos]
    classes = classes[pos]

    if len(classes) == 0:
        return None, None, None

    # NMS per class
    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        keep = nms_boxes(boxes[inds], scores[inds])
        if len(keep) != 0:
            nboxes.append(boxes[inds][keep])
            nclasses.append(classes[inds][keep])
            nscores.append(scores[inds][keep])

    if not nclasses:
        return None, None, None

    return (np.concatenate(nboxes),
            np.concatenate(nclasses),
            np.concatenate(nscores))


# =========================================================================
# 主程序逻辑
# =========================================================================

def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description='YOLO11 多模型对比：对比 PT / ONNX / RKNN 的检测效果和耗时'
    )

    # ---- 模型参数 ----
    parser.add_argument('--pt_model', type=str, default=None,
                        help='PyTorch 模型路径（.pt 文件），不指定则自动查找')
    parser.add_argument('--onnx_model', type=str, default=None,
                        help='ONNX 模型路径（.onnx 文件），不指定则自动查找')
    parser.add_argument('--rknn_model', type=str, default=None,
                        help='RKNN 模型路径（.rknn 文件），不指定则自动查找')

    # ---- RKNN 设备参数 ----
    parser.add_argument('--target', type=str, default='rk3588',
                        help='RKNN 目标平台（默认: rk3566），如 rk3588')
    parser.add_argument('--device_id', type=str, default=None,
                        help='RKNN 设备 ID（多设备时指定）')

    # ---- 数据参数 ----
    parser.add_argument('--img_folder', type=str, default=None,
                        help='测试图片文件夹路径（默认: ../imgs/）')

    # ---- 输出控制 ----
    parser.add_argument('--img_show', action='store_true', default=False,
                        help='显示带检测结果的图像')
    parser.add_argument('--img_save', action='store_true', default=False,
                        help='保存带检测结果的图像到 ./result/ 目录')

    # ---- 测试参数 ----
    parser.add_argument('--warmup', type=int, default=WARMUP_COUNT,
                        help=f'推理预热次数（默认: {WARMUP_COUNT}）')
    parser.add_argument('--test_count', type=int, default=TEST_COUNT,
                        help=f'正式测试推理次数（默认: {TEST_COUNT}）')

    return parser.parse_args()


def auto_find_models():
    """
    自动在当前目录下查找模型文件。

    查找规则:
        - 以 .pt / .torchscript 结尾的为 PyTorch 模型
        - 以 .onnx 结尾的为 ONNX 模型
        - 以 .rknn 结尾的为 RKNN 模型

    返回:
        pt_model    - PyTorch 模型路径（可能为 None）
        onnx_model  - ONNX 模型路径（可能为 None）
        rknn_model  - RKNN 模型路径（可能为 None）
    """
    all_files = os.listdir('.')
    pt_model = None
    onnx_model = None
    rknn_model = None

    for f in all_files:
        if f.endswith('.pt') or f.endswith('.torchscript'):
            if pt_model is None:  # 取第一个找到的
                pt_model = f
        elif f.endswith('.onnx'):
            if onnx_model is None:
                onnx_model = f
        elif f.endswith('.rknn'):
            if rknn_model is None:
                rknn_model = f

    return pt_model, onnx_model, rknn_model


def main():
    """
    主函数：多模型对比推理流程。
    """
    # ================================================================
    # 1. 解析参数
    # ================================================================
    args = parse_args()

    print("=" * 70)
    print("  YOLO11 多模型对比推理工具")
    print("=" * 70)

    # ================================================================
    # 2. 确定测试图片文件夹
    # ================================================================
    img_folder = args.img_folder
    if img_folder is None:
        # 尝试默认路径
        candidates = ['../model', '.']
        for c in candidates:
            if os.path.isdir(c):
                img_folder = c
                break
    print(f"\n[配置] 测试图片文件夹: {img_folder}")

    # 收集图片文件
    if not os.path.isdir(img_folder):
        print(f"[错误] 文件夹不存在: {img_folder}")
        sys.exit(1)

    img_list = sorted([
        f for f in os.listdir(img_folder)
        if is_image_file(f)
    ])
    if not img_list:
        print(f"[错误] 文件夹中没有图片文件: {img_folder}")
        sys.exit(1)
    print(f"       找到 {len(img_list)} 张测试图片: {img_list}")

    # ================================================================
    # 3. 确定模型文件
    # ================================================================
    pt_model = args.pt_model
    onnx_model = args.onnx_model
    rknn_model = args.rknn_model

    # 如果有未指定的模型，尝试自动查找
    auto_pt, auto_onnx, auto_rknn = auto_find_models()
    if pt_model is None:
        pt_model = auto_pt
    if onnx_model is None:
        onnx_model = auto_onnx
    if rknn_model is None:
        rknn_model = auto_rknn

    # 构建模型列表（只包含能找到的模型）
    model_configs = []  # [(显示名称, 模型路径, 平台), ...]
    if pt_model and os.path.exists(pt_model):
        model_configs.append(("PyTorch", pt_model, "pytorch"))
    if onnx_model and os.path.exists(onnx_model):
        model_configs.append(("ONNX", onnx_model, "onnx"))
    if rknn_model and os.path.exists(rknn_model):
        model_configs.append(("RKNN", rknn_model, "rknn"))

    if not model_configs:
        print("[错误] 未找到任何模型文件！")
        print("       请将 .pt / .onnx / .rknn 文件放在当前目录，")
        print("       或使用 --pt_model / --onnx_model / --rknn_model 指定。")
        sys.exit(1)

    print(f"\n[配置] 参与对比的模型:")
    for name, path, _ in model_configs:
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"       {name:<8}: {path} ({size_mb:.2f} MB)")

    print(f"\n[配置] RKNN 目标平台: {args.target}")
    print(f"[配置] 推理预热: {args.warmup} 次, 正式测试: {args.test_count} 次")
    print(f"[配置] 显示结果: {'是' if args.img_show else '否'}, "
          f"保存结果: {'是' if args.img_save else '否'}")

    # ================================================================
    # 4. 加载所有模型
    # ================================================================
    print(f"\n{'=' * 70}")
    print("  加载模型...")
    print(f"{'=' * 70}")

    models = {}  # {显示名称: {'model': ..., 'platform': ...}}
    for name, path, platform in model_configs:
        print(f"\n  [{name}]")
        try:
            model, actual_platform = load_model(
                path,
                target=args.target if platform == 'rknn' else None,
                device_id=args.device_id if platform == 'rknn' else None
            )
            models[name] = {
                'model': model,
                'platform': actual_platform,
            }
        except Exception as e:
            print(f"    [错误] 加载失败: {e}")
            print(f"    [跳过] {name} 模型")

    if not models:
        print("[错误] 所有模型加载失败，无法进行对比！")
        sys.exit(1)

    # ================================================================
    # 5. 初始化 COCO 测试助手
    # ================================================================
    co_helper = COCO_test_helper(enable_letter_box=True)

    # ================================================================
    # 6. 遍历所有图片进行推理和对比
    # ================================================================
    print(f"\n{'=' * 70}")
    print("  开始推理对比...")
    print(f"{'=' * 70}")

    # 用于汇总所有图片的耗时数据
    all_timing = {}  # {模型名: [每次推理耗时列表]}
    for name in models:
        all_timing[name] = []

    # 逐张图片处理
    for img_idx, img_name in enumerate(img_list):
        img_path = os.path.join(img_folder, img_name)
        print(f"\n{'=' * 70}")
        print(f"  图片 [{img_idx + 1}/{len(img_list)}]: {img_name}")
        print(f"{'=' * 70}")

        # 读取原图
        img_src = cv2.imread(img_path)
        if img_src is None:
            print(f"  [警告] 无法读取图片: {img_path}，跳过")
            continue

        # 保存各模型的检测结果
        detection_results = {}

        # 对每个模型进行推理
        for name, cfg in models.items():
            print(f"\n  --- {name} ---")

            # Step 1: 预处理
            input_data = preprocess_image(img_src, cfg['platform'], co_helper)

            # Step 2: 推理 + 测时
            avg_time, times, outputs = benchmark_inference(
                cfg['model'], [input_data], cfg['platform'],
                warmup=args.warmup, test_count=args.test_count
            )

            # 记录耗时
            all_timing[name].extend(times)

            # Step 3: 后处理
            boxes, classes, scores = post_process(outputs)

            # 打印检测概况
            if boxes is not None:
                print(f"    [结果] 检测到 {len(boxes)} 个目标")
                for box, score, cl in zip(boxes, scores, classes):
                    cls_name = CLASSES[int(cl)]
                    print(f"          {cls_name:<20} score={score:.3f} "
                          f"box=({int(box[0])},{int(box[1])},{int(box[2])},{int(box[3])})")
            else:
                print(f"    [结果] 未检测到目标")

            # 保存检测结果
            detection_results[name] = {
                'boxes': boxes,
                'classes': classes,
                'scores': scores,
            }

            # 如果需要可视化，复制一份绘制的图像
            if args.img_show or args.img_save:
                img_draw = img_src.copy()
                if boxes is not None:
                    # 将检测框从 letterbox 坐标还原到原图坐标
                    real_boxes = co_helper.get_real_box(boxes)
                    draw_detections(img_draw, real_boxes, scores, classes)

                save_dir = './result'
                if args.img_save:
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(
                        save_dir,
                        f"{os.path.splitext(img_name)[0]}_{name}.jpg"
                    )
                    cv2.imwrite(save_path, img_draw)
                    print(f"    [保存] 结果已保存: {save_path}")

                if args.img_show:
                    cv2.imshow(f"{name} - {img_name}", img_draw)

        # ================================================================
        # 各模型检测结果对比
        # ================================================================
        print(f"\n  {'=' * 55}")
        print(f"  检测结果对比")
        print(f"  {'=' * 55}")
        compare_detections(detection_results, img_name)

        # 显示图像后等待按键（只在 img_show 模式下）
        if args.img_show:
            print("\n  [提示] 按任意键查看下一张图片...")
            cv2.waitKeyEx(0)
            cv2.destroyAllWindows()

    # ================================================================
    # 7. 输出汇总：耗时对比
    # ================================================================
    print(f"\n\n{'=' * 70}")
    print("  耗时对比汇总")
    print(f"{'=' * 70}")

    timing_table = []
    for name in models:
        times = all_timing[name]
        if len(times) == 0:
            continue
        times_np = np.array(times)
        timing_table.append([
            name,
            f"{np.mean(times_np):.2f}",
            f"{np.min(times_np):.2f}",
            f"{np.max(times_np):.2f}",
            f"{np.std(times_np):.2f}",
            f"{np.median(times_np):.2f}",
        ])

    # 找最快的模型作为基准
    if len(timing_table) >= 2:
        base_speed = float(timing_table[0][1])
        base_name = timing_table[0][0]
        for row in timing_table:
            speed = float(row[1])
            ratio = base_speed / speed if speed > 0 else 0
            row.append(f"{ratio:.2f}x")

    timing_headers = ["模型", "平均(ms)", "最快(ms)", "最慢(ms)",
                      "标准差(ms)", "中位数(ms)"]
    if len(timing_table) >= 2:
        timing_headers.append("加速比")

    try:
        if _HAS_TABULATE:
            print(tabulate(timing_table, headers=timing_headers, tablefmt="grid"))
        else:
            raise ImportError
    except Exception:
        print("    " + "-" * 80)
        print("    " + "".join(f"{h:<15}" for h in timing_headers))
        print("    " + "-" * 80)
        for row in timing_table:
            print("    " + "".join(f"{str(v):<15}" for v in row))
        print("    " + "-" * 80)

    # ================================================================
    # 8. 释放所有模型资源
    # ================================================================
    print(f"\n{'=' * 70}")
    print("  释放模型资源...")
    print(f"{'=' * 70}")
    for name, cfg in models.items():
        print(f"  [释放] {name} ...")
        cfg['model'].release()
    print("  所有模型已释放。")

    # ================================================================
    # 9. 最终结论
    # ================================================================
    print(f"\n{'=' * 70}")
    print("  对比结论")
    print(f"{'=' * 70}")
    print(f"  测试图片: {len(img_list)} 张")
    print(f"  测试模型: {', '.join(models.keys())}")
    for row in timing_table:
        print(f"    {row[0]:<10}: 平均 {row[1]} ms/帧")
    print(f"\n  [提示] 检测框结果保存目录: ./result/")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
