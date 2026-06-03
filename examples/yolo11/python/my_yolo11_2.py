
"""
===========================================================================
my_yolo11_2.py —— YOLO11 多模型对比推理（增强版）
===========================================================================
功能:
    在 my_yolo11.py 基础上，额外输出：
      - 每张图片 × 每个模型的 JSON 检测结果文件
      - 经过 NMS 的标注图像（与 my_yolo11.py 的 --img_save 区分）
      - comparison_report_2.md 汇总报告（含各类别检出明细）

用法:
    python3 my_yolo11_2.py --img_folder ./imgs

输出:
    ./result/{img}_{model}_2.jpg      — NMS 后的标注图像
    ./result/{img}_{model}_2.json     — 检测结果（类别 + 置信度）
    ./result/comparison_report_2.md   — 汇总报告
===========================================================================
"""

import os
import cv2
import sys
import json
import time
import argparse
import numpy as np

# =========================================================================
# 路径处理
# =========================================================================
realpath = os.path.abspath(__file__)
_sep = os.path.sep
realpath = realpath.split(_sep)
try:
    zoo_idx = realpath.index('rknn_model_zoo-v2.3.0')
    sys.path.append(os.path.join(realpath[0] + _sep, *realpath[1:zoo_idx + 1]))
except ValueError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(realpath))))

# =========================================================================
# 从 my_yolo11.py 复用核心函数（保持原文件不变）
# =========================================================================
from my_yolo11 import (
    draw_detections, preprocess_image,
    load_model, auto_find_models, is_image_file, compare_detections,
    CLASSES, IMG_SIZE, WARMUP_COUNT, TEST_COUNT,
)
from py_utils.coco_utils import COCO_test_helper

# =========================================================================
# 检测超参数（独立定义，不依赖 my_yolo11.py）
# =========================================================================
OBJ_THRESH = 0.55      # 置信度阈值：低于此分的框直接丢弃
NMS_THRESH = 0.45      # NMS IoU 阈值：越小重叠抑制越严格（原 0.45 太松）
MAX_DETECTIONS = 100   # NMS 后最多保留的检测框数（按置信度取 top-K）

# 用于命名后缀，与 my_yolo11.py 的结果区分
OUTPUT_SUFFIX = "_3"


# =========================================================================
# NMS 后处理（本地定义，使用本文件的超参数）
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
    """YOLO11 后处理。使用本文件定义的 OBJ_THRESH / NMS_THRESH / MAX_DETECTIONS。"""
    for data in input_data:
        if len(data.shape) == 3 and data.shape[1] == 15:
            return _post_process_single(data)
    return _post_process_multi(input_data)


def _post_process_single(pred):
    """处理 ultralytics 导出的单输出格式 [1, 15, 8400]。
    
    15 通道 = 4(bbox cx,cy,w,h) + 11(class probs)
    YOLOv8/v11 无 objectness 分支，类别概率已含目标置信度。
    """
    if pred.shape[1] != 15 and pred.shape[2] == 15:
        pred = pred.transpose(0, 2, 1)
    if pred.shape[1] != 15:
        raise ValueError(f"Unexpected shape: {pred.shape}")

    pred = pred[0].T                     # (8400, 15)

    # YOLOv8/v11: 15 = 4(bbox) + 11(class)，无 objectness
    bbox = pred[:, :4]                   # cx, cy, w, h
    class_probs = pred[:, 4:]            # 11 个类别的原始 logits

    # Sigmoid 得到概率
    class_probs = 1.0 / (1.0 + np.exp(-class_probs))

    # 直接取最大概率作为综合得分（YOLOv8/v11 无独立 objectness）
    classes = np.argmax(class_probs, axis=1)
    max_scores = np.max(class_probs, axis=1)

    keep_mask = max_scores >= OBJ_THRESH
    if not np.any(keep_mask):
        return None, None, None

    boxes = bbox[keep_mask]
    classes = classes[keep_mask]
    scores = max_scores[keep_mask]

    # cx,cy,w,h -> x1,y1,x2,y2
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

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    # MAX_DETECTIONS：按置信度取 top-K
    if len(scores) > MAX_DETECTIONS:
        topk = np.argsort(scores)[::-1][:MAX_DETECTIONS]
        boxes = boxes[topk]
        classes = classes[topk]
        scores = scores[topk]

    return boxes, classes, scores


def _post_process_multi(input_data):
    """后备方案：处理原始 9 输出格式（3 检测头 × 3 分支）。"""
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

    box_conf = scores.reshape(-1)
    class_max = np.max(classes_conf, axis=-1)
    classes = np.argmax(classes_conf, axis=-1)
    pos = np.where(class_max * box_conf >= OBJ_THRESH)
    scores = (class_max * box_conf)[pos]
    boxes = boxes[pos]
    classes = classes[pos]

    if len(classes) == 0:
        return None, None, None

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

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    # MAX_DETECTIONS
    if len(scores) > MAX_DETECTIONS:
        topk = np.argsort(scores)[::-1][:MAX_DETECTIONS]
        boxes = boxes[topk]
        classes = classes[topk]
        scores = scores[topk]

    return boxes, classes, scores


# =========================================================================
# 命令行参数
# =========================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description='YOLO11 多模型对比（增强版）：输出 JSON + md 汇总报告'
    )

    # ---- 模型参数 ----
    parser.add_argument('--pt_model', type=str, default=None,
                        help='PyTorch 模型路径')
    parser.add_argument('--onnx_model', type=str, default=None,
                        help='ONNX 模型路径')
    parser.add_argument('--rknn_model', type=str, default=None,
                        help='RKNN 模型路径')
    parser.add_argument('--target', type=str, default='rk3588',
                        help='RKNN 目标平台（默认: rk3588）')
    parser.add_argument('--device_id', type=str, default=None)

    # ---- 数据参数 ----
    parser.add_argument('--img_folder', type=str, default=None,
                        help='测试图片文件夹')

    # ---- 测试参数 ----
    parser.add_argument('--warmup', type=int, default=WARMUP_COUNT,
                        help=f'预热次数（默认: {WARMUP_COUNT}）')
    parser.add_argument('--test_count', type=int, default=5,
                        help=f'耗时测试次数，取平均（默认: 5）')

    return parser.parse_args()


# =========================================================================
# 单次推理 + 5次平均耗时
# =========================================================================
def infer_with_timing(model, input_data, warmup=1, test_count=5):
    """
    执行推理并测量平均耗时。

    返回:
        outputs  - 最后一次推理的 raw 输出
        avg_ms   - test_count 次的平均耗时 (ms)
        times_ms - 每次耗时的列表 (ms)
    """
    # 预热
    for _ in range(warmup):
        _ = model.run(input_data)

    # 正式测试
    times = []
    outputs = None
    for i in range(test_count):
        start = time.perf_counter()
        outputs = model.run(input_data)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)

    avg_ms = np.mean(times)
    return outputs, avg_ms, times


# =========================================================================
# 保存 JSON 检测结果
# =========================================================================
def save_json_result(save_dir, img_name, model_name, avg_ms, boxes, classes, scores):
    """
    保存检测结果为 JSON 文件（不含坐标，只含类别+置信度）。
    文件: ./result/{img}_{model}_2.json
    """
    detections = []
    if boxes is not None:
        for cls_id, score in zip(classes, scores):
            detections.append({
                "class": CLASSES[int(cls_id)],
                "score": round(float(score), 4),
            })

    data = {
        "image": img_name,
        "model": model_name,
        "inference_time_ms": round(avg_ms, 2),
        "num_detections": len(detections),
        "detections": detections,
    }

    base = os.path.splitext(img_name)[0]
    filename = f"{base}_{model_name}{OUTPUT_SUFFIX}.json"
    filepath = os.path.join(save_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return filepath


# =========================================================================
# 保存 NMS 后的标注图像
# =========================================================================
def save_annotated_image(save_dir, img_src, img_name, model_name,
                         boxes, classes, scores, co_helper):
    """
    在图像上绘制 NMS 后的检测框并保存。
    文件: ./result/{img}_{model}_2.jpg
    """
    img_draw = img_src.copy()
    if boxes is not None:
        real_boxes = co_helper.get_real_box(boxes)
        draw_detections(img_draw, real_boxes, scores, classes)

    base = os.path.splitext(img_name)[0]
    filename = f"{base}_{model_name}{OUTPUT_SUFFIX}.jpg"
    filepath = os.path.join(save_dir, filename)
    cv2.imwrite(filepath, img_draw)
    return filepath


# =========================================================================
# 按类别统计检出数（用于 md 对比表）
# =========================================================================
def count_by_class(boxes, classes, scores):
    """统计每个类别的检出数量。返回 {类别名: 数量}"""
    counts = {}
    if boxes is None:
        return counts
    for cls_id in classes:
        name = CLASSES[int(cls_id)]
        counts[name] = counts.get(name, 0) + 1
    return counts


# =========================================================================
# 生成 comparison_report_2.md
# =========================================================================
def generate_md_report(save_dir, img_list, per_image_data, model_names,
                       all_classes, class_totals, timing_summary):
    """
    生成汇总 md 报告。

    参数:
        per_image_data - [{img_name, dets: {模型名: {counts, avg_ms, num_det}}}]
        model_names    - 模型名称列表
        all_classes    - 所有出现的类别名列表（排序后）
        class_totals   - {模型名: {类别名: 总检出数}}
        timing_summary - {模型名: {mean, min, max, std, median}}
    """
    pass  # 未使用，由下方增量函数替代


# =========================================================================
# 增量写入 md（每张图片处理完后追加）
# =========================================================================
def md_append_image(md_path, img_name, model_names, img_dets):
    """
    将一张图片的检测结果追加到 md 文件中。
    这样即使脚本中途中断，已处理的图片数据也不会丢失。
    """
    lines = []
    lines.append("---")
    lines.append("")
    lines.append(f"## {img_name}")
    lines.append("")

    for model_name in model_names:
        det = img_dets.get(model_name, {})
        avg_ms = det.get("avg_ms", 0)
        dets_list = det.get("dets_list", [])
        num_det = det.get("num_det", 0)

        lines.append(f"### {model_name}")
        lines.append("")
        lines.append(f"- 推理耗时: {avg_ms:.2f} ms")
        lines.append(f"- 检测到 {num_det} 个目标:")
        lines.append("")

        if dets_list:
            for d in dets_list:
                lines.append(f"  - {d['class']} ({d['score']:.4f})")
        else:
            lines.append("  - *无检测结果*")
        lines.append("")

    with open(md_path, 'a', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def md_write_header(md_path, img_list, model_names):
    """写入 md 文件头部。"""
    lines = []
    lines.append("# YOLO11 多模型对比推理报告")
    lines.append("")
    lines.append(f"- 测试图片: {len(img_list)} 张")
    lines.append(f"- 测试模型: {', '.join(model_names)}")
    lines.append(f"- 生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def md_append_comparison(md_path, model_names, all_classes, class_totals, timing_summary):
    """在 md 末尾追加全局对比表格。"""
    lines = []
    lines.append("---")
    lines.append("")
    lines.append("## 全局对比")
    lines.append("")

    # --- 耗时对比 ---
    lines.append("### 耗时对比")
    lines.append("")
    header = ["模型", "平均(ms)", "最快(ms)", "最慢(ms)", "标准差(ms)", "中位数(ms)"]
    if len(model_names) >= 2:
        header.append("加速比")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")

    base_mean = timing_summary.get(model_names[0], {}).get("mean", 1) if model_names else 1

    for model_name in model_names:
        ts = timing_summary.get(model_name, {})
        row = [
            model_name,
            f"{ts.get('mean', 0):.2f}",
            f"{ts.get('min', 0):.2f}",
            f"{ts.get('max', 0):.2f}",
            f"{ts.get('std', 0):.2f}",
            f"{ts.get('median', 0):.2f}",
        ]
        if base_mean and base_mean > 0:
            ratio = base_mean / ts.get("mean", 1) if ts.get("mean", 1) > 0 else 0
            row.append(f"{ratio:.2f}x")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")

    # --- 各类别检出数量对比 ---
    lines.append("### 各类别检出数量")
    lines.append("")
    cl_header = ["模型", "总检出"] + all_classes
    lines.append("| " + " | ".join(cl_header) + " |")
    lines.append("|" + "|".join(["---"] * len(cl_header)) + "|")

    for model_name in model_names:
        ct = class_totals.get(model_name, {})
        total = sum(ct.values())
        row = [model_name, str(total)]
        for cls_name in all_classes:
            row.append(str(ct.get(cls_name, 0)))
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")

    # --- 结论 ---
    lines.append("### 结论")
    lines.append("")
    for model_name in model_names:
        ts = timing_summary.get(model_name, {})
        ct = class_totals.get(model_name, {})
        total_det = sum(ct.values())
        lines.append(f"- **{model_name}**: 平均 {ts.get('mean', 0):.2f} ms/帧，共检出 {total_det} 个目标")
    lines.append("")

    with open(md_path, 'a', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


# =========================================================================
# 主函数
# =========================================================================
def main():
    args = parse_args()

    print("=" * 70)
    print("  YOLO11 多模型对比推理工具 (增强版)")
    print("=" * 70)

    # ================================================================
    # 1. 确定图片文件夹
    # ================================================================
    img_folder = args.img_folder
    if img_folder is None:
        candidates = ['../model', '.']
        for c in candidates:
            if os.path.isdir(c):
                img_folder = c
                break

    print(f"\n[配置] 测试图片文件夹: {img_folder}")

    if not os.path.isdir(img_folder):
        print(f"[错误] 文件夹不存在: {img_folder}")
        sys.exit(1)

    img_list = sorted([f for f in os.listdir(img_folder) if is_image_file(f)])
    if not img_list:
        print(f"[错误] 文件夹中没有图片文件: {img_folder}")
        sys.exit(1)
    print(f"       找到 {len(img_list)} 张测试图片: {img_list}")

    # ================================================================
    # 2. 确定模型文件
    # ================================================================
    pt_model = args.pt_model
    onnx_model = args.onnx_model
    rknn_model = args.rknn_model

    auto_pt, auto_onnx, auto_rknn = auto_find_models()
    if pt_model is None:
        pt_model = auto_pt
    if onnx_model is None:
        onnx_model = auto_onnx
    if rknn_model is None:
        rknn_model = auto_rknn

    model_configs = []
    if pt_model and os.path.exists(pt_model):
        model_configs.append(("PyTorch", pt_model, "pytorch"))
    if onnx_model and os.path.exists(onnx_model):
        model_configs.append(("ONNX", onnx_model, "onnx"))
    if rknn_model and os.path.exists(rknn_model):
        model_configs.append(("RKNN", rknn_model, "rknn"))

    if not model_configs:
        print("[错误] 未找到任何模型文件！")
        sys.exit(1)

    print(f"\n[配置] 参与对比的模型:")
    for name, path, _ in model_configs:
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"       {name:<8}: {path} ({size_mb:.2f} MB)")

    print(f"[配置] RKNN 目标平台: {args.target}")
    print(f"[配置] 预热: {args.warmup} 次, 耗时测试: {args.test_count} 次(平均)")

    # ================================================================
    # 3. 加载所有模型
    # ================================================================
    print(f"\n{'=' * 70}")
    print("  加载模型...")
    print(f"{'=' * 70}")

    models = {}
    for name, path, platform in model_configs:
        print(f"\n  [{name}]")
        try:
            model, actual_platform = load_model(
                path,
                target=args.target if platform == 'rknn' else None,
                device_id=args.device_id if platform == 'rknn' else None,
            )
            models[name] = {'model': model, 'platform': actual_platform}
        except Exception as e:
            print(f"    [错误] 加载失败: {e}")

    if not models:
        print("[错误] 所有模型加载失败！")
        sys.exit(1)

    # ================================================================
    # 4. 初始化
    # ================================================================
    co_helper = COCO_test_helper(enable_letter_box=True)
    save_dir = './result/3'
    os.makedirs(save_dir, exist_ok=True)

    # ---- 汇总统计 ----
    model_names = list(models.keys())
    all_timing = {name: [] for name in model_names}
    # 全局类别检出统计: {模型名: {类别名: 数量}}
    class_totals = {name: {} for name in model_names}
    # 收集所有出现过的类别
    all_classes_set = set()

    # ---- 初始化 md 文件（写头部） ----
    md_path = os.path.join(save_dir, f"comparison_report{OUTPUT_SUFFIX}.md")
    md_write_header(md_path, img_list, model_names)

    # ================================================================
    # 5. 逐张图片推理
    # ================================================================
    print(f"\n{'=' * 70}")
    print("  开始推理...")
    print(f"{'=' * 70}")

    for img_idx, img_name in enumerate(img_list):
        img_path = os.path.join(img_folder, img_name)
        print(f"\n  [{img_idx + 1}/{len(img_list)}] {img_name}")

        img_src = cv2.imread(img_path)
        if img_src is None:
            print(f"    [跳过] 无法读取")
            continue

        # 本张图片各模型的数据
        img_dets = {}

        for name, cfg in models.items():
            print(f"    --- {name} ---")

            # 预处理
            input_data = preprocess_image(img_src, cfg['platform'], co_helper)

            # 推理 + test_count 次平均耗时
            outputs, avg_ms, times = infer_with_timing(
                cfg['model'], [input_data],
                warmup=args.warmup, test_count=args.test_count
            )
            all_timing[name].extend(times)

            # NMS 后处理
            boxes, classes, scores = post_process(outputs)

            num_det = len(boxes) if boxes is not None else 0
            print(f"      耗时: {avg_ms:.2f} ms, 检出: {num_det} 个目标")

            # 保存 JSON 检测结果
            json_path = save_json_result(
                save_dir, img_name, name, avg_ms, boxes, classes, scores
            )
            print(f"      JSON: {json_path}")

            # 保存 NMS 后的标注图像
            img_path_saved = save_annotated_image(
                save_dir, img_src, img_name, name,
                boxes, classes, scores, co_helper
            )
            print(f"      图像: {img_path_saved}")

            # 收集本图检测结果（用于追加 md）
            dets_list = []
            if boxes is not None:
                for cls_id, score in zip(classes, scores):
                    dets_list.append({
                        "class": CLASSES[int(cls_id)],
                        "score": round(float(score), 4),
                    })

            img_dets[name] = {
                "avg_ms": round(avg_ms, 2),
                "num_det": num_det,
                "dets_list": dets_list,
            }

            # 按类别统计（全局）
            counts = count_by_class(boxes, classes, scores)
            for cls_name, cnt in counts.items():
                class_totals[name][cls_name] = class_totals[name].get(cls_name, 0) + cnt
                all_classes_set.add(cls_name)

        # 本图结果追加到 md
        md_append_image(md_path, img_name, model_names, img_dets)
        print()

    # ================================================================
    # 6. 耗时汇总 + 写入 md 全局对比表
    # ================================================================
    print(f"\n{'=' * 70}")
    print("  耗时汇总")
    print(f"{'=' * 70}")

    timing_summary = {}
    for name in model_names:
        times = np.array(all_timing[name])
        if len(times) == 0:
            continue
        timing_summary[name] = {
            "mean": float(np.mean(times)),
            "min": float(np.min(times)),
            "max": float(np.max(times)),
            "std": float(np.std(times)),
            "median": float(np.median(times)),
        }
        print(f"  {name:<8}: 平均 {timing_summary[name]['mean']:.2f} ms")

    all_classes_sorted = sorted(all_classes_set,
                                key=lambda x: list(CLASSES).index(x) if x in CLASSES else 999)

    # 追加全局对比表到 md
    md_append_comparison(md_path, model_names, all_classes_sorted,
                         class_totals, timing_summary)
    print(f"\n[md 报告] {md_path}")

    # ================================================================
    # 7. 释放模型
    # ================================================================
    print(f"\n{'=' * 70}")
    print("  释放模型...")
    for name, cfg in models.items():
        cfg['model'].release()
    print("  完成。")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
