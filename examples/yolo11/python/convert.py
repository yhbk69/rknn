#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO11 ONNX 转 RKNN 模型转换脚本

功能：将 YOLO11 的 ONNX 模型转换为 Rockchip NPU 可用的 RKNN 格式，支持量化

使用方法:
    cd /home/ztl/dltt/code/rknn/rknn_model_zoo-v2.3.0/examples/yolo11/python

    # 基本用法（默认量化为 int8，输出到 ../model/yolo11.rknn）
    python convert.py ../model/yolo11n.onnx rk3588

    # 指定平台和不量化（fp16 精度）
    python convert.py ../model/yolo11n.onnx rk3588 fp

    # 指定平台、量化类型和输出路径
    python convert.py ../model/yolo11n.onnx rk3566 i8 ../model/yolo11n_int8.rknn

    # RV1109 平台（使用 u8 量化）
    python convert.py ../model/yolo11n.onnx rv1109 u8 ../model/yolo11n_rv1109.rknn

参数说明:
    参数1: onnx_model_path  - ONNX 模型路径（必填）
    参数2: platform          - 目标平台（必填），可选：rk3562, rk3566, rk3568, rk3588, rk3576, rk1808, rv1109, rv1126
    参数3: dtype             - 数据类型（可选），i8/u8 表示量化，fp 表示浮点
    参数4: output_rknn_path  - 输出路径（可选），默认 ../model/yolo11.rknn

注意事项:
    1. 环境要求：
       - 必须安装 rknn-toolkit2（pip install rknn-toolkit2）
       - 建议在 x86_64 Linux 系统上运行转换（NPU 开发板上通常只能推理不能转换）
       - 需要 Python 3.6+

    2. 平台兼容性：
       - rk3562/rk3566/rk3568/rk3588/rk3576 支持 i8（int8）量化和 fp（float16）
       - rk1808/rv1109/rv1126 支持 u8（uint8）量化和 fp（float16）
       - 不同平台生成的 .rknn 文件不通用，必须为每个目标平台单独转换

    3. 量化说明：
       - 量化（i8/u8）：模型更小、推理更快，但精度可能略有下降
       - 不量化（fp）：精度更高，但模型较大、推理较慢
       - 量化需要校准数据集（DATASET_PATH），默认使用 COCO 子集
       - 如果量化失败，可尝试检查数据集路径是否正确或改用 fp 模式

    4. 数据集要求：
       - DATASET_PATH 指向一个文本文件，每行包含一张校准图片的路径
       - 图片应该是待检测场景的代表性样本（建议 100-1000 张）
       - 默认路径：../../../datasets/COCO/coco_subset_20.txt
       - 如果数据集不存在，量化会失败

    5. 模型要求：
       - 输入的 ONNX 模型必须是 YOLO11 导出的标准格式
       - 建议使用 ultralytics 导出：yolo export model=yolo11n.pt format=onnx
       - ONNX 模型的输入尺寸应与推理时一致（默认 640x640）

    6. 常见问题：
       - "Load model failed"：检查 ONNX 模型路径是否正确、文件是否损坏
       - "Build model failed"：检查数据集路径、平台名称是否正确
       - 内存不足：转换大模型时需要较多内存（建议 8GB+）
       - 权限错误：确保输出目录有写入权限

    7. 性能优化建议：
       - 生产环境推荐使用量化模型（i8/u8）以获得最佳性能
       - 如果检测精度不满足要求，可尝试 fp 模式
       - 可以为不同平台生成不同量化精度的模型进行测试对比
"""

import sys
from rknn.api import RKNN

# 校准数据集路径：用于量化的参考图片列表（每行一个图片路径）
DATASET_PATH = '../../../datasets/COCO/coco_subset_20.txt'
# 默认输出路径
DEFAULT_RKNN_PATH = '../model/yolo11.rknn'
# 默认启用量化
DEFAULT_QUANT = True

def parse_arg():
    """
    解析命令行参数
    
    返回:
        model_path: ONNX 模型路径
        platform: 目标平台名称
        do_quant: 是否量化
        output_path: 输出 RKNN 模型路径
    """
    if len(sys.argv) < 3:
        print("Usage: python3 {} onnx_model_path [platform] [dtype(optional)] [output_rknn_path(optional)]".format(sys.argv[0]))
        print("       platform choose from [rk3562,rk3566,rk3568,rk3588,rk3576,rk1808,rv1109,rv1126]")
        print("       dtype choose from [i8, fp] for [rk3562,rk3566,rk3568,rk3588,rk3576]")
        print("       dtype choose from [u8, fp] for [rk1808,rv1109,rv1126]")
        exit(1)

    model_path = sys.argv[1]
    platform = sys.argv[2]

    do_quant = DEFAULT_QUANT
    if len(sys.argv) > 3:
        model_type = sys.argv[3]
        if model_type not in ['i8', 'u8', 'fp']:
            print("ERROR: Invalid model type: {}".format(model_type))
            exit(1)
        elif model_type in ['i8', 'u8']:
            do_quant = True
        else:
            do_quant = False

    if len(sys.argv) > 4:
        output_path = sys.argv[4]
    else:
        output_path = DEFAULT_RKNN_PATH

    return model_path, platform, do_quant, output_path

if __name__ == '__main__':
    # 解析命令行参数
    model_path, platform, do_quant, output_path = parse_arg()

    # 创建 RKNN 对象
    rknn = RKNN(verbose=False)

    # 配置预处理参数
    # mean_values: 均值，std_values: 标准差，用于图像归一化
    # 这里配置为 (x - 0) / 255，即将像素值归一化到 [0, 1]
    print('--> Config model')
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform=platform)
    print('done')

    # 加载 ONNX 模型
    print('--> Loading model')
    ret = rknn.load_onnx(model=model_path)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # 构建 RKNN 模型
    # do_quantization: 是否量化，dataset: 校准数据集路径（仅量化时需要）
    print('--> Building model')
    ret = rknn.build(do_quantization=do_quant, dataset=DATASET_PATH)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # 导出 RKNN 模型到文件
    print('--> Export rknn model')
    ret = rknn.export_rknn(output_path)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    # 释放资源
    rknn.release()
    print('模型转换完成: {}'.format(output_path))
