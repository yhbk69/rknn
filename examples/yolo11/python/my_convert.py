"""
===========================================================================
my_convert.py —— ONNX 模型转 RKNN 模型脚本
===========================================================================
用法:
    python3 my_convert.py <onnx_model_path> <platform> [dtype] [output_path]

示例:
    # 将 best.onnx 转为 RKNN，目标平台 rk3588，使用 int8 量化
    python3 my_convert.py best.onnx rk3588 i8

    # 转为 FP16 推理（不量化）
    python3 my_convert.py best.onnx rk3588 fp

参数说明:
    onnx_model_path  - ONNX 模型文件路径（支持相对/绝对路径）
    platform         - 目标 RKNPU 平台，可选值:
                       rk3562, rk3566, rk3568, rk3588, rk3576, rk1808, rv1109, rv1126
    dtype            - (可选) 量化类型:
                       i8/u8  -> 量化（默认），fp -> 不量化（FP16 推理）
    output_path      - (可选) 输出 RKNN 模型路径，默认: ../model/<文件名>.rknn
===========================================================================
"""

import os
import sys
import glob

# ------------------------------------------------------------
# 导入 RKNN 工具包（需先安装 rknn-toolkit2）
# ------------------------------------------------------------
from rknn.api import RKNN


def print_usage():
    """打印使用说明。"""
    script = sys.argv[0]
    print("用法:")
    print("  python3 {} <onnx_model_path> <platform> [dtype] [output_path]".format(script))
    print()
    print("示例:")
    print("  # 将 best.onnx 转为 RKNN，目标平台 rk3588，使用 int8 量化")
    print("  python3 {} best.onnx rk3588 i8".format(script))
    print()
    print("  # 转为 FP16 推理（不量化）")
    print("  python3 {} best.onnx rk3588 fp".format(script))
    print()
    print("参数说明:")
    print("  onnx_model_path  - ONNX 模型文件路径（支持相对/绝对路径）")
    print("  platform         - 目标 RKNPU 平台")
    print("                     可选值: rk3562, rk3566, rk3568, rk3588, rk3576, rk1808, rv1109, rv1126")
    print("  dtype            - (可选) 量化类型")
    print("                     i8/u8 -> 量化（默认），fp -> 不量化（FP16 推理）")
    print("  output_path      - (可选) 输出 RKNN 模型路径，默认: ../model/<文件名>.rknn")

# ------------------------------------------------------------
# 默认配置常量
# ------------------------------------------------------------
# 量化校准数据集路径（用于 int8 量化时估计激活值范围）
DATASET_PATH = '../../../datasets/COCO/coco_subset_20.txt'

# 默认是否启用量化（True = int8 量化，False = FP16 推理）
DEFAULT_QUANT = True


def find_onnx_files():
    """
    自动查找当前目录下的所有 .onnx 文件。

    返回:
        list[str]: 找到的 .onnx 文件路径列表
    """
    # 使用 glob 模式匹配当前目录下所有 .onnx 文件
    onnx_files = glob.glob("*.onnx")
    if not onnx_files:
        print("[错误] 当前目录下未找到任何 .onnx 文件！")
        print("       请将 ONNX 模型放在当前目录，或通过命令行参数指定路径。")
        sys.exit(1)
    return onnx_files


def parse_args():
    """
    解析命令行参数，支持灵活的参数传递方式。

    支持的调用方式:
        1. python3 my_convert.py --help / -h      -> 打印帮助信息
        2. python3 my_convert.py                      -> 自动查找当前目录下的 ONNX 文件
        3. python3 my_convert.py <onnx_path> <platform>  -> 转换单个文件
        4. python3 my_convert.py <onnx_path> <platform> <dtype>
        5. python3 my_convert.py <onnx_path> <platform> <dtype> <output_path>

    返回:
        model_path   - ONNX 模型路径
        platform     - 目标平台
        do_quant     - 是否启用量化
        output_path  - 输出 RKNN 模型路径
    """
    # ================================================================
    # 帮助信息
    # ================================================================
    if len(sys.argv) >= 2 and sys.argv[1] in ('--help', '-h'):
        print_usage()
        sys.exit(0)

    # ================================================================
    # 情况 1: 没有提供任何参数 -> 自动查找并转换所有 ONNX 文件
    # ================================================================
    if len(sys.argv) < 2:
        print("[信息] 未指定模型文件，将自动查找当前目录下的所有 .onnx 文件...")
        onnx_files = find_onnx_files()
        print(f"[信息] 找到以下 ONNX 文件:")
        for i, f in enumerate(onnx_files):
            print(f"       [{i}] {f}")

        # 如果有多个文件，让用户选择
        if len(onnx_files) > 1:
            print("\n[提示] 请使用完整命令指定要转换的文件:")
            print(f"       python3 {sys.argv[0]} <onnx_path> <platform> [dtype]")
            print("       例如: python3 {} {} rk3588 i8".format(
                sys.argv[0], onnx_files[0]))
            sys.exit(1)

        # 只有一个文件，直接使用
        model_path = onnx_files[0]
        platform = input("请输入目标平台 (如 rk3588, rk3566 等): ").strip()
        if not platform:
            print("[错误] 平台不能为空！")
            sys.exit(1)
        do_quant = DEFAULT_QUANT
        output_path = None
        return model_path, platform, do_quant, output_path

    # ================================================================
    # 情况 2: 提供了至少模型路径和平台
    # ================================================================
    model_path = sys.argv[1]
    platform = sys.argv[2]

    # 验证平台是否合法
    valid_platforms = [
        'rk3562', 'rk3566', 'rk3568', 'rk3588',
        'rk3576', 'rk1808', 'rv1109', 'rv1126'
    ]
    if platform not in valid_platforms:
        print(f"[错误] 不支持的平台: {platform}")
        print(f"       可选平台: {', '.join(valid_platforms)}")
        sys.exit(1)

    # ================================================================
    # 可选参数: 量化类型 (i8/u8 -> 量化, fp -> 浮点)
    # ================================================================
    do_quant = DEFAULT_QUANT
    if len(sys.argv) > 3:
        dtype = sys.argv[3]
        if dtype not in ['i8', 'u8', 'fp']:
            print(f"[错误] 无效的量化类型: {dtype}")
            print("       可选值: i8 (int8量化), u8 (uint8量化), fp (浮点推理)")
            sys.exit(1)
        do_quant = (dtype in ['i8', 'u8'])

    # ================================================================
    # 可选参数: 输出路径（不指定则自动生成）
    # ================================================================
    if len(sys.argv) > 4:
        output_path = sys.argv[4]
    else:
        output_path = None  # 后续自动生成

    return model_path, platform, do_quant, output_path


def get_output_path(model_path: str, do_quant: bool) -> str:
    """
    根据输入模型路径和量化设置，自动生成输出 RKNN 模型路径。

    规则:
        - 输出路径为 ../model/<原文件名>_<量化类型>.rknn
        - 例如: best.onnx -> ../model/best_i8.rknn (量化)
        - 例如: best.onnx -> ../model/best_fp.rknn (浮点)

    参数:
        model_path  - 输入 ONNX 模型路径
        do_quant    - 是否启用量化

    返回:
        str: 自动生成的输出路径
    """
    # 获取不带扩展名的文件名
    base_name = os.path.splitext(os.path.basename(model_path))[0]
    # 根据是否量化添加后缀
    suffix = "i8" if do_quant else "fp"
    # 构建输出路径: ../model/<name>_<suffix>.rknn
    output_path = os.path.join("../model", f"{base_name}_{suffix}.rknn")
    return output_path


def convert_onnx_to_rknn(
    model_path: str,
    platform: str,
    do_quant: bool,
    output_path: str = None
):
    """
    ONNX -> RKNN 模型转换主函数。

    转换流程:
        1. 创建 RKNN 对象
        2. 配置模型参数（均值、方差、目标平台）
        3. 加载 ONNX 模型
        4. 构建 RKNN 模型（量化和优化）
        5. 导出 .rknn 文件
        6. 释放资源

    参数:
        model_path  - ONNX 模型文件路径
        platform    - 目标 RKNPU 平台
        do_quant    - 是否启用 int8 量化
        output_path - 输出 RKNN 文件路径（None 则自动生成）
    """
    # ----------------------------------------------------------------
    # 如果未指定输出路径，自动生成
    # ----------------------------------------------------------------
    if output_path is None:
        output_path = get_output_path(model_path, do_quant)

    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"[信息] 已创建输出目录: {output_dir}")

    # ----------------------------------------------------------------
    # 打印转换配置信息
    # ----------------------------------------------------------------
    print("=" * 60)
    print(f"ONNX -> RKNN 模型转换")
    print("=" * 60)
    print(f"  输入模型 : {model_path}")
    print(f"  输出模型 : {output_path}")
    print(f"  目标平台 : {platform}")
    print(f"  量化类型 : {'int8 量化' if do_quant else 'FP16 浮点'}")
    print(f"  数据集   : {DATASET_PATH if do_quant else '(浮点模式无需数据集)'}")
    print("=" * 60)

    # ----------------------------------------------------------------
    # 检查 ONNX 文件是否存在
    # ----------------------------------------------------------------
    if not os.path.exists(model_path):
        print(f"[错误] ONNX 模型文件不存在: {model_path}")
        sys.exit(1)

    # ----------------------------------------------------------------
    # 检查量化数据集是否存在（量化模式下需要）
    # ----------------------------------------------------------------
    if do_quant and not os.path.exists(DATASET_PATH):
        print(f"[警告] 量化数据集不存在: {DATASET_PATH}")
        print("       量化过程可能失败或精度下降！")
        print(f"       请确保 {DATASET_PATH} 文件包含约 20 张图片路径，每行一个。")

    # ----------------------------------------------------------------
    # Step 1: 创建 RKNN 对象
    # ----------------------------------------------------------------
    print("\n[Step 1/6] 创建 RKNN 对象...")
    rknn = RKNN(verbose=False)  # verbose=False 减少输出冗余
    print("[完成]")

    # ----------------------------------------------------------------
    # Step 2: 配置模型参数
    # ----------------------------------------------------------------
    print("\n[Step 2/6] 配置模型参数...")
    ret = rknn.config(
        # 均值: YOLO 模型通常不需要减均值，设为 [0,0,0]
        mean_values=[[0, 0, 0]],
        # 标准差: 输入归一化到 [0,1] 区间，所以除以 255
        std_values=[[255, 255, 255]],
        # 目标平台
        target_platform=platform
    )
    if ret != 0:
        print(f"[错误] 配置模型失败，错误码: {ret}")
        exit(ret)
    print("[完成]")

    # ----------------------------------------------------------------
    # Step 3: 加载 ONNX 模型
    # ----------------------------------------------------------------
    print(f"\n[Step 3/6] 加载 ONNX 模型: {model_path}...")
    ret = rknn.load_onnx(model=model_path)
    if ret != 0:
        print(f"[错误] 加载 ONNX 模型失败，错误码: {ret}")
        print("       可能的原因:")
        print("         - ONNX 文件损坏")
        print("         - ONNX opset 版本不兼容")
        print("         - 模型包含不支持的算子")
        exit(ret)
    print("[完成]")

    # ----------------------------------------------------------------
    # Step 4: 构建 RKNN 模型
    # ----------------------------------------------------------------
    quant_str = "int8 量化" if do_quant else "FP16 浮点"
    print(f"\n[Step 4/6] 构建 RKNN 模型 ({quant_str})...")
    print(f"       量化数据集: {DATASET_PATH if do_quant else 'N/A'}")
    print("       这可能需要几分钟，请耐心等待...")
    ret = rknn.build(
        do_quantization=do_quant,
        dataset=DATASET_PATH  # 量化时需要数据集用于校准
    )
    if ret != 0:
        print(f"[错误] 构建 RKNN 模型失败，错误码: {ret}")
        print("       可能的原因:")
        print("         - 量化数据集路径不正确")
        print("         - 数据集中的图片路径无效")
        print("         - DKNN 版本与 ONNX 模型不兼容")
        exit(ret)
    print("[完成]")

    # ----------------------------------------------------------------
    # Step 5: 导出 RKNN 模型文件
    # ----------------------------------------------------------------
    print(f"\n[Step 5/6] 导出 RKNN 模型: {output_path}...")
    ret = rknn.export_rknn(output_path)
    if ret != 0:
        print(f"[错误] 导出 RKNN 模型失败，错误码: {ret}")
        exit(ret)
    print("[完成]")

    # ----------------------------------------------------------------
    # Step 6: 释放 RKNN 资源
    # ----------------------------------------------------------------
    print("\n[Step 6/6] 释放 RKNN 资源...")
    rknn.release()
    print("[完成]")

    # ----------------------------------------------------------------
    # 输出文件大小信息
    # ----------------------------------------------------------------
    if os.path.exists(output_path):
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"\n{'=' * 60}")
        print(f"转换成功！")
        print(f"  输出文件: {output_path}")
        print(f"  文件大小: {file_size_mb:.2f} MB")
        print(f"{'=' * 60}")
    else:
        print(f"\n[警告] 输出文件未找到，可能导出失败。")


def main():
    """
    主入口函数。
    """
    # ================================================================
    # 解析命令行参数
    # ================================================================
    model_path, platform, do_quant, output_path = parse_args()

    # ================================================================
    # 执行 ONNX -> RKNN 转换
    # ================================================================
    convert_onnx_to_rknn(model_path, platform, do_quant, output_path)


if __name__ == '__main__':
    main()
