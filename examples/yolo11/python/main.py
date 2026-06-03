#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
硬件测试脚本 - 检测 CPU、GPU 和 NPU 是否可用

支持的测试：
1. CPU 测试 - 使用 numpy 进行矩阵运算
2. GPU 测试 - 检测 CUDA 和 OpenCL 是否可用
3. NPU 测试 - 检测 RKNN Runtime 是否可用
"""

import os
import sys
import time

def test_cpu():
    """测试 CPU 是否可用"""
    print("=" * 60)
    print("1. CPU 测试")
    print("=" * 60)
    
    try:
        import numpy as np
        print("✓ NumPy 已安装")
        
        # 执行简单的矩阵运算
        start = time.time()
        a = np.random.rand(1000, 1000)
        b = np.random.rand(1000, 1000)
        c = np.dot(a, b)
        elapsed = time.time() - start
        
        print(f"✓ CPU 矩阵运算成功 (1000x1000)")
        print(f"  耗时: {elapsed:.4f} 秒")
        print(f"  结果形状: {c.shape}")
        print(f"  结果示例: {c[0, 0]:.4f}")
        return True
    except Exception as e:
        print(f"✗ CPU 测试失败: {e}")
        return False

def test_gpu_cuda():
    """测试 NVIDIA GPU (CUDA) 是否可用"""
    print("\n" + "=" * 60)
    print("2. GPU (CUDA) 测试")
    print("=" * 60)
    
    try:
        import torch
        print("✓ PyTorch 已安装")
        
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"✓ CUDA 可用")
            print(f"  GPU 数量: {torch.cuda.device_count()}")
            print(f"  当前 GPU: {torch.cuda.get_device_name(0)}")
            
            # 执行简单的 GPU 运算
            start = time.time()
            a = torch.rand(1000, 1000, device=device)
            b = torch.rand(1000, 1000, device=device)
            c = torch.matmul(a, b)
            torch.cuda.synchronize()
            elapsed = time.time() - start
            
            print(f"✓ GPU 矩阵运算成功 (1000x1000)")
            print(f"  耗时: {elapsed:.4f} 秒")
            print(f"  结果形状: {c.shape}")
            return True
        else:
            print("✗ CUDA 不可用")
            return False
    except ImportError:
        print("✗ PyTorch 未安装，跳过 CUDA 测试")
        return False
    except Exception as e:
        print(f"✗ CUDA 测试失败: {e}")
        return False

def test_gpu_opencl():
    """测试 OpenCL GPU 是否可用"""
    print("\n" + "=" * 60)
    print("3. GPU (OpenCL) 测试")
    print("=" * 60)
    
    try:
        import pyopencl as cl
        print("✓ PyOpenCL 已安装")
        
        platforms = cl.get_platforms()
        if platforms:
            print(f"✓ OpenCL 平台数量: {len(platforms)}")
            
            for i, platform in enumerate(platforms):
                print(f"\n  平台 {i+1}: {platform.name}")
                devices = platform.get_devices()
                print(f"    设备数量: {len(devices)}")
                
                for j, device in enumerate(devices):
                    print(f"    设备 {j+1}: {device.name}")
                    print(f"      类型: {device.type}")
                    print(f"      最大计算单元: {device.max_compute_units}")
                    print(f"      最大工作组大小: {device.max_work_group_size}")
            
            return True
        else:
            print("✗ 未找到 OpenCL 平台")
            return False
    except ImportError:
        print("✗ PyOpenCL 未安装，跳过 OpenCL 测试")
        return False
    except Exception as e:
        print(f"✗ OpenCL 测试失败: {e}")
        return False

def test_npu():
    """测试 NPU (RKNN) 是否可用"""
    print("\n" + "=" * 60)
    print("4. NPU (RKNN) 测试")
    print("=" * 60)
    
    try:
        # 添加 rknn_model_zoo 路径
        realpath = os.path.abspath(__file__)
        _sep = os.path.sep
        realpath = realpath.split(_sep)
        try:
            zoo_idx = realpath.index('rknn_model_zoo-v2.3.0')
            sys.path.append(os.path.join(realpath[0]+_sep, *realpath[1:zoo_idx+1]))
        except ValueError:
            sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(realpath))))
        
        import rknn
        print("✓ RKNN Runtime 已安装")
        print(f"  RKNN 版本: {rknn.__version__}")
        
        # 尝试初始化 RKNN 上下文
        rknn_model = rknn.RKNN()
        print("✓ RKNN 上下文初始化成功")
        
        # 检查可用设备
        try:
            devices = rknn_model.list_devices()
            if devices:
                print(f"✓ 检测到 {len(devices)} 个 NPU 设备")
                for i, device in enumerate(devices):
                    print(f"  设备 {i+1}: {device}")
            else:
                print("✗ 未检测到 NPU 设备")
        except Exception as e:
            print(f"  设备列表查询: {e}")
        
        rknn_model.release()
        return True
    except ImportError:
        print("✗ RKNN Runtime 未安装")
        return False
    except Exception as e:
        print(f"✗ RKNN 测试失败: {e}")
        return False

def test_opencv():
    """测试 OpenCV 是否可用"""
    print("\n" + "=" * 60)
    print("5. OpenCV 测试")
    print("=" * 60)
    
    try:
        import cv2
        print(f"✓ OpenCV 已安装")
        print(f"  版本: {cv2.__version__}")
        
        # 测试基本功能
        img = cv2.imread("../model/bus.jpg")
        if img is not None:
            print(f"✓ 图像读取成功")
            print(f"  图像形状: {img.shape}")
            print(f"  图像大小: {img.size} 字节")
        else:
            print("✗ 图像读取失败（bus.jpg 可能不存在）")
        
        return True
    except ImportError:
        print("✗ OpenCV 未安装")
        return False
    except Exception as e:
        print(f"✗ OpenCV 测试失败: {e}")
        return False

def main():
    """主函数"""
    print("=" * 60)
    print("硬件检测测试脚本")
    print("=" * 60)
    print("日期:", time.strftime("%Y-%m-%d %H:%M:%S"))
    print("Python 版本:", sys.version)
    print("")
    
    results = {}
    
    # 测试各个硬件
    results['CPU'] = test_cpu()
    results['GPU_CUDA'] = test_gpu_cuda()
    results['GPU_OpenCL'] = test_gpu_opencl()
    results['NPU'] = test_npu()
    results['OpenCV'] = test_opencv()
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    total = len(results)
    passed = sum(results.values())
    
    print(f"\n测试总数: {total}")
    print(f"通过: {passed}")
    print(f"失败: {total - passed}")
    print(f"通过率: {passed/total*100:.1f}%")
    
    print("\n详细结果:")
    for name, status in results.items():
        status_str = "✓ 通过" if status else "✗ 失败"
        print(f"  {name}: {status_str}")
    
    print("\n" + "=" * 60)
    if passed == total:
        print("恭喜！所有硬件测试均通过！")
    else:
        print("部分测试未通过，请检查相关依赖是否安装")
    print("=" * 60)

if __name__ == "__main__":
    main()
