// ============================================================================
// my_yolo11.h — YOLO11 模型上下文结构体 + 接口声明（带中文注释）
//
// 【本文件作用】
//   定义 RKNN 推理所需的上下文结构体 rknn_app_context_t，
//   以及三个核心接口函数的声明：
//     1. init_yolo11_model()       — 加载模型、初始化上下文
//     2. inference_yolo11_model()  — 执行推理 + 后处理
//     3. release_yolo11_model()    — 释放资源
//
// 【如何使用】
//   rknn_app_context_t ctx;
//   memset(&ctx, 0, sizeof(ctx));
//   init_yolo11_model("model.rknn", &ctx);       // 加载模型
//   inference_yolo11_model(&ctx, &img, &result); // 推理
//   release_yolo11_model(&ctx);                    // 清理
// ============================================================================

#ifndef _RKNN_DEMO_MY_YOLO11_H_
#define _RKNN_DEMO_MY_YOLO11_H_

#include "rknn_api.h"    // RKNPU SDK 核心头文件（定义 rknn_context 等）
#include "common.h"      // RKNPU 公共定义（rknn_tensor_attr 等辅助宏）

// ---- 针对 RV1106/RV1103 平台的 DMA 缓冲区结构 ----
// 这两个芯片的 RGA（2D 加速）要求输入输出内存由 DMA 分配
#if defined(RV1106_1103)
    typedef struct {
        char *dma_buf_virt_addr;  // DMA 缓冲区虚拟地址
        int dma_buf_fd;           // DMA 缓冲区文件描述符
        int size;                 // 缓冲区大小
    } rknn_dma_buf;
#endif

// ---- RKNN 应用上下文结构体（核心数据结构） ----
// 所有推理函数都通过这个结构体传递模型状态
// 新增自定义字段（如后处理参数）可以加在这里
typedef struct {
    /*--- RKNN 运行时上下文 ---*/
    rknn_context rknn_ctx;              // RKNN 运行时句柄（最重要的对象）
    rknn_input_output_num io_num;       // 输入/输出 tensor 的数量
    rknn_tensor_attr* input_attrs;      // 输入 tensor 属性数组（维度、类型、量化参数等）
    rknn_tensor_attr* output_attrs;     // 输出 tensor 属性数组

    /*--- 零拷贝模式专用字段（条件编译） ---*/
    // 零拷贝：直接用 rknn_set_io_mem 绑定内存，跳过 rknn_inputs_set/outputs_get
    // 减少数据拷贝，提升性能
#if defined(RV1106_1103)
    rknn_tensor_mem* input_mems[1];     // 输入内存（RV1106 零拷贝用）
    rknn_tensor_mem* output_mems[9];    // 输出内存（YOLO11 有 9 个输出，3 分支 x 3 个）
    rknn_dma_buf img_dma_buf;           // DMA 缓冲区（RV1106 RGA 需要）
#endif
#if defined(ZERO_COPY)
    rknn_tensor_mem* input_mems[1];     // 输入内存（标准零拷贝用）
    rknn_tensor_mem* output_mems[9];    // 输出内存
    rknn_tensor_attr* input_native_attrs;   // 原生输入 tensor 属性（NC1HWC2 排布）
    rknn_tensor_attr* output_native_attrs;  // 原生输出 tensor 属性
#endif

    /*--- 模型输入尺寸 ---*/
    int model_channel;  // 输入通道数（通常 3，RGB）
    int model_width;    // 输入宽度（如 640）
    int model_height;   // 输入高度（如 640）

    /*--- 量化标记 ---*/
    bool is_quant;      // true = INT8 量化模型，false = FP16 非量化模型
    // 量化模型的输出是 int8 值，需要反量化（乘以 scale + 加 zp）转 float
    // 非量化模型的输出直接是 float

} rknn_app_context_t;

// 后处理结果结构体声明
#include "my_postprocess.h"


// ============================================================================
// 三个核心接口函数的声明
// ============================================================================

/**
 * @brief 初始化 YOLO11 模型
 *  1. 读取 .rknn 文件
 *  2. 调用 rknn_init() 加载模型到 NPU
 *  3. 查询输入/输出 tensor 属性（维度、量化参数等）
 *  4. 填充 app_ctx 结构体
 *
 * @param model_path  .rknn 模型文件路径
 * @param app_ctx     [out] 初始化后的上下文
 * @return 0 成功，非 0 失败
 */
int init_yolo11_model(const char* model_path, rknn_app_context_t* app_ctx);

/**
 * @brief 释放模型资源
 *  释放 input_attrs / output_attrs / rknn_ctx 等
 *
 * @param app_ctx  要释放的上下文
 * @return 0 成功
 */
int release_yolo11_model(rknn_app_context_t* app_ctx);

/**
 * @brief 执行推理 + 后处理
 *  1. 将输入图做 letterbox 缩放（保持长宽比填充）
 *  2. 调用 rknn_run() 执行 NPU 推理
 *  3. 读取输出 tensor
 *  4. 调用 post_process() 做 NMS 等后处理
 *
 * @param app_ctx   模型上下文
 * @param img       输入图像
 * @param od_results [out] 检测结果列表
 * @return 0 成功
 */
int inference_yolo11_model(rknn_app_context_t* app_ctx, image_buffer_t* img, object_detect_result_list* od_results);

#endif //_RKNN_DEMO_MY_YOLO11_H_
