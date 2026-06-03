// ============================================================================
// rknpu2/my_yolo11.cc — RKNPU2 标准接口推理实现（带中文注释）
//
// 【本文件作用】
//   面向 RKNPU2 平台（RK3562/RK3566/RK3568/RK3588/RK3576）的 YOLO11 推理实现。
//   使用标准 API 接口：rknn_inputs_set + rknn_run + rknn_outputs_get
//   这是最通用的推理方式，所有 RKNPU2 平台都支持。
//
// 【与零拷贝版的区别】
//   标准版：inputs_set/outputs_get 三步走，API 简单，内部有数据拷贝
//   零拷贝版 (yolo11_zero_copy.cc)：set_io_mem 直接绑定内存，性能好但需要手动转格式
//
// 【与 RKNPU1 版的区别】
//   - RKNPU1：量化输出为 uint8，dims 索引顺序不同（dims[0]=w, dims[1]=h）
//   - RKNPU2：量化输出为 int8，dims 索引顺序为 dims[2]=h, dims[3]=w
// ============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "my_yolo11.h"      // 改为 my_ 版本
#include "common.h"
#include "file_utils.h"
#include "image_utils.h"

// ---- 工具函数：打印 tensor 属性 ----
// 用于调试时查看模型输入输出的维度、类型、量化参数等信息
static void dump_tensor_attr(rknn_tensor_attr *attr)
{
    printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
           "zp=%d, scale=%f\n",
           attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
           attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
           get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

// ============================================================================
// 初始化模型：加载 .rknn → 查询 tensor 属性 → 填充上下文
// ============================================================================
//
// 【流程详解】
//   1. read_data_from_file() 将 .rknn 文件读到内存
//   2. rknn_init() 初始化 NPU 推理上下文
//   3. rknn_query(RKNN_QUERY_IN_OUT_NUM) 查询输入输出数量
//   4. rknn_query(RKNN_QUERY_INPUT_ATTR) 查询每个输入 tensor 的属性
//   5. rknn_query(RKNN_QUERY_OUTPUT_ATTR) 查询每个输出 tensor 的属性
//   6. 判断是否为量化模型（int8 + 非对称量化）
//   7. 确定模型输入尺寸（NCHW 或 NHWC 布局）
//   8. 将结果存入 app_ctx
//
// 【为什么要查 tensor 属性？】
//   - 知道输入尺寸才能正确做 letterbox 缩放
//   - 知道量化参数（zp, scale）才能反量化输出
//   - 知道 dims 顺序才能正确解读输出数据排布
//
int init_yolo11_model(const char *model_path, rknn_app_context_t *app_ctx)
{
    int ret;
    int model_len = 0;
    char *model;
    rknn_context ctx = 0;

    // ---- 1. 读取 .rknn 模型文件 ----
    model_len = read_data_from_file(model_path, &model);
    if (model == NULL)
    {
        printf("load_model fail!\n");
        return -1;
    }

    // ---- 2. 初始化 RKNN 上下文 ----
    // rknn_init() 是 RKNPU SDK 的核心函数
    // 参数 0 表示同步模式（阻塞直到初始化完成）
    ret = rknn_init(&ctx, model, model_len, 0, NULL);
    free(model);  // 模型数据加载到 NPU 后，可以释放文件缓存
    if (ret < 0)
    {
        printf("rknn_init fail! ret=%d\n", ret);
        return -1;
    }

    // ---- 3. 查询输入输出数量 ----
    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC)
    {
        printf("rknn_query fail! ret=%d\n", ret);
        return -1;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);
    // YOLO11 通常 1 个输入 + 9 个输出（3 分支 x 3 个 tensor = box + score + score_sum）

    // ---- 4. 查询输入 tensor 属性 ----
    printf("input tensors:\n");
    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        dump_tensor_attr(&(input_attrs[i]));
    }

    // ---- 5. 查询输出 tensor 属性 ----
    printf("output tensors:\n");
    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        dump_tensor_attr(&(output_attrs[i]));
    }

    // ---- 6. 判断是否为量化模型 ----
    // RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC + INT8 表示非对称 INT8 量化
    // 量化模型的输出需要反量化：float_val = (int8_val - zp) * scale
    // 非量化模型的输出直接就是 float
    if (output_attrs[0].qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC && output_attrs[0].type == RKNN_TENSOR_INT8)
    {
        app_ctx->is_quant = true;
    }
    else
    {
        app_ctx->is_quant = false;
    }

    // ---- 7. 保存到上下文 ----
    app_ctx->rknn_ctx = ctx;
    app_ctx->io_num = io_num;
    app_ctx->input_attrs = (rknn_tensor_attr *)malloc(io_num.n_input * sizeof(rknn_tensor_attr));
    memcpy(app_ctx->input_attrs, input_attrs, io_num.n_input * sizeof(rknn_tensor_attr));
    app_ctx->output_attrs = (rknn_tensor_attr *)malloc(io_num.n_output * sizeof(rknn_tensor_attr));
    memcpy(app_ctx->output_attrs, output_attrs, io_num.n_output * sizeof(rknn_tensor_attr));

    // ---- 8. 确定模型输入尺寸 ----
    // NCHW: dims[0]=N(batch), dims[1]=C(channel), dims[2]=H, dims[3]=W
    // NHWC: dims[0]=N, dims[1]=H, dims[2]=W, dims[3]=C
    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW)
    {
        printf("model is NCHW input fmt\n");
        app_ctx->model_channel = input_attrs[0].dims[1];
        app_ctx->model_height = input_attrs[0].dims[2];
        app_ctx->model_width = input_attrs[0].dims[3];
    }
    else
    {
        printf("model is NHWC input fmt\n");
        app_ctx->model_height = input_attrs[0].dims[1];
        app_ctx->model_width = input_attrs[0].dims[2];
        app_ctx->model_channel = input_attrs[0].dims[3];
    }
    printf("model input height=%d, width=%d, channel=%d\n",
           app_ctx->model_height, app_ctx->model_width, app_ctx->model_channel);

    return 0;
}

// ============================================================================
// 释放模型资源
// ============================================================================
int release_yolo11_model(rknn_app_context_t *app_ctx)
{
    if (app_ctx->input_attrs != NULL)
    {
        free(app_ctx->input_attrs);
        app_ctx->input_attrs = NULL;
    }
    if (app_ctx->output_attrs != NULL)
    {
        free(app_ctx->output_attrs);
        app_ctx->output_attrs = NULL;
    }
    if (app_ctx->rknn_ctx != 0)
    {
        rknn_destroy(app_ctx->rknn_ctx);  // 销毁 NPU 上下文，释放 NPU 内存
        app_ctx->rknn_ctx = 0;
    }
    return 0;
}

// ============================================================================
// 推理接口（标准版）
// ============================================================================
//
// 【处理流程】
//   1. 申请 dst_img 用于存放 letterbox 后的图片
//   2. convert_image_with_letterbox()：将原图缩放到模型输入尺寸，保持长宽比 + 灰色填充
//   3. rknn_inputs_set()：将图片数据传给 NPU 输入
//   4. rknn_run()：在 NPU 上执行推理
//   5. rknn_outputs_get()：获取 NPU 输出 tensor
//       - 如果是量化模型，want_float=false，拿到 int8 原始值
//       - 如果是非量化模型，want_float=true，拿到 float 值
//   6. post_process()：后处理解码 → 检测结果
//   7. rknn_outputs_release()：释放输出缓冲区
//   8. 释放 dst_img
//
int inference_yolo11_model(rknn_app_context_t *app_ctx, image_buffer_t *img, object_detect_result_list *od_results)
{
    int ret;
    image_buffer_t dst_img;
    letterbox_t letter_box;
    rknn_input inputs[app_ctx->io_num.n_input];
    rknn_output outputs[app_ctx->io_num.n_output];
    const float nms_threshold = NMS_THRESH;      // 默认的NMS阈值
    const float box_conf_threshold = BOX_THRESH; // 默认的置信度阈值
    int bg_color = 114;  // letterbox 填充色（灰色）

    if ((!app_ctx) || !(img) || (!od_results))
    {
        return -1;
    }

    memset(od_results, 0x00, sizeof(*od_results));
    memset(&letter_box, 0, sizeof(letterbox_t));
    memset(&dst_img, 0, sizeof(image_buffer_t));
    memset(inputs, 0, sizeof(inputs));
    memset(outputs, 0, sizeof(outputs));

    // ---- 预处理: letterbox + resize ----
    // 创建目标图像缓冲区（大小 = 模型输入尺寸）
    dst_img.width = app_ctx->model_width;
    dst_img.height = app_ctx->model_height;
    dst_img.format = IMAGE_FORMAT_RGB888;  // 模型输入是 RGB 三通道
    dst_img.size = get_image_size(&dst_img);
    dst_img.virt_addr = (unsigned char *)malloc(dst_img.size);
    if (dst_img.virt_addr == NULL)
    {
        printf("malloc buffer size:%d fail!\n", dst_img.size);
        return -1;
    }

    // letterbox：将原图等比例缩放到模型尺寸，多余部分用灰色填充
    // 这样保证图片不变形，模型推理效果更好
    ret = convert_image_with_letterbox(img, &dst_img, &letter_box, bg_color);
    if (ret < 0)
    {
        printf("convert_image_with_letterbox fail! ret=%d\n", ret);
        return -1;
    }

    // ---- 设置输入 tensor ----
    // 输入类型为 UINT8（0-255），布局 NHWC，数据来自 dst_img
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].size = app_ctx->model_width * app_ctx->model_height * app_ctx->model_channel;
    inputs[0].buf = dst_img.virt_addr;

    // 将输入数据传递给 NPU
    ret = rknn_inputs_set(app_ctx->rknn_ctx, app_ctx->io_num.n_input, inputs);
    if (ret < 0)
    {
        printf("rknn_input_set fail! ret=%d\n", ret);
        return -1;
    }

    // ---- 执行 NPU 推理 ----
    printf("rknn_run\n");
    ret = rknn_run(app_ctx->rknn_ctx, nullptr);
    if (ret < 0)
    {
        printf("rknn_run fail! ret=%d\n", ret);
        return -1;
    }

    // ---- 获取输出 tensor ----
    // 对于量化模型，want_float=false 拿到 int8 原始数据（需要反量化）
    // 对于非量化模型，want_float=true 直接拿 float 数据
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < app_ctx->io_num.n_output; i++)
    {
        outputs[i].index = i;
        outputs[i].want_float = (!app_ctx->is_quant);
    }
    ret = rknn_outputs_get(app_ctx->rknn_ctx, app_ctx->io_num.n_output, outputs, NULL);
    if (ret < 0)
    {
        printf("rknn_outputs_get fail! ret=%d\n", ret);
        goto out;
    }

    // ---- 后处理：将 NPU 原始输出解码为检测框 ----
    post_process(app_ctx, outputs, &letter_box, box_conf_threshold, nms_threshold, od_results);

    // 释放 NPU 输出缓冲区
    rknn_outputs_release(app_ctx->rknn_ctx, app_ctx->io_num.n_output, outputs);

out:
    if (dst_img.virt_addr != NULL)
    {
        free(dst_img.virt_addr);
    }

    return ret;
}
