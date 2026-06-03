// ============================================================================
// rknpu2/my_yolo11_rv1106_1103.cc — RV1106/RV1103 零拷贝推理实现（带中文注释）
//
// 【本文件作用】
//   面向 RV1106/RV1103 低功耗平台的 YOLO11 推理实现。
//   这两个芯片与 RK356x/RK3588 的主要区别：
//     1. NPU 只支持 NHWC 排布的零拷贝模式（不支持 NCHW）
//     2. RGA 硬件缩放需要 DMA 分配的内存
//     3. 只支持量化模型（不支持 FP16）
//
// 【零拷贝模式】
//   使用 rknn_set_io_mem 绑定内存，与 yolo11_zero_copy.cc 类似。
//   但输出格式是 NHWC（而非 NC1HWC2）：
//     - score tensor: [1, grid_h, grid_w, 80]  → 偏移 = (i*W+j)*80 + c
//     - box tensor:   [1, grid_h, grid_w, 4*16] → 偏移 = (i*W+j)*4*dfl_len + k
//   所以不需要 NC1HWC2 → NCHW 转换，直接读取即可。
//   对应的后处理函数是 process_i8_rv1106()（在 my_postprocess.cc 中）
//
// 【查询接口】
//   使用 RKNN_QUERY_NATIVE_INPUT_ATTR + RKNN_QUERY_NATIVE_NHWC_OUTPUT_ATTR
//   注意 NHWC 版本的输出查询使用的是 NATIVE_NHWC 而非 NATIVE_OUTPUT
// ============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "my_yolo11.h"
#include "common.h"
#include "file_utils.h"
#include "image_utils.h"

// ---- 打印 tensor 属性 ----
static void dump_tensor_attr(rknn_tensor_attr *attr)
{
    printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
           "zp=%d, scale=%f\n",
           attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
           attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
           get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

// ============================================================================
// 初始化模型（RV1106/1103 零拷贝版）
// ============================================================================
int init_yolo11_model(const char *model_path, rknn_app_context_t *app_ctx)
{
    int ret;
    int model_len = 0;
    char *model;
    rknn_context ctx = 0;

    // ---- 1. 加载模型（注意：rv1106 的 rknn_init 传路径名而非文件内容） ----
    // 与标准版不同，RV1106 的 rknn_init 第一个参数直接传模型路径字符串
    // 第二个参数传 0 表示按路径加载
    ret = rknn_init(&ctx, (char *)model_path, 0, 0, NULL);
    if (ret < 0)
    {
        printf("rknn_init fail! ret=%d\n", ret);
        return -1;
    }

    // ---- 2. 查询输入输出数量 ----
    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC)
    {
        printf("rknn_query fail! ret=%d\n", ret);
        return -1;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    // ---- 3. 查询 NATIVE 输入属性 ----
    printf("input tensors:\n");
    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_NATIVE_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        dump_tensor_attr(&(input_attrs[i]));
    }

    // 设置输入为 uint8 类型 + NHWC 布局
    // RV1106 零拷贝模式只支持 NHWC
    input_attrs[0].type = RKNN_TENSOR_UINT8;
    input_attrs[0].fmt = RKNN_TENSOR_NHWC;
    printf("input_attrs[0].size_with_stride=%d\n", input_attrs[0].size_with_stride);
    app_ctx->input_mems[0] = rknn_create_mem(ctx, input_attrs[0].size_with_stride);

    // 绑定输入内存
    ret = rknn_set_io_mem(ctx, app_ctx->input_mems[0], &input_attrs[0]);
    if (ret < 0) {
        printf("input_mems rknn_set_io_mem fail! ret=%d\n", ret);
        return -1;
    }

    // ---- 4. 查询 NATIVE NHWC 输出属性 ----
    printf("output tensors:\n");
    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        // ★ 关键：RV1106 使用 NATIVE_NHWC_OUTPUT_ATTR 而非 NATIVE_OUTPUT_ATTR
        // 因为 RV1106 的零拷贝输出就是 NHWC 布局（不需要额外转换）
        ret = rknn_query(ctx, RKNN_QUERY_NATIVE_NHWC_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        dump_tensor_attr(&(output_attrs[i]));
    }

    // 为每个输出创建 NPU 内存并绑定
    for (uint32_t i = 0; i < io_num.n_output; ++i) {
        app_ctx->output_mems[i] = rknn_create_mem(ctx, output_attrs[i].size_with_stride);
        ret = rknn_set_io_mem(ctx, app_ctx->output_mems[i], &output_attrs[i]);
        if (ret < 0) {
            printf("output_mems rknn_set_io_mem fail! ret=%d\n", ret);
            return -1;
        }
    }

    // ---- 5. 判断是否为量化模型 ----
    if (output_attrs[0].qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC)
    {
        app_ctx->is_quant = true;
    }
    else
    {
        app_ctx->is_quant = false;
    }

    // ---- 6. 保存到上下文 ----
    app_ctx->rknn_ctx = ctx;
    app_ctx->io_num = io_num;
    app_ctx->input_attrs = (rknn_tensor_attr *)malloc(io_num.n_input * sizeof(rknn_tensor_attr));
    memcpy(app_ctx->input_attrs, input_attrs, io_num.n_input * sizeof(rknn_tensor_attr));
    app_ctx->output_attrs = (rknn_tensor_attr *)malloc(io_num.n_output * sizeof(rknn_tensor_attr));
    memcpy(app_ctx->output_attrs, output_attrs, io_num.n_output * sizeof(rknn_tensor_attr));

    // ---- 确定输入尺寸（NHWC） ----
    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW)
    {
        printf("model is NCHW input fmt\n");
        app_ctx->model_channel = input_attrs[0].dims[1];
        app_ctx->model_height  = input_attrs[0].dims[2];
        app_ctx->model_width   = input_attrs[0].dims[3];
    } else
    {
        printf("model is NHWC input fmt\n");
        app_ctx->model_height  = input_attrs[0].dims[1];
        app_ctx->model_width   = input_attrs[0].dims[2];
        app_ctx->model_channel = input_attrs[0].dims[3];
    }
    printf("model input height=%d, width=%d, channel=%d\n",
           app_ctx->model_height, app_ctx->model_width, app_ctx->model_channel);

    return 0;
}

// ============================================================================
// 释放模型资源（RV1106/1103 零拷贝版）
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
    // 释放绑定的内存
    for (int i = 0; i < app_ctx->io_num.n_input; i++) {
        if (app_ctx->input_mems[i] != NULL) {
            rknn_destroy_mem(app_ctx->rknn_ctx, app_ctx->input_mems[i]);
        }
    }
    for (int i = 0; i < app_ctx->io_num.n_output; i++) {
        if (app_ctx->output_mems[i] != NULL) {
            rknn_destroy_mem(app_ctx->rknn_ctx, app_ctx->output_mems[i]);
        }
    }
    if (app_ctx->rknn_ctx != 0)
    {
        rknn_destroy(app_ctx->rknn_ctx);
        app_ctx->rknn_ctx = 0;
    }
    return 0;
}

// ============================================================================
// 推理接口（RV1106/1103 零拷贝版）
// ============================================================================
//
// 【与标准版的区别】
//   - letterbox 的结果直接写入绑定的 NPU 内存
//   - rknn_run 后，output_mems[i] 中就是 NHWC 格式的输出
//   - 直接将 output_mems 传给 post_process()（由 #if defined(RV1106_1103) 分支处理）
//   - 不需要 NC1HWC2 → NCHW 转换
//
int inference_yolo11_model(rknn_app_context_t *app_ctx, image_buffer_t *img, object_detect_result_list *od_results)
{
    int ret;
    image_buffer_t dst_img;
    letterbox_t letter_box;
    const float nms_threshold = NMS_THRESH;
    const float box_conf_threshold = BOX_THRESH;
    int bg_color = 114;

    if ((!app_ctx) || !(img) || (!od_results))
    {
        return -1;
    }
    memset(od_results, 0x00, sizeof(*od_results));
    memset(&letter_box, 0, sizeof(letterbox_t));
    memset(&dst_img, 0, sizeof(image_buffer_t));

    // ---- 预处理: letterbox（直接写入绑定的 NPU 内存） ----
    dst_img.width = app_ctx->model_width;
    dst_img.height = app_ctx->model_height;
    dst_img.format = IMAGE_FORMAT_RGB888;
    dst_img.size = get_image_size(&dst_img);
    dst_img.fd = app_ctx->input_mems[0]->fd;
    // virt_addr 已经在 main.cc 中通过 dma_buf_alloc 设置为 DMA 内存地址
    if (dst_img.virt_addr == NULL && dst_img.fd == 0)
    {
        printf("malloc buffer size:%d fail!\n", dst_img.size);
        return -1;
    }

    // letterbox 直接写入 DMA 内存（NPU 可以直接访问）
    ret = convert_image_with_letterbox(img, &dst_img, &letter_box, bg_color);
    if (ret < 0)
    {
        printf("convert_image_with_letterbox fail! ret=%d\n", ret);
        return -1;
    }

    // ---- 执行 NPU 推理 ----
    printf("rknn_run\n");
    ret = rknn_run(app_ctx->rknn_ctx, nullptr);
    if (ret < 0) {
        printf("rknn_run fail! ret=%d\n", ret);
        return -1;
    }

    // ---- 后处理 ----
    // 直接传入 output_mems（类型是 rknn_tensor_mem**）
    // post_process() 内部通过 #if defined(RV1106_1103) 分支使用 NHWC 版本的处理函数
    post_process(app_ctx, app_ctx->output_mems, &letter_box, box_conf_threshold, nms_threshold, od_results);
out:
    return ret;
}
