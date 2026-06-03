// ============================================================================
// rknpu2/my_yolo11_zero_copy.cc — RKNPU2 零拷贝推理实现（带中文注释）
//
// 【本文件作用】
//   面向 RKNPU2 非受限平台（RK356x/RK3588/RK3576 等）的零拷贝推理实现。
//   使用 rknn_set_io_mem 直接绑定输入输出内存，跳过 rknn_inputs_set / rknn_outputs_get，
//   减少数据拷贝次数，提升推理性能。
//
// 【零拷贝 vs 标准接口】
//   标准版 (yolo11.cc)：
//     分配 CPU 内存 → rknn_inputs_set(拷贝到 NPU) → rknn_run → rknn_outputs_get(拷贝回 CPU)
//     每次推理都有两次内存拷贝
//
//   零拷贝版 (本文件)：
//     rknn_create_mem(NPU 可访问内存) → rknn_set_io_mem(绑定) → rknn_run(直接读写)
//     letterbox 结果直接写入 NPU 内存，推理结果直接在 NPU 内存中读取
//     减少了 CPU↔NPU 之间的数据搬运
//
// 【NC1HWC2 格式说明】
//   NPU 内部为了计算效率，量化模型输出采用 NC1HWC2 格式：
//     C1 = ceil(channel / C2)，C2 = 16（或 8）
//   post_process() 期望 NCHW 格式，所以需要 NC1HWC2_i8_to_NCHW_i8() 做格式转换
//
// 【局限】
//   目前零拷贝模式仅支持 INT8 量化模型，FP16 暂不支持
// ============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "my_yolo11.h"
#include "common.h"
#include "file_utils.h"
#include "image_utils.h"

// ---- 打印 tensor 属性（含 stride 信息） ----
// 零拷贝模式下，w_stride 和 size_with_stride 很有用
// size_with_stride 是 NPU 实际分配的内存大小（可能大于 n_elems，因为对齐）
static void dump_tensor_attr(rknn_tensor_attr *attr) {
    char dims[128] = {0};
    for (int i = 0; i < attr->n_dims; ++i) {
        int idx = strlen(dims);
        sprintf(&dims[idx], "%d%s", attr->dims[i], (i == attr->n_dims - 1) ? "" : ", ");
    }
    printf("  index=%d, name=%s, n_dims=%d, dims=[%s], n_elems=%d, size=%d, w_stride = %d, size_with_stride = %d, "
           "fmt=%s, type=%s, qnt_type=%s, "
           "zp=%d, scale=%f\n",
           attr->index, attr->name, attr->n_dims, dims, attr->n_elems, attr->size, attr->w_stride, attr->size_with_stride,
           get_format_string(attr->fmt), get_type_string(attr->type), get_qnt_type_string(attr->qnt_type), attr->zp,
           attr->scale);
}

// ============================================================================
// 初始化模型（零拷贝版）
// ============================================================================
//
// 【与标准版的区别】
//   1. 使用 RKNN_QUERY_NATIVE_INPUT_ATTR / OUTPUT_ATTR 查询"原生"tensor 属性
//      原生属性反映 NPU 内部实际的数据排布（而不是 API 暴露的逻辑排布）
//   2. rknn_create_mem() 创建 NPU 可访问的内存
//   3. rknn_set_io_mem() 将内存绑定到输入输出
//   4. 同时也会查询标准 INPUT_ATTR / OUTPUT_ATTR（用于后处理获取维度信息）
//
int init_yolo11_model(const char *model_path, rknn_app_context_t *app_ctx) {
    int ret;
    int model_len = 0;
    char *model;
    rknn_context ctx = 0;

    // ---- 1. 读取模型文件 ----
    model_len = read_data_from_file(model_path, &model);
    if (model == NULL) {
        printf("load_model fail!\n");
        return -1;
    }

    // ---- 2. 初始化 RKNN 上下文 ----
    ret = rknn_init(&ctx, model, model_len, 0, NULL);
    free(model);
    if (ret < 0) {
        printf("rknn_init fail! ret=%d\n", ret);
        return -1;
    }

    // ---- 3. 查询输入输出数量 ----
    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC) {
        printf("rknn_query fail! ret=%d\n", ret);
        return -1;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    // ---- 4. 查询 NATIVE 输入属性（零拷贝必需） ----
    printf("input tensors:\n");
    rknn_tensor_attr input_native_attrs[io_num.n_input];
    memset(input_native_attrs, 0, sizeof(input_native_attrs));
    for (int i = 0; i < io_num.n_input; i++) {
        input_native_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_NATIVE_INPUT_ATTR, &(input_native_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        dump_tensor_attr(&(input_native_attrs[i]));
    }

    // 默认 native 输入类型是 int8（需要在外部自己做 normalize 和量化）
    // 改成 UINT8：让 NPU 自动做 normalize 和量化，letterbox 后直接传 RGB 原始值即可
    // 这意味着传入的数据范围是 0-255 的 uint8，不需要先转成 float 再量化
    input_native_attrs[0].type = RKNN_TENSOR_UINT8;
    app_ctx->input_mems[0] = rknn_create_mem(ctx, input_native_attrs[0].size_with_stride);

    // 绑定输入内存
    ret = rknn_set_io_mem(ctx, app_ctx->input_mems[0], &input_native_attrs[0]);
    if (ret < 0) {
        printf("input_mems rknn_set_io_mem fail! ret=%d\n", ret);
        return -1;
    }

    // ---- 5. 查询 NATIVE 输出属性 ----
    printf("output tensors:\n");
    rknn_tensor_attr output_native_attrs[io_num.n_output];
    memset(output_native_attrs, 0, sizeof(output_native_attrs));
    for (int i = 0; i < io_num.n_output; i++) {
        output_native_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_NATIVE_OUTPUT_ATTR, &(output_native_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        dump_tensor_attr(&(output_native_attrs[i]));
    }

    // 为每个输出创建 NPU 内存并绑定
    for (uint32_t i = 0; i < io_num.n_output; ++i) {
        app_ctx->output_mems[i] = rknn_create_mem(ctx, output_native_attrs[i].size_with_stride);
        ret = rknn_set_io_mem(ctx, app_ctx->output_mems[i], &output_native_attrs[i]);
        if (ret < 0) {
            printf("output_mems rknn_set_io_mem fail! ret=%d\n", ret);
            return -1;
        }
    }

    // ---- 6. 判断是否为量化模型 ----
    if (output_native_attrs[0].qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC &&
        output_native_attrs[0].type == RKNN_TENSOR_INT8) {
        app_ctx->is_quant = true;
    } else {
        app_ctx->is_quant = false;
    }

    // ---- 7. 额外查询标准 INPUT_ATTR / OUTPUT_ATTR ----
    // 后处理 post_process() 需要标准属性中的 dims 信息来知道 H/W 维度
    // 原生属性的 dims 是 NC1HWC2 格式，不能直接用
    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++) {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) { printf("rknn_query fail! ret=%d\n", ret); return -1; }
    }

    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++) {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) { printf("rknn_query fail! ret=%d\n", ret); return -1; }
    }

    // ---- 8. 全部保存到上下文 ----
    app_ctx->rknn_ctx = ctx;
    app_ctx->io_num = io_num;
    app_ctx->input_attrs = (rknn_tensor_attr *)malloc(io_num.n_input * sizeof(rknn_tensor_attr));
    memcpy(app_ctx->input_attrs, input_attrs, io_num.n_input * sizeof(rknn_tensor_attr));
    app_ctx->output_attrs = (rknn_tensor_attr *)malloc(io_num.n_output * sizeof(rknn_tensor_attr));
    memcpy(app_ctx->output_attrs, output_attrs, io_num.n_output * sizeof(rknn_tensor_attr));

    app_ctx->input_native_attrs = (rknn_tensor_attr *)malloc(io_num.n_input * sizeof(rknn_tensor_attr));
    memcpy(app_ctx->input_native_attrs, input_native_attrs, io_num.n_input * sizeof(rknn_tensor_attr));
    app_ctx->output_native_attrs = (rknn_tensor_attr *)malloc(io_num.n_output * sizeof(rknn_tensor_attr));
    memcpy(app_ctx->output_native_attrs, output_native_attrs, io_num.n_output * sizeof(rknn_tensor_attr));

    // ---- 确定输入尺寸 ----
    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
        printf("model is NCHW input fmt\n");
        app_ctx->model_channel = input_attrs[0].dims[1];
        app_ctx->model_height  = input_attrs[0].dims[2];
        app_ctx->model_width   = input_attrs[0].dims[3];
    } else {
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
// NC1HWC2 → NCHW 格式转换工具函数
// ============================================================================
//
// 【为什么需要这个转换？】
//   NPU 为了内部计算效率，量化模型的输出以 NC1HWC2 格式存储：
//     dims = [batch, C1, H, W, C2]
//     其中 C1 = ceil(channel / C2)，C2 通常为 16
//   后处理期望 NCHW 格式 [batch, channel, H, W] 或 [1, ch, h, w] 的连续排布，
//   所以需要做重排。
//
// 【转换逻辑】
//   对于每个 batch、每个 channel：
//     找到该 channel 在 NC1HWC2 中的位置（plane = c/C2, offset = c%C2）
//     将 C2 个值从 [plane][H][W][C2] 中取出，放入 [c][H][W] 的连续内存
//
int NC1HWC2_i8_to_NCHW_i8(const int8_t *src, int8_t *dst, int *dims, int channel, int h, int w, int zp, float scale) {
    int batch  = dims[0];
    int C1     = dims[1];
    int C2     = dims[4];
    int hw_src = dims[2] * dims[3];  // NC1HWC2 中的 H*W
    int hw_dst = h * w;              // 实际需要的 H*W
    for (int i = 0; i < batch; i++) {
        const int8_t *src_b = src + i * C1 * hw_src * C2;
        int8_t *dst_b = dst + i * channel * hw_dst;
        for (int c = 0; c < channel; ++c) {
            int plane  = c / C2;
            const int8_t *src_bc = plane * hw_src * C2 + src_b;
            int offset = c % C2;
            for (int cur_h = 0; cur_h < h; ++cur_h)
                for (int cur_w = 0; cur_w < w; ++cur_w) {
                    int cur_hw = cur_h * w + cur_w;
                    dst_b[c * hw_dst + cur_hw] = src_bc[C2 * cur_hw + offset];
                }
        }
    }
    return 0;
}

// ============================================================================
// 释放模型资源（零拷贝版）
// ============================================================================
// 注意：零拷贝模式下需要用 rknn_destroy_mem() 释放创建的内存
int release_yolo11_model(rknn_app_context_t *app_ctx) {
    int ret;
    if (app_ctx->input_attrs != NULL) {
        free(app_ctx->input_attrs);
        app_ctx->input_attrs = NULL;
    }
    if (app_ctx->output_attrs != NULL) {
        free(app_ctx->output_attrs);
        app_ctx->output_attrs = NULL;
    }
    if (app_ctx->input_native_attrs != NULL) {
        free(app_ctx->input_native_attrs);
        app_ctx->input_native_attrs = NULL;
    }
    if (app_ctx->output_native_attrs != NULL) {
        free(app_ctx->output_native_attrs);
        app_ctx->output_native_attrs = NULL;
    }

    // 释放零拷贝绑定的输入内存
    for (int i = 0; i < app_ctx->io_num.n_input; i++) {
        if (app_ctx->input_mems[i] != NULL) {
            ret = rknn_destroy_mem(app_ctx->rknn_ctx, app_ctx->input_mems[i]);
            if (ret != RKNN_SUCC) {
                printf("rknn_destroy_mem fail! ret=%d\n", ret);
                return -1;
            }
        }
    }
    // 释放零拷贝绑定的输出内存
    for (int i = 0; i < app_ctx->io_num.n_output; i++) {
        if (app_ctx->output_mems[i] != NULL) {
            ret = rknn_destroy_mem(app_ctx->rknn_ctx, app_ctx->output_mems[i]);
            if (ret != RKNN_SUCC) {
                printf("rknn_destroy_mem fail! ret=%d\n", ret);
                return -1;
            }
        }
    }
    if (app_ctx->rknn_ctx != 0) {
        ret = rknn_destroy(app_ctx->rknn_ctx);
        if (ret != RKNN_SUCC) {
            printf("rknn_destroy fail! ret=%d\n", ret);
            return -1;
        }
        app_ctx->rknn_ctx = 0;
    }
    return 0;
}

// ============================================================================
// 推理接口（零拷贝版）
// ============================================================================
//
// 【与标准版的区别】
//   - 输入：直接将 letterbox 结果写入 app_ctx->input_mems[0] 绑定的内存
//   - 输出：rknn_run 后直接从 app_ctx->output_mems[i] 读取结果
//   - 量化输出需要 NC1HWC2 → NCHW 格式转换
//   - 不调用 rknn_inputs_set / rknn_outputs_get
//
int inference_yolo11_model(rknn_app_context_t *app_ctx, image_buffer_t *img, object_detect_result_list *od_results) {
    int ret;
    image_buffer_t dst_img;
    letterbox_t letter_box;
    const float nms_threshold = NMS_THRESH;
    const float box_conf_threshold = BOX_THRESH;
    int bg_color = 114;

    if ((!app_ctx) || !(img) || (!od_results)) {
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
    dst_img.fd = app_ctx->input_mems[0]->fd;              // 使用绑定内存的 fd
    dst_img.virt_addr = (unsigned char*)app_ctx->input_mems[0]->virt_addr;  // 使用绑定内存的地址

    if (dst_img.virt_addr == NULL && dst_img.fd == 0) {
        printf("malloc buffer size:%d fail!\n", dst_img.size);
        return -1;
    }

    // letterbox 结果直接写入 NPU 绑定的内存，零拷贝
    ret = convert_image_with_letterbox(img, &dst_img, &letter_box, bg_color);
    if (ret < 0) {
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

    // ---- NC1HWC2 → NCHW 格式转换 ----
    // 从 output_mems 中读取 NPU 输出，转为后处理所需的 NCHW 格式
    rknn_output outputs[app_ctx->io_num.n_output];
    memset(outputs, 0, sizeof(outputs));
    for (uint32_t i = 0; i < app_ctx->io_num.n_output; i++) {
        int   channel = app_ctx->output_attrs[i].dims[1];
        int   h       = app_ctx->output_attrs[i].n_dims > 2 ? app_ctx->output_attrs[i].dims[2] : 1;
        int   w       = app_ctx->output_attrs[i].n_dims > 3 ? app_ctx->output_attrs[i].dims[3] : 1;
        int   hw      = h * w;
        int   zp      = app_ctx->output_native_attrs[i].zp;
        float scale   = app_ctx->output_native_attrs[i].scale;
        if (app_ctx->is_quant) {
            outputs[i].size = app_ctx->output_native_attrs[i].n_elems * sizeof(int8_t);
            outputs[i].buf = (int8_t *)malloc(outputs[i].size);
            if (app_ctx->output_native_attrs[i].fmt == RKNN_TENSOR_NC1HWC2) {
                // NC1HWC2 → NCHW 重排
                NC1HWC2_i8_to_NCHW_i8((int8_t *)app_ctx->output_mems[i]->virt_addr, (int8_t *)outputs[i].buf,
                                      (int *)app_ctx->output_native_attrs[i].dims, channel, h, w, zp, scale);
            } else {
                // 已经是 NCHW，直接拷贝
                memcpy(outputs[i].buf, app_ctx->output_mems[i]->virt_addr, outputs[i].size);
            }
        } else {
            printf("Currently zero copy does not support fp16!\n");
            goto out;
        }
    }

    // ---- 后处理（与标准版相同的 post_process） ----
    post_process(app_ctx, outputs, &letter_box, box_conf_threshold, nms_threshold, od_results);

    // 释放临时分配的输出缓冲区
    for (int i = 0; i < app_ctx->io_num.n_output; i++) {
        free(outputs[i].buf);
    }

out:
    return ret;
}
