// ============================================================================
// my_main.cc — YOLO11 C++ Demo 主入口（带中文注释）
//
// 【本文件作用】
//   整个程序的入口点。负责：
//     1. 解析命令行参数（模型路径 + 图片路径）
//     2. 加载 RKNN 模型
//     3. 读取输入图片
//     4. 调用推理接口（NPU 推理 + 后处理）
//     5. 将检测结果画框保存为 out.png
//     6. 清理资源
//
// 【执行流程】
//   main()
//     ├── init_post_process()             ← 加载 COCO 标签
//     ├── init_yolo11_model()             ← 加载 .rknn 模型到 NPU
//     ├── read_image()                    ← 读取输入图片
//     ├── [RV1106] dma_buf_alloc()        ← DMA 内存分配（仅 RV1106/1103）
//     ├── inference_yolo11_model()         ← 推理 + 后处理（核心调用）
//     ├── 打印 + 画框 (draw_rectangle/draw_text)
//     ├── write_image("out.png")           ← 保存结果
//     ├── deinit_post_process()
//     ├── release_yolo11_model()
//     └── 释放图片内存
//
// 【编译与运行】
//   # 编译（用项目根目录的 build-linux.sh，不要直接 cmake）
//   ./build-linux.sh -t rk3588 -a aarch64 -d yolo11
//
//   # 在开发板上运行
//   ./rknn_my_yolo11_demo model/yolo11n.rknn model/bus.jpg
// ============================================================================

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// 注意：这里 include 的是 my_ 版本的头文件
// my_yolo11.h 内部又 include 了 my_postprocess.h
#include "my_yolo11.h"

// utils/ 中的通用工具
#include "image_utils.h"    // 图像读写、resize、letterbox 等
#include "file_utils.h"     // 文件读取
#include "image_drawing.h"  // 画框 (draw_rectangle) 和写文字 (draw_text)

// RV1106/1103 需要 DMA 内存分配器
#if defined(RV1106_1103)
    #include "dma_alloc.hpp"
#endif

// ============================================================================
// 主函数
// ============================================================================

/**
 * 用法：./rknn_my_yolo11_demo <model_path> <image_path>
 *
 * 示例：./rknn_my_yolo11_demo model/yolo11n.rknn model/bus.jpg
 *
 * @param argc 参数个数（必须 = 3）
 * @param argv[0] 可执行文件名
 * @param argv[1] .rknn 模型文件路径
 * @param argv[2] 输入图片路径
 * @return 0 成功
 */
int main(int argc, char **argv)
{
    // ---- 1. 参数检查 ----
    if (argc != 3)
    {
        printf("用法: %s <model_path> <image_path>\n", argv[0]);
        printf("示例: %s model/yolo11n.rknn model/bus.jpg\n", argv[0]);
        return -1;
    }

    const char *model_path = argv[1];   // .rknn 模型路径
    const char *image_path = argv[2];   // 输入图片路径

    int ret;
    rknn_app_context_t rknn_app_ctx;    // 模型上下文（所有状态都放这里）
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));

    // ---- 2. 初始化后处理（加载 COCO 标签） ----
    init_post_process();

    // ---- 3. 加载 RKNN 模型 ----
    // init_yolo11_model 内部：
    //   - 读取 .rknn 文件到内存
    //   - 调用 rknn_init() 初始化 NPU 上下文
    //   - 查询输入/输出 tensor 的维度、量化参数等
    ret = init_yolo11_model(model_path, &rknn_app_ctx);
    if (ret != 0)
    {
        printf("init_yolo11_model failed! ret=%d model_path=%s\n", ret, model_path);
        goto out;   // 跳转到清理代码
    }

    // ---- 4. 读取输入图片 ----
    image_buffer_t src_image;
    memset(&src_image, 0, sizeof(image_buffer_t));
    ret = read_image(image_path, &src_image);

    // ---- [RV1106/1103 特殊处理] ----
    // RV1106 的 RGA 硬件加速器要求输入输出内存由 DMA 分配
    // 所以这里把读入的图片数据拷贝到 DMA 内存中
#if defined(RV1106_1103)
    ret = dma_buf_alloc(RV1106_CMA_HEAP_PATH, src_image.size,
                        &rknn_app_ctx.img_dma_buf.dma_buf_fd,
                        (void **) & (rknn_app_ctx.img_dma_buf.dma_buf_virt_addr));
    memcpy(rknn_app_ctx.img_dma_buf.dma_buf_virt_addr, src_image.virt_addr, src_image.size);
    dma_sync_cpu_to_device(rknn_app_ctx.img_dma_buf.dma_buf_fd);
    free(src_image.virt_addr);
    src_image.virt_addr = (unsigned char *)rknn_app_ctx.img_dma_buf.dma_buf_virt_addr;
    src_image.fd = rknn_app_ctx.img_dma_buf.dma_buf_fd;
    rknn_app_ctx.img_dma_buf.size = src_image.size;
#endif

    if (ret != 0)
    {
        printf("read image failed! ret=%d image_path=%s\n", ret, image_path);
        goto out;
    }

    // ---- 5. 执行推理 + 后处理 ----
    // 这是最核心的一行调用：
    //   inference_yolo11_model 内部会做：
    //     a) letterbox 缩放（保持长宽比，填充灰色边）
    //     b) rknn_run() 在 NPU 上执行推理
    //     c) 读取输出 tensor
    //     d) post_process() 做 NMS 等后处理
    //   od_results 中就是最终的检测结果列表
    object_detect_result_list od_results;
    ret = inference_yolo11_model(&rknn_app_ctx, &src_image, &od_results);
    if (ret != 0)
    {
        printf("inference_yolo11_model failed! ret=%d\n", ret);
        goto out;
    }

    // ---- 6. 打印结果并画框 ----
    char text[256];
    for (int i = 0; i < od_results.count; i++)
    {
        object_detect_result *det_result = &(od_results.results[i]);

        // 终端打印检测结果
        // 格式：类别 @ (左上_x 左上_y 右下_x 右下_y) 置信度
        // 例如：person @ (112 84 388 572) 0.873
        printf("%s @ (%d %d %d %d) %.3f\n",
               coco_cls_to_name(det_result->cls_id),               // 类别名
               det_result->box.left, det_result->box.top,          // 左上角
               det_result->box.right, det_result->box.bottom,      // 右下角
               det_result->prop);                                  // 置信度

        int x1 = det_result->box.left;
        int y1 = det_result->box.top;
        int x2 = det_result->box.right;
        int y2 = det_result->box.bottom;

        // 在图片上画蓝色矩形框（粗细 3 像素）
        draw_rectangle(&src_image, x1, y1, x2 - x1, y2 - y1, COLOR_BLUE, 3);

        // 在框上方写类别和置信度
        sprintf(text, "%s %.1f%%", coco_cls_to_name(det_result->cls_id), det_result->prop * 100);
        draw_text(&src_image, text, x1, y1 - 20, COLOR_RED, 10);
    }

    // ---- 7. 保存结果图片 ----
    write_image("out.png", &src_image);
    printf("结果已保存到 out.png\n");

// ---- 8. 清理资源（goto 目标） ----
out:
    deinit_post_process();          // 释放标签字符串

    ret = release_yolo11_model(&rknn_app_ctx);  // 释放 NPU 模型资源
    if (ret != 0)
    {
        printf("release_yolo11_model failed! ret=%d\n", ret);
    }

    // 释放图片内存
    if (src_image.virt_addr != NULL)
    {
#if defined(RV1106_1103)
        dma_buf_free(rknn_app_ctx.img_dma_buf.size,
                     &rknn_app_ctx.img_dma_buf.dma_buf_fd,
                     rknn_app_ctx.img_dma_buf.dma_buf_virt_addr);
#else
        free(src_image.virt_addr);
#endif
    }

    return 0;
}
