# Compilation Environment Setup Guide

It`s needed to set up a cross-compilation environment before compiling the C/C++ Demo of examples in this project on the x86 Linux system.


## Linux Platform

When the target device is a `Linux` system, use the `build-linux.sh` script in the root directory to compile the C/C++ Demo of the specific model.  
Before using this script to compile C/C++ Demo, please specify the path of the cross-compilation tool through the environment variable `GCC_COMPILER`.

### Download cross-compilation tools

*(If the cross-compilation tool is already installed on your system, please ignore this step)*

1. Different system architectures rely on different cross-compilation tools.The following are download links for cross-compilation tools recommended for different system architectures.：
   - aarch64: https://releases.linaro.org/components/toolchain/binaries/6.3-2017.05/aarch64-linux-gnu/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu.tar.xz
   - armhf: https://developer.arm.com/-/media/Files/downloads/gnu-a/8.3-2019.03/binrel/gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf.tar.xz?revision=e09a1c45-0ed3-4a8e-b06b-db3978fd8d56&rev=e09a1c450ed34a8eb06bdb3978fd8d56&hash=9C4F2E8255CB4D87EABF5769A2E65733
   - armhf-uclibcgnueabihf(RV1103/RV1106): https://console.zbox.filez.com/l/H1fV9a (fetch code: rknn)
2. Decompress the downloaded cross-compilation tool and remember the specific path, which will be used later during compilation.

### Compile C/C++ Demo

The command reference for compiling C/C++ Demo is as follows：
```shell
# go to the rknn_model_zoo root directory
cd <rknn_model_zoo_root_path>

# if GCC_COMPILER not found while building, please set GCC_COMPILER path
export GCC_COMPILER=<GCC_COMPILER_PATH>

./build-linux.sh -t <TARGET_PLATFORM> -a <ARCH> -d <model_name>

# for RK3588
./build-linux.sh -t rk3588 -a aarch64 -d mobilenet
# for RK3566
./build-linux.sh -t rk3566 -a aarch64 -d mobilenet
# for RK3568
./build-linux.sh -t rk3568 -a aarch64 -d mobilenet
# for RK1808
./build-linux.sh -t rk1808 -a aarch64 -d mobilenet
# for RV1109
./build-linux.sh -t rv1109 -a armhf -d mobilenet
# for RV1126
./build-linux.sh -t rv1126 -a armhf -d mobilenet
# for RV1103
./build-linux.sh -t rv1103 -a armhf -d mobilenet
# for RV1106
./build-linux.sh -t rv1106 -a armhf -d mobilenet
```

*Description:*
- `<GCC_COMPILER_PATH>`: Specify the cross-compilation path. Different system architectures use different cross-compilation tools.
    - `GCC_COMPILE_PATH` examples:
        - aarch64: ~/tools/cross_compiler/arm/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu
        - armhf: ~/tools/cross_compiler/arm/gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf/bin/arm-linux-gnueabihf
        - armhf-uclibcgnueabihf(RV1103/RV1106): ~/tools/cross_compiler/arm/arm-rockchip830-linux-uclibcgnueabihf/bin/arm-rockchip830-linux-uclibcgnueabihf
- `<TARGET_PLATFORM>`: Specify target platform. For example：`rk3588`. **Note: The target platforms currently supported by each model may be different, please refer to the `README.md` document in the specific model directory.**
- `<ARCH>`: Specify the system architecture. To query the system architecture, refer to the following command: 
  ```shell
  # Query architecture. For Linux, ['aarch64' or 'armhf'] should shown in log.
  adb shell cat /proc/version
  ```
- `model_name`: The model name. It is the folder name of each model in the examples directory.
