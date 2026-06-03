import os
import ctypes

# =========================================================================
# Monkey-patch: 某些 RK3588 系统上 librknnrt.so 安装在 /usr/lib/aarch64-linux-gnu/
# 但 rknn-toolkit2 尝试从 /usr/lib64/librknnrt.so 加载。此处拦截 ctypes.CDLL
# 调用，将 librknnrt.so 重定向到实际路径。
# =========================================================================
_original_CDLL = ctypes.CDLL

def _patched_CDLL(name, mode=ctypes.DEFAULT_MODE, handle=None):
    if isinstance(name, str) and ('librknnrt' in name):
        # 搜索已知的库路径
        for libpath in [
            '/usr/lib/aarch64-linux-gnu/librknnrt.so',
            '/lib/aarch64-linux-gnu/librknnrt.so',
        ]:
            if os.path.exists(libpath):
                return _original_CDLL(libpath, mode, handle)
    return _original_CDLL(name, mode, handle)

ctypes.CDLL = _patched_CDLL
# =========================================================================

from rknn.api import RKNN


class RKNN_model_container():
    def __init__(self, model_path, target=None, device_id=None) -> None:
        rknn = RKNN()

        # Direct Load RKNN Model
        rknn.load_rknn(model_path)

        print('--> Init runtime environment')
        if target==None:
            ret = rknn.init_runtime()
        else:
            ret = rknn.init_runtime(target=target, device_id=device_id)
        if ret != 0:
            print('Init runtime environment failed')
            exit(ret)
        print('done')
        
        self.rknn = rknn

    # def __del__(self):
    #     self.release()

    def run(self, inputs):
        if self.rknn is None:
            print("ERROR: rknn has been released")
            return []

        if isinstance(inputs, list) or isinstance(inputs, tuple):
            pass
        else:
            inputs = [inputs]

        result = self.rknn.inference(inputs=inputs)
    
        return result

    def release(self):
        self.rknn.release()
        self.rknn = None