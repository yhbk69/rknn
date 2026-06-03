import os
import numpy as np
import torch
torch.backends.quantized.engine = 'qnnpack'

def multi_list_unfold(tl):
    def unfold(_inl, target):
        if not isinstance(_inl, list) and not isinstance(_inl, tuple):
            target.append(_inl)
        else:
            unfold(_inl)

def flatten_list(in_list):
    flatten = lambda x: [subitem for item in x for subitem in flatten(item)] if type(x) is list else [x]
    return flatten(in_list)

# =========================================================================
# 尝试导入 ultralytics（用于加载 .pt checkpoint 而非 TorchScript）
# =========================================================================
try:
    from ultralytics import YOLO as _YOLO
    _HAS_ULTRALYTICS = True
except ImportError:
    _HAS_ULTRALYTICS = False


class Torch_model_container:
    def __init__(self, model_path, qnnpack=False) -> None:
        if qnnpack is True:
            torch.backends.quantized.engine = 'qnnpack'

        #! Backends must be set before load model.
        self.pt_model = self._load_model(model_path)
        if self.pt_model is not None:
            self.pt_model.eval()

    def _load_model(self, model_path):
        """尝试以 TorchScript 加载，失败则尝试 ultralytics checkpoint。"""
        # 1) 先尝试 TorchScript (torch.jit.load)
        try:
            model = torch.jit.load(model_path)
            print(f'    [加载] TorchScript 模型: {model_path}')
            return model
        except Exception as e:
            print(f'    [信息] TorchScript 加载失败 ({e})，尝试 ultralytics 方式...')

        # 2) 再尝试 ultralytics checkpoint
        if _HAS_ULTRALYTICS:
            try:
                yolo = _YOLO(model_path)
                pt_model = yolo.model  # ultralytics.nn.tasks.DetectionModel
                pt_model.eval()
                print(f'    [加载] ultralytics 模型: {model_path}')
                return pt_model

            except Exception as e:
                print(f'    [错误] ultralytics 加载失败: {e}')
        else:
            print('    [信息] ultralytics 未安装，无法加载 .pt checkpoint')

        raise RuntimeError(f'无法加载模型: {model_path} (不是 TorchScript，也非 ultralytics checkpoint)')

    # def __del__(self):
    #     self.release()

    def run(self, input_datas):
        if self.pt_model is None:
            print("ERROR: pt_model has been released")
            return []

        assert isinstance(input_datas, list), "input_datas should be a list, like [np.ndarray, np.ndarray]"

        input_datas_torch_type = []
        for _data in input_datas:
            input_datas_torch_type.append(torch.tensor(_data))

        for i,val in enumerate(input_datas_torch_type):
            if val.dtype == torch.float64:
                input_datas_torch_type[i] = input_datas_torch_type[i].float()

        result = self.pt_model(*input_datas_torch_type)

        if isinstance(result, tuple):
            result = list(result)
        if not isinstance(result, list):
            result = [result]
        
        result = flatten_list(result)

        for i in range(len(result)):
            # 处理量化 tensor 和非量化 tensor
            if result[i].is_quantized:
                result[i] = torch.dequantize(result[i])
            result[i] = result[i].cpu().detach().numpy()

        return result

    def release(self):
        del self.pt_model
        self.pt_model = None