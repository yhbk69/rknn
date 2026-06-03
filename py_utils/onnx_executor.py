import os
import tempfile
import numpy as np
import onnxruntime as rt

type_map = {
    'tensor(int32)' : np.int32,
    'tensor(int64)' : np.int64,
    'tensor(float32)' : np.float32,
    'tensor(float64)' : np.float64,
    'tensor(float)' : np.float32,
}
if getattr(np, 'bool', False):
    type_map['tensor(bool)'] = np.bool
else:
    type_map['tensor(bool)'] = bool


def _downgrade_onnx_ir(model_path):
    """
    当 onnxruntime 不支持模型 IR version 时，尝试降级 IR version 并返回临时文件路径。
    如果降级失败，返回原始路径。
    """
    try:
        import onnx
        model = onnx.load(model_path)
        ir_version = model.ir_version
        # onnxruntime 1.19 最高支持 IR 10，高于此版本尝试降级
        if ir_version > 10:
            print(f'    [信息] ONNX IR version {ir_version} 过高，尝试降级到 10...')
            tmp = tempfile.NamedTemporaryFile(suffix='.onnx', delete=False)
            model.ir_version = 10
            with open(tmp.name, 'wb') as f:
                f.write(model.SerializeToString())
            print(f'    [信息] 已降级 IR version {ir_version} -> 10: {tmp.name}')
            return tmp.name, tmp
        return model_path, None
    except Exception as e:
        print(f'    [警告] ONNX IR 降级失败 ({e})，使用原始模型')
        return model_path, None


class ONNX_model_container_py:
    def __init__(self, model_path) -> None:
        self._tmp_file = None  # 持有临时文件引用，避免被 GC 删除
        self.model_path = model_path
        self.sess = self._create_session(model_path)

    def _create_session(self, model_path):
        sp_options = rt.SessionOptions()
        sp_options.log_severity_level = 3
        # [1 for info, 2 for warning, 3 for error, 4 for fatal]
        try:
            return rt.InferenceSession(model_path, sess_options=sp_options, providers=['CPUExecutionProvider'])
        except Exception as e:
            err_str = str(e)
            if 'Unsupported model IR version' in err_str:
                print(f'    [信息] onnxruntime 不支持此 ONNX IR version，尝试降级...')
                new_path, tmp_ref = _downgrade_onnx_ir(model_path)
                self._tmp_file = tmp_ref  # 持有引用
                self.model_path = new_path
                return rt.InferenceSession(new_path, sess_options=sp_options, providers=['CPUExecutionProvider'])
            raise

    # def __del__(self):
    #     self.release()

    def run(self, input_datas):
        if self.sess is None:
            print("ERROR: sess has been released")
            return []

        if len(input_datas) < len(self.sess.get_inputs()):
            assert False,'inputs_datas number not match onnx model{} input'.format(self.model_path)
        elif len(input_datas) > len(self.sess.get_inputs()):
            print('WARNING: input datas number large than onnx input node')

        input_dict = {}
        for i, _input in enumerate(self.sess.get_inputs()):
            # convert type
            if _input.type in type_map and \
                type_map[_input.type] != input_datas[i].dtype:
                print('WARNING: force data-{} from {} to {}'.format(i, input_datas[i].dtype, type_map[_input.type]))
                input_datas[i] = input_datas[i].astype(type_map[_input.type])
            
            # reshape if need
            if _input.shape != list(input_datas[i].shape):
                if ignore_dim_with_zero(input_datas[i].shape,_input.shape):
                    input_datas[i] = input_datas[i].reshape(_input.shape)
                    print("WARNING: reshape inputdata-{}: from {} to {}".format(i, input_datas[i].shape, _input.shape))
                else:
                    assert False, 'input shape{} not match real data shape{}'.format(_input.shape, input_datas[i].shape)
            input_dict[_input.name] = input_datas[i]

        output_list = []
        for i in range(len(self.sess.get_outputs())):
            output_list.append(self.sess.get_outputs()[i].name)

        #forward model
        res = self.sess.run(output_list, input_dict)
        return res

    def release(self):
        del self.sess
        self.sess = None


class ONNX_model_container_cpp:
    def __init__(self, model_path) -> None:
        pass

    def run(self, input_datas):
        pass


def ONNX_model_container(model_path, backend='py'):
    if backend == 'py':
        return ONNX_model_container_py(model_path)
    elif backend == 'cpp':
        return ONNX_model_container_cpp(model_path)


def reset_onnx_shape(onnx_model_path, output_path, input_shapes):
    if isinstance(input_shapes[0], int):
        command = "python -m onnxsim {} {} --input-shape {}".format(onnx_model_path, output_path, ','.join([str(v) for v in input_shapes]))
    else:
        if len(input_shapes)!= 1:
            print("RESET ONNX SHAPE with more than one input, try to match input name")
            sess = rt.InferenceSession(onnx_model_path)
            input_names = [input.name for input in sess.get_inputs()]
            command = "python -m onnxsim {} {} --input-shape ".format(onnx_model_path, output_path)
            for i, input_name in enumerate(input_names):
                command += "{}:{} ".format(input_name, ','.join([str(v) for v in input_shapes[i]]))
        else:
            command = "python -m onnxsim {} {} --input-shape {}".format(onnx_model_path, output_path, ','.join([str(v) for v in input_shapes[0]]))
    
    print(command)
    os.system(command)
    return output_path
    