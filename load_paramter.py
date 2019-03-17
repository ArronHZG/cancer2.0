import glob
import torch
from collections import OrderedDict
from trainer import START_TIME

def load_parameter(model: torch.nn.Module, model_name, imageNet = False, pre_weight = None):
    '''
    加载模型
    pre_weight and imageNet 为空加载当前训练最优模型
    pre_weight and imageNet 不能同时出现
    :param model:
    :param model_name:
    :param imageNet:
    :param pre_weight:
    :return:
    '''
    if pre_weight and imageNet:
        print("pre_weight和imageNet只能二选一")
        return model
    if pre_weight:
        model_dict = model.state_dict()
        parameter_dict = {k: v for k, v in torch.load(pre_weight).items() if k in model_dict}
        model_dict.update(parameter_dict)
        model.load_state_dict(model_dict)
    else:
        if imageNet:
            path_list = glob.glob(f"models_weight/imageNetWeight/*{model_name}*.pth")
            if path_list:
                model_dict = model.state_dict()
                parameter_dict = {k: v for k, v in torch.load(path_list[0]).items() if k in model_dict}
                model_dict.update(parameter_dict)
                model.load_state_dict(model_dict)
            else:
                print("没有imageNet预训练模型")
        else:  # 加载最优模型
            path_list = glob.glob(f"models_weight/MyWeight/{START_TIME}/*{model_name}-*.pth")
            if path_list:
                # 找到最大的acc权重
                dic = OrderedDict({path.split("--")[-1].split(".")[-2]: path for path in path_list})
                keys = sorted(dic.keys())
                # print(keys)
                # 得到对应路径
                path = dic[keys[-1]]
                print(f"load: {path}")
                model_dict = model.state_dict()
                parameter_dict = {k: v for k, v in torch.load(path).items() if k in model_dict}
                model_dict.update(parameter_dict)
                model.load_state_dict(model_dict)
            else:
                print("没有acc预训练模型")
                # 正态初始化
    return model


if __name__ == '__main__':
    load_parameter(None, model_name="resnet18")
