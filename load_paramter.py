import glob
import torch
from collections import OrderedDict
from trainer import START_TIME

def load_parameter(model: torch.nn.Module, model_name, imageNet=False):
    if imageNet:
        path_list = glob.glob(f"models_weight/imageNetWeight/*{model_name}*.pth")
        if path_list:
            model_dict=model.state_dict()
            parameter_dict = {k: v for k, v in torch.load(path_list[0]).items() if k in model_dict}
            model_dict.update(parameter_dict)
            model.load_state_dict(model_dict)
        else:
            print("没有imageNet预训练模型")
    else:  # 加载最优模型
        path_list = glob.glob(f"models_weight/MyWeight/{START_TIME}/{model_name}-*.pth")
        if path_list:
            # 找到最大的acc权重
            dic = OrderedDict({float(path.split("-")[-2]): path for path in path_list})
            keys = sorted(dic.keys())
            # 得到对应路径
            path = dic[keys[-1]]
            print(f"load: {path}")
            model_dict=model.state_dict()
            parameter_dict = {k: v for k, v in torch.load(path).items() if k in model_dict}
            model_dict.update(parameter_dict)
            model.load_state_dict(model_dict)
        else:
            print("没有acc预训练模型")
    return model


if __name__ == '__main__':
    load_parameter(None, model_name="resnet18", imageNet=False)
