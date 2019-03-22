import glob
import torch
from collections import OrderedDict
from trainer import START_TIME

def load_parameter(model, model_name, type, pre_model = None):

    if not type:
        print("未选择模式，kaiming_uniform 初始化参数")
        model_dict = model.state_dict()
        parameter_dict = {k: torch.nn.init.kaiming_uniform(v) for k, v in model_dict}
        model_dict.update(parameter_dict)
        model.load_state_dict(model_dict)
        return model

    if type == "pre_model":
        if pre_model:
            model_dict = model.state_dict()
            parameter_dict = {k: v for k, v in torch.load(pre_model).items() if k in model_dict}
            model_dict.update(parameter_dict)
            model.load_state_dict(model_dict)
            print("加载model预训练模型")
        else:
            print("没有model预训练模型")

    if type == 'imageNet':
        path_list = glob.glob(f"models_weight/imageNetWeight/*{model_name}*.pth")
        if path_list:
            model_dict = model.state_dict()
            parameter_dict = {k: v for k, v in torch.load(path_list[0]).items() if k in model_dict}
            model_dict.update(parameter_dict)
            model.load_state_dict(model_dict)
        else:
            print("没有imageNet预训练模型")

    if type == 'acc_model':
        # 加载最优模型
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
    load_parameter(None, model_name="resnet18.txt")
