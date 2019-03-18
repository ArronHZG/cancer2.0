from models.resnet import resnet18
# from trainer import train_model, writer
from torchsummary import summary
from models.densenet import densenet201
import torch

# 加载模型
model = densenet201(num_classes=2, pretrained=False)
model_name = 'densenet201'
# 模型打印
summary(model, (3, 96, 96))
# model可视化
# x = torch.rand(1, 3, 96, 96)  # 随便定义一个输入
# writer.add_graph(model, x)


