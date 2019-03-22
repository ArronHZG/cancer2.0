from models.resnet import resnet18
# from trainer import train_model, writer
from torchsummary import summary
from models.pnasnet import pnasnet5large
import torch

# 加载模型
model = pnasnet5large(num_classes=2, pretrained=False)
model_name = 'pnasnet5large'
# 模型打印
summary(model, (3, 384, 384),device="cpu")
# model可视化
# x = torch.rand(1, 3, 96, 96)  # 随便定义一个输入
# writer.add_graph(model, x)


