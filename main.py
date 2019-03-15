# 损失函数
    # HingeEmbeddingLoss
    # BCEWithLogitsLoss
# 优化器
    # ASGD
    # AMSGrad
# 学习率调整方法
    # MultiStepLR
    # CosineAnnealingLR
import pandas as pd
import torch
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau, CosineAnnealingLR
from torch import nn
from sklearn.model_selection import train_test_split
import torchvision

from GradualWarmupScheduler import GradualWarmupScheduler
from PCam_data_set import PCam_data_set
from models.pnasnet5large import PNASNet5Large
from models.resnet import resnet50
from trainer import train_model
from torchvision.models import densenet201

BATCH_SIZE = 128
NUM_WORKERS = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载数据
csv_url = '../input/train_labels.csv'
data = pd.read_csv(csv_url)
train_path = '../input/train/'
test_path = '../input/test/'
data['label'].value_counts()
# 切分训练集和验证集
tr, vd = train_test_split(data, test_size=0.1, random_state=123)
train_set = PCam_data_set(tr, train_path, 'train')
valid_set = PCam_data_set(vd, train_path, 'valid')
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE,
                                           shuffle=True, num_workers=NUM_WORKERS)
valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=BATCH_SIZE,
                                           shuffle=False, num_workers=NUM_WORKERS)
dataloaders = {'train': train_loader, 'val': valid_loader}
# 加载模型
model = densenet201(pretrained=True)
model_name = 'dense201'
# model = PNASNet5Large(2)

# 开启多个GPU
# if torch.cuda.device_count() > 1:
#     print("Let's use", torch.cuda.device_count(), "GPUs!")
#     # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
#     model = nn.DataParallel(model)
# 恢复模型
# PATH = '../weight/resnet50-7-Loss-0.7759 Acc-0.5334-model.pth'
# model.load_state_dict(torch.load(PATH))
# model.eval()
# 加载到GPU
model.to(device)
# 优化器
params_to_update = model.parameters()
optimizer = torch.optim.Adam(params_to_update, lr=1e-2, amsgrad=True)

# 使用warm_up和余弦退火
scheduler_cos = CosineAnnealingLR(optimizer, T_max=5, eta_min=4e-08)
scheduler = GradualWarmupScheduler(optimizer, multiplier=8, total_epoch=10,
                                   after_scheduler=scheduler_cos)
# 损失函数
criterion = torch.nn.CrossEntropyLoss().to(device)
# model可视化
from torchsummary import summary

summary(model, (3, 90, 90))

# print(next(iter(train_loader))[0].shape)
# with SummaryWriter(comment='constantModel') as w:
#     w.add_graph(model, next(iter(train_loader))[0], True)

# 训练和评估
model_ft, hist = train_model(model, model_name, dataloaders, criterion, optimizer, scheduler, device, num_epochs=120)
