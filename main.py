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
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from sklearn.model_selection import train_test_split
from PCam_data_set import PCam_data_set
from load_paramter import load_parameter
from models.resnet import resnet18
from trainer import train_model, writer
from torchsummary import summary

BATCH_SIZE = 128
NUM_WORKERS = 8
device = 1
# 加载数据
INPUT_PATH = "/home/arron/文档/notebook/侯正罡/cancer/input"
train_csv_url = INPUT_PATH + '/train_labels.csv'
test_csv_url = INPUT_PATH + '/sample_submission.csv'
data = pd.read_csv(train_csv_url)
train_path = INPUT_PATH + '/train/'
test_path = INPUT_PATH + '/test/'
data['label'].value_counts()
# 切分训练集和验证集
test_size = 0.1
tr, vd = train_test_split(data, test_size=test_size, random_state=123)
train_set = PCam_data_set(tr, train_path, 'train')
valid_set = PCam_data_set(vd, train_path, 'valid')
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE,
                                           shuffle=True, num_workers=NUM_WORKERS)
valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=BATCH_SIZE,
                                           shuffle=False, num_workers=NUM_WORKERS)
dataloaders = {'train': train_loader, 'valid': valid_loader}
# 加载模型
model = resnet18(num_classes=2, pretrained=False)
model_name = 'resnet18'
# 模型参数加载
model = load_parameter(model,
                       model_name,
                       pre_weight='models_weight/MyWeight/' +
                                  '2019-03-17--14:34:45/' +
                                  '2019-03-17--17:38:43--resnet18--46--Loss--0.0699--Acc--0.9767.pth')
# 加载到GPU
model.cuda(device)
# 损失函数
criterion = torch.nn.CrossEntropyLoss().cuda(device)

# 训练
optimizer = torch.optim.ASGD(model.parameters(), lr=1e-1, lambd=1e-4, alpha=0.75, t0=1e6, weight_decay=1e-6)
train_model(model, model_name, dataloaders,
            criterion, optimizer, device, scheduler=None, test_size=test_size, num_epochs=[0, 20])
# 加载最优模型
model = load_parameter(model, model_name)
optimizer = torch.optim.ASGD(model.parameters(), lr=1e-2, lambd=1e-4, alpha=0.75, t0=1e6, weight_decay=1e-6)
scheduler = CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-3)
train_model(model, model_name, dataloaders,
            criterion, optimizer, device, scheduler, test_size=test_size, num_epochs=[20, 50])
# 加载最优模型
model = load_parameter(model, model_name)
optimizer = torch.optim.ASGD(model.parameters(), lr=1e-3, lambd=1e-4, alpha=0.75, t0=1e6, weight_decay=1e-6)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5,
                              verbose=True, threshold=1e-4, threshold_mode='rel',
                              cooldown=0, min_lr=0, eps=1e-86)
train_model(model, model_name, dataloaders,
            criterion, optimizer, device, scheduler, test_size=test_size, num_epochs=[50, 80])





# models = PNASNet5Large(2)

# 开启多个GPU
# if torch.cuda.device_count() > 1:
#     print("Let's use", torch.cuda.device_count(), "GPUs!")
#     # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
#     models = nn.DataParallel(models)
# 恢复模型
# PATH = '../weight/resnet50-7-Loss-0.7759 Acc-0.5334-models.pth'
# models.load_state_dict(torch.load(PATH))
# models.eval()
