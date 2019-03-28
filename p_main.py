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
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR
from sklearn.model_selection import train_test_split
from PCam_data_set import PCam_data_set
from load_paramter import load_parameter
from models.pnasnet import pnasnet5large
from trainer import train_model, logs_toCSV
from my_lr_scheduler.gradualWarmupScheduler import GradualWarmupScheduler
from torch.utils.data import DataLoader

BATCH_SIZE = 80
NUM_WORKERS = 8
DEVICE = 1
DOUBLE_CUDA = False

# s 加载数据
INPUT_PATH = "/home/arron/文档/notebook/侯正罡/cancer/input"
train_csv_url = INPUT_PATH + '/train_labels.csv'
data = pd.read_csv( train_csv_url )
train_path = INPUT_PATH + '/train/'
data['label'].value_counts()
# 切分训练集和验证集
test_size = 0.1
tr, vd = train_test_split( data, test_size=test_size, random_state=123 )
train_set = PCam_data_set( tr, train_path, 'train' )
valid_set = PCam_data_set( vd, train_path, 'valid' )
train_loader = DataLoader( train_set, batch_size=BATCH_SIZE,
                           shuffle=True, num_workers=NUM_WORKERS )
valid_loader = DataLoader( valid_set, batch_size=BATCH_SIZE,
                           shuffle=False, num_workers=NUM_WORKERS )
dataloaders = {'train': train_loader, 'valid': valid_loader}
# 加载模型
model = pnasnet5large( num_classes=2, pretrained=False )
model_name = 'pnasnet5large'
# 模型参数加载
# model = load_parameter(model,model_name)
model = load_parameter( model,
                        model_name)
# 双卡开启
if torch.cuda.device_count() > 1 and DOUBLE_CUDA:
    print( "Let's use", torch.cuda.device_count(), "GPUs!" )
    model = torch.nn.DataParallel( model )
    DEVICE = 0

# 加载到GPU
if torch.cuda.is_available():
    model.cuda( DEVICE )
# 损失函数
criterion = torch.nn.CrossEntropyLoss().cuda( DEVICE )

# 日志CSV头设置
logs_toCSV( header=True )

# 训练
optimizer = torch.optim.ASGD( model.parameters(), lr=1e-4, lambd=1e-4, alpha=0.75, t0=1e6, weight_decay=1e-4 )
scheduler = GradualWarmupScheduler( optimizer, multiplier=1000, total_epoch=10)
train_model( model, model_name, dataloaders,
             criterion, optimizer, DEVICE, scheduler=scheduler, test_size=test_size, num_epochs=[0, 20] )
# 加载最优模型
model = load_parameter( model, model_name, type='acc_model' )
optimizer = torch.optim.ASGD( model.parameters(), lr=1e-2, lambd=1e-4, alpha=0.75, t0=1e6, weight_decay=1e-4 )
scheduler = ReduceLROnPlateau( optimizer, mode='min', factor=0.8, patience=3,
                               verbose=True, threshold=1e-4, threshold_mode='rel',
                               cooldown=0, min_lr=0, eps=1e-86 )
train_model( model, model_name, dataloaders,
             criterion, optimizer, DEVICE, scheduler, test_size=test_size, num_epochs=[20, 200] )

# python -u p_main.py 2>&1 | tee logs/csv/20190323_train.log
