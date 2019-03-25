import os

from sklearn.model_selection import train_test_split
from tqdm import tqdm
from PCam_data_set import PCam_data_set
from load_paramter import load_parameter
from models.densenet import densenet201
import pandas as pd
import torch

from models.resnet import resnet18
from trainer import START_TIME, valid_epoch

BATCH_SIZE = 128
NUM_WORKERS = 8


def test_epoch(model, data_loaders, device):
    list=[]
    model.eval()
    for inputs, labels in tqdm(data_loaders):
        inputs = inputs.cuda(device)
        pred = model(inputs)
        pred = torch.argmax(pred,1)
        list.extend(pred.cpu().numpy().tolist())
    return list


def submit(model, model_name, device, test_path, csv_path):
    sample_df = pd.read_csv(csv_path)
    valid_set = PCam_data_set(sample_df, test_path, 'test')
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=BATCH_SIZE,
                                               shuffle=False, num_workers=NUM_WORKERS)
    np_list = test_epoch(model, valid_loader, device)
    sample_df["label"]=np_list
    if not os.path.exists("submit"):
        os.makedirs("submit")
    sample_df.to_csv(f"submit/{START_TIME}--{model_name}_submit.csv", index=False)


if __name__ == '__main__':
    # 加载数据
    INPUT_PATH = "/home/arron/文档/notebook/侯正罡/cancer/input"
    test_path = INPUT_PATH + '/test/'
    test_csv_url = INPUT_PATH + '/sample_submission.csv'
    device = 0

    # 加载模型
    model = densenet201(num_classes=2, pretrained=False)
    model_name = 'densenet201'
    # 模型参数加载
    model = load_parameter(model,
                           model_name,
                           type='pre_model',
                           pre_model='models_weight/MyWeight/' +
                                      '2019-03-22--19:50:11/' +
                                      '2019-03-22--21:08:38--densenet201--11--Loss--0.0880--Acc--0.9697.pth')

    # 加载到GPU
    model.cuda(device)
    submit(model, model_name, device, test_path, test_csv_url)
    # train_csv_url = INPUT_PATH + '/train_labels.csv'
    # data = pd.read_csv(train_csv_url)
    # train_path = INPUT_PATH + '/train/'
    # tr, vd = train_test_split(data, test_size=0.1, random_state=123)
    # train_set = PCam_data_set(tr, train_path, 'train')
    # valid_set = PCam_data_set(vd, train_path, 'valid')
    # train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE,
    #                                            shuffle=True, num_workers=NUM_WORKERS)
    # valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=BATCH_SIZE,
    #                                            shuffle=False, num_workers=NUM_WORKERS)
    # dataloaders = {'train': train_loader, 'valid': valid_loader}
    # # 损失函数
    # criterion = torch.nn.CrossEntropyLoss().cuda(device)
    # valid_acc, valid_loss = valid_epoch(model, dataloaders, device, criterion, model_name, 1, 0, 10)
    # print(valid_acc,valid_loss)