import os

from tqdm import tqdm
from PCam_data_set import PCam_data_set
from load_paramter import load_parameter
from models.densenet import densenet201
import pandas as pd
import torch

from models.resnet import resnet18
from trainer import START_TIME

BATCH_SIZE = 20
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
    device = 1

    # 加载模型
    model = densenet201(num_classes=2, pretrained=False)
    model_name = 'densenet201'
    # 模型参数加载
    model = load_parameter(model,
                           model_name,
                           type='pre_model',
                           pre_model='models_weight/MyWeight/' +
                                      '2019-03-22--21:28:29/' +
                                      '2019-03-23--06:27:47--densenet201--93--Loss--0.0748--Acc--0.9755.pth')

    # 加载到GPU
    model.cuda(device)
    submit(model, model_name, device, test_path, test_csv_url)