import os

from sklearn.model_selection import train_test_split
from tqdm import tqdm
from load_paramter import load_parameter
import pandas as pd
import torch

from models.densenet import densenet201
from models.pnasnet import pnasnet5large
from models.resnet import resnet18
from trainer import START_TIME
from tta_data_set import TTA_data_set

BATCH_SIZE = 80
NUM_WORKERS = 8


def test_epoch(model, valid_set, device):
    # 加载到GPU
    model.cuda(device)
    model.eval()
    list = []
    for idx in tqdm(range(len(valid_set))):
        pics, _ = valid_set[idx]
        pics = torch.stack(pics)
        inputs = pics.cuda(device)
        pred = model(inputs)
        pred = torch.argmax(pred.mean(dim=0))
        list.append(pred.cpu().item())
    return list


def submit(model, model_name, device, test_path, csv_path):
    sample_df = pd.read_csv(csv_path)
    valid_set = TTA_data_set(sample_df, test_path, tta_times=9)
    np_list = test_epoch(model, valid_set, device)
    sample_df["label"] = np_list
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
    # resnet18
    # densenet201
    # pnasnet5large
    model = densenet201(num_classes=2, pretrained=False)
    model_name = 'densenet201'
    # 模型参数加载
    model = load_parameter(model,
                           model_name,
                           type='pre_model',
                           pre_model='models_weight/MyWeight/' +
                                     '2019-03-23--15:32:09/' +
                                     '2019-03-24--03:02:37--densenet201--105--Loss--0.0722--Acc--0.9769.pth')
    submit(model, model_name, device, test_path, test_csv_url)

    test = False
    if test:
        train_csv_url = INPUT_PATH + '/train_labels.csv'
        data = pd.read_csv(train_csv_url)
        train_path = INPUT_PATH + '/train/'
        _, vd = train_test_split(data, test_size=0.1, random_state=123)
        valid_set = TTA_data_set(vd, train_path, tta_times=9)
        valid_len = vd.count()["id"]

        # 加载模型
        # resnet18
        # densenet201
        # pnasnet5large
        model = densenet201(num_classes=2, pretrained=False)
        model_name = "densenet201"
        # 模型参数加载
        model = load_parameter(model,
                               model_name,
                               type='pre_model',
                               pre_model='models_weight/MyWeight/' +
                                          '2019-03-23--15:32:09/' +
                                          '2019-03-24--03:02:37--densenet201--105--Loss--0.0722--Acc--0.9769.pth')

        model.cuda(device)
        model.eval()

        # 损失函数
        criterion = torch.nn.CrossEntropyLoss().cuda(device)
        # 评估
        valid_acc = 0
        for idx in tqdm(range(valid_len)):
            inputs, label = valid_set[idx]
            # 10个
            inputs = torch.stack(inputs).cuda(device)
            # 1个
            pred = model(inputs)
            pred = torch.argmax(pred.mean(dim=0))
            valid_acc += int(pred.cpu().item() == label)
            # print(pred.cpu().item(),"==",label)
        valid_acc /= valid_len
        print(f"valid_acc: {valid_acc}")
        # valid_acc: 0.9777292973366057