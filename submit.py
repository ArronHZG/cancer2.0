from tqdm import tqdm

from PCam_data_set import PCam_data_set
from load_paramter import load_parameter
from models.resnet import resnet18
import pandas as pd
import torch
from trainer import START_TIME

BATCH_SIZE = 128
NUM_WORKERS = 8


def test_epoch(model, data_loaders, device):
    list=[]
    model.eval()
    idx=0
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
    sample_df.to_csv(f"submit/{START_TIME}--{model_name}_submit.csv", index=False)


if __name__ == '__main__':
    # 加载数据
    INPUT_PATH = "/home/arron/文档/notebook/侯正罡/cancer/input"
    test_path = INPUT_PATH + '/test/'
    test_csv_url = INPUT_PATH + '/sample_submission.csv'
    device = 0

    # 加载模型
    model = resnet18(num_classes=2, pretrained=False)
    model_name = 'resnet18'
    # 模型参数加载
    # 模型参数加载
    model = load_parameter(model,
                           model_name,
                           pre_weight='models_weight/MyWeight/' +
                                      '2019-03-17--14:34:45/' +
                                      '2019-03-17--17:38:43--resnet18--46--Loss--0.0699--Acc--0.9767.pth')

    # 加载到GPU
    model.cuda(device)
    submit(model, model_name, device, test_path, test_csv_url)