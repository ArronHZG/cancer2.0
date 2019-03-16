# get test id's from the sample_submission.csv and keep their original order
from tqdm import tqdm

from PCam_data_set import PCam_data_set
from load_paramter import load_parameter
from models.resnet import resnet18

SAMPLE_SUB =HOME_PATH+'/input/sample_submission.csv'
sample_df = pd.read_csv(SAMPLE_SUB)
sample_list = list(sample_df.id)

# List of tumor preds.
# These are in the order of our test dataset and not necessarily in the same order as in sample_submission
pred_list = [p for p in tumor_preds]
# print(pred_list)
# To know the id's, we create a dict of id:pred
pred_dic = dict((key, value) for (key, value) in zip(learner.data.test_ds.items,pred_list))
# print(learner.data.test_ds.items)

# Now, we can create a new list with the same order as in sample_submission
pred_list_cor = [pred_dic['./'+ HOME_PATH+'/input/test/' + id + '.tif'] for id in sample_list]

# Next, a Pandas dataframe with id and label columns.
df_sub = pd.DataFrame({'id':sample_list,'label':pred_list_cor})

# Export to csv
df_sub.to_csv('{0}_submission.csv'.format(MODEL_PATH), header=True, index=False)

import pandas as pd
import torch
BATCH_SIZE = 128
NUM_WORKERS = 8


def test_epoch(model, data_loaders, device):
    model.eval()
    for inputs, labels in tqdm(data_loaders):
        inputs = inputs.cuda(device)
        pred = model(inputs)
        print(pred)
    return 0


def submit(model,model_name,device,csv_path):

    sample_df = pd.read_csv(csv_path)
    valid_set = PCam_data_set(sample_df, train_path, 'valid')
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=BATCH_SIZE,
                                               shuffle=False, num_workers=NUM_WORKERS)
    test_epoch(model, valid_loader, device)

if __name__ == '__main__':
    # 加载数据
    INPUT_PATH = "/home/arron/文档/notebook/侯正罡/cancer/input"
    csv_url = INPUT_PATH + '/train_labels.csv'
    data = pd.read_csv(csv_url)
    train_path = INPUT_PATH + '/train/'
    test_path = INPUT_PATH + '/test/'


    # 加载模型
    model = resnet18(num_classes=2, pretrained=False)
    model_name = 'resnet18'
    # 模型参数加载
    model = load_parameter(model, model_name)
    submit(model,model_name,1,)
