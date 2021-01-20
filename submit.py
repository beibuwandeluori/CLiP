import torch
from tqdm import tqdm
from model.models import RANZCRResNet200D
from utils import seed_everything
import pandas as pd
import numpy as np
from data.dataset import RANZERDataset, get_transforms
import os


def inference_func(test_loader):
    model.eval()
    bar = tqdm(test_loader)
    LOGITS = []
    PREDS = []

    with torch.no_grad():
        for batch_idx, images in enumerate(bar):
            x = images.to(device)
            logits = model(x)
            LOGITS.append(logits.cpu())
            PREDS += [logits.sigmoid().detach().cpu()]

        PREDS = torch.cat(PREDS).cpu().numpy()
        LOGITS = torch.cat(LOGITS).cpu().numpy()
    return PREDS


def tta_inference_func(test_loader):
    model.eval()
    bar = tqdm(test_loader)
    PREDS = []
    LOGITS = []

    with torch.no_grad():
        for batch_idx, images in enumerate(bar):
            x = images.to(device)
            #             print(x.shape)
            x = torch.stack([x, x.flip(-1)], 0)  # hflip
            x = x.view(-1, 3, image_size, image_size)
            logits = model(x)
            #             logits = logits.view(batch_size, 2, -1).mean(1)
            PREDS += [logits.sigmoid().detach().cpu()]
            LOGITS.append(logits.cpu())
        PREDS = torch.cat(PREDS).cpu().numpy()

    return PREDS


def tta_inference_func_new(test_loader, tta_time=2):
    model.eval()
    PREDS = []

    with torch.no_grad():
        for t in range(tta_time):
            PREDS_t = test_on_one_epoch(test_loader)
            PREDS_t = torch.cat(PREDS_t).cpu().numpy()
            PREDS += [PREDS_t]

        PREDS = np.mean(PREDS, axis=0)

    return PREDS


def test_on_one_epoch(test_loader):
    bar = tqdm(test_loader)
    PREDS = []
    for batch_idx, images in enumerate(bar):
        images = images.to(device)
        #         print(images.shape)
        logits = model(images)
        PREDS += [logits.sigmoid()]

    return PREDS


if __name__ == '__main__':
    tta = True
    tta_time = 3
    batch_size = 8
    seed_everything(seed=42)
    image_size = 512
    device = torch.device('cuda:0')
    enet_type = ['resnet200d'] * 5
    model_path = ['/data1/cby/py_project/CLiP/output/weights/resnet200d/f0_e18_AUC0.946.pth',
                  '/data1/cby/py_project/CLiP/output/weights/resnet200d/f1_e19_AUC0.951.pth',
                  '/data1/cby/py_project/CLiP/output/weights/resnet200d/f2_e15_AUC0.952.pth',
                  '/data1/cby/py_project/CLiP/output/weights/resnet200d/f3_e16_AUC0.953.pth',
                  '/data1/cby/py_project/CLiP/output/weights/resnet200d/f4_e16_AUC0.952.pth']

    test = pd.read_csv('/raid/chenby/CLiP/sample_submission.csv')
    test['file_path'] = test.StudyInstanceUID.apply(
        lambda x: os.path.join('/raid/chenby/CLiP/test', f'{x}.jpg'))
    target_cols = test.iloc[:, 1:12].columns.tolist()

    # test_dataset = RANZCRDataset(test, 'test', transform=transforms_test)
    test_dataset = RANZERDataset(test, 'test', transform=get_transforms(mode_type='test_tta'))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = RANZCRResNet200D(out_dim=11, pretrained=True)

    test_preds = []
    for i in range(len(enet_type)):
        if enet_type[i] == 'resnet200d':
            print('resnet200d loaded', i + 1)
            model = RANZCRResNet200D(enet_type[i], out_dim=len(target_cols))
            model = model.to(device)
        model.load_state_dict(torch.load(model_path[i], map_location='cuda:0'))
        if tta:
            #             test_preds += [tta_inference_func(test_loader)]
            test_preds += [tta_inference_func_new(test_loader, tta_time=tta_time)]
        else:
            test_preds += [inference_func(test_loader)]

    submission = pd.read_csv('/raid/chenby/CLiP/sample_submission.csv')
    submission[target_cols] = np.mean(test_preds, axis=0)
    submission.to_csv('/data1/cby/py_project/CLiP/output/results/submission.csv', index=False)
