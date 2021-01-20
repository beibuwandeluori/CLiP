import argparse
from warnings import filterwarnings
filterwarnings("ignore")
import torch
from tqdm import tqdm
import numpy as np
import torch.nn as nn
from data.dataset import RANZERDataset
from model.models import RANZCRResNet200D, EfficientNet_ns
from utils import seed_everything
import os
import pandas as pd
from sklearn.metrics import roc_auc_score
from data.dataset import get_transforms


def macro_multilabel_auc(label, pred):
    aucs = []
    for i in range(len(target_cols)):
        aucs.append(roc_auc_score(label[:, i], pred[:, i]))
    print(np.round(aucs, 4))
    return np.mean(aucs)


def valid_on_one_epoch(valid_loader):
    bar = tqdm(valid_loader)
    TARGETS = []
    losses = []
    PREDS = []
    for batch_idx, (images, targets) in enumerate(bar):
        images, targets = images.to(device), targets.to(device)
        logits = model(images)
        PREDS += [logits.sigmoid()]
        # PREDS += [nn.Softmax(dim=1)(logits)]
        TARGETS += [targets.detach().cpu()]
        loss = criterion(logits, targets)
        losses.append(loss.item())
        smooth_loss = np.mean(losses[-30:])
        bar.set_description(f'valid, loss: {loss.item():.5f}, smth: {smooth_loss:.5f}')

    return TARGETS, losses, PREDS


def valid_func(valid_loader):
    model.eval()

    with torch.no_grad():
        if not is_tta:
            TARGETS, losses, PREDS = valid_on_one_epoch(valid_loader)
            PREDS = torch.cat(PREDS).cpu().numpy()
        else:
            losses = []
            PREDS = []
            for t in range(tta_time):
                print(f'tta {t+1}/{tta_time}')
                TARGETS, losses_t, PREDS_t = valid_on_one_epoch(valid_loader)
                PREDS_t = torch.cat(PREDS_t).cpu().numpy()

                PREDS += [PREDS_t]
                losses += [losses_t]

            PREDS = np.mean(PREDS, axis=0)
            losses = np.mean(losses, axis=0)

    TARGETS = torch.cat(TARGETS).cpu().numpy()
    # roc_auc = roc_auc_score(TARGETS.reshape(-1), PREDS.reshape(-1))
    roc_auc = macro_multilabel_auc(TARGETS, PREDS)
    loss_valid = np.mean(losses)
    return loss_valid, roc_auc


def ensemble():
    pass


parser = argparse.ArgumentParser(description="Cassava Leaf Disease Classification  @cby Training")
parser.add_argument(
    "--model_name", default='tf_efficientnet_b2_ns', help="Model name", type=str
)
parser.add_argument(
    "--device_id", default=6, help="Setting the GPU id", type=int
)
parser.add_argument(
    "--input_size", default=512, help="Input size", type=int
)
parser.add_argument(
    "--batch_size", default=16, help="Input size", type=int
)
parser.add_argument(
    "--k", default=0, help="The value of K Fold", type=int
)
parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                    nargs=argparse.REMAINDER)
args = parser.parse_args()


if __name__ == '__main__':
    is_tta = True
    tta_time = 3
    # model_paths = ['/data1/cby/py_project/CLiP/output/pretrained_weights/resnet200d/resnet200d_fold0_cv953.pth',
    #                '/data1/cby/py_project/CLiP/output/pretrained_weights/resnet200d/resnet200d_fold1_cv955.pth',
    #                '/data1/cby/py_project/CLiP/output/pretrained_weights/resnet200d/resnet200d_fold2_cv955.pth',
    #                '/data1/cby/py_project/CLiP/output/pretrained_weights/resnet200d/resnet200d_fold3_cv957.pth',
    #                '/data1/cby/py_project/CLiP/output/pretrained_weights/resnet200d/resnet200d_fold4_cv954.pth']
    model_paths = ['/data1/cby/py_project/CLiP/output/weights/resnet200d/f0_e18_AUC0.946.pth',
                   '/data1/cby/py_project/CLiP/output/weights/resnet200d/f1_e19_AUC0.951.pth',
                   '/data1/cby/py_project/CLiP/output/weights/resnet200d/f2_e15_AUC0.952.pth',
                   '/data1/cby/py_project/CLiP/output/weights/resnet200d/f3_e16_AUC0.953.pth',
                   '/data1/cby/py_project/CLiP/output/weights/resnet200d/f4_e16_AUC0.952.pth']
    fold_id = args.k
    image_size = args.input_size
    seed = 42
    warmup_epo = 1
    init_lr = 1e-4
    batch_size = args.batch_size
    valid_batch_size = args.batch_size
    warmup_factor = 10
    num_workers = 4

    use_amp = False
    debug = False  # change this to run on full data
    early_stop = 5

    kernel_type = args.model_name
    data_dir = '/raid/chenby/CLiP/train'  # 修改这个路劲
    model_dir = f'output/weights/{kernel_type}'
    write_dir = f'output/logs/{kernel_type}'
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    if not os.path.isdir(write_dir):
        os.makedirs(write_dir)
    write_file = f'{write_dir}/{fold_id}.out'

    seed_everything(seed)
    device = torch.device(f'cuda:{args.device_id}')

    csv_path = '/data1/cby/py_project/CLiP/data/csv/train_folds.csv'  # 修改这个路劲
    df_train = pd.read_csv(csv_path)
    df_train['file_path'] = df_train.StudyInstanceUID.apply(lambda x: os.path.join(data_dir, f'{x}.jpg'))
    if debug:
        df_train = df_train.sample(frac=0.1)
    target_cols = df_train.iloc[:, 1:12].columns.tolist()
    if 'efficientnet' not in kernel_type:
        model = RANZCRResNet200D(out_dim=len(target_cols), pretrained=True)
    else:
        model = EfficientNet_ns(model_arch=kernel_type, n_class=11, pretrained=True)

    model_path = model_paths[fold_id]
    if model_path is not None:
        # model = torch.load(model_path)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print('Model found in {}'.format(model_path))
    else:
        print('No model found, initializing random model.')

    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    df_valid_this = df_train[df_train['fold'] == fold_id]
    if not is_tta:
        dataset_valid = RANZERDataset(df_valid_this, 'valid', transform=get_transforms(mode_type='valid'))
    else:
        dataset_valid = RANZERDataset(df_valid_this, 'valid', transform=get_transforms(mode_type='tta_valid'))

    valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=valid_batch_size, shuffle=False,
                                               num_workers=num_workers, pin_memory=False)
    loss_valid, roc_auc = valid_func(valid_loader)
    print(f'loss_valid:{loss_valid:.5f}, roc_auc:{roc_auc:.5f}')
