import argparse
from warnings import filterwarnings
filterwarnings("ignore")
import torch
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch.optim as optim
from data.dataset import RANZERDataset
from model.models import RANZCRResNet200D, EfficientNet_ns
from utils import seed_everything, GradualWarmupSchedulerV2
import os
import pandas as pd
from sklearn.metrics import roc_auc_score
from data.dataset import get_transforms
import time


def macro_multilabel_auc(label, pred):
    aucs = []
    for i in range(len(target_cols)):
        aucs.append(roc_auc_score(label[:, i], pred[:, i]))
    print(np.round(aucs, 4))
    return np.mean(aucs)


def train_func(train_loader, epoch):
    print(f'train in epoch:{epoch}')
    model.train()
    bar = tqdm(train_loader)
    if use_amp:
        scaler = torch.cuda.amp.GradScaler()
    losses = []
    for batch_idx, (images, targets) in enumerate(bar):

        images, targets = images.to(device), targets.to(device)

        if use_amp:
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        else:
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        losses.append(loss.item())
        smooth_loss = np.mean(losses[-30:])

        bar.set_description(f'train epoch:{epoch}, loss: {loss.item():.5f}, smth: {smooth_loss:.5f}')

    loss_train = np.mean(losses)
    return loss_train


def valid_func(valid_loader, epoch):
    print(f'valid in epoch:{epoch}')
    model.eval()
    bar = tqdm(valid_loader)
    PROB = []
    TARGETS = []
    losses = []
    PREDS = []

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(bar):
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            PREDS += [logits.sigmoid()]
            TARGETS += [targets.detach().cpu()]
            loss = criterion(logits, targets)
            losses.append(loss.item())
            smooth_loss = np.mean(losses[-30:])
            bar.set_description(f'valid epoch{epoch}, loss: {loss.item():.5f}, smth: {smooth_loss:.5f}')

    PREDS = torch.cat(PREDS).cpu().numpy()
    TARGETS = torch.cat(TARGETS).cpu().numpy()
    # roc_auc = roc_auc_score(TARGETS.reshape(-1), PREDS.reshape(-1))
    roc_auc = macro_multilabel_auc(TARGETS, PREDS)
    loss_valid = np.mean(losses)
    return loss_valid, roc_auc


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
    fold_id = args.k
    image_size = args.input_size
    seed = 42
    warmup_epo = 1
    init_lr = 1e-4
    batch_size = args.batch_size
    valid_batch_size = args.batch_size
    n_epochs = 20
    warmup_factor = 10
    num_workers = 4

    use_amp = False
    debug = False  # change this to run on full data
    early_stop = 5

    kernel_type = args.model_name
    data_dir = '/raid/chenby/CLiP/train'
    model_dir = f'output/weights/' + kernel_type
    write_dir = f'output/logs/{kernel_type}'
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    if not os.path.isdir(write_dir):
        os.makedirs(write_dir)
    write_file = f'{write_dir}/{fold_id}.out'

    seed_everything(seed)
    device = torch.device(f'cuda:{args.device_id}')

    df_train = pd.read_csv('/data1/cby/py_project/CLiP/data/csv/train_folds.csv')
    df_train['file_path'] = df_train.StudyInstanceUID.apply(lambda x: os.path.join(data_dir, f'{x}.jpg'))
    if debug:
        df_train = df_train.sample(frac=0.1)
    target_cols = df_train.iloc[:, 1:12].columns.tolist()
    # model = RANZCRResNet200D(out_dim=len(target_cols), pretrained=True)
    model = EfficientNet_ns(model_arch=kernel_type, n_class=11, pretrained=True)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=init_lr / warmup_factor)
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs, eta_min=1e-7)
    scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=10, total_epoch=warmup_epo,
                                                after_scheduler=scheduler_cosine)

    df_train_this = df_train[df_train['fold'] != fold_id]
    df_valid_this = df_train[df_train['fold'] == fold_id]

    dataset_train = RANZERDataset(df_train_this, 'train', transform=get_transforms(mode_type='train'))
    dataset_valid = RANZERDataset(df_valid_this, 'valid', transform=get_transforms(mode_type='valid'))

    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers, pin_memory=False)
    valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=valid_batch_size, shuffle=False,
                                               num_workers=num_workers, pin_memory=False)

    log = {}
    roc_auc_max = 0.
    loss_min = 99999
    not_improving = 0

    for epoch in range(1, n_epochs + 1):
        scheduler_warmup.step(epoch - 1)
        loss_train = train_func(train_loader, epoch)
        loss_valid, roc_auc = valid_func(valid_loader, epoch)

        log['loss_train'] = log.get('loss_train', []) + [loss_train]
        log['loss_valid'] = log.get('loss_valid', []) + [loss_valid]
        log['lr'] = log.get('lr', []) + [optimizer.param_groups[0]["lr"]]
        log['roc_auc'] = log.get('roc_auc', []) + [roc_auc]

        content = time.ctime() + ' ' + f'Fold {fold_id}, Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, ' \
            f'loss_train: {loss_train:.5f}, loss_valid: {loss_valid:.5f}, roc_auc: {roc_auc:.6f}.'
        print(content)
        with open(write_file, 'a') as outF:
            outF.write(content)
        not_improving += 1

        if roc_auc > roc_auc_max:
            print(f'roc_auc_max ({roc_auc_max:.6f} --> {roc_auc:.6f}). Saving model ...')
            torch.save(model.state_dict(), f'{model_dir}/f{fold_id}_e{epoch}_AUC{roc_auc:.3f}.pth')
            roc_auc_max = roc_auc
            not_improving = 0

        # if loss_valid < loss_min:
        #     loss_min = loss_valid
        #     torch.save(model.state_dict(), f'{model_dir}/fold{fold_id}_e{epoch}_loss{loss_min:.3f}.pth')

        if not_improving == early_stop:
            print('Early Stopping...')
            break

    torch.save(model.state_dict(), f'{model_dir}{kernel_type}_fold{fold_id}_final.pth')