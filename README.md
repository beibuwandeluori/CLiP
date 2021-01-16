# CLiP
Code for RANZCR CLiP - Catheter and Line Position Challenge

# Requirements
python>=3.6 torch>=1.5
```bash
pip install -q pip install git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git
pip install efficientnet_pytorch
pip install tensorboardX
pip install timm

```
#训练
训练前，需要修改train.py中的路劲：data_dir 和 csv_path

```bash
chmod +x train.sh 
./train.sh 
```
