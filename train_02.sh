BATCH_SIZE=12
DEVICE_ID=3
INPUT_SIZE=600
EPOCH=10
python train.py --model_name=resnet200d --device_id=${DEVICE_ID} --input_size=${INPUT_SIZE} --batch_size=${BATCH_SIZE} --epoch=${EPOCH} --k=0
python train.py --model_name=resnet200d --device_id=${DEVICE_ID} --input_size=${INPUT_SIZE} --batch_size=${BATCH_SIZE} --epoch=${EPOCH} --k=1
python train.py --model_name=resnet200d --device_id=${DEVICE_ID} --input_size=${INPUT_SIZE} --batch_size=${BATCH_SIZE} --epoch=${EPOCH} --k=2
python train.py --model_name=resnet200d --device_id=${DEVICE_ID} --input_size=${INPUT_SIZE} --batch_size=${BATCH_SIZE} --epoch=${EPOCH} --k=3
python train.py --model_name=resnet200d --device_id=${DEVICE_ID} --input_size=${INPUT_SIZE} --batch_size=${BATCH_SIZE} --epoch=${EPOCH} --k=4