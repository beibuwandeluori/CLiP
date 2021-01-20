BATCH_SIZE=16
DEVICE_ID=4
INPUT_SIZE=512
FOLD_ID=3
python test.py --model_name=resnet200d --device_id=${DEVICE_ID} --input_size=${INPUT_SIZE} --batch_size=${BATCH_SIZE} --k=${FOLD_ID}