# DACON-236107-ObjectDetection

합성데이터 기반 객체 탐지 AI 경진대회  
https://dacon.io/competitions/official/236107/overview/description

## Requirement

- Ultralytics Docker Container
- pandas

```shell
docker run --name yolo --ipc=host --runtime=nvidia --gpus all -i -t -v ./:/workspace -w /workspace ultralytics/ultralytics
```

```shell
pip install pandas
```

## Prepare Data

```shell
python scripts/prepare_yolo_labels.py
python scripts/split_train_val_symlink.py
```

## Train

```shell
python train.py
```

## Predict

> Need `Train` step

```shell
python predict.py
```
