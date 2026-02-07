# DACON-236092-SemanticSegmentation

SW중심대학 공동 AI 경진대회 2023  
https://dacon.io/competitions/official/236092/overview/description

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
