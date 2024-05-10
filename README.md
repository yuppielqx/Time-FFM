
## Getting Started

### Requirements
Our experimental environments include Python 3.9, Pytorch 1.13.1 with CUDA 11.6, and transformers 4.31.0. To install all dependencies, please use the below command.
```
pip install -r requirements.txt
```

### Datasets
The pre-processed datasets can be obtained from the link [here](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2?usp=sharing). Then you may choose to download `all_datasets.zip`, place this zip file into the `dataset` folder, and finally unzip the file.

### Running
In general, we use a csv file to indicate the executing tasks (including training and evaluations) during the cross-domain learning process. There are five columns in the file.

(1) Data: the name of a dataset, corresponding to a config file in the folder `data_configs`.

(2) Prediction: the prediction length.

(3) Train: the indicator for training.

(4) Valid: the indicator for validation.

(5) Test: the indicator for testing.

For example, the below command is used to train one model for the tasks listed in the file `execute_list/train_all.csv`. Note that the argument `max_token_num` should be set to a value larger than the combined number of tokens in both language instructions and time series patches.
```
python run.py --gpu 0 --training_list execute_list/train_all.csv --percent 100
```


## Acknowledgement
We appreciate the following github repository for sharing the valuable code base and datasets:

https://github.com/thuml/Time-Series-Library

https://github.com/liuxu77/UniTime