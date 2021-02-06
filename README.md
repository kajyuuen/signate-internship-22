# 【学生限定】SIGNATE 22卒インターン選考コンペティション private3位解法

https://signate.jp/competitions/402/discussions

## プログラムの実行

```
$ ./run.sh
$ ./run.sh +defaults.models=cat
```

### ① Preprocessing

```
$ python -m src.preprocessing
```

### ② Learning

```
$ python learning.py
```

### ③ Predicting

```
$ python predicting.py
```

## Jupyter Notebookの起動

```
$ python -m jupyter-notebook --port=23013
```

## Tensor Boardの起動

```
$ tensorboard --logdir ./lightning_logs
```
