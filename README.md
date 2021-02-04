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