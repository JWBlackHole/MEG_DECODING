# MEG_DECODING

-------

# PLOT EVOKED RESPONSE

**JUST RUN THIS CONFIG**

[plot_evo.json](app/config/plot_evo.json)

------------

## 安裝requirements

`pip install -r requirements.txt`

## How to run
cd to root dir (`app` 的parent directory)

`python -m app.main`

## 主程式
程式都在`app` 資料夾

## 程式起點
[./main.py](app/main.py)

## Main Files
[train_config.json](train_config.json)  
- template config file

[nnModelRunner.py](app/models/nn/nnModelRunner.py)
- NN model 的前置作業

[nnModel.py](app/models/nn/nnModel.py)
- NN Model

[ldaModelRunner.py](app/models/lda/ldaModelRunner.py)
- Linear Discriminant Analysis (LDA) 的前置作業
  
[ldaModel.py](app/models/lda/ldaModel.py)
- LDA Model
- 
[preprocessor.py](app/signal/preprocessor.py)
- load data, set up MNE epoch & meta data, 過high, low pass filter

## Config Explain

target label:
- "voiced"
- "word_onset"
- plot_word_evo

### 畫圖evoke response 圖
"flow": "plot_word_evo"
"target_label": "plot_word",


## graph

evoked response of each event
https://drive.google.com/drive/u/0/folders/1ZoWMuvWO3q_cWMu2V3O5MAw7kJiG3BQ-?hl=zh-TW

## Remark

- save_path, 開檔案等等路徑需要relative to root dir (MEG_DECODE 這個資料夾)
- preprocess_low_pass 要小, preprocess_low_high 要大!!

## Test
### 測試 custom import

自己在test_main.py import 新的custom modules
```
cd to root
python -m app.test_main
```

# TO DO
改先叫model, model 再load data from disk per batch
研究如何一個一個batch load

# sensor topology
![alt text](graphs/sensor_topology.png)

# 方向
decim 改1 (then should be 800 time point each epoch), 用cnn sliding window 每n 個取一個average, n=100 之類

# status after merge
## NN
everything ok
## svm
```
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "C:\Users\taliah\code\fyp\meg-shared\MEG_DECODING\main.py", line 168, in <module>
    svmRunner = svmModel(X, y, target_label)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\taliah\code\fyp\meg-shared\MEG_DECODING\app\my_models\svm\svmModel.py", line 14, in __init__
    X = torch.tensor(X).to(torch.float32).reshape(-1, 208*81)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: shape '[-1, 16848]' is invalid for input of size 15299232
(gw) PS C:\Users\taliah\code\fyp\meg-shared\MEG_DECODING>
```


/home/dataset/Data/manho/MEG_DECODING/app/signal/megSignal.py:213: RuntimeWarning: The measurement information indicates a low-pass frequency of 180.0 Hz. The decim=10 parameter will result in a sampling frequency of 100.0 Hz, which can cause aliasing artifacts.

# mem and cpu usage
batch size 2000 + load 1 subject all ses, all task->memory max usage 87%
~70% when reading event
~87% when apply baseline, apply filter

## memory issue update
cannot call `concatenate_epochs`!!! **this will load data!!!**

## whats sure

1-8Hz + voiced phoneme -> better
offet onset time for  -0.05s --> ???