# MEG_DECODING
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
## cnn
```
ile "C:\Users\taliah\code\fyp\meg-shared\MEG_DECODING\main.py", line 193, in <module>
    cnnRunner = cnnModel(X, y)
                ^^^^^^^^^^^^^^
  File "C:\Users\taliah\code\fyp\meg-shared\MEG_DECODING\app\my_models\cnn\cnnModel.py", line 18, in __init__
    X = torch.tensor(X).to(torch.float32).reshape(-1, 208, 81, 1)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: shape '[-1, 208, 81, 1]' is invalid for input of size 15299232
```

# reference and resources

## CNN 超簡單解釋
https://medium.com/jameslearningnote/%E8%B3%87%E6%96%99%E5%88%86%E6%9E%90-%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-%E7%AC%AC5-1%E8%AC%9B-%E5%8D%B7%E7%A9%8D%E7%A5%9E%E7%B6%93%E7%B6%B2%E7%B5%A1%E4%BB%8B%E7%B4%B9-convolutional-neural-network-4f8249d65d4f