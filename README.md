# MEG_DECODING
## 安裝requirements

`pip install -r requirements.txt`

## How to run
cd to root dir (`app` 的parent directory)

`python -m app.main`

## 主程式
程式都在`app` 資料夾

## 程式起點
[./app/main.py](app/main.py)

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

## Remark

- save_path, 開檔案等等路徑需要relative to root dir (MEG_DECODE 這個資料夾)

## Test
### 測試 custom import

自己在test_main.py import 新的custom modules
```
cd to root
python -m app.test_main
```

# TO DO
處理一次train 一個人n 個task n 個session or n個人n 個task n 個session 

