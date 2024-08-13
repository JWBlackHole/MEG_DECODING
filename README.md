# MEG_DECODING

## 程式起點
[main.py](main.py)

## Main Files
[train_config.json](train_config.json)  
- 預計將來在這裡放training 及一些house keeping的參數
- 目前只有少量參數作為template

[nnModelRunner.py](nnModelRunner.py)
- NN model 的前置作業

[MyModel.py](MyModel.py)
- NN Model

[ldaModelRunner.py](ldaModelRunner.py)
- Linear Discriminant Analysis (LDA) 的前置作業
  
[ldaModel.py](ldaModel.py)
- LDA Model
- 
[preprocessor.py](preprocessor.py)
- load data, set up MNE epoch & meta data, 過high, low pass filter