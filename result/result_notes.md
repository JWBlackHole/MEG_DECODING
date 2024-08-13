# LDA

## KFold
setting:
```
"training": {
        "flow": "lda",
        "until_subject": "01",
        "until_session": "1",
        "until_task": "1",
        "preprocess_low_pass": 30,
        "preprocess_high_pass": 0.5
    }
```
```
scores (returned from model):
     fold_0    fold_1    fold_2    fold_3    fold_4    fold_5    fold_6    fold_7    fold_8    fold_9   fold_10   fold_11   fold_12   fold_13   fold_14        
0  0.621884  0.616343  0.594183  0.610803  0.613573  0.603878  0.612188  0.596953  0.595568  0.608033  0.594183  0.602493  0.601108  0.591413  0.610803        
```

## NN
```
Epoch: 0 | Loss: 0.63942, Accuracy: 69.67% | Test loss: 0.64662, Test acc: 67.17%
Epoch: 10 | Loss: 0.62503, Accuracy: 69.67% | Test loss: 0.63716, Test acc: 67.17%
Epoch: 20 | Loss: 0.61865, Accuracy: 69.67% | Test loss: 0.63385, Test acc: 67.17%
Epoch: 30 | Loss: 0.61584, Accuracy: 69.67% | Test loss: 0.63299, Test acc: 67.17%
Epoch: 40 | Loss: 0.61460, Accuracy: 69.67% | Test loss: 0.63301, Test acc: 67.17%
Epoch: 50 | Loss: 0.61406, Accuracy: 69.67% | Test loss: 0.63328, Test acc: 67.17%
Epoch: 60 | Loss: 0.61383, Accuracy: 69.67% | Test loss: 0.63357, Test acc: 67.17%
Epoch: 70 | Loss: 0.61373, Accuracy: 69.67% | Test loss: 0.63382, Test acc: 67.17%
Epoch: 80 | Loss: 0.61369, Accuracy: 69.67% | Test loss: 0.63400, Test acc: 67.17%
Epoch: 90 | Loss: 0.61367, Accuracy: 69.67% | Test loss: 0.63412, Test acc: 67.17%
Epoch: 100 | Loss: 0.61366, Accuracy: 69.67% | Test loss: 0.63421, Test acc: 67.17%
```