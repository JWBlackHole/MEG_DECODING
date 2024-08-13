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