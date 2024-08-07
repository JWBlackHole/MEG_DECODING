import torch 
from torch import nn

from MyModel import MyModel

class NNModelRunner():
    def __init__(self,X, y) -> None:

        self.X = X
        self.y = y
        X = torch.tensor(X)
        y = torch.tensor(y)
        print(type(X), type(y))
        
        # print(len(X), len(y))
        # print(len(X[0]), len(X[1]))
        # print(len(X[0][0]), len(X[1][0]))
        # print(X[0][0])
        # print(X, y)
        
        train_split = int(0.8 * len(X))
        X_train, y_train = X[:train_split], y[:train_split]
        X_test,  y_test  = X[train_split:], y[train_split:]

        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_0 = MyModel(num_classes = 2).to(device)
        
        loss_function = torch.nn.BCELoss
        optimizer = torch.optim.SGD(params = model_0.parameters(), lr = 0.1)
        
        # for epoch in range(100):