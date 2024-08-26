import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC #2nd
from sklearn.metrics import accuracy_score #3rd
from sklearn.model_selection import GridSearchCV #4th
import torch


class svmModel():
    def __init__(self, X, y, target_label) -> None:

        X = torch.tensor(X).to(torch.float32).reshape(-1, 208*81)
        y = torch.tensor(y.astype(bool)).to(torch.float32)

        self.X = X
        self.y = y
        self.target_label = target_label

    def train(self) -> None:


        ## 1. Preprocess

        # 分割資料集為訓練集和測試集
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        # 標準化特徵
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)


        ## 2. Train SVM model

        # 創建SVM分類器，這裡使用線性核函數
        svm = SVC(kernel='linear', C=0.1)

        # 訓練模型
        svm.fit(X_train, y_train)


        ## 3. Prediction & Evaluation

        # 預測測試集
        y_pred = svm.predict(X_test)

        # 計算準確率
        accuracy = accuracy_score(y_test, y_pred)

        print(f'Accuracy: {accuracy:.2f}')


        # ## 4. Modify the hyperparameters

        # # 定義參數網格
        # param_grid = {
        #     'C': [0.1, 1, 10, 100],
        #     'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        #     'gamma': ['scale', 'auto']
        # }

        # # 創建網格搜索
        # grid_search = GridSearchCV(SVC(), param_grid, cv=5)

        # # 執行網格搜索
        # grid_search.fit(X_train, y_train)

        # # 找到最佳參數
        # best_params = grid_search.best_params_
        # print(f'Best parameters: {best_params}')


        # ## 5. Re-train model & Evaluation

        # # 使用最佳參數創建SVM分類器
        # svm = SVC(C=best_params['C'], kernel=best_params['kernel'], gamma=best_params['gamma'])

        # # 訓練模型
        # svm.fit(X_train, y_train)

        # # 預測測試集
        # y_pred = svm.predict(X_test)

        # # 計算準確率
        # accuracy = accuracy_score(y_test, y_pred)
        # print(f'Accuracy with best parameters: {accuracy:.2f}')