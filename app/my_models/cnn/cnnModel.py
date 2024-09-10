import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
import torch


class cnnModel():
    def __init__(self, X, y) -> None:
        # 假設我們有一個包含特徵和標籤的數據集
        # 如果X是數字特徵而非圖像，可能需要reshape，例如 reshape為 (樣本數, 高, 寬, 通道數)
        # 假設X原本是數值類型，這裡僅舉例變成 (28, 28, 1) 的格式
        X = torch.tensor(X).to(torch.float32).reshape(-1, 208, 81, 1)
        y = torch.tensor(y.astype(bool)).to(torch.float32)

        self.X = X
        self.y = y

    def train(self) -> None:
        # 分割資料集為訓練集和測試集
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=33)

        # ##################
        
        # # 假設 y_train 是原始標籤數據
        # X_train_balanced = X_train.copy()
        # y_train_balanced = y_train.copy()

        # # 將數據分為多數類別和少數類別
        # X_train_majority = X_train_balanced[y_train_balanced[:, 0] == 0]
        # y_train_majority = y_train_balanced[y_train_balanced[:, 0] == 0]
        # X_train_minority = X_train_balanced[y_train_balanced[:, 0] == 1]
        # y_train_minority = y_train_balanced[y_train_balanced[:, 0] == 1]

        # # 欠採樣多數類別
        # X_train_majority_downsampled, y_train_majority_downsampled = resample(
        #     X_train_majority, y_train_majority,
        #     replace=False,  # 不重複抽樣
        #     n_samples=len(y_train_minority),  # 使其與少數類別數量相同
        #     random_state=42
        # )

        # # 合併少數類別和欠採樣後的多數類別
        # X_train = np.vstack((X_train_majority_downsampled, X_train_minority))
        # y_train = np.vstack((y_train_majority_downsampled, y_train_minority))

        # ##################

        # # 如果是數值特徵則可能需要reshape
        # X_train = X_train.reshape(-1, 28, 28, 1)
        # X_test = X_test.reshape(-1, 28, 28, 1)

        # # 標準化數據（例如，如果是像素值，可以除以255使數據範圍在0到1之間）
        # X_train = X_train.astype('float32') / 255.0
        # X_test = X_test.astype('float32') / 255.0

        # 將標籤轉換為one-hot編碼
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

        # 建立CNN模型
        model = Sequential([
            Conv2D(32, (10, 20), activation='relu', input_shape=(208, 81, 1)),  # 第一層卷積 #(1000, 20)
            # MaxPooling2D(pool_size=(2, 2)),  # 池化層
            Conv2D(64, (10, 20), activation='relu'),  # 第二層卷積
            # MaxPooling2D(pool_size=(2, 2)),  # 池化層
            Flatten(),  # 展平層
            Dense(128, activation='relu'),  # 全連接層
            Dropout(0.5),  # 防止過擬合
            Dense(y_train.shape[1], activation='sigmoid')  # 輸出層  # softmax
        ])

        # 編譯模型
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) # loss='categorical_crossentropy'

        # 訓練模型
        history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

        # 評估模型
        # loss, accuracy = model.evaluate(X_test, y_test)
        # print(f'Loss: {loss:.2f} Accuracy: {accuracy:.2f}')

        # 使用模型進行預測，得到的結果是概率
        y_pred_prob = model.predict(X_test)
        print(y_pred_prob[:100])

        # 將概率轉換為 0 或 1
        y_pred = (y_pred_prob >= 0.5).astype(int)

        # 打印一些預測結果查看
        # print(y_pred[:100])  # 查看前100個預測值

        # 計算模型的準確率
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Accuracy: {accuracy:.2f}')
