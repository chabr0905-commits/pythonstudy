import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


#미국 대학원 입학여부를 분류하는 모델을 작성하시오. 

df = pd.read_csv('binary.csv')
df = df.dropna()
print(df.head(3))
print(df.info())

#전처리: rank는 범주형 자료이므로 원핫 처리
df  = pd.get_dummies(df, columns=['rank'], dtype=int)
print(df.head(3))

# feature, label로 구분
x = df.drop('admit', axis=1)
y = df['admit']
print(x.head(3))
print(y.head(3))

#스케일링
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

#test/train 나누기
x_train, x_test, y_train, y_test = train_test_split(
    x_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y            # 클래스(0, 1) 비율을 원본과 동일하게 유지 (중요!)
    )

# model
print(x_train.shape[1])
model = Sequential([
    Input(shape=(x_train.shape[1],)),
    Dense(units=16, activation='relu'),
    Dense(units=8, activation='relu'),
    Dense(units=1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
print(model.summary())

history = model.fit(
    x_train, y_train,
    validation_data=(x_test,y_test),
    epochs=50, batch_size=32, verbose=2
)

loss,acc = model.evaluate(x_test, y_test, verbose=0)
print(f'테스트결과 손실:{loss:.4f}, 정확도:{acc:.4f}')


# acc
plt.subplot(1,2,2)
plt.plot(history.history['acc'],label='acc')
plt.plot(history.history['val_acc'], label='val acc')
plt.xlabel('epoch')
plt.ylabel('acc')
plt.legend()
plt.show()


#사용자 입력 결과 예측
