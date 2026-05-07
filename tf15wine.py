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

wdf = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/wine.csv")
print(wdf.head(2))
print(wdf.info())

print(wdf.iloc[:,12].unique())
print(len(wdf[wdf.iloc[:,12]==0]))
print(len(wdf[wdf.iloc[:,12]==1]))

#array로 변환
dataset = wdf.values
x = dataset[:, 0:12]
y = dataset[:, -1]
print(x[:2])
print(y[:2])

x_train,x_test,y_train,y_test= train_test_split(x,y, \
                                                test_size=0.3, random_state=12,stratify=y,shuffle=True)
print(x_train[:2],x_train.shape)
print(y_train[:2],y_train.shape)


# 신경망(ANN) 모델
model = Sequential()
model.add(Input(shape=(12,)))
    # Hidden Layers
model.add(Dense(units=24, activation='relu')) #활성화함수: ReLU, 시그모이드,Leaky ReLU, ELU
model.add(Dense(units=12, activation='relu'))
model.add(Dense(units=8, activation='relu'))
    # Output Layer
model.add(Dense(units=1, activation='sigmoid'))
print(model.summary())

model.compile(loss='binary_crossentropy', optimizer ='adam', metrics=['accuracy'])
# fit() 전에 훈련되지 않은 모델의 정확도

loss, acc = model.evaluate(x_train, y_train, verbose = 0)
print(f'훈련되지 않은 모델의 정확도:{acc * 100}%')

# 조기 종료
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# 모델 지정
MODEL_DIR = './winemodel/'
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# 조건 설정
# modelpath = 'model/{epoch:02d}-{val_loss:.3f}.keras'
modelpath = './winemodel/winemodel.keras'
chkpoint = ModelCheckpoint(filepath=modelpath, monitor='val_loss',\
                            mode='auto', save_best_only=True)

# 학습 모델
history = model.fit(x_train,y_train, epochs=1000,\
                    validation_split=0.2, batch_size=64,\
                        callbacks=[early_stop])

loss,acc = model.evaluate(x_test, y_test, verbose = 0)
print(f'훈련된 모델의 정확도:{acc * 100}%')

# 시각화 


# 지정된 모델로 예측
from tensorflow.keras.models import load_model

mymodel = load_model(modelpath)
new_data = x_test[:5, :]
print(new_data)
new_pred = mymodel.predict(new_data)
print('예측결과:', np.where(new_pred >= 0.5, 1,0).ravel())
