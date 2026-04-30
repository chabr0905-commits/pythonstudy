# 다층 신경망 구성: Keras(Deep Learning 라이브러리) 모듈 사용
# 일관성 있게 API 제공 받을 수 있음
# 머신러닝 모델을 쉽게 작성이 가능

# 실습: 논리회로 처리를 위한 분류 간단한 모델

import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Activation
from tensorflow.keras.optimizers import SGD, RMSprop, Adam

# 1) 데이터 수집 및 가공
x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[1]]) # y = np.array([0,1,1,1]) 같음

#2) 모델 설정
model = Sequential([
    Input(shape=(2, )),
    Dense(units=1),
    Activation('sigmoid')
])

# 이것도 똑같음
# model = Sequential()
# model.add(Input(shape=(2,)))
# model.add(Dense(units=1))
# model.add(Activation('sigmoid'))


# 3) 모델 학습 과정 설정
model.compile(loss='binary_crossentropy', optimizer='sgd',metrics=['accuracy'])
# loss 손실함수: 훈련데이터에서 신경망의 성능을 측정하는 방법으로 모델이 정확히 학습할 수 있다
# optimizer : 입력데이터와 손실함수를 기반으로 모델을 갱신함
# metrics: 훈련단계와 검정단게를 모니터링 하기 위해 사용 (모델 성능 지표)
model.compile(loss='binary_crossentropy', optimizer='rmsprop',metrics=['accuracy'])

# 4) 모델 학습 (더 나은 결과를 찾는 자동화된 과정): train data 사용
model.fit(x=x,y=y,epochs=5, batch_size=1, verbose=1)

# 5) 모델 평가: test data 사용
loss_metrics = model.evaluate(x=x, y=y)
print('loss_metrics:',loss_metrics)

# 6) 학습결과 확인 (예측)
pred = model.predict(x=x)
print('예측결과: ', pred)

# epochs 횟수 올릴수록 정확도는 올라가고 손실은 줄어듦 -> 경사하강법

proba = model.predict(x=x, verbose=0)
pred = (proba > 0.5).astype('int32')
print('예측값: ',pred.ravel())
print('실제 값: ', y.ravel())

