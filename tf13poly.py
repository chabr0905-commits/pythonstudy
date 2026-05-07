# 다항회귀
# 매출 = 광고비 * w + b
# 매출 = 광고비 * w1 + 광고비2 * w2 + b

import numpy as np
import pandas as pd
import koreanize_matplotlib
import tensorflow as tf
import matplotlib.pyplot as plt

np.random.seed(7)
tf.random.set_seed(7)

# 광고비가 증가하면 매출도 증가하나, 어느정도 이후에는 증가폭이 둔화되는 곡선 데이터
ad_cost = np.linspace(0,100,80) # 광고비 데이터
# sales는 광고비에 따른 매출 데이터를 만드는 부분 2차함수
# sales = 광고비제곱*-0.06 +7.5 * 광고비 + 40 + noise 인위적으로 수식 작성
sales = (-0.06 * (ad_cost **2) + 7.5 * ad_cost + 40) + \
    np.random.normal(0, 25, size=len(ad_cost))

df = pd.DataFrame({'광고비':ad_cost,'매출':sales})
print(df.head())
df.to_csv('ad_sales.csv', index=False, encoding='utf8')
print('csv저장성공')

df = pd.read_csv('ad_sales.csv')
print(df.info())

# 결측치 있으면 해당 행 삭제
df = df.dropna()
print('데이터크기:',df.shape)

# feature 
x = df[['광고비']].values.astype(np.float32)
y = df[['매출']].values.astype(np.float32)
print(x[:3])
print(y[:3])

#산점도 
plt.figure(figsize=(8,5))

# traim test 분리
indices = np.arange(len(x))
np.random.shuffle(indices)
x = x[indices]
y = y[indices]
train_size = int(len(x)*0.8)

x_train = x[:train_size]
x_test = x[train_size:]
y_train = y[:train_size]
y_test = y[train_size:]
print('x:',x_train.shape,x_test.shape)
print('y:',y_train.shape,y_test.shape)
