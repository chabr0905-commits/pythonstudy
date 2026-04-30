# 연산자와 기초함수
import tensorflow as tf
import numpy as np


x = tf.constant(7)
y = tf.constant(3)

# cond(): 삼항연산 #if-else 문과 유사
result1 = tf.cond(x>y,lambda:tf.add(x,y), lambda:tf.subtract(x,y))
print(result1)

# case() 조건 연산 # switch-case 문과 유사
f1 = lambda:tf.constant(1) # lambda에 의해 1을 반환
f2 = lambda:tf.constant(tf.multiply(2,3))
result2 = tf.case([(tf.less(x,y),f1)], default=f2) 
# if (x < y) return 1 else return 6
print(result2)

print('관계연산---')
print(tf.equal(1,2))
print(tf.not_equal(1,2))
print(tf.less(1,2))
print(tf.greater(1,2))
print(tf.greater_equal(1,2))

print('논리연산---')
print(tf.logical_and(True,False))
print(tf.logical_or(True,False))
print(tf.logical_not(True))

print('유일 합집합---') # 중복제거 인덱싱에 도움
kbs = tf.constant([1,2,2,3,2])
val, idx = tf.unique(kbs) #유일값과 인덱스 반환
print('val: ', val)
print('idx: ', idx)

print('reduce ~ 함수 ---')
ar = tf.constant([[1.0, 2.0], [3.0, 4.0]])
print(tf.reduce_mean(ar, axis =1).numpy())
print(tf.reduce_max(ar).numpy()) # 4.0

print('reshape 함수 ---')
t = np.array([[0,1,2],[3,4,5],[6,7,8],[9,10,11]])
print(t.shape)
print(tf.reshape(t, shape=[12]))
print(tf.reshape(t, shape=[2,6]))
print(tf.reshape(t, shape=[-1,6]))
print(tf.reshape(t, shape=[2,-1]))

print('squeeze 함수 ---')
print(tf.squeeze(t))
t2 = np.array([[0],[3],[6],[9]])
print(t2.shape)
print(tf.squeeze(t2))

print('expend 함수: 차원확대---')
tarr = tf.constant([[1,2,3],[4,5,6]])
print(tarr.shape)
sbs = tf.expand_dims(tarr,0) #첫번째 차원을 추가해 확장
print(sbs.numpy())
sbs = tf.expand_dims(tarr,1) #두번째 차원을 추가해 확장
print(sbs.numpy())
sbs = tf.expand_dims(tarr,2) #세번째 차원을 추가해 확장
print(sbs.numpy())
sbs = tf.expand_dims(tarr,-1) #axis=-1은 마지막 위치에 새 차원 추가
print(sbs.numpy())

print('cast 함수: 자료형 변환---')
num =  tf.constant([1,2,3])
num2 = tf.cast(num, tf.float32)
print(num2, num2.dtype)

