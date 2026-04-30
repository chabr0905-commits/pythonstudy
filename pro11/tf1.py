import tensorflow as tf
import numpy as np

print(tf.__version__)
print('즉시 실행 모드:',tf.executing_eagerly())
print('GPU 사용 정보 확인:', tf.config.list_physical_devices('GPU'))

print('\nTensor: 텐서플로에서 데이터를 담는 기본 자료구조(숫자 데이터 저장용 다차원 배열)')
# ndarray 와 비슷하지만 텐서플로에서 연산에 사용되도록 만들어진 객체
print(12, type(12)) #12 <class 'int'>
print(tf.constant(12)) #(12, shape=(), dtype=int32)
print(tf.constant([12])) #([12], shape=(1,), dtype=int32)
print(tf.constant([[12]])) #([[12]], shape=(1, 1), dtype=int32)
print(tf.constant([[12,1]]))
print(tf.rank(tf.constant([[12,1]])))
tf.print(tf.constant(12))

print('파이썬 기본함수, 객체 자체를 문자열로 반환 후 출력, 정보 중심 출력')
tf.print('텐서플로 전용 출력함수, 텐서 실제값을 중심으로 출력')

print()
imsi = np.array([1,2])
print(type(imsi)) # <class 'numpy.ndarray'>
imsi[0] = 10 # 값 변경 가능

a = tf.constant([1,2])
print(type(a)) # <class 'tensorflow.python.framework.ops.EagerTensor'>

# a[0] = 10 에러 값변경 불가
b = tf.constant([3,4]) 


print(7)

print(tf.convert_to_tensor(7))



