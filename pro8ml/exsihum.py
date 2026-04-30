import numpy as np
from scipy import stats

# 1. 귀무가설: 강사에 따라 성적의 차이가 없다
# 2. 대립가설: 강사에 따라 성적의 차이가 있다

t1 = np.array([71, 58, 92, 78, 71, 68, 67, 88, 88, 60, 80, 70, 68, 82, 78])
t2 = np.array([50, 65, 75, 91, 67, 39, 81, 68, 97, 86, 66, 60, 65, 55, 58])

# 정규성 검정

# 독립표본 t-검정 
t_stat, p_value = stats.ttest_ind(t1, t2)

print(f"t-통계량: {t_stat}")
print(f"p-value: {p_value}")

