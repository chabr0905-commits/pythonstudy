import pandas as pd
data = pd.read_csv('titanic_data.csv', usecols=['Survived', 'Pclass', 'Sex', 'Age','Fare'])
print(data.head(2), data.shape)    # (891, 12)
data.loc[data["Sex"] == "male","Sex"] = 0
data.loc[data["Sex"] == "female", "Sex"] = 1

data["Sex"] = data["Sex"].astype(int)

print(data["Sex"].head(2))
print(data.columns)

feature = data[["Pclass", "Sex", "Fare"]]
label = data["Survived"]

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(
    feature, label, test_size=0.3, random_state=12
)

# 2) 의사결정나무 클래스를 사용해 분류 모델 작성
dt_model = DecisionTreeClassifier(random_state=12)
dt_model.fit(X_train, y_train)

# 3) 예측결과로 분류 정확도를 출력
y_pred = dt_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"의사결정나무 분류 정확도: {accuracy:.4f}")
