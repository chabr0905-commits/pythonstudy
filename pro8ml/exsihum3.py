import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

df = pd.read_csv('bike_dataset.csv')

features = ['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed']
X = df[features]
y = df['count']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

importances = model.feature_importances_
for name, val in zip(features, importances):
    print(f"{name}: {val}")

y_pred = model.predict(X_test)
comp = pd.DataFrame({'실제값': y_test.values, '예측값': y_pred}).head(1)
print(comp)