from DataLoad import data_rounded
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
import joblib

x_train, y_train, x_test, y_test = data_rounded()
x_train = x_train.reshape(x_train.shape[0],-1)
x_test = x_test.reshape(x_test.shape[0], -1)
scaler = StandardScaler()

x_train_std = scaler.fit_transform(x_train)
x_test_std = scaler.transform(x_test)
x_train_std_sample, _, y_train_sample, _ = train_test_split(x_train_std, y_train, train_size=0.01, random_state=1)
x_train_std_sample = np.array(x_train_std_sample)
y_train_sample = np.array(y_train_sample)
clf = RandomForestClassifier(n_estimators=100, random_state=1)
clf.fit(x_train_std_sample, y_train_sample)
selector = SelectFromModel(clf, threshold=0.001)
selector.fit(x_train_std_sample, y_train_sample)
x_train_selected = selector.transform(x_train_std)
x_test_selected = selector.transform(x_test_std)
joblib.dump(selector, 'selector.pkl')
clf.fit(x_train_selected, y_train)
accuracy = clf.score(x_test_selected, y_test)
print("Current accuracy:", accuracy)
print(x_train_selected[1])