from DataLoad import np_data_loader
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split

x_train, y_train, x_test, y_test = np_data_loader()
x_train = x_train.reshape(x_train.shape[0],-1)
x_test = x_test.reshape(x_test.shape[0], -1)
scaler = StandardScaler()

x_train_std = scaler.fit_transform(x_train)
x_test_std = scaler.transform(x_test)
x_train_std_sample, _, y_train_sample, _ = train_test_split(x_train_std, y_train, train_size=0.01, random_state=1)
x_train_std_sample = np.array(x_train_std_sample)
y_train_sample = np.array(y_train_sample)
feat_labels = np.arange(0, x_train_std_sample.shape[1], 1)
forest = RandomForestClassifier(criterion='gini', n_estimators=500 , random_state=1)
forest.fit(x_train_std_sample,y_train_sample)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
sfm = SelectFromModel(forest,threshold=0.1 , prefit=True)
X_selected = sfm.transform(x_train_std_sample)
for f in range(x_train_std_sample.shape[1]):
    print ("%2d) %-*s %f" % (f + 1, 30,feat_labels[indices[f]],importances[indices[f]]))
plt.title('Feature Importance')
plt.bar(range(x_train_std_sample.shape[1]),importances[indices],align='center')
plt.xticks(range(x_train_std_sample.shape[1]),feat_labels, rotation=90)
plt.xlim([-1, x_train_std_sample.shape[1]])
plt.tight_layout()
plt.show()
