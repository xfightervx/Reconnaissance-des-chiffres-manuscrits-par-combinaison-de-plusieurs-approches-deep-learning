from DataLoading_FeatureSelection.DataLoad import data_std_reshaped_loader
import joblib
from sklearn.svm import SVC


x_train, y_train, x_test, y_test = data_std_reshaped_loader()
selector =joblib.load('selector.pkl')
x_train,x_test = selector.transform(x_train), selector.transform(x_test)
svc = SVC(kernel='rbf', C=10, gamma=0.01 ,max_iter=700)
svc.fit(x_train, y_train)
joblib.dump(svc, 'svc.pkl')
print(svc.score(x_test, y_test))
