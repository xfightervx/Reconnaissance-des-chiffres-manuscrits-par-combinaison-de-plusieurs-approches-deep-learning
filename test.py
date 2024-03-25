import joblib
from drawing import get_drawing_matrix
import numpy as np
import joblib
svc = joblib.load('svc.pkl')
selector = joblib.load('selector.pkl')
matrix = get_drawing_matrix()
matrix = np.array(matrix)
matrix = matrix.flatten()
sc = joblib.load('scaler.pkl')
matrix = np.array([matrix])
matrix = sc.transform(matrix)
matrix = selector.transform(matrix)
prediction = svc.predict(matrix)
print(prediction)