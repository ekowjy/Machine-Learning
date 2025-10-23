import numpy as np
from sklearn.linear_model import LinearRegression

# Data
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)  # jam belajar
y = np.array([50, 60, 65, 70, 75])            # nilai ujian

# Buat model regresi linear
model = LinearRegression()
model.fit(X, y)

# Tampilkan hasil
a = model.intercept_
b = model.coef_[0]
print(f"Persamaan: Y = {int(a)} + {int(b)}X")

# Prediksi jika belajar 6 jam
prediksi = model.predict([[6]])
print(f"Jika belajar 6 jam, nilai diprediksi: {int(prediksi[0])}")