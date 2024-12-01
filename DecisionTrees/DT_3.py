from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor  # Doğru sınıf: Regressor kullanılır
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Veri setini yükleme
diabetes = load_diabetes()

# Özellikler (features) ve hedef (target) değişkenlerini ayırma
X = diabetes.data  # Features
y = diabetes.target  # Target

# Eğitim ve test verilerini ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Karar ağacı regresyon modeli
tree_reg = DecisionTreeRegressor(random_state=42)  # Classifier yerine Regressor kullanılmalı
tree_reg.fit(X_train, y_train)

# Tahmin yapma
y_pred = tree_reg.predict(X_test)

# Hata hesaplama
mse = mean_squared_error(y_test, y_pred)  # Ortalama kare hatası
print("Mean Squared Error (MSE):", mse)

rmse = np.sqrt(mse)  # Kök ortalama kare hatası
print("Root Mean Squared Error (RMSE):", rmse)
