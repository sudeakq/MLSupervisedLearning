from sklearn.tree import DecisionTreeRegressor  # Karar ağacı regresör sınıfı
import numpy as np
import matplotlib.pyplot as plt

# Veri setini oluştur
X = np.sort(5 * np.random.rand(80, 1), axis=0)
y = np.sin(X).ravel()

# Gürültü ekle
y[::5] += 0.5 * (0.5 - np.random.rand(16))

# Karar ağacı regresör modelleri
regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor(max_depth=5)

# Modelleri eğit
regr_1.fit(X, y)
regr_2.fit(X, y)

# Test verisi
X_test = np.arange(0, 5, 0.05)[:, np.newaxis]

# Tahminler
y_pred_1 = regr_1.predict(X_test)
y_pred_2 = regr_2.predict(X_test)

# Tahminleri görselleştir
plt.figure()

# Veriyi çiz
plt.scatter(X, y, color="red", label="Data")

# Tahminleri çiz
plt.plot(X_test, y_pred_1, color="blue", label="Max Depth: 2", linewidth=2)
plt.plot(X_test, y_pred_2, color="green", label="Max Depth: 5", linewidth=2)

# Grafik ayarları
plt.xlabel("Data")
plt.ylabel("Target")
plt.legend()
plt.title("Karar Ağacı Regresör Tahminleri")
plt.show()
