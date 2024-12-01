# Sınıflandırma işlemi için Olivetti Faces veri seti kullanılıyor
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Olivetti Faces veri setini yükle
oli = fetch_olivetti_faces()

# Veri setinden ilk iki yüz görüntüsünü görselleştir
plt.figure()
for i in range(2):
    plt.subplot(1, 2, i + 1)  # 1 satır, 2 sütunlu bir alt grafik düzeni oluştur
    plt.imshow(oli.images[i + 40], cmap="gray")  # Yüz görüntülerini gri tonlamada göster
    plt.title(f"Face {i+1}")  # Görüntü başlıkları ekle

plt.show()

# Özellikler (X) ve hedef (y) değişkenlerini ayır
X = oli.data  # Yüzlerin pikselleri
y = oli.target  # Yüzlere ait sınıf etiketleri (kişiler)

# Eğitim ve test verilerini ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest sınıflandırıcısı oluştur ve eğitim verisiyle model eğit
rf_clfr = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clfr.fit(X_train, y_train)

# Test verisi ile tahmin yap
y_pred = rf_clfr.predict(X_test)

# Doğruluk oranını hesapla ve yazdır
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
