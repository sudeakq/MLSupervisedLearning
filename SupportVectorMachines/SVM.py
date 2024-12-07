from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Veri setini yükleme
digits = load_digits()

# Görselleri görselleştirme
figure, axes = plt.subplots(nrows=2, ncols=5, figsize=(10, 5),
                            subplot_kw={"xticks": [], "yticks": []})
for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap="binary", interpolation="nearest")
    ax.set_title(digits.target[i])
plt.show()

# Özellikler ve hedef değişken
X = digits.data
y = digits.target

# Eğitim ve test verisine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SVM modeli oluşturma ve eğitme
svm_clf = SVC(kernel="linear", random_state=42)
svm_clf.fit(X_train, y_train)

# Tahmin ve değerlendirme
y_pred = svm_clf.predict(X_test)
print(classification_report(y_test, y_pred))
