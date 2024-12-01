from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import DecisionBoundaryDisplay
import numpy as np
import matplotlib.pyplot as plt  # 'matplod' yerine 'matplotlib' olmalıydı

import warnings
warnings.filterwarnings("ignore")

# Iris veri setini yükleme
iris = load_iris()

# Hedef sınıf sayısını ve renkleri belirleme
n_classes = len(iris.target_names)
plot_colors = "ryb"

# Özellik çiftlerini doğru şekilde döngüye alalım
for pairidx, pair in enumerate([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]):
    X = iris.data[:, pair]  # Seçilen özellik çiftini al
    y = iris.target

    # Karar ağacı sınıflandırıcısını eğitme
    clf = DecisionTreeClassifier().fit(X, y)

    # Grafik oluşturma
    ax = plt.subplot(2, 3, pairidx + 1)
    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
    DecisionBoundaryDisplay.from_estimator(
        clf,
        X,
        cmap=plt.cm.RdYlBu,
        response_method="predict",
        ax=ax,
        xlabel=iris.feature_names[pair[0]],
        ylabel=iris.feature_names[pair[1]]
    )

    # Sınıfları grafiğe ekleme
    for i, color in zip(range(n_classes), plot_colors):  # 'zip' fonksiyonu hatalıydı
        idx = np.where(y == i)  # Sınıf etiketi filtresi düzeltildi
        plt.scatter(
            X[idx, 0], X[idx, 1], c=color, label=iris.target_names[i],
            cmap=plt.cm.RdYlBu, edgecolors="black"
        )

#  gösterim
plt.legend()
plt.show()
