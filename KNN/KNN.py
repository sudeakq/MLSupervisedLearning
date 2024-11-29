#sklern adında kütüphaneden veri cekecegiz
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd

#ilk adım veri seti incelemek
cancer= load_breast_cancer()
df=pd.DataFrame(data=cancer.data, columns = cancer.feature_names)
df["target"]=cancer.target

#iki knn sınıflandırıcı yontemının secilmesi
#modelin train edilmesi

X=cancer.data #features
y=cancer.target #target

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#olceklendirme
scaler=StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test =scaler.transform(X_test)

#knn modeli olustur ve train et
knn= KNeighborsClassifier(n_neighbors=3) #Model oluşturma komşu parametresini unutma
knn.fit(X_train, y_train) #fit fonksiyonu verimizi (samples + target) kullanarak knn algoritmasını eğitir

#sonuclarin değerlendirilmesi
y_pred =knn.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)
print("dogruluk:", accuracy)

conf_matrix= confusion_matrix(y_test, y_pred)
print("confusion matrix:")
print(conf_matrix)

#hiperparametre ayarlanması
"""
Knn: Hyperparameter = K
K: 1,2,3,....N
Accuracy:%A,%B,%C...

"""
accuracy_values=[]
k_values=[]
for k in range(1,21):
    knn= KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    accuracy_values.append(accuracy)
    k_values.append(k)

plt.figure()
plt.plot(k_values, accuracy_values, marker="0" , linestyle="-")
plt.title("k degerine gore dogruluk")
plt.xlabel("K degeri")
plt.ylabel("dogruluk")
plt.xsticks(k_values)
plt.grid(True)

