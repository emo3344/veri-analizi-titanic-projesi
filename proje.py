# ===============================
# TITANIC VERI ANALIZI PROJESI
# ===============================

# 1 KÜTÜPHANELER
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 2 VERIYI OKUMA
df = pd.read_csv("train.csv")

# 3 VERI KEŞFI (EDA)
print(df.head())
print(df.info())
print(df.describe())

# 4 EKSİK VERI ANALIZI
print(df.isnull().sum())

# Age sütunundaki boş değerleri ortalama ile doldurma
df["Age"].fillna(df["Age"].mean(), inplace=True)

# Embarked sütunundaki boş değerleri en sık görülen ile doldurma
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

# Cabin sütununu çok boş olduğu için siliyoruz
df.drop("Cabin", axis=1, inplace=True)

# 5 KATEGORIK VERILERI SAYISALA ÇEVIRME
le = LabelEncoder()
df["Sex"] = le.fit_transform(df["Sex"])
df["Embarked"] = le.fit_transform(df["Embarked"])

# 6 GÖRSELLEŞTIRME
plt.figure()
sns.countplot(x="Survived", data=df)
plt.title("Hayatta Kalanlar vs Kalmayanlar")
plt.show()

plt.figure()
sns.histplot(df["Age"], bins=30)
plt.title("Yas Dagilimi")
plt.show()

# 7 MODELE HAZIRLIK
X = df.drop(["Survived", "Name", "Ticket", "PassengerId"], axis=1)
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 8 MODEL KURULUMU (LOGISTIC REGRESSION)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 9 TAHMIN
y_pred = model.predict(X_test)

# 10 DEGERLENDIRME
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("Classification Report:")
print(classification_report(y_test, y_pred))
