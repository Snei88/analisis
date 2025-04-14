#ESNEYDER CARMONA MONTOYA
#ANALSIIS DE DATOS



# 1. Importaci�n de librer�as
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc

# 2. Cargar el dataset
df = pd.read_csv("heart.csv")  #RUTA

# 3. Exploraci�n b�sica
print("Primeros registros:\n", df.head())
print("\nInformaci�n del dataset:\n")
print(df.info())
print("\nEstad�sticas descriptivas:\n", df.describe())
print("\nValores nulos:\n", df.isnull().sum())

# 4. An�lisis exploratorio visual
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='target', palette='Set2')
plt.title('Distribuci�n de la variable objetivo (Enfermedad card�aca)')
plt.show()

# Correlaci�n entre variables
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Matriz de correlaci�n')
plt.show()

# 5. Selecci�n de caracter�sticas (usaremos todas para este ejemplo)
X = df.drop('target', axis=1)
y = df['target']

# 6. Escalado de variables
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 7. Divisi�n de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

# 8. Entrenamiento del modelo
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 9. Predicciones
y_pred = model.predict(X_test)

# 10. Evaluaci�n del modelo
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nMatriz de confusi�n:\n", confusion_matrix(y_test, y_pred))
print("\nReporte de clasificaci�n:\n", classification_report(y_test, y_pred))

# 11. Curva ROC
y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('Tasa de falsos positivos')
plt.ylabel('Tasa de verdaderos positivos')
plt.title('Curva ROC')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()
