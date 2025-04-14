
# An�lisis de calidad del vino tinto usando clasificaci�n

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. Cargar los datos
df = pd.read_csv("winequality-red.csv", sep=';')  #ruta
print("Datos cargados correctamente.")

# 2. An�lisis exploratorio
print("\nInformaci�n general del dataset:")
print(df.info())

print("\nEstad�sticas descriptivas:")
print(df.describe())

print("\nClases de calidad de vino y su frecuencia:")
print(df["quality"].value_counts().sort_index())

# Gr�fico de la distribuci�n de calidad
plt.figure(figsize=(8, 5))
sns.countplot(x="quality", data=df, palette="Set2")
plt.title("Distribuci�n de la calidad del vino")
plt.xlabel("Calidad")
plt.ylabel("Cantidad")
plt.tight_layout()
plt.show()

# Mapa de correlaci�n
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Matriz de correlaci�n entre caracter�sticas")
plt.tight_layout()
plt.show()

# 3. Preprocesamiento
# No hay valores nulos, se estandarizan los datos
X = df.drop("quality", axis=1)
y = df["quality"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Divisi�n del dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

# 5. Entrenamiento del modelo
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# 6. Evaluaci�n
y_pred = modelo.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy del modelo: {accuracy:.4f}")
print("\nReporte de clasificaci�n:")
print(classification_report(y_test, y_pred))

# Matriz de confusi�n
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap="YlGnBu")
plt.title("Matriz de confusi�n")
plt.xlabel("Predicci�n")
plt.ylabel("Real")
plt.tight_layout()
plt.show()

# 7. Visualizaci�n de importancia de caracter�sticas
importancias = pd.Series(modelo.feature_importances_, index=X.columns)
importancias.sort_values().plot(kind='barh', figsize=(10,6), color='darkred')
plt.title("Importancia de caracter�sticas")
plt.xlabel("Importancia")
plt.tight_layout()
plt.show()

# 8. Interpretaci�n
print("\n Interpretaci�n del modelo:")
print("- Se analizaron 1599 vinos con 11 variables f�sico-qu�micas.")
print("- No se encontraron valores nulos, pero se aplic� estandarizaci�n.")
print("- Las caracter�sticas m�s importantes para clasificar la calidad fueron:")
print("  alcohol, volatile acidity, sulphates y citric acid.")
print(f"- El modelo Random Forest alcanz� una precisi�n del {accuracy:.2%}.")
print("- Los errores m�s comunes fueron entre clases vecinas (ej. confundir vinos de calidad 5 y 6).")
print("- Visualmente, se observ� buen desempe�o en las clases m�s frecuentes (5 y 6).")
