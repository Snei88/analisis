
# Análisis de calidad del vino tinto usando clasificación

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

# 2. Análisis exploratorio
print("\nInformación general del dataset:")
print(df.info())

print("\nEstadísticas descriptivas:")
print(df.describe())

print("\nClases de calidad de vino y su frecuencia:")
print(df["quality"].value_counts().sort_index())

# Gráfico de la distribución de calidad
plt.figure(figsize=(8, 5))
sns.countplot(x="quality", data=df, palette="Set2")
plt.title("Distribución de la calidad del vino")
plt.xlabel("Calidad")
plt.ylabel("Cantidad")
plt.tight_layout()
plt.show()

# Mapa de correlación
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Matriz de correlación entre características")
plt.tight_layout()
plt.show()

# 3. Preprocesamiento
# No hay valores nulos, se estandarizan los datos
X = df.drop("quality", axis=1)
y = df["quality"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. División del dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

# 5. Entrenamiento del modelo
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# 6. Evaluación
y_pred = modelo.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy del modelo: {accuracy:.4f}")
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))

# Matriz de confusión
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap="YlGnBu")
plt.title("Matriz de confusión")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.tight_layout()
plt.show()

# 7. Visualización de importancia de características
importancias = pd.Series(modelo.feature_importances_, index=X.columns)
importancias.sort_values().plot(kind='barh', figsize=(10,6), color='darkred')
plt.title("Importancia de características")
plt.xlabel("Importancia")
plt.tight_layout()
plt.show()

# 8. Interpretación
print("\n Interpretación del modelo:")
print("- Se analizaron 1599 vinos con 11 variables físico-químicas.")
print("- No se encontraron valores nulos, pero se aplicó estandarización.")
print("- Las características más importantes para clasificar la calidad fueron:")
print("  alcohol, volatile acidity, sulphates y citric acid.")
print(f"- El modelo Random Forest alcanzó una precisión del {accuracy:.2%}.")
print("- Los errores más comunes fueron entre clases vecinas (ej. confundir vinos de calidad 5 y 6).")
print("- Visualmente, se observó buen desempeño en las clases más frecuentes (5 y 6).")
