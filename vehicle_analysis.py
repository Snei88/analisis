
# Análisis de precios de vehículos usados

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Cargar datos
df = pd.read_csv("vehicle_dataset.csv")  #ruta

# 2. Análisis exploratorio
print("Información general del dataset:")
print(df.info())
print("\nEstadísticas descriptivas:")
print(df.describe())

# Mapa de calor de correlaciones numéricas
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Mapa de calor de correlaciones")
plt.tight_layout()
plt.show()

# 3. Preprocesamiento de datos
# Verificar duplicados
duplicados = df.duplicated().sum()
print(f"\nDuplicados encontrados: {duplicados}")
df = df.drop_duplicates()

# Asegurar que los booleanos sean enteros (0 y 1)
df = df.astype(int)

# 4. Selección de características
X = df.drop("selling_price", axis=1)
y = df["selling_price"]

# 5. División Train/Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 6. Entrenamiento del modelo
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# 7. Evaluación del modelo
y_pred = modelo.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nError Cuadrático Medio (MSE): {mse}")
print(f"R2 Score: {r2}")

# 8. Visualización de importancia de características
importancias = pd.Series(modelo.coef_, index=X.columns)
importancias.sort_values().plot(kind='barh', figsize=(10,6), color='skyblue')
plt.title("Importancia de características (coeficientes)")
plt.xlabel("Valor del coeficiente")
plt.tight_layout()
plt.show()

# 9. Interpretación rápida
print("\nInterpretación:")
print("Valores negativos indican una relación inversa con el precio de venta.")
print("Por ejemplo, 'transmission_Manual' tiene un coeficiente negativo fuerte,")
print("lo que sugiere que los autos con transmisión manual tienden a tener precios más bajos.")
