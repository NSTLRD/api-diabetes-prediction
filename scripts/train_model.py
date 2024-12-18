import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from ucimlrepo import fetch_ucirepo

# Crear carpeta para modelos si no existe
MODEL_DIR = "C:/Users/starling/PycharmProjects/diabetes-prediction/models/"
os.makedirs(MODEL_DIR, exist_ok=True)

# Ruta para guardar el modelo entrenado
MODEL_PATH = os.path.join(MODEL_DIR, "diabetes_health_model.pkl")

# Descargar el dataset desde ucimlrepo
print("Descargando el dataset desde UCI ML Repository...")
cdc_diabetes_health_indicators = fetch_ucirepo(id=891)

# Extraer características (X) y objetivos (y)
X = cdc_diabetes_health_indicators.data.features
y = cdc_diabetes_health_indicators.data.targets

# Imprimir información del dataset (opcional)
print("Dataset cargado con éxito:")
print("Número de muestras:", X.shape[0])
print("Número de características:", X.shape[1])
print(X.columns)

# Dividir el dataset en entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo de Machine Learning
print("Entrenando el modelo...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluar el modelo en el conjunto de prueba
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo en datos de prueba: {accuracy * 100:.2f}%")

# Guardar el modelo entrenado
joblib.dump(model, MODEL_PATH)
print(f"Modelo guardado en: {MODEL_PATH}")
