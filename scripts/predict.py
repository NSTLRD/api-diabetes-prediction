import sys
import joblib
import numpy as np

# Cargar el modelo entrenado
model = joblib.load("../models/diabetes_model.pkl")

# Leer las características desde la consola
input_features = list(map(float, sys.argv[1].split(",")))
prediction = model.predict([input_features])

# Imprimir el resultado
print(prediction[0])
