from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

# Inicializar la aplicación Flask
app = Flask(__name__)

# Ruta al modelo entrenado
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "diabetes_health_model.pkl")

# Cargar el modelo entrenado
try:
    model = joblib.load(MODEL_PATH)
    print("Modelo cargado exitosamente.")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    model = None

# Nombres de las características utilizadas durante el entrenamiento
FEATURE_NAMES = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
    "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education",
    "Income"
]


@app.route("/predict", methods=["POST"])
def predict():
    """Realiza la predicción de diabetes con una interpretación extendida."""
    if not model:
        return jsonify({"error": "El modelo no está disponible"}), 500

    try:
        # Leer los datos enviados en el cuerpo de la solicitud (JSON)
        data = request.get_json()

        # Agregar valores predeterminados para características faltantes
        for feature in FEATURE_NAMES:
            if feature not in data:
                data[feature] = 0  # Valor predeterminado, ajusta si es necesario

        # Convertir los datos a un DataFrame de Pandas con los nombres correctos
        features_df = pd.DataFrame([data], columns=FEATURE_NAMES)

        # Realizar la predicción
        prediction = model.predict(features_df)
        binary_result = int(prediction[0])  # 0: No diabetes, 1: Prediabetes o diabetes

        # Clasificación extendida
        bmi = data["BMI"]
        age = data["Age"]
        high_bp = data["HighBP"]
        high_chol = data["HighChol"]
        phys_activity = data["PhysActivity"]

        # Evaluar clasificación extendida
        if bmi >= 25 and bmi < 30 and high_bp == 1 and phys_activity == 0:
            classification = "Prediabetes"
        elif binary_result == 0:
            if bmi >= 18.5 and bmi <= 25 and high_bp == 0 and high_chol == 0 and phys_activity == 1:
                classification = "Persona Sana"
            else:
                classification = "No Diabetes Detectada"
        else:
            if bmi < 18.5:
                classification = "Diabetes Tipo 1 (Posible)"
            else:
                classification = "Diabetes Tipo 2 (Posible)"

        # Respuesta
        response = {
            "prediction": binary_result,
            "classification": classification,
            "interpretation": "Diabetes detectada" if binary_result == 1 else "No se detectó diabetes"
        }
        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500




if __name__ == "__main__":
    app.run(debug=True)
