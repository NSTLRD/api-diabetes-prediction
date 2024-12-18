# Diabetes Prediction API

## Introduction
This project implements a Flask-based REST API for diabetes prediction using a Machine Learning model trained with the `cdc_diabetes_health_indicators` dataset from the UCI Machine Learning Repository. The model utilizes a Random Forest Classifier to predict the likelihood of diabetes based on various health indicators.

## Features
- **RESTful API**: Exposes a `/predict` endpoint for diabetes prediction.
- **Machine Learning**: Uses a Random Forest Classifier trained with real-world data.
- **Extensibility**: Modular design for easy enhancements or integration with other systems.

## Technologies Used
- **Python**: Programming language.
- **Flask**: Lightweight web framework for building the API.
- **scikit-learn**: For model training and evaluation.
- **Pandas**: For data manipulation.
- **joblib**: For saving and loading the trained model.

## Project Structure
```
flask-diabetes-prediction/
├── app.py                     # Main application file
├── models/
│   └── diabetes_health_model.pkl  # Trained model
├── requirements.txt           # Python dependencies
├── scripts/
│   └── train_model.py         # Script to train the ML model
├── data/
│   └── diabetes.csv           # Example dataset (optional)
└── README.md                  # Project documentation
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/flask-diabetes-prediction.git
   cd flask-diabetes-prediction
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Ensure the dataset and model are correctly set up:
   - Place the trained model (`diabetes_health_model.pkl`) in the `models/` directory.
   - Optionally, provide the dataset in the `data/` directory if retraining is needed.

## Usage

### Run the API
Start the Flask server:
```bash
python app.py
```

The server will run by default on `http://127.0.0.1:5000`.

### Endpoints

#### `POST /predict`
- **Description**: Predicts the likelihood of diabetes based on health indicators.
- **Request Body**:
  ```json
  {
      "HighBP": 1,
      "HighChol": 0,
      "CholCheck": 1,
      "BMI": 27,
      "Smoker": 0,
      "Stroke": 0,
      "HeartDiseaseorAttack": 0,
      "PhysActivity": 0,
      "Fruits": 0,
      "Veggies": 0,
      "HvyAlcoholConsump": 0,
      "AnyHealthcare": 1,
      "NoDocbcCost": 0,
      "GenHlth": 3,
      "MentHlth": 2,
      "PhysHlth": 5,
      "DiffWalk": 1,
      "Sex": 1,
      "Age": 45,
      "Education": 2,
      "Income": 4
  }
  ```
- **Response**:
  ```json
  {
      "prediction": 0,
      "classification": "No Diabetes Detected",
      "interpretation": "No diabetes detected"
  }
  ```

### Retrain the Model
If retraining the model is needed, run the training script:
```bash
python scripts/train_model.py
```
The retrained model will be saved in the `models/` directory.

## Testing
- Use **Postman** or **cURL** to send requests to the API.
- Example using `cURL`:
  ```bash
  curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
      "HighBP": 1,
      "HighChol": 0,
      "CholCheck": 1,
      "BMI": 27,
      "Smoker": 0,
      "Stroke": 0,
      "HeartDiseaseorAttack": 0,
      "PhysActivity": 0,
      "Fruits": 0,
      "Veggies": 0,
      "HvyAlcoholConsump": 0,
      "AnyHealthcare": 1,
      "NoDocbcCost": 0,
      "GenHlth": 3,
      "MentHlth": 2,
      "PhysHlth": 5,
      "DiffWalk": 1,
      "Sex": 1,
      "Age": 45,
      "Education": 2,
      "Income": 4
  }'
  ```

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Author
- **Starling Diaz**
- [GitHub](https://github.com/NSTLRD)
- [LinkedIn](https://www.linkedin.com/in/starling-diaz-908225181/)

