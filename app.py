from flask import Flask, request, jsonify
from flask_cors import CORS 
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)
model = joblib.load('health_score_model.pkl')

doctor_conditions = {'Diabetes': 2, 'Cancer': 1, 'Obesity': 3, 'Arthritis': 0}
medications = {'Ibuprofen': 0, 'Paracetamol': 1, 'Lipitor': 2, 'Aspirin': 3, 'Penicillin': 4}
test_results = {'Normal': 0, 'Inconclusive': 1, 'Abnormal': 2}

# ✅ Health check route (for browser or Render)
@app.route("/", methods=["GET"])
def home():
    return "✅ Health Score Predictor API is running!"

# ✅ Prediction route (for frontend)
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_df = pd.DataFrame([{
        'Age': data['age'],
        'Medical Condition_Encoded': doctor_conditions.get(data['condition'], 0),
        'Medication_Encoded': medications.get(data['medication'], 0),
        'Test Results_Encoded': test_results.get(data['test_result'], 0),
    }])
    prediction = model.predict(input_df)[0]
    return jsonify({'health_score': round(prediction, 2)})

if __name__ == '__main__':
    app.run(debug=True)
