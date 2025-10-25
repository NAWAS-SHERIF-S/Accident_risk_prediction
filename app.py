from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import os
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np

app = Flask(__name__)

# Load or train model
def load_model():
    model_path = 'model.pkl'
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    else:
        # Train a simple model if none exists
        try:
            X_train = pd.read_csv('processed_data/X_train_processed.csv')
            y_train = pd.read_csv('processed_data/y_train_processed.csv').values.ravel()
            
            model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
            model.fit(X_train, y_train)
            
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            return model
        except Exception as e:
            print(f"Error training model: {e}")
            return None

model = load_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Simple prediction based on 4 key factors
        weather = int(data.get('weather', 0))  # 0=Clear, 1=Rain, 2=Snow, 3=Fog
        traffic_density = int(data.get('traffic_density', 0))  # 0=Low, 1=Medium, 2=High
        road_type = int(data.get('road_type', 0))  # 0=Highway, 1=Urban, 2=Rural
        time_of_day = int(data.get('time_of_day', 0))  # 0=Morning, 1=Afternoon, 2=Evening, 3=Night
        
        # Calculate individual factor contributions
        weather_score = {0: 0, 1: 1, 2: 2, 3: 1}.get(weather, 0)
        traffic_score = traffic_density
        road_score = {0: 0, 1: 1, 2: 0.5}.get(road_type, 0)
        time_score = {0: 0, 1: 0, 2: 1, 3: 1.5}.get(time_of_day, 0)
        
        
        risk_score = weather_score + traffic_score + road_score + time_score
        
        # Determine risk level
        if risk_score <= 1.5:
            risk_level = 'Low'
            confidence = 85
        elif risk_score <= 3.5:
            risk_level = 'Medium'
            confidence = 80
        else:
            risk_level = 'High'
            confidence = 90
        
        return jsonify({
            'success': True,
            'risk_level': risk_level,
            'confidence': f"{confidence}%",
            'risk_score': round(risk_score, 1),
            'factor_breakdown': {
                'Weather': round(weather_score, 1),
                'Traffic': round(traffic_score, 1),
                'Road Type': round(road_score, 1),
                'Time of Day': round(time_score, 1)
            }
        })
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/stats')
def stats():
    try:
        df = pd.read_csv('saferoads_accident_risk_dataset.csv')
        stats = {
            'total_records': len(df),
            'risk_distribution': df['risk_level'].value_counts().to_dict(),
            'weather_distribution': df['weather'].value_counts().head(5).to_dict()
        }
        return jsonify(stats)
    except:
        return jsonify({'error': 'Data not available'})

if __name__ == '__main__':
    app.run(debug=True)