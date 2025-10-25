from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import os
import numpy as np
import random

app = Flask(__name__)

# Load model, encoders, and scaler
def load_artifacts():
    try:
        # Load model
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Load encoders and scaler
        with open('processed_data/encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)
        with open('processed_data/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
            
        return model, encoders, scaler
    except Exception as e:
        print(f"Error loading artifacts: {e}")
        return None, None, None

model, encoders, scaler = load_artifacts()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None or encoders is None or scaler is None:
            return jsonify({'success': False, 'error': 'Model not loaded properly'})
            
        data = request.json
        
        # Map frontend values to dataset values
        weather_map = {0: 'Sunny', 1: 'Rainy', 2: 'Snowy', 3: 'Foggy'}
        road_map = {0: 'Highway', 1: 'City', 2: 'Rural'}
        time_map = {0: 'Morning', 1: 'Afternoon', 2: 'Evening', 3: 'Night'}
        
        weather = weather_map.get(int(data.get('weather', 0)), 'Sunny')
        road_type = road_map.get(int(data.get('road_type', 0)), 'Highway')
        time_of_day = time_map.get(int(data.get('time_of_day', 0)), 'Morning')
        traffic_density = int(data.get('traffic_density', 0))
        
        # Create dynamic dummy values based on inputs with randomization
        base_visibility = 8.0
        base_accidents = 3
        base_vehicles = 100
        road_condition = 'Good'
        light_condition = 'Daylight'
        
        # Add some randomization to prevent identical results
        random_factor = random.uniform(0.8, 1.2)
        
        # Adjust values based on weather
        if weather == 'Snowy':
            visibility = 2.0 * random_factor
            past_accidents = int(8 * random_factor)
            vehicle_count = int(250 * random_factor)
            road_condition = 'Poor'
        elif weather == 'Foggy':
            visibility = 1.5 * random_factor
            past_accidents = int(7 * random_factor)
            vehicle_count = int(200 * random_factor)
            light_condition = 'Dark'
        elif weather == 'Rainy':
            visibility = 4.0 * random_factor
            past_accidents = int(5 * random_factor)
            vehicle_count = int(180 * random_factor)
            road_condition = 'Fair'
        else:
            visibility = base_visibility * random_factor
            past_accidents = int(base_accidents * random_factor)
            vehicle_count = int(base_vehicles * random_factor)
        
        # Adjust based on traffic density
        if traffic_density == 2:  # High traffic
            vehicle_count += int(150 * random_factor)
            past_accidents += int(3 * random_factor)
        elif traffic_density == 1:  # Medium traffic
            vehicle_count += int(75 * random_factor)
            past_accidents += int(1 * random_factor)
        
        # Adjust based on time of day
        if time_of_day == 'Night':
            light_condition = 'Dark'
            past_accidents += int(2 * random_factor)
        elif time_of_day == 'Evening':
            light_condition = 'Street Light'
            past_accidents += int(1 * random_factor)
        
        # Create input dataframe with all required features
        input_data = {
            'route_id': random.randint(100, 999),
            'latitude': round(random.uniform(8.0, 37.0), 4),
            'longitude': round(random.uniform(68.0, 97.0), 4),
            'weather': weather,
            'temperature_C': round(random.uniform(5.0, 45.0), 1),
            'humidity_%': round(random.uniform(30.0, 99.0), 1),
            'traffic_density': traffic_density,
            'road_type': road_type,
            'road_condition': road_condition,
            'num_lanes': 2,
            'speed_limit': 60,
            'vehicle_count': vehicle_count,
            'visibility_km': visibility,
            'time_of_day': time_of_day,
            'day_of_week': 'Mon',
            'holiday': 0,
            'past_accidents': past_accidents,
            'population_density': random.randint(100, 2000),
            'light_condition': light_condition
        }
        
        df_input = pd.DataFrame([input_data])
        
        # Apply encoders
        for col, encoder in encoders.items():
            if col in df_input.columns and col != 'risk_level':
                try:
                    df_input[col] = encoder.transform(df_input[col])
                except ValueError:
                    # Handle unseen categories
                    df_input[col] = 0
        
        # Scale numerical features
        numeric_cols = df_input.select_dtypes(include=['float64', 'int64']).columns
        df_input[numeric_cols] = scaler.transform(df_input[numeric_cols])
        
        # Make prediction
        prediction = model.predict(df_input)[0]
        prediction_proba = model.predict_proba(df_input)[0]
        
        # Debug: Check actual risk level mapping from encoders
        if 'risk_level' in encoders:
            risk_encoder = encoders['risk_level']
            risk_levels = list(risk_encoder.classes_)
        else:
            risk_levels = ['High', 'Low', 'Medium']
        
        risk_level = risk_levels[prediction]
        confidence = int(max(prediction_proba) * 100)
        

        
        # Calculate risk score for display
        risk_score = prediction_proba[0] * 2 + prediction_proba[2] * 4 + prediction_proba[1] * 6
        
        return jsonify({
            'success': True,
            'risk_level': risk_level,
            'confidence': f"{confidence}%",
            'risk_score': round(risk_score, 1)
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