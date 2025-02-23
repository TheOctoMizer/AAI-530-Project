import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

def preprocess_input(input_data, le):
    # Create DataFrame from input data
    df = pd.DataFrame([input_data])
    
    # Convert to proper datetime format
    base_date = pd.Timestamp('2023-10-01')
    df['datetime'] = base_date + pd.to_timedelta(df['Date'] - 1, unit='D')
    
    # Parse time component
    df['time'] = pd.to_datetime(df['Time'], format='%I:%M:%S %p').dt.time
    df['datetime'] = pd.to_datetime(
        df['datetime'].dt.strftime('%Y-%m-%d') + ' ' + df['time'].astype(str)
    )
    
    # Extract time features
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    
    # Encode day of week using the saved encoder
    df['day_of_week_encoded'] = le.transform(df['Day of the week'])
    
    # Create cyclical features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
    
    return df

def predict_traffic(input_data, model_path='traffic_xgboost_model.pkl'):
    # Load the saved model and components
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    features = model_data['features']
    le = pickle.loads(model_data['label_encoder'])
    
    # Preprocess the input data
    processed_df = preprocess_input(input_data, le)
    
    # Select features and predict
    X = processed_df[features]
    prediction = model.predict(X)[0]
    
    # Convert to integer (total vehicles)
    return int(round(prediction))

def get_traffic_situation(prediction):
    # Define thresholds based on your data analysis
    if prediction < 50:
        return "low"
    elif 50 <= prediction < 150:
        return "normal"
    elif 150 <= prediction < 200:
        return "high"
    else:
        return "heavy"

if __name__ == "__main__":
    # Example usage
    sample_input = {
        'Time': '8:00:00 AM',
        'Date': 10,
        'Day of the week': 'Tuesday',
        'CarCount': 134,
        'BikeCount': 18,
        'BusCount': 21,
        'TruckCount': 11
    }
    
    prediction = predict_traffic(sample_input)
    situation = get_traffic_situation(prediction)
    
    print(f"Predicted Total Vehicles: {prediction}")
    print(f"Traffic Situation: {situation}") 