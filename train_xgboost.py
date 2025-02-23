import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

def prepare_data(df):
    # Create proper datetime by assuming data spans two consecutive months
    base_date = pd.Timestamp('2023-10-01')  # Starting date assumption
    df['datetime'] = base_date + pd.to_timedelta(df['Date'] - 1, unit='D')
    
    # Extract time components from the Time column
    df['time'] = pd.to_datetime(df['Time'], format='%I:%M:%S %p').dt.time
    df['datetime'] = pd.to_datetime(
        df['datetime'].dt.strftime('%Y-%m-%d') + ' ' + df['time'].astype(str)
    )
    
    # Extract time-based features
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    
    # Encode day of week
    le = LabelEncoder()
    df['day_of_week_encoded'] = le.fit_transform(df['Day of the week'])
    
    # Create cyclical time features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
    
    # Select features for training
    features = ['hour_sin', 'hour_cos', 'day_of_week_encoded', 
               'CarCount', 'BikeCount', 'BusCount', 'TruckCount']
    
    return df, features, le

def main():
    # Read the data
    df = pd.read_csv('TrafficTwoMonth.csv')
    
    # Prepare the data
    df, features, le = prepare_data(df)
    
    # Split features and target
    X = df[features]
    y = df['Total']
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Initialize and train the model
    model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        eval_metric='rmse'
    )
    
    print("Training XGBoost model...")
    model.fit(
        X_train, 
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=True
    )
    
    # Calculate and print test score
    test_score = model.score(X_test, y_test)
    print(f"\nTest R² Score: {test_score:.4f}")
    
    # Save the model
    print("\nSaving model...")
    with open('traffic_xgboost_model.pkl', 'wb') as f:
        pickle.dump({
            'model': model,
            'features': features,
            'label_encoder': pickle.dumps(le)
        }, f)
    print("Model saved as 'traffic_xgboost_model.pkl'")

if __name__ == "__main__":
    main() 