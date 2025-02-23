import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Import the model architecture
from lstm import TrafficLSTM

# --- Data Preprocessing ---
def prepare_data(data_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(data_path)
    
    # Convert `Traffic Situation` to categorical data type
    df['Traffic Situation'] = df['Traffic Situation'].astype('category')
    
    # Create numerical mapping for traffic situations
    new_category_mapping = {category: i for i, category in enumerate(df['Traffic Situation'].cat.categories)}
    df['Traffic Situation Num'] = df['Traffic Situation'].map(new_category_mapping)
    
    # One-hot encode categorical features
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoded_features = encoder.fit_transform(df[['Day of the week']])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(['Day of the week']))
    
    # Scale numerical features
    scaler = MinMaxScaler()
    numerical_features = ['CarCount', 'BikeCount', 'BusCount', 'TruckCount', 'Total']
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    
    # Combine features
    X = pd.concat([df[['Time', 'Date'] + numerical_features], encoded_df], axis=1)
    
    # Convert Time to seconds since midnight
    X['Time'] = X['Time'].apply(convert_time_to_seconds)
    
    return X, df['Traffic Situation Num'], df['Total']

def convert_time_to_seconds(time_str):
    time_part, meridiem = time_str.split(' ')
    hours, minutes, seconds = map(int, time_part.split(':'))
    
    if meridiem == 'PM' and hours != 12:
        hours += 12
    elif meridiem == 'AM' and hours == 12:
        hours = 0
        
    return hours * 3600 + minutes * 60 + seconds

# --- Create PyTorch Dataset ---
class TrafficDataset(Dataset):
    def __init__(self, X, y_traffic_situation, y_total):
        self.X = torch.tensor(X.values, dtype=torch.float32).unsqueeze(1)
        self.y_traffic_situation = torch.tensor(y_traffic_situation.values, dtype=torch.long)
        self.y_total = torch.tensor(y_total.values, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_traffic_situation[idx], self.y_total[idx]

def train_model(model, train_loader, test_loader, num_epochs, device):
    # Use AdamW optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Loss functions
    criterion_classification = nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion_regression = nn.MSELoss()
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos'
    )
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        total_loss = 0
        
        for i, (inputs, labels_classification, labels_regression) in enumerate(pbar):
            inputs = inputs.to(device)
            labels_classification = labels_classification.to(device)
            labels_regression = labels_regression.to(device)

            optimizer.zero_grad()
            outputs_classification, outputs_regression = model(inputs)
            
            loss_classification = criterion_classification(outputs_classification, labels_classification)
            loss_regression = criterion_regression(outputs_regression.squeeze(), labels_regression)
            loss = loss_classification * 0.7 + loss_regression * 0.3
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{total_loss/(i+1):.4f}'})
    
    # Evaluation
    evaluate_model(model, test_loader, device)

def evaluate_model(model, test_loader, device):
    model.eval()
    criterion_regression = nn.MSELoss()
    
    with torch.no_grad():
        correct_classification = 0
        total_classification = 0
        total_loss_regression = 0
        
        pbar = tqdm(test_loader, desc='Evaluating')
        
        for inputs, labels_classification, labels_regression in pbar:
            inputs = inputs.to(device)
            labels_classification = labels_classification.to(device)
            labels_regression = labels_regression.to(device)

            outputs_classification, outputs_regression = model(inputs)
            _, predicted_classification = torch.max(outputs_classification.data, 1)
            total_classification += labels_classification.size(0)
            correct_classification += (predicted_classification == labels_classification).sum().item()
            loss_regression = criterion_regression(outputs_regression.squeeze(), labels_regression)
            total_loss_regression += loss_regression.item()
            
            current_accuracy = 100 * correct_classification / total_classification
            current_loss = total_loss_regression / (pbar.n + 1)
            pbar.set_postfix({
                'Accuracy': f'{current_accuracy:.2f}%',
                'Reg Loss': f'{current_loss:.4f}'
            })

        accuracy_classification = 100 * correct_classification / total_classification
        avg_loss_regression = total_loss_regression / len(test_loader)

        print(f"\nFinal Results:")
        print(f"Accuracy (Classification): {accuracy_classification:.2f}%")
        print(f"Average Loss (Regression): {avg_loss_regression:.4f}")

def main():
    # Configuration
    data_path = "TrafficTwoMonth.csv"
    input_size = 14  # Will be determined from data
    hidden_size = 256
    num_layers = 4
    num_classes = 4
    global learning_rate
    learning_rate = 0.001
    num_epochs = 50
    batch_size = 32
    
    # Set device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Prepare data
    print("Preparing data...")
    X, y_traffic_situation, y_total = prepare_data(data_path)
    
    # Split data
    X_train, X_test, y_traffic_situation_train, y_traffic_situation_test, y_total_train, y_total_test = \
        train_test_split(X, y_traffic_situation, y_total, test_size=0.2, random_state=42)
    
    # Create datasets and dataloaders
    train_dataset = TrafficDataset(X_train, y_traffic_situation_train, y_total_train)
    test_dataset = TrafficDataset(X_test, y_traffic_situation_test, y_total_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = TrafficLSTM(input_size, hidden_size, num_layers, num_classes)
    model = model.to(device)
    
    # Train model
    print("Starting training...")
    train_model(model, train_loader, test_loader, num_epochs, device)
    
    # Save the model
    torch.save(model.state_dict(), 'traffic_lstm_model.pth')
    print("Model saved to 'traffic_lstm_model.pth'")

if __name__ == "__main__":
    main() 