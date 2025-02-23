import torch
from lstm import TrafficLSTM

def predict_single_input(model, input_data, device):
    """Make a single prediction using the trained model."""
    model.eval()
    with torch.no_grad():
        # Ensure input is a tensor with correct shape [1, 1, input_size]
        if not isinstance(input_data, torch.Tensor):
            input_data = torch.tensor(input_data, dtype=torch.float32)
        input_data = input_data.view(1, 1, -1).to(device)
        
        # Get predictions
        traffic_situation, total_traffic = model(input_data)
        
        # Process predictions
        traffic_situation = torch.softmax(traffic_situation, dim=1)
        situation_prob = traffic_situation.squeeze().cpu().numpy()
        total_traffic = total_traffic.squeeze().cpu().numpy()
        
        return situation_prob, total_traffic

def main():
    # Set device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Model configuration
    input_size = 14  # Adjust based on your feature count
    hidden_size = 256
    num_layers = 4
    num_classes = 4
    
    # Initialize and load the model
    model = TrafficLSTM(input_size, hidden_size, num_layers, num_classes)
    model.load_state_dict(torch.load('traffic_lstm_model.pth'))
    model = model.to(device)
    
    # Example input (replace with your actual input)
    example_input = [0.5] * input_size  # Replace with your actual input features
    
    # Get prediction
    situation_prob, total_traffic = predict_single_input(model, example_input, device)
    
    # Convert to human-readable output
    situation_labels = ["Free Flow", "Moderate", "Heavy", "Stop and Go"]
    predicted_situation = situation_labels[situation_prob.argmax()]
    
    # Print results
    print(f"\nPredicted Traffic Situation: {predicted_situation}")
    print(f"Situation Probabilities: {situation_prob}")
    print(f"Predicted Total Traffic: {total_traffic}")

if __name__ == "__main__":
    main() 