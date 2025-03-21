import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

import load_data
import boids

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1) 

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        out, _ = self.lstm(x, (h0, c0)) 
        out = out[:, -1, :]              
        out = self.fc(out)               
        return out

def create_sequences(df, feature_cols, target_col, seq_length=10):
    data = df[feature_cols].values 
    target = df[target_col].values 

    X, y = [], []
    for i in range(len(df) - seq_length):
        seq_x = data[i : i + seq_length]
        seq_y = target[i + seq_length]
        X.append(seq_x)
        y.append(seq_y)
    X = np.array(X)
    y = np.array(y)
    return X, y

def train_test_split(X, y, train_size=0.8):
    split_idx = int(len(X) * train_size)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    return X_train, y_train, X_test, y_test


def train_model(model, X_train, y_train, X_test, y_test, epochs, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    X_train_t = torch.from_numpy(X_train).float().to(device)
    y_train_t = torch.from_numpy(y_train).float().to(device)
    X_test_t = torch.from_numpy(X_test).float().to(device)
    y_test_t = torch.from_numpy(y_test).float().to(device)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs.squeeze(), y_train_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test_t)
            val_loss = criterion(val_outputs.squeeze(), y_test_t)
        
        if (epoch+1) % 1 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}, Val Loss: {val_loss.item():.6f}")

def predict_future(model, X_last, n_future=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    preds = []
    current_seq = torch.from_numpy(X_last).float().unsqueeze(0).to(device)

    with torch.no_grad():
        for _ in range(n_future):
            out = model(current_seq)
            pred = out.item()
            preds.append(pred)

            new_step = torch.tensor(pred).float().to(device)
            
            current_seq = current_seq.squeeze(0)
            current_seq = current_seq[1:]
           
            new_row = current_seq[-1].clone()
            new_row[0] = new_step
            current_seq = torch.cat((current_seq, new_row.unsqueeze(0)), dim=0)
            
            current_seq = current_seq.unsqueeze(0)

    return preds

if __name__ == "__main__":
    df = load_data.load_and_preprocess(symbol="BTCUSDT", interval="1d", limit=10000, ma_window=5)
    print("Data loaded:", df.shape)

    boids_df = boids.generate_boids_features(num_days=len(df),
                                             num_boids=50,
                                             width=200,
                                             height=200,
                                             max_speed=5.0,
                                             perception_radius=50.0)
    print("Boids features:", boids_df.shape)

    df_boids = pd.concat([df.reset_index(drop=True), boids_df.reset_index(drop=True)], axis=1)

    feature_cols = ["close", "ma_close",
                    "boids_mean_x", "boids_mean_y", "boids_mean_vx", "boids_mean_vy",
                    "boids_std_x", "boids_std_y", "boids_std_vx", "boids_std_vy"]
    target_col = "future_close"

    df_boids.dropna(subset=feature_cols + [target_col], inplace=True)

    min_vals = df_boids[feature_cols].min()
    max_vals = df_boids[feature_cols].max()

    def normalize(col, val):
        return (val - min_vals[col]) / (max_vals[col] - min_vals[col] + 1e-9)

    for c in feature_cols:
        df_boids[c] = df_boids[c].apply(lambda x: normalize(c, x))

    target_min = df_boids[target_col].min()
    target_max = df_boids[target_col].max()
    df_boids[target_col] = (df_boids[target_col] - target_min) / (target_max - target_min + 1e-9)
    
    seq_length = 10
    X, y = create_sequences(df_boids, feature_cols, target_col, seq_length=seq_length)

    X_train, y_train, X_test, y_test = train_test_split(X, y, train_size=0.8)

    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_test:", X_test.shape, "y_test:", y_test.shape)

    input_dim = len(feature_cols)
    hidden_dim = 32
    num_layers = 1
    model = LSTMModel(input_dim, hidden_dim, num_layers)

    train_model(model, X_train, y_train, X_test, y_test, epochs=100, lr=1e-3)

    X_last = X_test[-1]
    future_preds_norm = predict_future(model, X_last, n_future=5)
    future_preds = [(p * (target_max - target_min + 1e-9)) + target_min for p in future_preds_norm]

    print("Normalized future preds:", future_preds_norm)
    print("Denormalized future preds:", future_preds)
