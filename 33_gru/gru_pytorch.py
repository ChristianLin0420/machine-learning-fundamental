import os
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


def ensure_plots_dir():
    plots_dir = os.path.join(os.path.dirname(__file__), 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    return plots_dir


def make_sine_sequences(n_samples=500, seq_len=30, noise=0.05, seed=0):
    rng = np.random.RandomState(seed)
    X, Y = [], []
    for _ in range(n_samples):
        phase = rng.rand() * 2 * np.pi
        x = np.linspace(0, 2 * np.pi, seq_len + 1) + phase
        s = np.sin(x) + noise * rng.randn(seq_len + 1)
        seq = s[:seq_len].reshape(seq_len, 1)
        target = s[1:].reshape(seq_len, 1)  # next-step targets
        X.append(seq.astype(np.float32))
        Y.append(target.astype(np.float32))
    return np.stack(X, axis=0), np.stack(Y, axis=0)


class SequenceDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X)  # (N, T, 1)
        self.Y = torch.from_numpy(Y)  # (N, T, 1)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class GRUModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1, output_size=1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (B, T, F)
        out, h = self.gru(x)  # out: (B, T, H)
        y = self.head(out)    # y: (B, T, O)
        return y


def train_model(device='cpu', epochs=20, batch_size=64):
    X, Y = make_sine_sequences(n_samples=600, seq_len=30)
    X_train, Y_train = X[:500], Y[:500]
    X_val, Y_val = X[500:], Y[500:]

    train_ds = SequenceDataset(X_train, Y_train)
    val_ds = SequenceDataset(X_val, Y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = GRUModel().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    train_losses, val_losses = [], []
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            running += loss.item()
        train_losses.append(running / len(train_loader))

        model.eval()
        val_running = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_running += loss.item()
        val_losses.append(val_running / len(val_loader))

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d} | Train {train_losses[-1]:.4f} | Val {val_losses[-1]:.4f}")

    return model, train_losses, val_losses


def plot_curves(train_losses, val_losses, save_path):
    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('GRU (PyTorch) Training Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    plots_dir = ensure_plots_dir()
    model, train_losses, val_losses = train_model(device=device)
    save_path = os.path.join(plots_dir, 'gru_pytorch_training_curve.png')
    plot_curves(train_losses, val_losses, save_path)
    print("Saved:", save_path)


if __name__ == '__main__':
    main()

