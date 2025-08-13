import os
import math
import numpy as np
import matplotlib.pyplot as plt


def ensure_plots_dir():
    plots_dir = os.path.join(os.path.dirname(__file__), 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    return plots_dir


class GRUFromScratch:
    """
    Minimal GRU from scratch using NumPy for educational purposes.
    Supports many-to-one next-step regression with MSE loss.
    Shapes:
      - inputs: list of length T, each (input_size, 1)
      - targets: list of length T, each (output_size, 1) or single (output_size, 1) for next-step
    """

    def __init__(self, input_size, hidden_size, output_size, seed=42):
        rng = np.random.RandomState(seed)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Parameters
        # Gates: z, r and candidate h~
        self.W_z = rng.randn(hidden_size, input_size) * 0.1
        self.U_z = rng.randn(hidden_size, hidden_size) * 0.1
        self.b_z = np.zeros((hidden_size, 1))

        self.W_r = rng.randn(hidden_size, input_size) * 0.1
        self.U_r = rng.randn(hidden_size, hidden_size) * 0.1
        self.b_r = np.zeros((hidden_size, 1))

        self.W_h = rng.randn(hidden_size, input_size) * 0.1
        self.U_h = rng.randn(hidden_size, hidden_size) * 0.1
        self.b_h = np.zeros((hidden_size, 1))

        # Output layer
        self.W_y = rng.randn(output_size, hidden_size) * 0.1
        self.b_y = np.zeros((output_size, 1))

        # Optimizer state (Adam)
        self.m = {k: np.zeros_like(v) for k, v in self.parameters().items()}
        self.v = {k: np.zeros_like(v) for k, v in self.parameters().items()}
        self.t = 0

    def parameters(self):
        return {
            'W_z': self.W_z, 'U_z': self.U_z, 'b_z': self.b_z,
            'W_r': self.W_r, 'U_r': self.U_r, 'b_r': self.b_r,
            'W_h': self.W_h, 'U_h': self.U_h, 'b_h': self.b_h,
            'W_y': self.W_y, 'b_y': self.b_y,
        }

    @staticmethod
    def sigmoid(x):
        x_clip = np.clip(x, -50, 50)
        return 1.0 / (1.0 + np.exp(-x_clip))

    def forward(self, inputs, h0=None):
        T = len(inputs)
        if h0 is None:
            h_prev = np.zeros((self.hidden_size, 1))
        else:
            h_prev = h0

        cache = {
            'x': [], 'z': [], 'r': [], 'h_tilde': [], 'h': [h_prev]
        }
        y = None
        for t in range(T):
            x_t = inputs[t]
            z_t = self.sigmoid(self.W_z @ x_t + self.U_z @ h_prev + self.b_z)
            r_t = self.sigmoid(self.W_r @ x_t + self.U_r @ h_prev + self.b_r)
            h_tilde_t = np.tanh(self.W_h @ x_t + self.U_h @ (r_t * h_prev) + self.b_h)
            h_t = (1 - z_t) * h_prev + z_t * h_tilde_t
            y = self.W_y @ h_t + self.b_y

            cache['x'].append(x_t)
            cache['z'].append(z_t)
            cache['r'].append(r_t)
            cache['h_tilde'].append(h_tilde_t)
            cache['h'].append(h_t)
            h_prev = h_t

        return y, cache

    def compute_loss(self, y_pred, y_true):
        diff = y_pred - y_true
        return 0.5 * np.sum(diff * diff)

    def backward(self, y_grad, cache):
        grads = {k: np.zeros_like(v) for k, v in self.parameters().items()}
        dh_next = self.W_y.T @ y_grad

        # Last timestep only (many-to-one)
        h_t = cache['h'][-1]
        h_prev = cache['h'][-2]
        x_t = cache['x'][-1]
        z_t = cache['z'][-1]
        r_t = cache['r'][-1]
        h_tilde_t = cache['h_tilde'][-1]

        # Output layer grads
        grads['W_y'] += y_grad @ h_t.T
        grads['b_y'] += y_grad

        # h_t = (1 - z_t) * h_prev + z_t * h_tilde_t
        dh = dh_next
        dz = dh * (h_tilde_t - h_prev)
        dh_tilde = dh * z_t
        dh_prev = dh * (1 - z_t)

        # h_tilde = tanh( ... )
        dh_tilde_pre = (1 - h_tilde_t ** 2) * dh_tilde
        grads['W_h'] += dh_tilde_pre @ x_t.T
        grads['U_h'] += dh_tilde_pre @ (r_t * h_prev).T
        grads['b_h'] += dh_tilde_pre
        dr = (self.U_h.T @ dh_tilde_pre) * h_prev

        # z gate
        dz_pre = z_t * (1 - z_t) * dz
        grads['W_z'] += dz_pre @ x_t.T
        grads['U_z'] += dz_pre @ h_prev.T
        grads['b_z'] += dz_pre
        dh_prev += self.U_z.T @ dz_pre

        # r gate
        dr_pre = r_t * (1 - r_t) * dr
        grads['W_r'] += dr_pre @ x_t.T
        grads['U_r'] += dr_pre @ h_prev.T
        grads['b_r'] += dr_pre
        dh_prev += self.U_r.T @ dr_pre

        # Add through candidate path
        dh_prev += (self.U_h.T @ dh_tilde_pre) * r_t

        return grads

    def adam_step(self, grads, lr=1e-2, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t += 1
        for k in self.parameters().keys():
            self.m[k] = beta1 * self.m[k] + (1 - beta1) * grads[k]
            self.v[k] = beta2 * self.v[k] + (1 - beta2) * (grads[k] ** 2)
            m_hat = self.m[k] / (1 - beta1 ** self.t)
            v_hat = self.v[k] / (1 - beta2 ** self.t)
            self.parameters()[k] -= lr * m_hat / (np.sqrt(v_hat) + eps)

    def train_step(self, inputs, target, lr=1e-2, clip=1.0):
        y_pred, cache = self.forward(inputs)
        loss = self.compute_loss(y_pred, target)
        y_grad = (y_pred - target)
        grads = self.backward(y_grad, cache)

        # Clip gradients
        for k in grads:
            np.clip(grads[k], -clip, clip, out=grads[k])

        self.adam_step(grads, lr=lr)
        return float(loss)


def make_sine_sequences(n_samples=400, seq_len=30, input_size=1, output_size=1, seed=0):
    rng = np.random.RandomState(seed)
    X, Y = [], []
    for _ in range(n_samples):
        phase = rng.rand() * 2 * math.pi
        x = np.linspace(0, 2 * math.pi, seq_len + 1) + phase
        s = np.sin(x) + 0.05 * rng.randn(seq_len + 1)
        seq = [s[t].reshape(input_size, 1) for t in range(seq_len)]
        target = s[seq_len].reshape(output_size, 1)
        X.append(seq)
        Y.append(target)
    return X, Y


def plot_training(losses, save_path):
    plt.figure(figsize=(6, 4))
    plt.plot(losses, label='Train Loss')
    plt.xlabel('Iteration')
    plt.ylabel('MSE Loss')
    plt.title('GRU (NumPy) Training Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


def main():
    print("Starting GRU (NumPy) demo...")
    plots_dir = ensure_plots_dir()

    # Data
    X, Y = make_sine_sequences(n_samples=300, seq_len=30)
    model = GRUFromScratch(input_size=1, hidden_size=32, output_size=1, seed=123)

    losses = []
    for i in range(500):
        idx = i % len(X)
        loss = model.train_step(X[idx], Y[idx], lr=5e-3, clip=1.0)
        losses.append(loss)
        if (i + 1) % 100 == 0:
            print(f"Iter {i+1:4d} - Loss: {loss:.6f}")

    plot_training(losses, os.path.join(plots_dir, 'gru_numpy_training_curve.png'))
    print("Saved:", os.path.join(plots_dir, 'gru_numpy_training_curve.png'))


if __name__ == '__main__':
    main()

