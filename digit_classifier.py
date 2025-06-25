import numpy as np

# Activation Functions
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    exp = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)

# Loss Functions
def cross_entropy_loss(y_pred, y_true):
    samples = y_pred.shape[0]
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    correct_conf = y_pred[range(samples), y_true]
    return -np.mean(np.log(correct_conf))

def cross_entropy_grad(y_pred, y_true):
    samples = y_pred.shape[0]
    grad = y_pred.copy()
    grad[range(samples), y_true] -= 1
    return grad / samples

# Data loader
def load_mnist_csv(train_file="mnist_train.csv", test_file="mnist_test.csv", train_samples=10000):
    train = np.loadtxt(train_file, delimiter=',', dtype=np.uint8)
    test = np.loadtxt(test_file, delimiter=',', dtype=np.uint8)

    X_train, y_train = train[:, 1:], train[:, 0]
    X_test, y_test = test[:, 1:], test[:, 0]

    # Normalize pixel values
    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0

    # Limit training samples
    X_train = X_train[:train_samples]
    y_train = y_train[:train_samples]

    return X_train, y_train, X_test, y_test

# Neural Network
class Layer_Node:
    def __init__(self, input_size, hidden_size, output_size, lr=0.1):
        self.lr = lr
        self.w1 = np.random.randn(input_size, hidden_size) * np.sqrt(1. / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.w2 = np.random.randn(hidden_size, output_size) * np.sqrt(1. / hidden_size)
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        self.z1 = X @ self.w1 + self.b1
        self.a1 = relu(self.z1)
        self.z2 = self.a1 @ self.w2 + self.b2
        self.a2 = softmax(self.z2)
        return self.a2

    def backward(self, X, y):
        grad_z2 = cross_entropy_grad(self.a2, y)
        grad_w2 = self.a1.T @ grad_z2
        grad_b2 = np.sum(grad_z2, axis=0, keepdims=True)

        grad_a1 = grad_z2 @ self.w2.T
        grad_z1 = grad_a1 * relu_derivative(self.z1)
        grad_w1 = X.T @ grad_z1
        grad_b1 = np.sum(grad_z1, axis=0, keepdims=True)

        self.w2 -= self.lr * grad_w2
        self.b2 -= self.lr * grad_b2
        self.w1 -= self.lr * grad_w1
        self.b1 -= self.lr * grad_b1

    def train(self, X, y, epochs=10, batch_size=64):
        for epoch in range(epochs):
            indices = np.random.permutation(len(X))
            X, y = X[indices], y[indices]
            for i in range(0, len(X), batch_size):
                xb = X[i:i+batch_size]
                yb = y[i:i+batch_size]
                self.forward(xb)
                self.backward(xb, yb)
            preds = self.predict(X)
            loss = cross_entropy_loss(self.a2, y)
            acc = np.mean(preds == y)
            print(f"Epoch {epoch+1}: Loss = {loss:.4f}, Accuracy = {acc:.4f}")

    def predict(self, X):
        self.forward(X)
        return np.argmax(self.a2, axis=1)

# Run
if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_mnist_csv()
    model = Layer_Node(input_size=784, hidden_size=64, output_size=10, lr=0.1)
    model.train(X_train, y_train, epochs=5)
    preds = model.predict(X_test)
    acc = np.mean(preds == y_test)
    print("Final Test Accuracy:", acc)

