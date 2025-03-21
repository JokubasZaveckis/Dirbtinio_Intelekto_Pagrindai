import numpy as np
import time

def sigmoid(z):
    # Sigmoidinė aktyvacijos funkcija: 1 / (1 + e^(-z))
    return 1.0 / (1.0 + np.exp(-z))

def forward(X, w, b):

    # Vieno neurono skaičiavimas:
    # z = X·w + b
    # output = sigmoid(z)

    return sigmoid(np.dot(X, w) + b)

def mse_loss(y_pred, y_true):

    # Vidutinio kvadrato paklaida:
    # mean((y_pred - y_true)^2)

    return np.mean((y_pred - y_true) ** 2)

def compute_accuracy(y_pred, y_true):

    # Apskaičiuoja klasifikavimo tikslumą (accuracy).
    # Jei y_pred >= 0.5, priskiriame klasę 1, kitaip 0.

    preds = (y_pred >= 0.5).astype(int)
    return np.mean(preds == y_true)

def train_batch_gd(X_train, y_train, X_val, y_val, lr=0.01, epochs=100):

    # Vieno neurono mokymas naudojant paketinį gradientinį nusileidimą.
    # Grąžina išmoktas reikšmes (w, b), epochų eigoje gautas paklaidas (train, val),
    # tikslumus (train, val) ir mokymo laiką sekundėmis.

    n_features = X_train.shape[1]

    # Inicijuoja svorius ir poslinkį nedidelėmis atsitiktinėmis reikšmėmis
    w = np.random.randn(n_features) * 0.01
    b = 0.0

    train_errors, val_errors = [], []
    train_accs, val_accs = [], []

    start_time = time.time()

    for epoch in range(epochs):
        # Perskaičiuoja išeities reikšmes mokymo ir validavimo aibėms
        y_pred_train = forward(X_train, w, b)
        y_pred_val   = forward(X_val,   w, b)

        # Apskaičiuoja paklaidas
        train_err = mse_loss(y_pred_train, y_train)
        val_err   = mse_loss(y_pred_val,   y_val)

        # Apskaičiuoja tikslumą
        train_acc = compute_accuracy(y_pred_train, y_train)
        val_acc   = compute_accuracy(y_pred_val,   y_val)

        train_errors.append(train_err)
        val_errors.append(val_err)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Apskaičiuoja gradientus
        diff = (y_pred_train - y_train)
        d_sig = y_pred_train * (1 - y_pred_train)  # išvestinė sigmoid(x)
        grad = diff * d_sig

        dw = np.dot(X_train.T, grad) * (2.0 / len(y_train))
        db = np.sum(grad) * (2.0 / len(y_train))

        w -= lr * dw
        b -= lr * db

    end_time = time.time()
    train_time = end_time - start_time

    return w, b, train_errors, val_errors, train_accs, val_accs, train_time

def train_stochastic_gd(X_train, y_train, X_val, y_val, lr=0.01, epochs=100):

    # Vieno neurono mokymas naudojant stochastinį gradientinį nusileidimą.
    # Grąžina išmoktas reikšmes (w, b), epochų eigoje gautas paklaidas (train, val),
    # tikslumus (train, val) ir mokymo laiką sekundėmis.

    n_features = X_train.shape[1]
    w = np.random.randn(n_features) * 0.01
    b = 0.0

    train_errors, val_errors = [], []
    train_accs, val_accs = [], []

    start_time = time.time()

    for epoch in range(epochs):
        # Sumaišo duomenis kiekvienoje epochoje
        permutation = np.random.permutation(len(y_train))
        X_train_shuffled = X_train[permutation]
        y_train_shuffled = y_train[permutation]

        # Eina per kiekvieną įrašą stochastiškai
        for i in range(len(y_train_shuffled)):
            x_i = X_train_shuffled[i]
            y_i = y_train_shuffled[i]

            # Perskaičiuoja neurono išeitį konkrečiam x_i
            z_i = np.dot(x_i, w) + b
            y_pred_i = sigmoid(z_i)

            diff = (y_pred_i - y_i)
            d_sig = y_pred_i * (1 - y_pred_i)
            grad = diff * d_sig

            dw = 2.0 * grad * x_i
            db = 2.0 * grad

            w -= lr * dw
            b -= lr * db

        # Apskaičiuoja paklaidas/tikslumus visam train ir val
        y_pred_train = forward(X_train, w, b)
        y_pred_val   = forward(X_val, w, b)

        train_err = mse_loss(y_pred_train, y_train)
        val_err   = mse_loss(y_pred_val, y_val)
        train_acc = compute_accuracy(y_pred_train, y_train)
        val_acc   = compute_accuracy(y_pred_val,   y_val)

        train_errors.append(train_err)
        val_errors.append(val_err)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

    end_time = time.time()
    train_time = end_time - start_time

    return w, b, train_errors, val_errors, train_accs, val_accs, train_time
