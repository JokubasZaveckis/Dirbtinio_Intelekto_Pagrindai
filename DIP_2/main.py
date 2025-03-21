import matplotlib.pyplot as plt
from data_preparation import load_data
from neuron import (
    forward,
    mse_loss,
    compute_accuracy,
    train_batch_gd,
    train_stochastic_gd
)

def run_experiment(X_train, y_train, X_val, y_val, X_test, y_test, epochs=50, lr=0.01):
    print(f"Epochos = {epochs}, mokymosi greitis lr = {lr} ...")

    # ----- Paketinis gradientinis nusileidimas -----
    (w_bgd, b_bgd,
     train_err_bgd, val_err_bgd,
     train_acc_bgd, val_acc_bgd,
     time_bgd) = train_batch_gd(X_train, y_train, X_val, y_val, lr=lr, epochs=epochs)

    y_pred_test_bgd = forward(X_test, w_bgd, b_bgd)
    test_err_bgd = mse_loss(y_pred_test_bgd, y_test)
    test_acc_bgd = compute_accuracy(y_pred_test_bgd, y_test)

    # ----- Stochastinis gradientinis nusileidimas -----
    (w_sgd, b_sgd,
     train_err_sgd, val_err_sgd,
     train_acc_sgd, val_acc_sgd,
     time_sgd) = train_stochastic_gd(X_train, y_train, X_val, y_val, lr=lr, epochs=epochs)

    y_pred_test_sgd = forward(X_test, w_sgd, b_sgd)
    test_err_sgd = mse_loss(y_pred_test_sgd, y_test)
    test_acc_sgd = compute_accuracy(y_pred_test_sgd, y_test)

    # Testavimo rezultatai BGD
    print("==== PGD TESTAVIMAS ====")
    preds_bgd = (y_pred_test_bgd >= 0.5).astype(int)
    for i, (pred, actual) in enumerate(zip(preds_bgd, y_test)):
        print(f"Testas {i}: Spėta = {pred}, Atsakymas = {actual}")

    # Testavimo rezultatai SGD
    print("==== SGN TESTAVIMAS ====")
    preds_sgd = (y_pred_test_sgd >= 0.5).astype(int)
    for i, (pred, actual) in enumerate(zip(preds_sgd, y_test)):
        print(f"Testas {i}: Spėta = {pred}, Atsakymas = {actual}")

    # Pavaizduojame paklaidos kitimą (mokymo ir validavimo) abiem metodais
    plt.figure()
    plt.plot(train_err_bgd, label="Mokymo paklaida(PGN)")
    plt.plot(val_err_bgd,   label="Vailidacijos paklaida (PGN)")
    plt.plot(train_err_sgd, label="Mokymo paklaida (SGN)")
    plt.plot(val_err_sgd,   label="Validacijos paklaida(SGN)")
    plt.xlabel("Epochos")
    plt.ylabel("MSE (Viduritnio kvadrato paklaida)")
    plt.title("Mokymo ir validacijos paklaida per epochas")
    plt.legend()
    plt.show()

    # Pavaizduojame tikslumo kitimą
    plt.figure()
    plt.plot(train_acc_bgd, label="Mokymo tikslumas (PGN)")
    plt.plot(val_acc_bgd,   label="Validacijos tikslumas (PGN)")
    plt.plot(train_acc_sgd, label="Mokymo tikslumas (SGN)")
    plt.plot(val_acc_sgd,   label="Validacijos tikslumas (SGN)")
    plt.xlabel("Epochos")
    plt.ylabel("Tikslumas")
    plt.title("Mokymo ir validacijos tikslumas per epochas")
    plt.legend()
    plt.show()

    print("==== PGN Rezultatai ====")
    print(f" Galutiniai svoriai: {w_bgd}")
    print(f" Galutinis poslinkis: {b_bgd}")
    print(f" Mokymo paklaida paskutinėje epochoje: {train_err_bgd[-1]:.4f}, mokymo tikslumas: {train_acc_bgd[-1]*100:.2f}%")
    print(f" Validacijos paklaida paskutinėje epochoje:   {val_err_bgd[-1]:.4f},   validacijos tikslumas: {val_acc_bgd[-1]*100:.2f}%")
    print(f" Testavimo paklaida: {test_err_bgd:.4f}, testavimo tikslumas: {test_acc_bgd*100:.2f}%")
    print(f" Mokymo laikas: {time_bgd:.4f} sekundės\n")

    print("==== SGN Rezultatai ====")
    print(f" Galutiniai svoriai: {w_sgd}")
    print(f" Galutinis poslinkis: {b_sgd}")
    print(f" Mokymo paklaida paskutinėje epochoje: {train_err_sgd[-1]:.4f}, mokymo tikslumas: {train_acc_sgd[-1]*100:.2f}%")
    print(f" Validacijos paklaida paskutinėje epochoje:   {val_err_sgd[-1]:.4f},   validacijos tikslumas: {val_acc_sgd[-1]*100:.2f}%")
    print(f" Testavimo paklaida: {test_err_sgd:.4f}, testavimo tikslumas: {test_acc_sgd*100:.2f}%")
    print(f" Mokymo laikas: {time_sgd:.4f} sekundės\n")


if __name__ == "__main__":
    DATA_PATH = "data/breast-cancer-wisconsin.data"
    
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(DATA_PATH)

    run_experiment(X_train, y_train, X_val, y_val, X_test, y_test,
                   epochs=50, lr=0.1)


