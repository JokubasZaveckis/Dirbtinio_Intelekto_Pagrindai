import numpy as np
import pandas as pd

def load_data(data_path: str, shuffle_seed: int = 42):

    # Grąžina X_train, y_train, X_val, y_val, X_test, y_test
    # iš nurodyto duomenų rinkinio kelio.


    # Apibrėžia stulpelių pavadinimus
    cols = [
        "ID",
        "Clump_Thickness",
        "Uniformity_Cell_Size",
        "Uniformity_Cell_Shape",
        "Marginal_Adhesion",
        "Single_Epithelial_Cell_Size",
        "Bare_Nuclei",
        "Bland_Chromatin",
        "Normal_Nucleoli",
        "Mitoses",
        "Class",
    ]

    # Nuskaitymas
    df = pd.read_csv(data_path, header=None, names=cols)

    # Išmeta eilutes su trūkstamomis reikšmėmis
    df = df[df["Bare_Nuclei"] != "?"].copy()
    df["Bare_Nuclei"] = pd.to_numeric(df["Bare_Nuclei"])

    # Išmeta ID stulpelį
    df.drop("ID", axis=1, inplace=True)

    # Pakeičia klases: 2 -> 0, 4 -> 1
    df["Class"] = df["Class"].map({2: 0, 4: 1})

    # Sumaišo eilutes atsitiktine tvarka
    df = df.sample(frac=1.0, random_state=shuffle_seed).reset_index(drop=True)

    # Paruošia X ir y masyvus
    X = df.drop("Class", axis=1).values.astype(float)
    y = df["Class"].values.astype(int)

    # Padalina į mokymo, validavimo ir testavimo aibes santykiu 80/10/10
    n_samples = X.shape[0]
    train_size = int(0.8 * n_samples)
    val_size = int(0.1 * n_samples)

    X_train = X[:train_size]
    y_train = y[:train_size]

    X_val = X[train_size : train_size + val_size]
    y_val = y[train_size : train_size + val_size]

    X_test = X[train_size + val_size :]
    y_test = y[train_size + val_size :]

    print(f"Duomenų eilučių skaičius: {df.shape[0]}")

    return X_train, y_train, X_val, y_val, X_test, y_test
