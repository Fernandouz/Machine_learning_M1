import pandas as pd
from preprocessing import FeatureEngineering
from processing import MLModeler

def main():
    print("\n========= PIPELINE ML MULTI-ALGO =========\n")

    # 1. Chargement des datasets
    print("[main] Chargement des splits X_train, y_train, X_val, y_val...")
    X_train = pd.read_csv('data_train/X_train.csv', low_memory=False)
    y_train = pd.read_csv('data_train/y_train.csv', low_memory=False).squeeze()
    X_val = pd.read_csv('data_train/X_val.csv', low_memory=False)
    y_val = pd.read_csv('data_train/y_val.csv', low_memory=False).squeeze()

    print(f"[main] X_train : {X_train.shape} | y_train : {y_train.shape}")
    print(f"[main] X_val   : {X_val.shape} | y_val   : {y_val.shape}")

    # 2. Feature engineering (fit sur le train, transform sur le val)
    print("[main] Feature engineering (fit_transform sur X_train, transform sur X_val)...")
    fe = FeatureEngineering()
    X_train_fe = fe.fit_transform(X_train)
    X_val_fe = fe.transform(X_val)
    print(f"[main] X_train_fe : {X_train_fe.shape} | X_val_fe : {X_val_fe.shape}")

    # 3. Liste des modèles à entraîner
    algos = [
        ("linreg", "models/linreg_ML.joblib"),
        ("rf",     "models/random_forest_ML.joblib"),
        ("xgb",    "models/xgb_ML.joblib"),
    ]

    # 4. Boucle sur chaque algo
    for algo, path in algos:
        print(f"\n[main] ====== Entraînement du modèle {algo} ======")
        model = MLModeler(model_type=algo, model_path=path)
        model.fit(X_train_fe, y_train)
        model.evaluate(X_val_fe, y_val)  # <--- save le scatter
        model.plot_learning_curve(X_train_fe, y_train)  # <--- save la learning curve
        model.save()
        print(f"[main] Modèle {algo} sauvegardé sous {path}\n")

    print("\n========= PIPELINE MULTI-ALGO TERMINÉ =========\n")

if __name__ == "__main__":
    main()
