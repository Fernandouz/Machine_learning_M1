import os
import pandas as pd
import numpy as np
from preprocessing import FeatureEngineering
from processing import MLModeler
from processing_dl import DLRegressor  # Assure-toi que ta classe est dispo dans ce fichier

def prepare_and_split_data():
    RAW_CSV = 'tlse_raw_data.csv'
    if all([os.path.exists(f) for f in [
        "data_train/X_train.csv", "data_train/y_train.csv",
        "data_train/X_val.csv",   "data_train/y_val.csv",
        "data_test/X_test.csv",   "data_test/y_test.csv"
    ]]):
        print("[main] Splits déjà présents, skip preprocess initial.")
        return

    print("[main] Lecture du fichier brut :", RAW_CSV)
    df = pd.read_csv(RAW_CSV, low_memory=False)
    print(f"[main] Lignes initiales : {len(df)}")

    df = df[
        (df["nature_mutation"] == "Vente") &
        (df["type_local"].notna()) &
        (df["type_local"].isin(["Appartement", "Maison"])) &
        (df["valeur_fonciere"].notna()) &
        (df["latitude"].notna())
    ].copy()
    print(f"[main] Après filtrage ventes appart/maison + valeurs non manquantes : {len(df)}")

    df["date_mutation"] = pd.to_datetime(df["date_mutation"], format="%Y-%m-%d", errors="coerce")
    for col in df.columns:
        if col not in ["type_local", "section_prefixe"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Float64")
    df["section_prefixe"] = df["section_prefixe"].astype(str).str[:3]
    df["section_prefixe"] = pd.to_numeric(df["section_prefixe"], errors="coerce").astype("Int32")

    def filtrer_groupe(groupe):
        q_low = groupe["valeur_fonciere"].quantile(0.1)
        q_high = groupe["valeur_fonciere"].quantile(0.95)
        return groupe[
            (groupe["valeur_fonciere"] >= q_low) & (groupe["valeur_fonciere"] <= 2000000) &
            (groupe["surface_reelle_bati"] > 9)
        ]
    df = df.groupby("section_prefixe", group_keys=False).apply(filtrer_groupe)
    print(f"[main] Après suppression des valeurs extrêmes : {len(df)}")

    colonnes = [
        "numero_disposition",
        "valeur_fonciere",
        "type_local",
        "surface_reelle_bati",
        "nombre_pieces_principales",
        "longitude",
        "latitude",
        "section_prefixe",
        "nombre_lots",
    ]
    df = df[colonnes]
    print(f"[main] Colonnes finales : {df.shape[1]}")

    from sklearn.model_selection import train_test_split
    X = df.drop(columns=["valeur_fonciere"])
    y = df["valeur_fonciere"]
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, shuffle=True
    )

    print(f"[main] Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    prix_m2_train = y_train / X_train["surface_reelle_bati"]
    medianes_par_quartier = prix_m2_train.groupby(X_train["section_prefixe"]).median().round(2)
    X_train["prix_m2_quartier"] = X_train["section_prefixe"].map(medianes_par_quartier)
    X_val["prix_m2_quartier"] = X_val["section_prefixe"].map(medianes_par_quartier)
    X_test["prix_m2_quartier"] = X_test["section_prefixe"].map(medianes_par_quartier)

    os.makedirs("data_train", exist_ok=True)
    os.makedirs("data_test", exist_ok=True)
    X_train.to_csv("data_train/X_train.csv", index=False)
    y_train.to_csv("data_train/y_train.csv", index=False)
    X_val.to_csv("data_train/X_val.csv", index=False)
    y_val.to_csv("data_train/y_val.csv", index=False)
    X_test.to_csv("data_test/X_test.csv", index=False)
    y_test.to_csv("data_test/y_test.csv", index=False)
    print("[main] Données prétraitées et splits sauvegardés.")

def main():
    print("\n========= PIPELINE ML/DL MULTI-ALGO =========\n")
    prepare_and_split_data()

    # Chargement des datasets
    print("[main] Chargement des splits X_train, y_train, X_val, y_val...")
    X_train = pd.read_csv('data_train/X_train.csv', low_memory=False)
    y_train = pd.read_csv('data_train/y_train.csv', low_memory=False).squeeze()
    X_val = pd.read_csv('data_train/X_val.csv', low_memory=False)
    y_val = pd.read_csv('data_train/y_val.csv', low_memory=False).squeeze()

    print(f"[main] X_train : {X_train.shape} | y_train : {y_train.shape}")
    print(f"[main] X_val   : {X_val.shape} | y_val   : {y_val.shape}")

    # Feature engineering
    print("[main] Feature engineering (fit_transform sur X_train, transform sur X_val)...")
    fe = FeatureEngineering()
    X_train_fe = fe.fit_transform(X_train)
    X_val_fe = fe.transform(X_val)
    print(f"[main] X_train_fe : {X_train_fe.shape} | X_val_fe : {X_val_fe.shape}")

    # Sélection du mode
    print("\n[CHOIX] Veux-tu lancer :\n"
          "   1 - ML classique (sklearn, GridSearch)\n"
          "   2 - Deep Learning (DLRegressor, Keras)\n"
          "   3 - Interprétation SHAP (RandomForest / XGBoost)\n")
    mode = input("Tape 1, 2 ou 3 : ").strip()

    if mode == "1":
        print("\n[main] Lancement du pipeline **ML** classique...")
        algos = [
            ("linreg", "models/linreg_ML.joblib"),
            ("rf",     "models/random_forest_ML.joblib"),
            ("xgb",    "models/xgb_ML.joblib"),
        ]
        for algo, path in algos:
            print(f"\n[main] ====== Entraînement du modèle {algo} ======")
            model = MLModeler(model_type=algo, model_path=path)
            model.fit(X_train_fe, y_train)
            model.evaluate(X_val_fe, y_val)  # save le scatter
            model.plot_learning_curve(X_train_fe, y_train)
            model.save()
            print(f"[main] Modèle {algo} sauvegardé sous {path}\n")


    elif mode == "2":

        print("\n[main] Lancement du pipeline **Deep Learning** (DLRegressor)...")

        from processing_dl import DLRegressor  # Si pas déjà importé au top

        # Architecture profonde + log(y)

        model = DLRegressor(

            hidden_layers=(128, 64, 32, 16),  # Plus profond = plus de flexibilité

            dropout=0.5,  # Dropout modéré (0.1 à 0.2)

            epochs=1000,  # On laisse longtemps mais EarlyStopping arrête au bon moment

            batch_size=128,

            scaler_X=True,  # OBLIGÉ sur tabulaire

            scaler_y=False,  # False car log_target déjà activé

            log_target=True,  # Activation log-transform de la target

            learning_rate=0.001,

            patience=20,  # Laisse-le respirer, ne stoppe pas trop vite (10-20 epochs)

            verbose=1,

            random_state=42

        )

        model.fit(X_train_fe, y_train.values if hasattr(y_train, "values") else y_train)

        y_pred = model.predict(X_val_fe)

        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        rmse = np.sqrt(mean_squared_error(y_val, y_pred))

        mae = mean_absolute_error(y_val, y_pred)

        r2 = r2_score(y_val, y_pred)

        print("\n--- Évaluation sur validation ---")

        print(f"RMSE : {rmse:,.2f}")

        print(f"MAE  : {mae:,.2f}")

        print(f"R²   : {r2:.3f}")

        # (OPTION) Sauvegarde prédictions et vrai/y_pred pour analyse

        pd.DataFrame({"y_true": y_val, "y_pred": y_pred}).to_csv("results/pred_dl_val.csv", index=False)

        print("[main] Prédictions DL sauvegardées dans results/pred_dl_val.csv")

        # (OPTION) Plot des résidus

        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 6))

        plt.scatter(y_val, y_pred, alpha=0.3)

        plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], "r--", lw=2)

        plt.xlabel("Valeur réelle")

        plt.ylabel("Prédiction DL")

        plt.title("DLRegressor : Prédiction vs Réel")

        plt.grid()

        plt.tight_layout()

        plt.savefig("results/plots/dl_pred_vs_reality.png", dpi=100)

        plt.show()

        print("[main] Plot DL sauvegardé dans results/plots/dl_pred_vs_reality.png")

    elif mode == "3":
        print("\n[main] Interprétation SHAP pour modèles ML existants...")

        for algo, path in [("rf", "models/random_forest_ML.joblib"), ("xgb", "models/xgb_ML.joblib")]:
            if not os.path.exists(path):
                print(f"[main] ⚠️ Modèle {algo} non trouvé ({path}). Skippé.")
                continue

            print(f"\n[main] ====== Analyse SHAP du modèle {algo} ======")
            model = MLModeler(model_type=algo, model_path=path)
            model.load(path)
            # model.plot_shap_values(X_val_fe)
            model.plot_shap_summary_bar(X_val_fe)
            model.plot_shap_dependence(
                X_val_fe,
            )
            model.plot_shap_waterfall(X_val_fe, instance_index=5)


    else:
        print("[main] Choix non reconnu. Abandon. Relance et tape 1 ou 2, sois pas borné.")

    print("\n========= PIPELINE TERMINÉ =========\n")

if __name__ == "__main__":
    main()
