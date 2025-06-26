import os
import pandas as pd
from preprocessing import FeatureEngineering
from processing import MLModeler

def prepare_and_split_data():
    RAW_CSV = 'tlse_raw_data.csv'
    # Vérifie si les fichiers existent déjà pour éviter les re-splits inutiles
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

    # --- FILTRAGE DES LIGNES ---
    df = df[
        (df["nature_mutation"] == "Vente") &
        (df["type_local"].notna()) &
        (df["type_local"].isin(["Appartement", "Maison"])) &
        (df["valeur_fonciere"].notna()) &
        (df["latitude"].notna())
    ].copy()
    print(f"[main] Après filtrage ventes appart/maison + valeurs non manquantes : {len(df)}")

    # --- CONVERSION DES TYPES ---
    df["date_mutation"] = pd.to_datetime(df["date_mutation"], format="%Y-%m-%d", errors="coerce")
    for col in df.columns:
        if col not in ["type_local", "section_prefixe"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Float64")
    df["section_prefixe"] = df["section_prefixe"].astype(str).str[:3]
    df["section_prefixe"] = pd.to_numeric(df["section_prefixe"], errors="coerce").astype("Int32")

    # --- SUPPRESSION DES VALEURS ABERRANTES PAR SECTION ---
    def filtrer_groupe(groupe):
        q_low = groupe["valeur_fonciere"].quantile(0.1)
        q_high = groupe["valeur_fonciere"].quantile(0.9)
        return groupe[
            (groupe["valeur_fonciere"] >= q_low) & (groupe["valeur_fonciere"] <= q_high) &
            (groupe["surface_reelle_bati"] > 9)
        ]
    df = df.groupby("section_prefixe", group_keys=False).apply(filtrer_groupe)
    print(f"[main] Après suppression des valeurs extrêmes : {len(df)}")

    # --- SÉLECTION DES COLONNES FINALES ---
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

    # --- SPLIT TRAIN / VAL / TEST ---
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

    # --- AJOUT DES PRIX AU M² MÉDIANS PAR QUARTIER ---
    prix_m2_train = y_train / X_train["surface_reelle_bati"]
    medianes_par_quartier = prix_m2_train.groupby(X_train["section_prefixe"]).median().round(2)
    X_train["prix_m2_quartier"] = X_train["section_prefixe"].map(medianes_par_quartier)
    X_val["prix_m2_quartier"] = X_val["section_prefixe"].map(medianes_par_quartier)
    X_test["prix_m2_quartier"] = X_test["section_prefixe"].map(medianes_par_quartier)

    # --- SAUVEGARDE DES SPLITS ---
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
    print("\n========= PIPELINE ML MULTI-ALGO =========\n")

    # 0. Préprocessing initial et split si besoin
    prepare_and_split_data()

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
