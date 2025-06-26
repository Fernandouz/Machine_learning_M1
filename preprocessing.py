import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import haversine_distances

class FeatureEngineering(BaseEstimator, TransformerMixin):
    """
    Transformer scikit-learn pour enrichir un dataset immobilier :
    - Encodage one-hot de 'type_local'
    - Ajout de features géographiques (distances aux POI)
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        # Pas d'apprentissage ici
        return self

    def transform(self, X):
        X = X.copy()
        X = self.encoder_type_local(X)
        X = self.ajouter_distances_poi(X)
        return X

    def encoder_type_local(self, X):
        if "type_local" in X.columns:
            X = pd.get_dummies(X, columns=["type_local"], prefix="type_local")
        else:
            raise KeyError("La colonne 'type_local' est requise pour l'encodage.")
        return X

    def ajouter_distances_poi(self, df_clean):
        poi_dict = {
            "creche": "creches.csv",
            "maternelle": "maternelles.csv",
            "elementaire": "ecole_elementaire.csv",
            "arret": "arret_physique.csv"
        }
        for mode, path in poi_dict.items():
            df_poi = charger_csv_sans_erreur(path)
            df_poi = extraire_lat_lon(df_poi)
            df_clean = compute_distances(df_clean, df_poi, mode, rayon_m=500)

        df_arrets = charger_csv_sans_erreur("arret_physique.csv")
        df_arrets = extraire_lat_lon(df_arrets)
        for mode_transport in ["bus", "metro", "tram"]:
            if "CONC_MODE" in df_arrets.columns:
                df_arrets_mode = df_arrets[df_arrets["CONC_MODE"].str.lower() == mode_transport]
                df_clean = compute_distances(df_clean, df_arrets_mode, f"arret_{mode_transport}", rayon_m=500)
        return df_clean

# Fonctions utilitaires (hors classe)
def charger_csv_sans_erreur(path, sep=";"):
    try:
        return pd.read_csv(path, sep=sep, on_bad_lines="skip")
    except Exception as e:
        print(f"Erreur lors du chargement de {path} : {e}")
        return pd.DataFrame()

def extraire_lat_lon(df):
    df = df.copy()
    if 'latitude' in df.columns and 'longitude' in df.columns:
        df = df[df['latitude'].notna() & df['longitude'].notna()]
        df['latitude'] = df['latitude'].astype(float)
        df['longitude'] = df['longitude'].astype(float)
        return df
    for col in df.columns:
        if df[col].dtype == object and df[col].str.contains(",", regex=False).any():
            try:
                df[['latitude', 'longitude']] = df[col].str.split(",", expand=True).astype(float)
                return df
            except Exception:
                continue
    raise ValueError("❌ Aucune colonne de coordonnées valides trouvée dans ce DataFrame.")

def compute_distances(df_clean, df_poi, mode, rayon_m=1000):
    df_clean_valid = df_clean.dropna(subset=["latitude", "longitude"]).copy()
    df_poi_valid = df_poi.dropna(subset=["latitude", "longitude"]).copy()
    if len(df_poi_valid) == 0 or len(df_clean_valid) == 0:
        return df_clean
    biens_coords = np.radians(df_clean_valid[["latitude", "longitude"]].values)
    poi_coords = np.radians(df_poi_valid[["latitude", "longitude"]].values)
    distances = haversine_distances(biens_coords, poi_coords) * 6371000
    df_clean_valid[f"dist_min_{mode}_m"] = distances.min(axis=1)
    df_clean_valid[f"nb_{mode}_moins_{rayon_m}m"] = (distances < rayon_m).sum(axis=1)
    df_clean = df_clean.merge(
        df_clean_valid[[f"dist_min_{mode}_m", f"nb_{mode}_moins_{rayon_m}m"]],
        left_index=True, right_index=True, how="left"
    )
    return df_clean
