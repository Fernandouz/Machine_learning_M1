import pandas as pd
import numpy as np

class MutationPreprocessor:
    """
    Pipeline de nettoyage 'ML Ready' pour données de mutation immobilière.
    Suit les bonnes pratiques du document fourni (light cleaning, split, deep cleaning, etc.).
    """

    def __init__(self):
        # Stockage des bornes pour prix et valeur foncière (IQR)
        self.iqr_bounds = {}

    ######################
    # 1. LIGHT CLEANING  #
    ######################
    def _clean_basic(self, df):
        """Nettoyage basique : doublons, types, date, filtres initiaux."""
        df = df.copy()

        # Supprimer les doublons exacts
        df.drop_duplicates(inplace=True)

        # Convertir la date de mutation et extraire ses composantes
        df["date_mutation"] = pd.to_datetime(df["date_mutation"], errors="coerce")
        df["annee_mutation"] = df["date_mutation"].dt.year
        df["mois_mutation"] = df["date_mutation"].dt.month
        df["jour_mutation"] = df["date_mutation"].dt.day
        df["jour_sem_mutation"] = df["date_mutation"].dt.weekday

        # Garder uniquement les ventes appartement/maison valides
        df = df[
            (df["nature_mutation"] == "Vente") &
            (df["type_local"].isin(["Appartement", "Maison"])) &
            (df["surface_reelle_bati"].notna()) &
            (df["valeur_fonciere"].notna())
        ]

        # Préfixe section (cast sur les 3 premiers caractères)
        df["section_prefixe"] = df["section_prefixe"].astype(str).str[:3]
        df["section_prefixe"] = pd.to_numeric(df["section_prefixe"], errors="coerce").astype("Int32")

        # Conversion numérique (hors colonnes catégorielles)
        for col in df.columns:
            if col not in ["type_local", "section_prefixe"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Création prix au m² (arrondi)
        df["prix_m2"] = (df["valeur_fonciere"] / df["surface_reelle_bati"]).round()

        return df

    def _select_features(self, df):
        """Colonnes utiles pour le modèle (pré-split)."""
        colonnes = [
            "numero_disposition", "valeur_fonciere", "code_postal", "code_type_local",
            "surface_reelle_bati", "nombre_pieces_principales", "longitude", "latitude",
            "section_prefixe", "nombre_lots", "annee_mutation", "mois_mutation",
            "jour_mutation", "jour_sem_mutation", "prix_m2", "type_local"
        ]
        return df[[col for col in colonnes if col in df.columns]]

    ######################
    # 2. DEEP CLEANING   #
    ######################
    def _compute_iqr_bounds(self, df, col):
        """Calcule les bornes IQR pour une variable donnée."""
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        return Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

    def _remove_outliers(self, df):
        """Filtrage des outliers sur les colonnes numériques clés."""
        df = df.copy()
        for col in ["valeur_fonciere", "prix_m2"]:
            q_low, q_high = self._compute_iqr_bounds(df, col)
            self.iqr_bounds[col] = (q_low, q_high)
            df = df[(df[col] >= q_low) & (df[col] <= q_high)]

        # Suppression des petits biens (< 9m²)
        df = df[df["surface_reelle_bati"] > 9]
        return df

    def _apply_outlier_filter(self, df):
        """Réapplique les filtres IQR appris (sans recalculer)."""
        for col, (q_low, q_high) in self.iqr_bounds.items():
            df = df[(df[col] >= q_low) & (df[col] <= q_high)]
        df = df[df["surface_reelle_bati"] > 9]
        return df

    #############################
    # 3. TRANSFORMATION (POST-SPLIT)
    #############################
    def _encode_features(self, df):
        """One-hot encoding des colonnes catégorielles après split (pour éviter fuite)."""
        return pd.get_dummies(df, columns=["type_local"], prefix="type_local")

    #########################
    # FIT_TRANSFORM & TRANSFORM
    #########################

    def fit_transform(self, df):
        """Pipeline complet d'entraînement + nettoyage (train uniquement)."""
        df = self._clean_basic(df)
        df = self._select_features(df)
        df = self._remove_outliers(df)
        df = self._encode_features(df)
        return df

    def transform(self, df):
        """Pipeline de transformation sur données de test/val (pas de recalcul)."""
        df = self._clean_basic(df)
        df = self._select_features(df)
        df = self._apply_outlier_filter(df)
        df = self._encode_features(df)
        return df
