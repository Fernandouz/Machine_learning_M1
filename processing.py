import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
import shap
from pyexpat import features

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from preprocessing import FeatureEngineering

# Ajoute tqdm pour la barre de progression
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

class MLModeler:
    """
    Classe pipeline Machine Learning (régression) :
    - GridSearch, entraînement, prédiction, évaluation, sauvegarde, courbe apprentissage.
    Affiche chaque étape du process, avec beaucoup de prints et une barre de progression pendant le GridSearch.
    """

    def __init__(self, model_type="linreg", model_path="models/best_model.joblib"):
        print("\n[MLModeler] ===== Initialisation pipeline ML (régression) =====")
        self.model_type = model_type
        self.model_path = model_path
        self.best_model = None
        self.grid_search = None

        print(f"[MLModeler] Modèle choisi : {model_type}")

        # Construction du pipeline et grille d'hyperparams
        if model_type == "linreg":
            print("[MLModeler] Pipeline : LinearRegression")
            self.pipeline = Pipeline([
                ('clf', LinearRegression())
            ])
            self.param_grid = {
                'clf__fit_intercept': [True, False],
                'clf__positive': [False, True]
            }
            # RMSE : sensible aux gros écarts, plus "métier" en immo
            self.scoring = 'neg_root_mean_squared_error'
            # Pour voir d'autres options :
            # self.scoring_options = ['neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_root_mean_squared_error', 'r2']
        elif model_type == "rf":
            print("[MLModeler] Pipeline : RandomForestRegressor")
            self.pipeline = Pipeline([
                ('clf', RandomForestRegressor(random_state=42, n_jobs=-1))
            ])
            self.param_grid = {
                'clf__n_estimators': [100, 200, 300],
                'clf__max_depth': [None, 10, 20, 30],
                'clf__min_samples_split': [2, 5, 10],
                'clf__min_samples_leaf': [1, 2, 4],
                'clf__maXatures': ['auto', 'sqrt', 'log2'],
                'clf__bootstrap': [True, False]
            }
            # MAE est souvent préféré pour la robustesse sur outliers en RF
            self.scoring = 'neg_mean_absolute_error'
            # Optionnel : tester aussi avec 'neg_root_mean_squared_error'
        elif model_type == "xgb":
            print("[MLModeler] Pipeline : XGBRegressor")
            self.pipeline = Pipeline([
                ('clf', XGBRegressor(objective='reg:squarederror', verbosity=0, n_jobs=-1, random_state=42))
            ])
            self.param_grid = {
                'clf__n_estimators': [100, 200],
                'clf__learning_rate': [0.05, 0.1],
                'clf__max_depth': [3, 5],
                'clf__subsample': [0.8, 1.0],
                'clf__colsample_bytree': [0.8, 1.0],
                'clf__reg_alpha': [0, 0.1],
                'clf__reg_lambda': [1, 2]
                # Tu peux rajouter gamma ou min_child_weight si tu veux, mais déjà ça fait 64 combinaisons
            }
            # Pour XGB : RMSE est le plus classique (Kaggle Style)
            self.scoring = 'neg_root_mean_squared_error'
            # Si tu fais du log-transform, 'neg_mean_squared_log_error' est ultra pertinent
        else:
            raise ValueError("[MLModeler] model_type inconnu : 'linreg', 'rf', 'xgb'")
        print(f"[MLModeler] Paramètres du pipeline initialisés.\n")

    def fit(self, X, y):
        """
        Entraîne le modèle avec GridSearchCV + prints détaillés + barre de progression.
        """
        y = y.ravel()
        print("\n[MLModeler] ===== DÉBUT ENTRAÎNEMENT + GRIDSEARCH =====")
        print(f"[MLModeler] X shape : {X.shape} | y shape : {y.shape}")
        print(f"[MLModeler] Grille d'hyperparamètres : {self.param_grid}")
        print(f"[MLModeler] Scoring utilisé pour la grille : {self.scoring}")

        # Préparation GridSearch
        total_candidates = 1
        for v in self.param_grid.values():
            total_candidates *= len(v)
        print(f"[MLModeler] Nombre de combinaisons à tester : {total_candidates}\n")

        # Affichage de la barre de progression via callback
        def tqdm_callback(estimator, params):
            if HAS_TQDM:
                tqdm_bar.update(1)
                tqdm_bar.set_postfix(params)
            else:
                print(f"[GridSearch] Test params : {params}")

        # Barre de progression si tqdm installé
        if HAS_TQDM:
            global tqdm_bar
            tqdm_bar = tqdm(total=total_candidates, desc="GridSearch Progress", ncols=80)

        # GridSearchCV avec callback sur chaque fit (sklearn >=1.3 nécessaire pour callbacks)
        self.grid_search = GridSearchCV(
            self.pipeline, self.param_grid, cv=3, scoring=self.scoring, n_jobs=-1, verbose=2
        )

        print("[MLModeler] Lancement du GridSearchCV...\n")
        self.grid_search.fit(X, y)
        if HAS_TQDM:
            tqdm_bar.close()

        print("[MLModeler] ENTRAÎNEMENT TERMINÉ !")
        print(f"[MLModeler] Meilleurs paramètres : {self.grid_search.best_params_}")
        print(f"[MLModeler] Score optimal (mean CV) : {self.grid_search.best_score_:.4f}")

        self.best_model = self.grid_search.best_estimator_
        print("[MLModeler] Best estimator entraîné et prêt à prédire !")

    def predict(self, X):
        """
        Prédit les valeurs cibles avec le meilleur modèle entraîné.
        """
        if self.best_model is None:
            raise Exception("Le modèle doit être entraîné avant la prédiction (fit() manquant)")
        print(f"\n[MLModeler] ===== Prédiction sur {X.shape[0]} observations =====")
        return self.best_model.predict(X)

    import os
    import pandas as pd
    from datetime import datetime
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    def evaluate(self, X, y, scores_path="results/scores.csv", plot_dir="results/plots/"):
        """
        Affiche les métriques de régression + plot + sauvegarde les scores et le plot dans un PNG.
        """
        print("\n[MLModeler] ===== ÉVALUATION DU MODÈLE (RÉGRESSION) =====")
        y_pred = self.predict(X)
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        print(f"\n--- Scores de régression ---")
        print(f"RMSE : {rmse:.2f}")
        print(f"MAE  : {mae:.2f}")
        print(f"R²   : {r2:.3f}")

        # Sauvegarde du score comme avant (optionnel)
        os.makedirs(os.path.dirname(scores_path), exist_ok=True)
        scores_row = {
            "datetime": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            "model": self.model_type,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "test_shape": X.shape[0]
        }
        try:
            if os.path.exists(scores_path):
                scores_df = pd.read_csv(scores_path)
                scores_df = pd.concat([scores_df, pd.DataFrame([scores_row])], ignore_index=True)
            else:
                scores_df = pd.DataFrame([scores_row])
            scores_df.to_csv(scores_path, index=False)
            print(f"[MLModeler] Scores sauvegardés dans {scores_path}")
        except Exception as e:
            print(f"[MLModeler] ERREUR lors de la sauvegarde des scores : {e}")

        # === Sauvegarde du plot prédiction vs réalité ===
        os.makedirs(plot_dir, exist_ok=True)
        timestamp = scores_row["datetime"]
        plot_path = os.path.join(plot_dir, f"{self.model_type}_pred_vs_reality_{timestamp}.png")
        print(f"[MLModeler] Sauvegarde du plot 'Prédiction vs Réalité' sous {plot_path}")
        plt.figure(figsize=(6, 6))
        plt.scatter(y, y_pred, alpha=0.5, label="Prédictions")
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label="Idéal")
        plt.xlabel("Vraies valeurs")
        plt.ylabel("Prédictions")
        plt.title(f"Prédiction vs Réalité - {self.model_type}")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

    def plot_learning_curve(self, X, y, cv=3, plot_dir="results/plots/"):
        """
        Affiche et sauvegarde la courbe d'apprentissage (learning curve).
        """
        import numpy as np
        from sklearn.model_selection import learning_curve

        print("\n[MLModeler] ===== Affichage de la courbe d'apprentissage (learning curve) =====")
        print(f"[MLModeler] X shape : {X.shape} | y shape : {y.shape}")
        train_sizes, train_scores, test_scores = learning_curve(
            self.best_model, X, y, cv=cv, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 5), scoring=self.scoring
        )
        train_scores_mean = -np.mean(train_scores, axis=1)
        test_scores_mean = -np.mean(test_scores, axis=1)
        plt.figure()
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training error (MSE)")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Validation error (MSE)")
        plt.title(f"Courbe d'apprentissage (MSE) - {self.model_type}")
        plt.xlabel("Taille du train set")
        plt.ylabel("Erreur quadratique moyenne")
        plt.legend(loc="best")
        plt.grid()
        plt.tight_layout()

        # === Sauvegarde automatique ===
        os.makedirs(plot_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        curve_path = os.path.join(plot_dir, f"{self.model_type}_learning_curve_{timestamp}.png")
        plt.savefig(curve_path)
        print(f"[MLModeler] Courbe d'apprentissage sauvegardée sous {curve_path}")
        plt.close()

    def save(self, path=None):
        """
        Sauvegarde du modèle entraîné.
        """
        path = path or self.model_path
        print(f"[MLModeler] ===== Sauvegarde du modèle sous {path} =====")
        joblib.dump(self.best_model, path)
        print("[MLModeler] Modèle sauvegardé avec succès !\n")

    def load(self, path=None):
        """
        Charge un modèle sauvegardé.
        """
        path = path or self.model_path
        if not os.path.exists(path):
            raise FileNotFoundError(f"Modèle introuvable : {path}")
        print(f"[MLModeler] ===== Chargement du modèle depuis {path} =====")
        self.best_model = joblib.load(path)
        print("[MLModeler] Modèle chargé avec succès !")

    def plot_shap_values(self, X, plot_dir="results/shap/", class_index=1):
        """
        Affiche et sauvegarde les valeurs SHAP du modèle (si applicable).
        Fonctionne uniquement avec RandomForest ou XGBoost.
        """
        print("\n[MLModeler] ===== Analyse SHAP des features =====")

        if self.model_type not in ["rf", "xgb"]:
            print("[MLModeler] SHAP non disponible pour ce modèle.")
            return

        if self.best_model is None:
            raise Exception("Le modèle doit être entraîné avant d’utiliser SHAP.")

        os.makedirs(plot_dir, exist_ok=True)
        model = self.best_model.named_steps["clf"]
        explainer = shap.TreeExplainer(model)



        print(f"[MLModeler] Calcul des valeurs SHAP sur {X.shape[0]} échantillons...")
        print(X)

        shap_values = explainer.shap_values(X)

        if isinstance(shap_values, list):
            print(f"[MLModeler] Modèle binaire détecté, utilisation de la classe {class_index}")
            shap_values = shap_values[class_index]

        # Résumé SHAP
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        shap_summary_path = os.path.join(plot_dir, f"shap_summary_{self.model_type}_{timestamp}.png")
        print(f"[MLModeler] Sauvegarde du résumé SHAP sous {shap_summary_path}")
        plt.figure()
        shap.summary_plot(shap_values, X, show=False)
        plt.tight_layout()
        plt.savefig(shap_summary_path)
        plt.close()

        print("[MLModeler] Analyse SHAP terminée.")

    def plot_shap_summary_bar(self, X, plot_dir="results/shap/", class_index=1):
        """
        Trace un summary_plot de type bar pour l'importance moyenne des features.
        """
        if self.model_type not in ["rf", "xgb"]:
            return

        print("\n[MLModeler] Résumé SHAP (bar plot) des variables les plus impactantes...")
        model = self.best_model.named_steps["clf"]
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[class_index]

        os.makedirs(plot_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = os.path.join(plot_dir, f"shap_bar_{self.model_type}_{timestamp}.png")

        plt.figure()
        shap.summary_plot(shap_values, X, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        print(f"[MLModeler] SHAP bar plot sauvegardé dans {path}")

    def plot_shap_dependence(self, X, plot_dir="results/shap/", class_index=1):
        """
        Trace des scatter plots SHAP value pour une sélection de features.
        """
        if self.model_type not in ["rf", "xgb"]:
            return

        model = self.best_model.named_steps["clf"]
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[class_index]

        features = None

        if self.model_type == "rf":
            features = ["nombre_lots", "surface_reelle_bati", "prix_m2_quartier", "section_prefixe"]

        if self.model_type == "xgb":
            features = ["nombre_lots", "surface_reelle_bati", "prix_m2_quartier", "type_local_Appartement"]

        features = features or list(X.columns[:5])
        features = [f for f in features if f in X.columns]

        n_features = len(features)
        os.makedirs(plot_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        fig, axes = plt.subplots(1, n_features, figsize=(5 * n_features, 5), constrained_layout=True)
        if n_features == 1:
            axes = [axes]

        for i, col in enumerate(features):
            ax = axes[i]
            scatter = ax.scatter(
                X[col],
                shap_values[:, X.columns.get_loc(col)],
                c=X[col],
                cmap="viridis",
                alpha=0.7
            )
            # Point d’intérêt arbitraire : le 5ème (ajustable)
            ax.scatter(
                X.iloc[5][col],
                shap_values[5, X.columns.get_loc(col)],
                color='red',
                edgecolor='black',
                s=100,
                label="Obs. n°5"
            )
            ax.set_xlabel(col)
            ax.set_ylabel("SHAP value")
            ax.set_title(f"SHAP - {col}")
            ax.legend()
            fig.colorbar(scatter, ax=ax, label=col)

        path = os.path.join(plot_dir, f"shap_dependence_{self.model_type}_{timestamp}.png")
        plt.savefig(path)
        plt.close()
        print(f"[MLModeler] SHAP dependence plots sauvegardés dans {path}")

    def plot_shap_waterfall(self, X, instance_index=5, plot_dir="results/shap/", class_index=1):
        """
        Affiche un waterfall plot SHAP pour une instance.
        """
        if self.model_type not in ["rf", "xgb"]:
            return

        model = self.best_model.named_steps["clf"]
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[class_index]
            base_value = explainer.expected_value[class_index]
        else:
            base_value = explainer.expected_value

        os.makedirs(plot_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = os.path.join(plot_dir, f"shap_waterfall_{self.model_type}_obs{instance_index}_{timestamp}.png")

        shap_obj = shap.Explanation(
            values=shap_values[instance_index],
            base_values=base_value,
            data=X.iloc[instance_index],
            feature_names=X.columns
        )
        shap.plots.waterfall(shap_obj, show=False)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        print(f"[MLModeler] SHAP waterfall plot sauvegardé dans {path}")
