import pickle
import joblib
import os
import matplotlib.pyplot as plt
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split, learning_curve
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

class MLModeler:
    """
    Classe pour automatiser tout le pipeline Machine Learning classique :
    - Prétraitement, split, GridSearch, évaluation, sauvegarde, graphique.
    """
    def __init__(self, model_type="logreg", model_path="best_model.joblib"):
        print("[MLModeler] Initialisation du pipeline ML...")
        self.model_type = model_type
        self.model_path = model_path
        self.best_model = None
        self.grid_search = None

        # Construction du pipeline de base (changeable selon use case)
        if model_type == "logreg":
            self.pipeline = Pipeline([
                ('clf', LogisticRegression(max_iter=1000))
            ])
            self.param_grid = {
                'clf__C': [0.01, 0.1, 1, 10]
            }
        elif model_type == "rf":
            self.pipeline = Pipeline([
                ('clf', RandomForestClassifier())
            ])
            self.param_grid = {
                'clf__n_estimators': [100, 200],
                'clf__max_depth': [None, 10, 20]
            }
        elif model_type == "svc":
            self.pipeline = Pipeline([
                ('clf', SVC())
            ])
            self.param_grid = {
                'clf__C': [0.1, 1, 10],
                'clf__kernel': ['linear', 'rbf']
            }
        else:
            raise ValueError("model_type inconnu : utilise 'logreg', 'rf', ou 'svc'")

    def fit(self, X, y):
        """
        Entraîne le modèle avec GridSearchCV.
        """
        print("[MLModeler] Début entraînement + GridSearch...")
        self.grid_search = GridSearchCV(self.pipeline, self.param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=2)
        self.grid_search.fit(X, y)
        self.best_model = self.grid_search.best_estimator_
        print("[MLModeler] Entraînement terminé !")
        print(f"[MLModeler] Meilleurs paramètres : {self.grid_search.best_params_}")

    def predict(self, X):
        """
        Prédit les labels avec le meilleur modèle.
        """
        if self.best_model is None:
            raise Exception("Le modèle doit être entraîné avant la prédiction (fit() manquant)")
        print("[MLModeler] Prédiction en cours...")
        return self.best_model.predict(X)

    def evaluate(self, X, y, class_names=None):
        """
        Affiche le rapport de classification et la matrice de confusion.
        """
        print("[MLModeler] Évaluation du modèle...")
        y_pred = self.predict(X)
        print("\n--- Rapport de classification ---")
        print(classification_report(y, y_pred))
        print("\n--- Matrice de confusion ---")
        cm = confusion_matrix(y, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Matrice de confusion")
        plt.show()

    def plot_learning_curve(self, X, y, cv=3):
        """
        Affiche la courbe d'apprentissage du modèle.
        """
        print("[MLModeler] Affichage de la courbe d'apprentissage...")
        train_sizes, train_scores, test_scores = learning_curve(
            self.best_model, X, y, cv=cv, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 5), scoring='accuracy'
        )
        train_scores_mean = np.mean(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        plt.figure()
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
        plt.title("Courbe d'apprentissage")
        plt.xlabel("Taille du train set")
        plt.ylabel("Score")
        plt.legend(loc="best")
        plt.grid()
        plt.show()

    def save(self, path=None):
        """
        Sauvegarde le modèle entraîné (format joblib).
        """
        path = path or self.model_path
        print(f"[MLModeler] Sauvegarde du modèle sous {path} ...")
        joblib.dump(self.best_model, path)
        print("[MLModeler] Modèle sauvegardé !")

    def load(self, path=None):
        """
        Charge un modèle déjà sauvegardé (joblib).
        """
        path = path or self.model_path
        if not os.path.exists(path):
            raise FileNotFoundError(f"Modèle introuvable : {path}")
        print(f"[MLModeler] Chargement du modèle depuis {path} ...")
        self.best_model = joblib.load(path)
        print("[MLModeler] Modèle chargé !")
