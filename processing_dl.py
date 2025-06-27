import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd

class DLRegressor(BaseEstimator, RegressorMixin):
    """
    Deep Learning Regressor (n-to-n, sklearn-compatible).
    - Standardisation X/Y en option
    - Log transformation de y (optionnel)
    - Architecture et paramètres customisables
    """

    def __init__(self, hidden_layers=(128, 64, 32, 16), dropout=0.2, epochs=150, batch_size=32,
                 scaler_X=True, scaler_y=True, log_target=True,
                 learning_rate=0.001, patience=10, verbose=1, random_state=42):
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y
        self.log_target = log_target
        self.learning_rate = learning_rate
        self.patience = patience
        self.verbose = verbose
        self.random_state = random_state
        # Pour le fit
        self.model_ = None
        self.scaler_X_ = None
        self.scaler_y_ = None

    def build_model(self, input_dim, output_dim):
        np.random.seed(self.random_state)
        import tensorflow as tf
        tf.random.set_seed(self.random_state)
        model = Sequential()
        model.add(Dense(self.hidden_layers[0], activation='relu', kernel_initializer="he_normal", input_dim=input_dim))
        if self.dropout:
            model.add(Dropout(self.dropout))
        for units in self.hidden_layers[1:]:
            model.add(Dense(units, activation='relu', kernel_initializer="he_normal"))
            if self.dropout:
                model.add(Dropout(self.dropout))
        model.add(Dense(output_dim, activation='linear'))
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='huber', metrics=['mae'])
        return model

    def fit(self, X, y):
        print("[DLRegressor] Standardisation + log(y) (si activé)...")
        # Standardisation
        if self.scaler_X:
            self.scaler_X_ = StandardScaler()
            X_scaled = self.scaler_X_.fit_transform(X)
            print("[DLRegressor] X standardisé !")
        else:
            X_scaled = X.values if isinstance(X, (pd.DataFrame, pd.Series)) else X

        # Log sur la target
        if self.log_target:
            y_tr = np.log1p(y)
            print("[DLRegressor] y log-transformé !")
        else:
            y_tr = y

        # Standardisation y (optionnelle mais rarement utile)
        if self.scaler_y:
            self.scaler_y_ = StandardScaler()
            y_scaled = self.scaler_y_.fit_transform(y_tr.values.reshape(-1, 1)).flatten()
            print("[DLRegressor] y standardisé !")
        else:
            y_scaled = y_tr.values if isinstance(y_tr, (pd.DataFrame, pd.Series)) else y_tr

        print("[DLRegressor] Construction du modèle keras...")
        input_dim = X_scaled.shape[1]
        output_dim = 1
        self.model_ = self.build_model(input_dim, output_dim)

        print("[DLRegressor] Entraînement du modèle...")
        es = EarlyStopping(monitor="val_loss", patience=self.patience, restore_best_weights=True)
        self.model_.fit(
            X_scaled, y_scaled,
            validation_split=0.15,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[es],
            verbose=self.verbose
        )
        print("[DLRegressor] Apprentissage terminé.")
        return self

    def predict(self, X):
        print("[DLRegressor] Prédiction en cours...")
        X_scaled = self.scaler_X_.transform(X) if self.scaler_X else (X.values if isinstance(X, (pd.DataFrame, pd.Series)) else X)
        y_pred = self.model_.predict(X_scaled)
        # Destandardisation y (si activé)
        if self.scaler_y:
            y_pred = self.scaler_y_.inverse_transform(y_pred)
        # Inverse log
        if self.log_target:
            y_pred = np.expm1(y_pred)
        return y_pred.flatten()

    def score(self, X, y):
        from sklearn.metrics import r2_score
        y_pred = self.predict(X)
        return r2_score(y, y_pred)
