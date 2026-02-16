import lightgbm as lgb
import numpy as np
import pandas as pd

class IncrementalLightGBM:
    def __init__(self, params=None):
        self.params = params if params else {}
        self.model = None

    def fit(self, X, y, batch_size=1024):
        # Initialize model if none exists
        if self.model is None:
            self.model = lgb.LGBMClassifier(**self.params)

        # Process data in batches
        for start in range(0, len(X), batch_size):
            end = min(start + batch_size, len(X))
            X_batch = X[start:end]
            y_batch = y[start:end]

            # Fit model incrementally
            self.model.fit(X_batch, y_batch, init_score=self.model.predict_proba(X_batch)[:, 1], verbose=False,
                           eval_set=[(X_batch, y_batch)], eval_names=['train'], eval_metric='logloss', early_stopping_rounds=10)

    def predict(self, X):
        if self.model is None:
            raise Exception('Model has not been trained yet!')
        return self.model.predict(X) 

    def save_model(self, filename):
        self.model.booster_.save_model(filename)

    def load_model(self, filename):
        self.model = lgb.Booster(model_file=filename)
