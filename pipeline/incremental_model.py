import lightgbm as lgb
import numpy as np
import pandas as pd

class IncrementalLightGBM:
    def __init__(self, **kwargs):
        self.model = lgb.LGBMClassifier(**kwargs)
        self.X = pd.DataFrame()
        self.y = pd.Series()

    def add_data(self, X_new, y_new):
        self.X = pd.concat([self.X, X_new], ignore_index=True)
        self.y = pd.concat([self.y, y_new], ignore_index=True)

    def incremental_train(self):
        self.model.fit(self.X, self.y, init_score=np.zeros(len(self.y)), verbose=False)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def save_model(self, filename):
        self.model.save_model(filename)

    def load_model(self, filename):
        self.model = lgb.Booster(model_file=filename)

# Example usage:
# model = IncrementalLightGBM()
# model.add_data(X_new_train, y_new_train)
# model.incremental_train()
# predictions = model.predict(X_test)