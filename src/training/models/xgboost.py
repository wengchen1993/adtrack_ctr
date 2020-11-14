from xgboost import XGBClassifier
import numpy as np


class XGBoost:
    def __init__(self, extra_grid_params={}):
        self.name = 'xgboost'
        self.model = XGBClassifier
        self.grid_params = {
            'clf__n_estimators': np.arange(100, 300, 100),
            'clf__learning_rate': [0.1],
            'clf__max_depth': np.arange(2, 8, 3)
        }
        self.grid_params.update(extra_grid_params)

    def gen_model_grid_params(self):
        model_content = {
            'model': self.model,
            'params': self.grid_params
        }
        return model_content
