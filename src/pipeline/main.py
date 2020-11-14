from src.pipeline.preprocessing import get_preprocessing_pipeline
from src.pipeline.model import get_model

from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV

import logging
logger = logging.getLogger(__name__)


class MLPipeline:
    RANDOM_STATE = 42

    def __init__(self, model_name, feature_config):
        self.model = None
        self.model_name = model_name

        model_ref = get_model(model_name)
        model = model_ref['model']
        model_params = model_ref['params']

        feature_pipeline = get_preprocessing_pipeline(
                feature_config.get('category', []),
                feature_config.get('datetime', [])
        )

        ml_flow_pipeline = Pipeline([
            ('features', feature_pipeline),
            ('clf', model())
        ])
        logger.info("Created ml flow pipeline.")

        self.pipeline = {
            'name': self.model_name,
            'model': ml_flow_pipeline,
            'params': model_params
        }

        logger.info("Packed pipeline properties.")

        self.n_jobs = 4
        self.n_split = 3
        self.scoring = 'roc_auc'

    def train(self, X, y, train_params=None):
        k_fold = StratifiedKFold(n_splits=self.n_split, random_state=self.RANDOM_STATE)

        # Perform GridSearch Cross Validation
        self.model = GridSearchCV(estimator=self.pipeline['model'],
                                  param_grid=self.pipeline['params'],
                                  scoring=self.scoring,
                                  cv=k_fold,
                                  n_jobs=self.n_jobs)

        logger.info("Created GridSearch object.")

        self.model.fit(X, y)

        return self.model

    def predict(self, data):
        if self.model is None:
            raise Exception("Please train model prior to testing it.")

        return self.model.predict(data)
