from sklearn.pipeline import Pipeline, FeatureUnion

from src.preprocessing.feature_generators.categorical import CategoricalReduceTransformer
from src.preprocessing.feature_generators.datetime import DateTimeTransformer
from src.preprocessing.feature_generators.extract import FeatureExtractorTransformer


def get_preprocessing_pipeline(categorical_config, datetime_feats):

    preprocessor_pipe = FeatureUnion([
            ('category', Pipeline([
                ('extract', FeatureExtractorTransformer(categorical_config['feature_list'])),
                ('one_hot', CategoricalReduceTransformer(categorical_config))
            ])),
            ('datetime', Pipeline([
                ('extract', FeatureExtractorTransformer(datetime_feats)),
                ('datetime', DateTimeTransformer())
            ]))
    ])

    return preprocessor_pipe
