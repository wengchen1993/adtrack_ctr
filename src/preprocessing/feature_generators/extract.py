from sklearn.base import TransformerMixin


class FeatureExtractorTransformer(TransformerMixin):

    def __init__(self, feature_list):
        self.feature_list = feature_list

    def transform(self, input_df):
        # Return selected features from dataframe
        return input_df[self.feature_list]

    def fit(self, *_):
        return self
