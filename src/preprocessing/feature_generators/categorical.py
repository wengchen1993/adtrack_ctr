from sklearn.base import TransformerMixin
import pandas as pd
from collections import Counter


class CategoricalReduceTransformer(TransformerMixin):

    def __init__(self, config={}):

        self.top_n = config.get('top_n', None)
        self.feature_list = config.get('feature_list', None)
        self.default_column_names = config.get('default_column_names', None)

        self.one_hot_columns = None
        self.all_columns = None
        self.top_n_cats = {}

    def transform(self, input_df, **transform_params):

        # Check if the input is dataframe, if not convert to dataframe with set columns
        if not isinstance(input_df, pd.DataFrame):
            input_df = pd.DataFrame(input_df, columns=self.default_column_names)

        if self.feature_list is not None:
            feat_list = self.feature_list
        else:
            feat_list = input_df.columns

        one_hot_cols = []
        # Apply one_hot coding for all features in feature list
        for col in feat_list:
            input_df[col] = input_df[col].map(str)

            if self.top_n_cats:
                # New / unknown value will be treated as minority
                # Top n most frequent categories and None values retained respective encoding
                input_df.loc[~input_df[col].isin(self.top_n_cats[col]), col] = '-1'

            # Get the one_hot coding
            one_hot_df = pd.get_dummies(input_df[col], prefix=col)

            # Drop original feature from dataset
            input_df = input_df.drop(columns=[col])

            # Add one hot coding instead of original feature
            input_df = pd.concat([input_df, one_hot_df], axis=1)

            # Keep track of one_hot columns in train set
            one_hot_cols.extend(one_hot_df.columns)

        '''
        # If transformer has stored state (fit was used), assign unknown / new category with 0
        missing_columns = list(set(self.one_hot_columns) - set(one_hot_cols))

        # Assign any missing columns as all zeros
        input_df[missing_columns] = 0
        '''
        # Check the order of columns are the same
        input_df = input_df[list(self.all_columns) + list(self.one_hot_columns)]

        return input_df

    def fit(self, input_df, *_):

        # Check if the input is dataframe
        if not isinstance(input_df, pd.DataFrame):
            input_df = pd.DataFrame(input_df, columns=self.default_column_names)

        if self.feature_list is not None:
            feat_list = self.feature_list
        else:
            feat_list = input_df.columns

        one_hot_cols = []

        # Apply one hot coding for all features in feature list
        for col in feat_list:
            input_df[col] = input_df[col].map(str)

            if self.top_n is not None:
                # Get top n most frequent categories, replace the minority as single class
                cat_counter = Counter(input_df[col]).most_common(self.top_n)
                self.top_n_cats[col] = set([c[0] for c in cat_counter] + ['nan'])
                input_df.loc[~input_df[col].isin(self.top_n_cats[col]), col] = '-1'

            # Get the one_hot coding
            one_hot_df = pd.get_dummies(input_df[col], prefix=col)

            # Drop original feature from dataset
            input_df = input_df.drop(columns=[col])

            # Keep track of one_hot columns in train set
            one_hot_cols.extend(one_hot_df.columns)

        self.one_hot_columns = one_hot_cols
        self.all_columns = input_df.columns

        return self

    def get_feature_list(self):
        return list(self.all_columns) + list(self.one_hot_columns)
