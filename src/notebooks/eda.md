---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.6.0
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python
from collections import Counter
import pandas as pd
import numpy as np
import time
import gc

import pprint
pp = pprint.PrettyPrinter(indent=4)

# Pipelining
from sklearn.base import TransformerMixin
from sklearn.model_selection import train_test_split
```

```python
RANDOM_STATE=42
```

# Objective


The challenge is to predict the CTR (click through rate). 
The original Kaggle challenge although focus on Fraud but we can still proceed with similar fashion of approach.


This notebook covers the core logic behind the code. Majority of the time was spent on structuring the ETL 
(outside of this notebook).


# Data Extraction


There are 7 features in total, where 5 features are categorical and the other 2 as date features.

```python
categorical_features = ['ip', 'app', 'device', 'os', 'channel']
datetime_features = ['click_time']
```

`attributed_time` is defined as :
```
if user download the app for after clicking an ad, this is the time of the app download
```

This would therefore make more sense to exclude `attributed_time` as training feature, we could use it for sanity
check instead.


### Type Specification

```python
# Variable types, will be a schema for actual external database
dtypes = {
    'ip': 'uint32',
    'app': 'uint16',
    'device': 'uint16',
    'os': 'uint16',
    'channel': 'uint16',
    'is_attributed': 'uint8',
    'click_id': 'uint32'
}
```

### Specify Data Source

```python
data_path = "../../data/"
```

### Data Loading


Due to the limiting size of machine used, total data points used for both train and test are a sub sample of the original
dataset.  

In the use case of actual large data consumption scenarios, distributed framework such as Spark will serve as a 
better tool for parallel batch processing.

```python
data_full = pd.read_csv(data_path + '/train/train.csv', 
                           dtype=dtypes)
```

### Imbalanced data


Imbalanced dataset can be countered using two general approaches:
    
1. Oversampling  
2. Undersampling  

(1) Oversampling strategies:  
    - Use SMOTE-NC to synthesize more data points using the attributes of nearest-neighbours  
    - Tune scale_pos_weight that increases the weight for minority class
    
(2) Undersampling strategies:  
    - Reduce the negative sample population to get closer of that of the positive ones


### Current sampling strategy


In this problem, the amount of data points are huge - 184,903,891 (about 184 million) and positive sample only
consists of 0.2% of the total population.

In order to let the model pick up more patterns of the positive sample, all positive samples are kept while we 
undersampled the negative samples randomly.

Total number of data points is predefined, minus all positive (minority)sample population which will then be the 
negative (majority) sample population.

```python
num_target_data_points = 1500000
```

```python
y_label = 'is_attributed'
```

```python
data_full[y_label].value_counts(0), data_full[y_label].value_counts(1)
```

```python
# Get all positive sample
data_pos_only = data_full[data_full[y_label] == 1]
```

```python
num_pos_points = data_pos_only.shape[0]
```

```python
data_neg_sampled = data_full[data_full[y_label] == 0].sample(n=(num_target_data_points - num_pos_points)) 
```

### Form sampled new dataset

```python
data_sampled = pd.concat([data_neg_sampled, data_pos_only], axis=0).reset_index(drop=True)
```

```python
labels = data_sampled[y_label]
features = data_sampled.drop(columns=[y_label, 'attributed_time'])
```

```python
num_data_points, num_features = data_sampled.shape
print(f"{num_data_points} data points with {num_features} features.")
```

```python
labels.value_counts(0), labels.value_counts(1)
```

```python
data_sampled.head(n=5)
```

# Train Test Split

```python
# Split the dataset into train and test (label name is pre-defined)
features, labels = data_sampled.drop(columns=[y_label]), data_sampled[y_label]

train_X, test_X, train_y, test_y = train_test_split(features, labels,
                                                    test_size=0.2,
                                                    random_state=RANDOM_STATE,
                                                    stratify=labels)
```

# Basic Stats


### Check Nulls

```python
for col in train_X.columns:
    nan_abs_val = train_X[col].isna().sum()
    nan_percent_val = nan_abs_val / num_data_points * 100
    print(f"{col} have {nan_abs_val} nulls, {nan_percent_val} %.")
```

### Check Unique (only for categorical)

```python
for col in categorical_features:
    num_cat_uniq = train_X[col].nunique()
    percent_cat_uniq = num_cat_uniq / num_data_points * 100
    print(f"{col} have unique number of categories: {num_cat_uniq}, {percent_cat_uniq:.3g}% .")
```

### Check categorical encoded range and common elements

```python
for col in categorical_features:
    print('='*30)
    print(f"For {col}, min is {train_X[col].min()}, max is {train_X[col].max()}")
    c = Counter(train_X[col]).most_common(10)
    top_10_common_elements = sorted(c, key=lambda x:x[1], reverse=True)
    
    print(f"Top 10 most common {col} categories with frequency:") 
    pp.pprint(top_10_common_elements)
```

### Check label distribution

```python
train_y.value_counts(1)
```

# Feature Generation Strategy


Features would be generated via sklearn transformer to retain any states and for clearer code structural consturction.

```python
class FeatureExtractorTransformer(TransformerMixin):

    def __init__(self, feature_list):
        self.feature_list = feature_list

    def transform(self, input_df):
        # Return selected features from dataframe
        return input_df[self.feature_list]

    def fit(self, *_):
        return self
```

## 1. Categorical


Most common and simplest way to deal with categorical features is one-hot encoding. 
However, the dimensions to deal with here are quite high.

We can reduce the cardinality by picking only top n most frequent cateogries and group the rest into a single category.

```python
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
            
        # If transformer has stored state (fit was used), assign unknown / new category with 0
        missing_columns = list(set(self.one_hot_columns) - set(one_hot_cols))

        # Assign any missing columns as all zeros
        input_df[missing_columns] = 0

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
```

## 2. Datetime


Datetime feature can be generated by breaking down to individual temporal dimension i.e hour, minutes etc.

These can then be further break down into combination of sin and cos - cyclic representation of features.

```python
dt_mappings = {
    'month': 12,
    'day': 31,
    'weekday': 7,
    'hour': 24,
    'minute': 60,
    'second': 60
}
```

```python
class DateTimeTransformer(TransformerMixin):

    def __init__(self, feature_list=None):
        self.feature_list = feature_list

    def transform(self, df):
        # Only collect transformed feature columns
        dt_feats = pd.DataFrame()

        for col in df.columns:
            # Extract datetime series
            dt_series = df[col]

            if isinstance(dt_series, pd.Series):
                try:
                    dt_series = pd.to_datetime(dt_series)

                    for dt_scale, dt_mval in dt_mappings.items():
                        # Since sin-cos is represented in a circular fashion, (0,0) is never reached
                        # This is hence used to indicate None
                        conv_dt_series = 2*np.pi*getattr(dt_series.dt, dt_scale) / dt_mval
                        dt_feats[f'{col}_sin_{dt_scale}'] = np.sin(conv_dt_series).fillna(0)
                        dt_feats[f'{col}_cos_{dt_scale}'] = np.cos(conv_dt_series).fillna(0)

                except Exception:
                    raise ValueError("Invalid datetime object or string.")
            else:
                raise ValueError("Input must be Pandas Series type.")

        return dt_feats

    def fit(self, *_):
        # This should be a stateless transformer
        return self

```

# Pipeline Construction

```python
from sklearn.pipeline import Pipeline, FeatureUnion
```

```python
features_config = {
    'category': {
        'top_n': 5,
        'feature_list': ['ip', 'app', 'device', 'os', 'channel'],
    },
    'datetime': ['click_time']
}
```

```python
categorical_config = features_config['category']
datetime_feats = features_config['datetime']
```

## 1. Feature Block

```python
preprocessor_pipeline = FeatureUnion([
        ('category', Pipeline([
            ('extract', FeatureExtractorTransformer(categorical_config['feature_list'])),
            ('one_hot', CategoricalReduceTransformer(categorical_config))
        ])),
        ('datetime', Pipeline([
            ('extract', FeatureExtractorTransformer(datetime_feats)),
            ('datetime', DateTimeTransformer())
        ]))
])
```

## 2. Model Block

```python
from xgboost import XGBClassifier

class XGBoost:
    def __init__(self, extra_grid_params={}):
        self.name = 'xgboost'
        self.model = XGBClassifier
        self.grid_params = {
            'clf__n_estimators': np.arange(100, 300, 100),  # number of trees
            'clf__learning_rate': [0.1],
            'clf__max_depth': np.arange(2, 8, 3),  # max number of levels in each decision tree,
        }
        self.grid_params.update(extra_grid_params)

    def gen_model_grid_params(self):
        model_content = {
            'model': self.model,
            'params': self.grid_params
        }
        return model_content
```

```python
ml_pipeline = XGBoost()
```

## ML Flow Pipeline

```python
ml_flow_pipeline = Pipeline([
    ('features', preprocessor_pipeline),
    ('clf', ml_pipeline.model())
])
```

# Model Training

```python
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV
```

```python
score_metric = 'roc_auc'
n_jobs = 4
n_inner_fold = 3
```

```python
model_train_start = time.time()

# Stratified K Fold validation as to maintain the imbalanced data distribution within folds
k_fold = StratifiedKFold(n_splits=n_inner_fold, random_state=RANDOM_STATE)

# Perform GridSearch Cross Validation    
model = GridSearchCV(estimator=ml_flow_pipeline, 
                     param_grid=ml_pipeline.grid_params, 
                     scoring=score_metric, 
                     cv=k_fold, 
                     n_jobs=n_jobs)

# Train the model
model.fit(train_X, train_y)

model_train_end = time.time()

print(f"Model training took {model_train_end-model_train_start:.5} seconds.")
```

# Model Evaluation


Considering we might be interested to use multiple metrics to measure the performance, we can create a class
to accept the trained model and targeted performance metric. This can then be used to generate or save any 
visualisations (if needed). 

```python
from sklearn.metrics import roc_auc_score, log_loss, recall_score

available_metrics = {
    'roc_auc_score': roc_auc_score,
    'log_loss': log_loss,
    'recall_score': recall_score
}

def get_feval(eval_metric):
    if eval_metric not in available_metrics:
        raise ValueError(f"{eval_metric} is not available. Available metrics are {list(available_metrics.keys())}. ")

    return available_metrics[eval_metric]
```

```python
class ModelPerformance:
    def __init__(self, model, metrics=None):
        self.model = model
        if metrics is None:
            self.metrics = set()
        else:
            self.metrics = metrics

    def set_metrics(self, metrics):
        if isinstance(metrics, list):
            metrics = set(metrics)

        self.metrics = metrics

    def evaluate(self, data, y_true):
        # Performance
        performance_cache = {}

        # Get prediction
        y_pred = self.model.predict(data)

        for metric in self.metrics:
            performance_cache[metric] = get_feval(metric)(y_true, y_pred)

        return performance_cache
```

### Performance Measurement

```python
ml_metrics = ['recall_score', 'roc_auc_score']
ml_eval = ModelPerformance(model, ml_metrics)
```

```python
performance_metrics = ml_eval.evaluate(test_X, test_y)
```

```python
for metric_name, metric_val in performance_metrics.items():
    print(f"Trained model achieved score {metric_val} of {metric_name}.")
```

```python

```
