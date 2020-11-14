from sklearn.base import TransformerMixin
import numpy as np
import pandas as pd

import logging

dt_mappings = {
    'month': 12,
    'day': 31,
    'weekday': 7,
    'hour': 24,
    'minute': 60,
    'second': 60
}


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
