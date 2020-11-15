import argparse
import dill as pickle
import os
import time

import datetime
import pandas as pd
from typing import Tuple, Dict
from sklearn.model_selection import train_test_split

from src.pipeline.main import MLPipeline
from src.evaluation.model_performance import ModelPerformance

import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)-2s %(name)-4s: %(levelname)-8s %(message)s',
                    handlers=[logging.StreamHandler(), logging.FileHandler(f"logs/etl.log")])

logger = logging.getLogger(__name__)

RANDOM_STATE = 42


def extract(data_source: str, num_sampled_data: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract data from given directory, expects fixed train and test directories.

    Parameters
    ----------
    data_source: str

    Returns
    -------
    (train_data, test_data): Tuple[pd.DataFrame, pd.DataFrame]
        Loaded train and test dataset in dataframe.
    """
    # Specify dtypes for small memory footprint (use a schema for )
    dtypes = {
        'ip': 'uint32',
        'app': 'uint16',
        'device': 'uint16',
        'os': 'uint16',
        'channel': 'uint16',
        'is_attributed': 'uint8',
        'click_id': 'uint32'
    }

    # Read in dataset for training stage
    dataset_path = os.path.join(data_source, 'train', 'train.csv')
    data_full = pd.read_csv(dataset_path, dtype=dtypes)

    # Split the dataset into train and test (label name is pre-defined)
    y_label = 'is_attributed'

    logger.info(f"Absolute counts of full data points population: {data_full[y_label].value_counts(0)} .")
    logger.info(f"Ratio of full data points population: {data_full[y_label].value_counts(1)} .")

    # Perform undersampling while keep all positive samples
    data_pos_only = data_full[data_full[y_label] == 1]

    # Random undersampling by selecting amount of predefined target number of points minus all positive
    num_pos_points = data_pos_only.shape[0]
    num_neg_points_required = num_sampled_data - num_pos_points
    data_neg_sampled = data_full[data_full[y_label] == 0].sample(n=num_neg_points_required)

    # Concatenate to reform the dataset
    data_sampled = pd.concat([data_neg_sampled, data_pos_only], axis=0).reset_index(drop=True)

    logger.info(f"Absolute counts of sampled data points population: {data_sampled[y_label].value_counts(0)} .")
    logger.info(f"Ratio of sampled data points population: {data_sampled[y_label].value_counts(1)} .")

    features, labels = data_sampled.drop(columns=[y_label]), data_sampled[y_label]

    train_X, test_X, train_y, test_y = train_test_split(features, labels,
                                                        test_size=0.2,
                                                        random_state=RANDOM_STATE,
                                                        stratify=labels)

    train_data = {
        'data': train_X,
        'label': train_y
    }

    test_data = {
        'data': test_X,
        'label': test_y
    }

    return train_data, test_data


def transform(train_data: pd.DataFrame, 
              test_data: pd.DataFrame, 
              model_name: str = "logit") -> Tuple[MLPipeline, Dict[str, float]]:
    """
    Contains all ML training and evaluation logic.

    Parameters
    ----------
    train_data
    test_data
    model_name

    Returns
    -------
    model, performance_metric
        Trained model artefact and performance metric score.
    """
    # Setup features dtypes, would ideally be a separate config for more features
    features_config = {
        'category': {
            'top_n': 5,
            'feature_list': ['ip', 'app', 'device', 'os', 'channel'],
        },
        'datetime': ['click_time']
    }

    train_X, train_y = train_data['data'], train_data['label']
    test_X, test_y = test_data['data'], test_data['label']

    # Setup ML pipeline for training and test
    ml_pipe = MLPipeline(model_name, features_config)

    logger.info("Begin model training ...")
    model_train_start = time.time()
    model = ml_pipe.train(train_X, train_y)
    model_train_end = time.time()
    logger.info(f"Model training took {model_train_end - model_train_start:.5} seconds.")

    # Setup evaluation object for multiple performance metrics query
    ml_metrics = ['recall_score']
    ml_eval = ModelPerformance(model, ml_metrics)
    performance_metric = ml_eval.evaluate(test_X, test_y)

    return model, performance_metric


def load(model, storage_dir="artefacts"):
    """
    Serialise model for portable format.

    Parameters
    ----------
    model
    storage_dir
    """
    # Timestamp based storage directory for tracking
    model_destination_dir = os.path.join(storage_dir,
                                         datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    # Create storage directory if not exists
    os.makedirs(model_destination_dir, exist_ok=True)

    model_destination_path = os.path.join(model_destination_dir, "model.pkl")

    with open(model_destination_path, "wb") as f:
        pickle.dump(model, f)

    logger.info(f"Model serialised into {model_destination_path} .")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_source",
                        help="Directory where data is stored. It can be S3 path if AWS is used.",
                        type=str,
                        required=True)
    parser.add_argument("--model_name",
                        help="Type of model to use.",
                        type=str,
                        default="logit")
    parser.add_argument("--num_sampled_data",
                        help="Total number of data points to be sampled for training.",
                        default=1500000,
                        type=int)

    args = parser.parse_args()

    logger.info("Launching ETL job ...")

    logger.info(f"Preparing data extraction in the given directory {args.data_source}")
    train_data, test_data = extract(args.data_source, args.num_sampled_data)
    logger.info("Completed data extraction to generate train and test dataset.")

    logger.info("Initialise transform job - feature engineering and model training ... ")
    model, performance_metrics = transform(train_data, test_data, args.model_name)
    logger.info("Transform job completed.")

    for metric_name, metric_val in performance_metrics.items():
        logger.info(f"Trained model achieved score {metric_val} of {metric_name}.")

    logger.info("Serialising model into artefacts ...")
    load(model)
    logger.info("Model serialization completed using pickle.")

    logger.info("ETL job completed.")
