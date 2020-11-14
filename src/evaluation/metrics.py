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
