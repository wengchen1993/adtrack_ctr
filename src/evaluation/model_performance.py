from src.evaluation.metrics import get_feval


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
