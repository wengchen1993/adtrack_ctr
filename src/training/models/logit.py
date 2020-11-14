from sklearn.linear_model import LogisticRegression


class Logit:
    def __init__(self, extra_grid_params={}):
        self.name = 'logit'
        self.model = LogisticRegression
        self.grid_params = {'clf__penalty': ['none', 'l2']}
        self.grid_params.update(extra_grid_params)

    def gen_model_grid_params(self):
        model_content = {
            'model': self.model,
            'params': self.grid_params
        }
        return model_content
