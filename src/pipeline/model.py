from src.training.models.logit import Logit

available_models = {
    'logit': Logit
}


def get_model(model_name, extra_params={}):

    if model_name in available_models:
        model_ref = available_models[model_name](extra_params).gen_model_grid_params()
        return model_ref
    else:
        raise ValueError(f"{model_name} is not available. Choices available are {list(available_models.keys())}")
