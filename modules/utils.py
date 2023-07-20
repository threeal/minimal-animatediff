import os

def get_model_path(model_name: str):
    cache_path = os.environ.get('ANIMATEDIFF_MODELS_CACHE')
    models_path = cache_path if cache_path is not None else 'models'
    return os.path.join(models_path, model_name)
