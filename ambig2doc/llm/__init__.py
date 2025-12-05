"""
LLM pipeline module for different model configurations.
"""
import importlib


AVAILABLE_MODELS = {
    "llama3.1-8b": "llm.llama3_1_8b_instruct",
    "llama3.3-70b-fp8": "llm.llama3_3_70b_instruct_fp8",
}


def get_llm(model_name):
    """
    Load and return a pipeline for the specified model.
    
    Args:
        model_name: Name of the model (e.g., 'llama3.1-8b', 'llama3.3-70b-fp8')
    
    Returns:
        Tuple of (pipeline, model_identifier)
    
    Raises:
        ValueError: If model_name is not recognized
    """
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(
            f"Unknown model '{model_name}'. Available models: {', '.join(AVAILABLE_MODELS.keys())}"
        )
    
    module_path = AVAILABLE_MODELS[model_name]
    module = importlib.import_module(module_path)
    
    llm = module.create_llm()
    model_id = module.get_model_name()
    
    return llm, model_id


def list_available_models():
    """Return a list of available model names."""
    return list(AVAILABLE_MODELS.keys())
