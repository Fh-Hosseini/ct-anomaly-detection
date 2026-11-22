def get_reconstruction(model_output):
    """
    Unpack the model output to get the reconstructed volumes.
    If the model output is a tuple, return the first element (reconstructed volumes). Otherwise, return the model output as is.
    """
    if isinstance(model_output, tuple):
        return model_output[0]
    return model_output
