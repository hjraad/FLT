from .extractor_models import ConvAutoencoder, ConvAutoencoderCIFARResidual
from .nets import CNNCifar, CNNMnist, CNNLeaf

def get_model(name: str):
    
    available_models = {
        'convae': ConvAutoencoder,
        'convaeres': ConvAutoencoderCIFARResidual,
        'cnncifar': CNNCifar,
        'cnnmnist': CNNMnist,
        'cnnleaf': CNNLeaf,
    }

    return available_models[name]