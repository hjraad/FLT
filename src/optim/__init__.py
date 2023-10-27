from .flt import FLT

def get_method(name: str):
    
    available_methods = {
        'flt': FLT,
    }

    return available_methods[name]