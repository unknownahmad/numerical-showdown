import numpy as np

def parse_function(func_str):
    safe_dict = {
        'x': None,
        'sin': np.sin,
        'cos': np.cos,
        'tan': np.tan,
        'exp': np.exp,
        'log': np.log,
        'sqrt': np.sqrt,
        'pi': np.pi,
        'e': np.e
    }
    
    return lambda x_val: eval(func_str, {"__builtins__": None}, {**safe_dict, 'x': x_val})

def get_derivative(f, x, h=1e-7):
    return (f(x + h) - f(x - h)) / (2 * h)