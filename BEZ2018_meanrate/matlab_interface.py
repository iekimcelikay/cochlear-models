# matlab_interface.py
import numpy as np
import matlab.engine

def setup_matlab_engine(path):
    eng = matlab.engine.start_matlab("-desktop")
    model_path = eng.genpath(path)
    eng.addpath(model_path, nargout=0)
    return eng

def matlab_double_to_np_array(matlab_double_obj):
	np_array = np.array(matlab_double_obj).squeeze()
	return np_array

def ensure_matlab_row_vector(arr):
    if isinstance(arr, matlab.double):
        # Check if it's a column vector and transpose it
        if len(arr) > 0 and isinstance(arr[0], list) and len(arr[0]) == 1:
            return matlab.double([[x[0] for x in arr]])  # flatten column to row
        return arr
    elif isinstance(arr, np.ndarray):
        return matlab.double([arr.tolist()])
    elif isinstance(arr, list):
        return matlab.double([arr])
    else:
        raise TypeError(f"Unsupported vihc type: {type(arr)}")