# matlab_interface.py

import matlab.engine

def setup_matlab_engine(path):
    eng = matlab.engine.start_matlab("-desktop")
    model_path = eng.genpath(path)
    eng.addpath(model_path, nargout=0)
    return eng