class ModelBuilderRegistry:
    """Central registry for all model name builders."""

    def __init__(self):
        self._builders = {}

    def register(self, model_name):
        """Decorator to register a model name builder function."""
        def decorator(builder_func):
            self._builders[model_name] = builder_func
            return builder_func
        return decorator
    def get(self, model_name):
        """Retrieve a registered model name builder function."""
        if model_name not in self._builders:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(self._builders.keys())}")
        return self._builders[model_name]
    
    def list_models(self):
        """List all registered model names."""
        return list(self._builders.keys())


# Global registry instance
model_builders = ModelBuilderRegistry()

# Register builders with decarator

@model_builders.register("bez2018")
def bez2018_name_builder(params, timestamp):
    lsr, msr, hsr = params['num_ANF']
    num_runs = params['num_runs']
    num_cf = params['num_cf']
    min_cf = params['min_cf']
    max_cf = params['max_cf']
    return (f"bez2018_psth_batch_{num_runs}runs_{num_cf}cfs_{min_cf}-{max_cf}Hz_{lsr}-{msr}-{hsr}fibers_{timestamp}")

@model_builders.register("cochlea_zilany2014")
def cochlea_zilany2014_name_builder(params, timestamp):
    num_runs = params['num_runs']
    num_cf = params['num_cf']
    min_cf = params['min_cf']
    max_cf = params['max_cf']
    return (f"cochlea_zilany2014_psth_batch_{num_runs}runs_{num_cf}cfs_{min_cf}-{max_cf}Hz_{timestamp}")        

@model_builders.register("wsr_model")
def wsr_model_name_builder(params, timestamp):
    num_channels = params['num_channels']
    frame_length = params['frame_length']
    time_constant = params['time_constant']
    factor = params['factor']
    shift = params['shift']
    return (f"wsr_model_psth_batch_{num_channels}chans_{frame_length}ms_{time_constant}tc_{factor}factor_{shift}shift_{timestamp}")
