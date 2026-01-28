import numpy as np
from typing import Dict
import logging

logger = logging.getLogger(__name__)

def calculate_population_rate(
        mean_rates: Dict[str, np.ndarray],
        hsr_weight: float = 0.63,
        msr_weight: float = 0.25,
        lsr_weight: float = 0.12
        ) -> np.ndarray:
    """Calculate weighted population rate across fiber types for each CF.

    Typical auditory nerve fiber proportions:
    - 63% High Spontaneous Rate (HSR)
    - 25% Medium Spontaneous Rate (MSR)
    - 12% Low Spontaneous Rate (LSR)

    Args:
        mean_rates: Dict[fiber_type, array(num_cf)] - mean rates per fiber type
        hsr_weight: Weight for HSR fibers (default: 0.63)
        msr_weight: Weight for MSR fibers (default: 0.25)
        lsr_weight: Weight for LSR fibers (default: 0.12)

    Returns:
        np.ndarray of shape (num_cf,) - weighted population rate per CF
    """
    weights = {'hsr': hsr_weight, 'msr': msr_weight, 'lsr': lsr_weight}

    # Validate weights sum to 1
    weight_sum = sum(weights.values())
    if not np.isclose(weight_sum, 1.0):
        logger.warning(f"Weights sum to {weight_sum:.3f}, normalizing to 1.0")
        weights = {k: v/weight_sum for k, v in weights.items()}

    # Calculate weighted sum
    population_rate = np.zeros_like(mean_rates['hsr'])

    for fiber_type, weight in weights.items():
        if fiber_type in mean_rates:
            population_rate += weight * mean_rates[fiber_type]
        else:
            logger.warning(f"Missing {fiber_type} in mean_rates")

    return population_rate