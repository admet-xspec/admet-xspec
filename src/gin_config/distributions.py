"""
This module defines various probability s used for hyperparameter optimization.
It includes the following continuous distributions:
- Normal, LogNormal (mu, var)
- Uniform, LogUniform (min, max)
And one discrete distribution:
The s are defined using the `scipy.stats.rv_continuous` superclass.
"""

from scipy.stats import rv_continuous, rv_discrete
import numpy as np
import gin


@gin.configurable
class LogUniform(rv_continuous):
    def __init__(self, min=0, max=1):
        super().__init__()
        self.lower = min
        self.upper = max

    def _pdf(self, x):
        if self.lower <= x <= self.upper:
            return 1 / (x * np.log(self.upper / self.lower))
        else:
            return 0

    def __str__(self):
        return f"LogUniform(lower={self.lower}, upper={self.upper})"


@gin.configurable
class UniformDiscrete(rv_discrete):
    def __init__(self, min=0, max=1):
        super().__init__()
        self.lower = min
        self.upper = max

    def _pmf(self, x):
        if self.lower <= x <= self.upper:
            return 1 / (self.upper - self.lower + 1)
        else:
            return 0

    def __str__(self):
        return f"UniformDiscrete(lower={self.lower}, upper={self.upper})"


@gin.configurable
class LogUniformDiscrete(rv_discrete):
    def __init__(self, min=0, max=1):
        super().__init__()
        self.lower = min
        self.upper = max

    def _pmf(self, x):
        if self.lower <= x <= self.upper:
            return 1 / (x * np.log(self.upper / self.lower))
        else:
            return 0

    def __str__(self):
        return f"UniformLogDiscrete(lower={self.lower}, upper={self.upper})"
