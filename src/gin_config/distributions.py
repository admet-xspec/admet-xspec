"""
This module defines various probability distributions used for hyperparameter optimization.
It includes the following distributions, all parametrized by a lower and upper bound (min and max):
- Uniform
- LogUniform
- QUniform (discrete uniform)
- QLogUniform (discrete log uniform)
The distributions are defined under the `scipy.stats.rv_continuous` or `scipy.stats.rv_discrete` superclass.
"""

from scipy.stats import rv_continuous, rv_discrete
import numpy as np
import ray.tune as tune
import gin


@gin.register
class Uniform(rv_continuous):
    def __init__(self, min=0, max=1):
        super().__init__()
        self.lower = min
        self.upper = max

    def _pdf(self, x):
        if self.lower <= x <= self.upper:
            return 1 / (self.upper - self.lower)
        else:
            return 0

    def __str__(self):
        return f"Uniform (lower={self.lower}, upper={self.upper})"

    def get_ray_distrib(self):
        """Get the corresponding Ray library distribution object - used by ChemProp models"""
        return tune.uniform(self.lower, self.upper)


@gin.register
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
        return f"LogUniform (lower={self.lower}, upper={self.upper})"

    def get_ray_distrib(self):
        """Get the corresponding Ray library distribution object - used by ChemProp models"""
        return tune.loguniform(self.lower, self.upper)


@gin.register
class QUniform(rv_continuous):
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
        return f"QUniform (lower={self.lower}, upper={self.upper})"

    def get_ray_distrib(self):
        """Get the corresponding Ray library distribution object - used by ChemProp models"""
        return tune.randint(self.lower, self.upper + 1)


@gin.register
class QLogUniform(rv_continuous):
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
        return f"QLogUniform (lower={self.lower}, upper={self.upper})"

    def get_ray_distrib(self):
        """Get the corresponding Ray library distribution object - used by ChemProp models"""
        return tune.qloguniform(self.lower, self.upper, q=1, base=10)
