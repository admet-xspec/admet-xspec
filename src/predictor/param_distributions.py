"""
This module defines various probability distributions used for hyperparameter optimization.
It includes:
- NormalDistribution (mu, var)
- UniformDistribution (lower, upper)
- LogNormalDistribution (mu, var)
- LogUniformDistribution (lower, upper)
The distributions are defined using the `scipy.stats.rv_continuous` superclass.
"""

from scipy.stats import rv_continuous
import numpy as np
import gin

@gin.configurable
class NormalDistribution(rv_continuous):
    def __init__(self, mean=0, var=1):
        super().__init__()
        self.mean = mean
        self.var = var
    def _pdf(self, x):
        return (1 / (np.sqrt(2 * np.pi * self.var))) * \
               np.exp(-0.5 * ((x - self.mean) / self.var) ** 2)
    def __str__(self):
        return f"NormalDistribution(mean={self.mean}, var={self.var})"

@gin.configurable
class UniformDistribution(rv_continuous):
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
        return f"UniformDistribution(lower={self.lower}, upper={self.upper})"

@gin.configurable
class LogNormalDistribution(rv_continuous):
    def __init__(self, mean=0, var=1):
        super().__init__()
        self.mean = mean
        self.var = var

    def _pdf(self, x):
        return (1 / (x * np.sqrt(2 * np.pi * self.var))) * \
               np.exp(-0.5 * ((np.log(x) - self.mean) / self.var) ** 2)\

    def __str__(self):
        return f"LogNormalDitribution(mean={self.mean}, var={self.var})"

@gin.configurable
class LogUniformDistribution(rv_continuous):
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
        return f"LogUniformDistribution(lower={self.lower}, upper={self.upper})"