"""
This module defines various probability distributions used for hyperparameter optimization.
It includes the following distributions, all parametrized by a lower and upper bound (a and b):
- Uniform
- LogUniform
- QUniform (discrete uniform)
- QLogUniform (discrete log uniform)
"""

from scipy.stats import uniform, loguniform, rv_discrete
import numpy as np
import ray.tune as tune
import gin
import abc


class Distribution(abc.ABC):

    def __init__(self, min=0, max=1):
        super().__init__()
        self.lower = min
        self.upper = max
        self.distribution = self._init_distribution()

    def rvs(self, size=1, **kwargs):
        return self.distribution.rvs(size=size, **kwargs)

    @abc.abstractmethod
    def _init_distribution(self):
        """
        This method must be implemented by subclasses.
        It should return a scipy.stats distribution object.
        """
        pass


@gin.register
class Uniform(Distribution):

    def _init_distribution(self):
        return uniform(self.lower, self.upper)

    def __str__(self):
        return f"Uniform (lower={self.lower}, upper={self.upper})"

    def get_ray_distrib(self):
        """Get the corresponding Ray library distribution object - used by ChemProp models"""
        return tune.uniform(self.lower, self.upper)


@gin.register
class LogUniform(Distribution):

    def _init_distribution(self):
        return loguniform(self.lower, self.upper)

    def __str__(self):
        return f"LogUniform (lower={self.lower}, upper={self.upper})"

    def get_ray_distrib(self):
        """Get the corresponding Ray library distribution object - used by ChemProp models"""
        return tune.loguniform(self.lower, self.upper)


@gin.register
class QUniform(Distribution):

    def _init_distribution(self):
        return uniform(self.lower, self.upper)

    def rvs(self, size=1, **kwargs):
        samples = self.distribution.rvs(size=size, **kwargs)
        return [int(sample) for sample in samples] if len(samples) > 1 else int(samples)

    def __str__(self):
        return f"QUniform (lower={self.lower}, upper={self.upper})"

    def get_ray_distrib(self):
        """Get the corresponding Ray library distribution object - used by ChemProp models"""
        return tune.randint(self.lower, self.upper + 1)


@gin.register
class QLogUniform(Distribution):

    def _init_distribution(self):
        return loguniform(self.lower, self.upper)

    def rvs(self, size=1, **kwargs):
        samples = self.distribution.rvs(size=size, **kwargs)
        return [int(sample) for sample in samples] if len(samples) > 1 else int(samples)

    def __str__(self):
        return f"QLogUniform (lower={self.lower}, upper={self.upper})"

    def get_ray_distrib(self):
        """Get the corresponding Ray library distribution object - used by ChemProp models"""
        return tune.qloguniform(self.lower, self.upper, q=1, base=10)
