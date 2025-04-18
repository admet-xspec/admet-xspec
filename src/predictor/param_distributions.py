from scipy.stats import rv_continuous
import numpy as np
import gin

@gin.configurable
class NormalDistribution(rv_continuous):
    def _pdf(self, x, mean=0, var=1):
        return (1 / (var * (2 * np.pi) ** 0.5)) * \
               np.exp(-0.5 * ((x - mean) / var) ** 2)

@gin.configurable
class UniformDistribution(rv_continuous):
    def _pdf(self, x, lower=0, upper=1):
        if lower <= x <= upper:
            return 1 / (upper - lower)
        else:
            return 0

@gin.configurable
class LogNormalDistribution(rv_continuous):
    def _pdf(self, x, mean=0, var=1):
        return (1 / (x * var * (2 * np.pi) ** 0.5)) * \
               np.exp(-0.5 * ((np.log(x) - mean) / var) ** 2)

@gin.configurable
class LogUniformDistribution(rv_continuous):
    def _pdf(self, x, lower=0, upper=1):
        if lower <= x <= upper:
            return 1 / (upper - lower) * (1 / x)
        else:
            return 0