from chemprop import nn, featurizers
import gin
import numpy as np
from optuna import samplers

gin.external_configurable(nn.AtomMessagePassing, module="nn")
gin.external_configurable(nn.BondMessagePassing, module="nn")

gin.external_configurable(nn.MeanAggregation, module="nn")
gin.external_configurable(nn.SumAggregation, module="nn")
gin.external_configurable(nn.NormAggregation, module="nn")

gin.external_configurable(
    featurizers.SimpleMoleculeMolGraphFeaturizer, module="featurizers"
)

gin.external_configurable(np.logspace, module="np")
gin.external_configurable(np.linspace, module="np")

gin.external_configurable(samplers.GridSampler, module="samplers")
gin.external_configurable(samplers.RandomSampler, module="samplers")
gin.external_configurable(samplers.GPSampler, module="samplers")
gin.external_configurable(samplers.TPESampler, module="samplers")
gin.external_configurable(samplers.NSGAIISampler, module="samplers")
