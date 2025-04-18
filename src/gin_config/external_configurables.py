from chemprop import nn, featurizers
from scipy import stats
import gin

# Chemprop external configurables

gin.external_configurable(nn.AtomMessagePassing, module="nn")
gin.external_configurable(nn.BondMessagePassing, module="nn")

gin.external_configurable(nn.MeanAggregation, module="nn")
gin.external_configurable(nn.SumAggregation, module="nn")
gin.external_configurable(nn.NormAggregation, module="nn")

gin.external_configurable(
    featurizers.SimpleMoleculeMolGraphFeaturizer, module="featurizers"
)

# Scipy and sklearn external configurables

gin.external_configurable(stats.uniform, module="stats")
gin.external_configurable(stats.norm, module="stats")
gin.external_configurable(stats.lognorm, module="stats")
