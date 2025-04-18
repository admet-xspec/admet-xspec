from chemprop import nn, featurizers
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
