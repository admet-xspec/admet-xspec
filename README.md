# admet-prediction
ADMET prediction module for GenProSyn
  
Author: Hubert Rybka

## Training

Before working with this repository, please get a look at [gin-config docs](https://github.com/google/gin-config). 
All of the configuration in this project is done exclusively with this package.

If you want to train your own predictive model, identify the `configs/config.gin` file first.
This is the main configuration file for the training script. Take a look below - It is an example of one such file. It includes
settings and parameters used by the training script. The config file is well documented, and before we move on, I encourage
you to read and try to understand this short config file:

### config.gin

```
#   TRAINING SCRIPT GENERAL CONFIG FILE

#   Add paths to the .gin config files of a featurizer and a predictor (classifier or regressor)
    include 'configs/featurizers/ecfp.gin'
    include 'configs/classifiers/svm.gin'

#==========================================================================================================#
#   Initializing predictor and featurizer - do not modify
    train.featurizer = @featurizer/gin.singleton()
    train.predictor = @predictor/gin.singleton()
#==========================================================================================================#

#   Provide path to the dataset
    train.data_path = 'data/permeability/bbbp_pampa.csv'
    train.test_size = 0.2
    train.strafity_test=True

#   Provide a name for the model
    NAME = 'MODEL'

#   Name of the directory in which every trained model will get its own subdirectory
#   to store output files, metrics and trained weights in. Default: 'models'.
    MODELS_DIR = 'models'
```

### Workflow

Now, let's discuss the general strategy. Each of the classifiers, regressors and featurizers are implemented as classes
and have their own .gin config files in different `configs` subdirectories. The `configs/config.gin` file is designed in
such a way, that it only contains model-agnostic parameters and settings. All the model hyperparameters, as well as settings 
that influence the training process of only one specific model or a family of models are included in `configs/predictor`
subdirectory.

**Example:** 
If we would want to train a simple svm-based classifier and use ECFP4 fingerprint featurizer for data preparation, we 
should start by including the paths to the configs of both classes at the top of `configs/config.gin`.

    include 'configs/featurizers/ecfp.gin'
    include 'configs/classifiers/svm.gin'

Next, if we know our datasets well, we can take care of providing path to a dataset in .csv format. The dataset should
contain two columns: 'smiles', containing only SMILES strings, and and 'y', containing the labels (target values).

    train.data_path = 'data/permeability/bbbp_pampa.csv'
    train.test_size = 0.2
    train.strafity_test=True

All the saved states, metrics, logs and different output files will be saved in `MODEL_DIR/NAME` directory.
We can choose the name under which our model will be saved by modifying the NAME macro.

    NAME = 'MODEL'
    MODEL_DIR = 'models'

## configs/predictors/svm.gin

The configuration fil `svm.gin` e of a predictor (SvmClassifier) is presented below. It can be found in 
`configs/classifiers` directory. All the options relevent to SvmClassifier class are configurable here.

The structure of the config file allows us to start by listing all the metrics we would like to include in
the final evalluaton of our trained predicitive model. The config files themselves are well-documented and
hopefully require little-to-no addtional explainations.

```
    predictor/gin.singleton.constructor = @SvmClassifier
    #==========================================================================================================#
    #   Supported metrics are: "mean_squared_error", "r2_score", "roc_auc_score", "accuracy_score",
    #                          "f1_score", "precision_score", "recall_score"
    #==========================================================================================================#
    #   Define all the metrics used in final evaluation of the model:                   # list of strings
        ScikitPredictorBase.metrics = ['roc_auc_score', 'accuracy_score', 'precision_score', 'recall_score']
    
    #   Define the primary metric, one to be optimized during hyperparameter search:    # string
        ScikitPredictorBase.primary_metric = 'roc_auc_score'
    #==========================================================================================================#
    #   If the parameter below is set to false, the script will not execute hyperparameter optimization step.
    #   Instead, the model will be trained using fixed hyperparameters provided in params dict.
    
        ScikitPredictorBase.optimize_hyperparameters = False
    
        ScikitPredictorBase.params = {
            'C': 1,
            'kernel': 'rbf',
            'gamma': 'scale',
            }
    #==========================================================================================================#
    #   If optimize_hyperparameters is set to True, the training script will first optimize the values of
    #   hyperparameters using random search - CV strategy
    
        ScikitPredictorBase.optimization_iterations = 20    # maximum times the script is alowed to draw and
                                                            # evaluate a new set of hyperparameter values
    
        ScikitPredictorBase.n_jobs = 8                      # number of CPUs to use
        ScikitPredictorBase.n_folds = 5                     # number of cross-validation folds
    
    #   Dictionary of distributions to sample hyperparameter values from. To each of the models' hyperparameters
    #   either a discreete list or a continous distribution may be assigned. Below is a brief demonstration on
    #   how to configure continuous probability distributions for the hyperparameters.
    
    #   There are 2 continuous distributions you can use:
    #       - `Uniform`
    #       - `LogUniform`
    #   And 2 discrete distributions:
    #       - `UniformDiscrete`
    #       - `LogUniformDiscrete`
    #   The first two are used to sample continuous values, while the last two are used to sample discrete values. 
    #   All of those are parametrized by their min and max values.
    
    #   If you wanted to provide a LogNormal distribution for the parameter C, remember to put it in
    #   some scope (ex. @C/LogNormal, where C/ is the scope). That way, when setting C/LogUniform.min
    #   and C/LogUniform.max distribution parameters later on, those will only change for our C hyperparam
    #   distribution, and not for any other LogNormals used in this config. The @ is essential when passing
    #   the distribution itself (which is a class) to the dictionary, but not needed while setting attributes.
    
        ScikitPredictorBase.params_distribution = {
            'C': @C/LogUniform,
            'gamma': ['scale', 'auto'],
            'kernel': ['rbf']
            }
    
    #   Parametrizing the distribution functions:
    
            C/LogUniform.min = 0.1
            C/LogUniform.max = 1000
    
    #==========================================================================================================#```

Some important parameters are:

    ScikitPredictorBase.optimize_hyperparameters = False

If this parameter is set to False, the model will be trained with a set of fixed hyperparameters, which can
configured in a dictionary below.

    ScikitPredictorBase.params = {'C': 1, 'kernel': 'rbf', 'gamma': 'scale'}

If, however, we set `ScikitPredictorBase.optimize_hyperparameters = True`, the model will first be tuned using
`slearn.model_selection.RandomizedSearchCV` cross-validation randomized search strategy, and the best set of 
hyperparameters will be used to re-train the final model on the whole training set. The user can modify parameters
of this search with:
more common
    ScikitPredictorBase.optimization_iterations = 20        # max no. sets of hyperparameter values to be drawn
    ScikitPredictorBase.n_jobs = 8                          # no. of CPUs to use
    ScikitPredictorBase.n_folds = 5                         # no. of folds to employ in cross-validation
```

A somewhat more in-depth explanation should be given regarding the configuration of randomized search protocol in
model tuning. Again, the hyperparameters and their distributions are defined in a dictionary, such as this:

    ScikitPredictorBase.params_distribution = {
                'C': @C/LogUniform,
                'gamma': ['scale', 'auto'],
                'kernel': ['rbf']
                }

- Passing a python list to one of the hyperparameter keys is equivalent to passing a discrete distribution to it.
- If we wanted intead to assign a continuous distribution to one (or a few) of the hyperparameters, some basic understanding 
of gin-config would be of great help. Currently, there are two continuous and two discrete distributions at our disposal. 
Those are implemented as class wrappers for well-known `scipy.stat` functions and can be found in `src/gin_config/distributions.py'

**Example:** Let's take a look on how we would defne a distribution of parameters to tune a feed-forward neural network.
Imagine, that the param dict for this hypotetical model looks like this:

    FFN.params_distribution = {
                'layer_1_size': ?
                'layer_2_size': ?
                'layer_3_size': ?
                'dropout_rate': ?
                'learning rate': ?
                }

Let's say that for our for `layer_1_size`, `layer_2_size` and `layer_3_size` hyperparameters we want to sample *integers*, 
from a discrete LogUniform distribution, for all layer size hyperparameters. We cannot just pass the same
distribution class to all three of them. **As gin-config works on the principle of replacing default arguments in classes'
constructors with the ones provided in config file**, if we would pass the same distribution class to all three of them,
we wouldn't be able to parametrize them separately. 

The solution is to use **scopes**. Scopes are a gin-config feature that allows us to create a unique instance of a class
for each scope. Scopes are denoted with `/`, so `@C/LogUniform` that we saw in previous example is an object of class
`LogUniform`, in scope `C/`, which was chosen as it was the C hyperparameter that we wanted to tune. 

Now, If we were interested in creating a separate LogUniformDiscrete object for each of the three layer sizes, we can 
approach it like this:

    FFN.params_distribution = {in
                    'layer_1_size': @layer_1_size/LogUniform
                    'layer_2_size': @layer_2_size/LogUniform
                    'layer_3_size': @layer_3_size/LogUniform
                    'dropout_rate': ?
                    'learning rate': ?
                    }

Now, we can set the parameters for each distribution class separately, using the scope name as a prefix:

    layer_1_size/LogUniform.min = 256
    layer_1_size/LogUniform.max = 2048

    layer_2_size/LogUniform.min = 128
    layer_2_size/LogUniform.max = 512

    layer_3_size/LogUniform.min = 16
    layer_3_size/LogUniform.max = 64

For the dropout rate, which is a float in range, we can use a continuous distribution, such as `Uniform` or `LogUniform`, 
and parametrize it in the same way. Learning rate is also a float, so we can use `Uniform` here as well.

    dropout_rate/Uniform.min = 0.1
    dropout_rate/Uniform.max = 0.5
    learning_rate/Uniform.min = 0.0005
    learning_rate/Uniform.max = 0.02

