# admet-prediction
ADMET prediction module for GenProSyn

## Training

Before working with this repository, please take a look at [gin-config docs](https://github.com/google/gin-config). 
All of the configuration in this project is done exclusively with this package.

If you want to train your own predictive model, identify the `configs/train.gin` config file first.
This is the main configuration file for the training script.

### Workflow

Now, let's discuss the general strategy. Each of the classifiers, regressors and featurizers is implemented as a class
and has their own .gin config file somewhere in different `configs` subdirectories. The `configs/train.gin` file is constructed in
such that it only takes care of model-agnostic parameters and settings, as well as gathers (imports) other .gin files needed for 
configuration of different machine leraning models and data preparation protocols. All the model hyperparameters, as well as settings 
that influence the training process of only one specific model or a family of models are included in `configs/classifiers` and
`configs/regressors` subdirectories.

**Example:** 
If we would want to train a simple svm-based classifier, perform a random train-test split and use ECFP4 fingerprint featurizer for 
construction of the feature matix, we should start by including paths to the configs of the classes somewhere in `configs/train.gin`.

    include 'configs/data_splitters/random.gin'
    include 'configs/featurizers/ecfp.gin'
    include 'configs/classifiers/svm.gin'

Next we can take care of providing a path to the dataset in .csv format. The data file should contain at least two columns: 'smiles', 
containing only SMILES strings, and 'y', containing true labels (target values) as floats.

    TrainingPipeline.data_path = 'data/permeability/bbbp_pampa.csv'

All the saved states, metrics, logs and different output files will be saved in `out_dir/model_name` directory.
We can choose the name under which our model will be saved by modifying the `TrainingPipeline.model_name` parameter.

    TrainingPipeline.out_dir = 'models'
    TrainingPipeline.model_name = 'MyRandomForest'

## configs/predictors/svm.gin

An example configuration file `svm.gin` for a predictor class (SvmClassifier) is presented below. It can be found in 
`configs/classifiers` directory. All the options relevent to SvmClassifier class are to be configured here.

```
predictor/gin.singleton.constructor = @SvmClassifier
predictor = @predictor/gin.singleton()

#==========================================================================================================#
#   If the parameter below is set to false, the script will not execute hyperparameter optimization step.
#   Instead, the model will be trained using fixed hyperparameters provided in params dict.

    SvmClassifier.optimize_hyperparameters = True

    SvmClassifier.params = {
        'C': 3,
        'kernel': 'rbf',
        'gamma': 'scale',
        }

#==========================================================================================================#
#   Define the target metric, one to be optimized during hyperparameter search.
#   Supported metrics are: 'accuracy', 'roc_auc', 'f1', 'recision', 'recall'

    SvmClassifier.target_metric = 'roc_auc'

#==========================================================================================================#
#   If optimize_hyperparameters is set to True, the training script will first optimize the values of
#   hyperparameters using random search - CV strategy

    SvmClassifier.optimization_iterations = 50    # maximum times the script is allowed to draw and
                                                        # evaluate a new set of hyperparameter values

    SvmClassifier.n_jobs = 20                      # number of CPUs to use
    SvmClassifier.n_folds = 5                     # number of cross-validation folds

#   Dictionary of distributions to sample hyperparameter values from. To each of the models' hyperparameters
#   either a discrete list or a continuous distribution may be assigned. Below is a brief demonstration on
#   how to configure probability distributions for the hyperparameters.

#   There are four distributions you can use, all parametrized by min and max:
#       * Uniform       (continuous)
#       * LogUniform    (continuous)
#       * QUniform      (discrete)
#       * QLogUniform   (discrete)

#   If you wanted to provide a LogUniform distribution for the parameter C, remember to put it in
#   some scope (ex. @C/LogUniform, where C/ is the scope). That way, when setting C/LogUniform.min
#   and C/LogUniform.max distribution parameters later on, those will only change for our C hyperparam
#   distribution, and not for any other LogUniforms used in this config. The @ is essential when passing
#   the distribution itself (which is a class) to the dictionary, but not needed while setting attributes.

    SvmClassifier.params_distribution = {
        'C': @C/LogUniform(),
        'gamma': @gamma/LogUniform(),
        'kernel': ['rbf']
        }

#   Parametrizing the distribution functions:

        C/LogUniform.min = 0.1
        C/LogUniform.max = 1000

        gamma/LogUniform.min = 0.001
        gamma/LogUniform.max = 1

#==========================================================================================================#
```

Some important parameters are:

    SvmClassifier.optimize_hyperparameters = False

If this parameter is set to False, the model will be trained with a set of fixed hyperparameters, which can
configured in `params` dictionary.

    SvmClassifier.params = {'C': 1, 'kernel': 'rbf', 'gamma': 'scale'}

If, however, we set `SvmClassifier.optimize_hyperparameters = True`, the model will first be tuned using
k-fold cross-validation randomized search strategy, and the best set of hyperparameters will be used to re-train 
the final model on the whole training set. The user can modify parameters of hyperparameter search with:

    SvmClassifier.optimization_iterations = 20        # how many sets of hyperparameter values to be drawn
    SvmClassifier.n_jobs = 8                          # no. of CPUs to use
    SvmClassifier.n_folds = 5                         # no. of folds in cross-validation

### Randomized Hyperaparameter Tuning

A somewhat more in-depth explanation should be given regarding the configuration of randomized search protocol in
model tuning. Again, the hyperparameters and their distributions are defined in a dictionary, such as this:

    SvmClassifier.params_distribution = {
                'C': @C/LogUniform,
                'gamma': @gamma/LogUniform,
                'kernel': ['rbf']
                }

- Passing a python list to one of the hyperparameter keys is equivalent to passing a uniform discrete distribution to it.
- If we intead wanted to assign a continuous non-uniform distribution to one (or a few) of the hyperparameters, some basic understanding 
of gin-config is needed. Currently, there are two continuous and two discrete distributions at our disposal: `Uniform`, `LogUniform`, `QUniform`, `QLogUniform`,
each parametrized by `min` and `max` values.

The distributions are implemented as class wrappers for `scipy.stat` functions and can be found in `src/gin_config/distributions.py`.

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

