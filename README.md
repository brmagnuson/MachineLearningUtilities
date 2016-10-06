# Machine Learning Utilities
mlutilities is a Python 3.4 package that helps you explore and evaluate combinations of machine learning techniques, feature transformations, and datasets to find which will predict the best results.

## Dependencies

sklearn 0.16.1, matplotlib 0.17.0, pandas, numpy, and some base Python packages.

## Using mlutilities to experiment with machine learning approaches

If you'd like to run the below code snippets easily and see the results, check out the [demo folder](https://github.com/brmagnuson/MachineLearningUtilities/tree/master/demo).

Most mlutilities operations are built around a DataSet object, which is instantiated by giving it a description, pointing toward a .csv file, and specifying information about the location of the dependent variable/predictor variables columns if necessary. The code below shows a simple example of reading in a dataset and splitting it into a training set and a testing set.

```python
# Read data sets
myData = mlutilities.types.DataSet('My Training Data',
                                   pathToData)
splitData = mlutilities.dataTransformation.splitDataSet(myData,
                                                        testProportion=0.3,
                                                        randomSeed=89271)
trainingData = splitData.trainDataSet
testingData = splitData.testDataSet
```

Once a DataSet has been created, use the tuneModels function is used to tune one or more models for one or more training DataSets. It expects a list of DataSets and a list of TuneModelConfigurations, which contain the necessary information to tune each individual model. For example, the below code sets up two models, a random forest model with possible tree numbers of 50, 75, and 100 and a k-nearest neighbor model that uses either 2 or 5 nearest neighbors. The tuneModels function performs a default 5-fold cross-validation grid search for each TuneModelConfiguration across all the possible combinations of supplied parameters (using scikit-learn’s default values for unspecified parameters). By default, the internal cross-validation grid search is scored based on R-squared value. It returns a list of TuneModelResults, which contain the best set of model parameters for each model. 

```python
# Tune models for training data set
tuneScoringMethod = 'r2'

rfParameters = [{'n_estimators': [50, 75, 100]}]
rfMethod = mlutilities.types.ModellingMethod('Random Forest',
                                           sklearn.ensemble.RandomForestRegressor)
rfConfig = mlutilities.types.TuneModelConfiguration('Tune Random Forest',
                                                    rfMethod,
                                                    rfParameters,
                                                    tuneScoringMethod)

knnParameters = [{'n_neighbors': [2, 5]}]
knnMethod = mlutilities.types.ModellingMethod('K Nearest Neighbors',
                                            sklearn.neighbors.KNeighborsRegressor)
knnConfig = mlutilities.types.TuneModelConfiguration('Tune KNN',
                                                     knnMethod,
                                                     knnParameters,
                                                     tuneScoringMethod)

predictorConfigs = [rfConfig, knnConfig]
tunedModelResults = mlutilities.modeling.tuneModels([trainingData],
                                                    predictorConfigs)
```

Once the models have been tuned using training data, they can be applied to testing data using the applyModels function and their results scored on one or more performance metrics using the scoreModels function. I use R-squared and the mean observed/expected ratio value below as the performance metrics. Functions to evaluate the mean and standard deviation of the observed/expected ratio were created in mlutilities to supplement the existing scikit-learn performance metrics such as R-squared and mean squared error.

```python
# Apply tuned models to some test data
applyModelConfigs = []
for tunedModelResult in tunedModelResults:
    applyModelConfig = mlutilities.types.ApplyModelConfiguration(
                           tunedModelResult.description,
                           tunedModelResult.modellingMethod,
                           tunedModelResult.parameters,
                           trainingData,
                           testingData)
    applyModelConfigs.append(applyModelConfig)
applyModelResults = mlutilities.modeling.applyModels(applyModelConfigs)

# Score test results
r2Method = mlutilities.types.ModelScoreMethod('R Squared', 
                                              sklearn.metrics.r2_score)
meanOEMethod = mlutilities.types.ModelScoreMethod('Mean O/E', 
                                   mlutilities.modeling.meanObservedExpectedScore)
testScoringMethods = [r2Method, meanOEMethod]
testScoreModelResults = mlutilities.modeling.scoreModels(applyModelResults, 
                                                         testScoringMethods)
```

To easily display results, scoreModelResults can be converted to a pandas DataFrame, which can be printed out, written to a CSV file, or visualized.

```python
scoreModelResultsDF = mlutilities.utilities.createScoreDataFrame(
                                                            testScoreModelResults)
mlutilities.utilities.barChart(scoreModelResultsDF, 'R Squared', 
                'R Squared for Each Model', 'ExampleData/rSquared.png', '#2d974d')
```

The output of the above barChart function generates a simple chart with a bar for each model, numbered by its index in the results DataFrame.

![Example Random Forest vs. K-Nearest Neighbor R-squared](https://github.com/brmagnuson/MachineLearningUtilities/blob/master/demo/rSquared.png "Random Forest vs. K-Nearest Neighbor R-squared") 


mlutilities also allows for DataSets to be scaled from 0 to 1 before any models are used, as some machine learning algorithms perform better with numbers reduced to this range, and for other DataSets to be scaled using the same scaling process. The scaleDataSet and scaleDataSetByScaler functions are demonstrated below.

```python
# Scale data
scaledTrainingData, scaler = mlutilities.dataTransformation.scaleDataSet(
                                                                     trainingData)
scaledTestingData = mlutilities.dataTransformation.scaleDataSetByScaler(
                                                              testingData, scaler)
```

Likewise, mlutilities can be used to perform feature engineering (the set of possible variable selection and dimensionality reduction techniques) on a DataSet before a model is applied. This consists of either variable selection approaches, such as using a variance threshold technique to only keep predictor variables that have relatively high variance or specifying a list of predictor variables to keep, or variable extraction approaches, such as principle component analysis (PCA) or independent component analysis (ICA) to generate a reduced set of predictor variables through dimensionality reduction. The same feature engineering process can then be used to transform another DataSet as well. The below code demonstrates this process, using PCA to create new DataSets with 5 principle components.

```python
# Perform feature engineering
pcaConfig = mlutilities.types.FeatureEngineeringConfiguration('PCA n5',
                     'extraction', sklearn.decomposition.PCA, {'n_components': 5})

pcaTrainingData, transformer = mlutilities.dataTransformation.\
    engineerFeaturesForDataSet(trainingData, pcaConfig)
pcaTestingData = mlutilities.dataTransformation.engineerFeaturesByTransformer(
                                                         testingData, transformer)
```

mlutilities provides a way to average or stack models to test improvement due to model aggregation. An example of building a StackingEnsemble is shown below. The tunedModelResults for the random forest and k-nearest neighbors models are processed to extract the tuned model configurations. These are then used as the base predictors for the stacking ApplyModelConfiguration, with the random forest model arbitrarily chosen (by specifying predictorConfigs[0]) as the second-level model that predicts based on the base predictors’ results. (An AveragingEnsemble can also be created in a similar way. For that approach, rather than using a second-level model, the base predictors’ results are all averaged together for a final prediction. By default, a regular arithmetic mean is calculated, but if desired, the user can specify weights for each base predictor, which results in a weighted average instead.)

```python
# Create stacking ensemble
predictorConfigs = []
for tunedModelResult in tunedModelResults:
    predictorConfig = mlutilities.types.PredictorConfiguration(
                          tunedModelResult.modellingMethod.description,
                          tunedModelResult.modellingMethod.function,
                          tunedModelResult.parameters)
    predictorConfigs.append(predictorConfig)

stackMethod = mlutilities.types.ModellingMethod('Stacking Ensemble',
                                               mlutilities.types.StackingEnsemble)
stackParameters = {'basePredictorConfigurations': predictorConfigs,
                   'stackingPredictorConfiguration': predictorConfigs[0]}
stackApplyModelConfig = mlutilities.types.ApplyModelConfiguration(
    'Stacking Ensemble', stackMethod, stackParameters, trainingData, testingData)

stackResult = mlutilities.modeling.applyModel(stackApplyModelConfig)
```

Hope you enjoy!
