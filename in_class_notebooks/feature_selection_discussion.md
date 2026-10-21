# Additional Discussion About Feature Selection

## Document Purpose
This .md file is intended to be a companion to the following in-class notebook: ```feat_selection_pt3_cross_validation.ipynb```
   
This document will provide further clarification around feature selection with categorical variables. I glossed over this topic when walking through the notebook in class, and wanted to take some time to provide more context and nuance. 

Please note also that this .md file is talking specifically about techniques for regression. You can and will likely revisit this topic when you begin to learn about trees and ensemble methods. 

I've added a note at the top of the notebook as well, referencing this markdown file. 

## Feature Selection and Cross-Validation: What the first notebook gets right

The first notebook shows a number of implementations of techniques we discussed in class. The following take home messages from that notebook are correct and don't need revision:

* Using cross-validation gives a better estimate of out-of-sample RMSE than a single train/test split. Everything we covered in Part 1 is correct with no change. 
* Forward Selection, Backward Selection and Recursive Feature Elimination (RFE) will often select different features from the same data set for a given value of k. (k = number of features to include in the model) 
* Forward Selection and Backward Selection, as implemented in sklearn via the SequentialFeatureSelector method, use cross-validation to determine the best out-of-sample performance at a given value of k. RFE does not do this, and instead trains a model and drops the feature with the smallest coefficient. This is why feature scaling is very important when doing RFE. 
* If you don't remove features with multicollinearity, all of these feature selection techniques can select a set of features that includes two or more multicollinear variables, and the presence of multicollinear variables can impact the other features that are selected. If all you care about is prediction, this might be ok, but if you care about interpretability, removing these features a priori can be a good idea, especially if you have limited compute resources and a large dataset. 

**Note:** Although we did not discuss explicitly when we first reviewed this notebook, you can also use regularization as a feature selection technique. We will discuss this more in class, but for purposes of this notebook, we will ignore that as an option for now. 

## Feature Selection and Cross-Validation: Where the first notebook requires more nuance

There are two main areas that were a bit misleading in the original notebook, and so I wanted to provide some additional discussion to help clarify. 

* Whether or not to scale categorical variables
* How to include categorical variables as part of feature selection. 

### Scaled Categorical Variables

First: In [the original notebook](feat_selection_pt3_cross_validation.ipynb), I scaled both categorical and numeric variables using the following code:

```python
preprocess = make_column_transformer(
    (OneHotEncoder(drop="first", sparse_output=False), cat),
    remainder="passthrough",
    verbose_feature_names_out=False,
)

forward = make_pipeline(
    preprocess, StandardScaler(),
    SequentialFeatureSelector(LinearRegression(), n_features_to_select=6,
                              direction="forward", cv=cv,
                              scoring="neg_root_mean_squared_error"),
    LinearRegression())
forward.fit(X, y)
```

As a point of comparison, if I wanted to repeat this and scale the numeric variables only, I would write the code like this:

```python
preprocess = make_column_transformer(
    (OneHotEncoder(drop="first", sparse_output=False), cat),
    (StandardScaler(), num),
    verbose_feature_names_out=False,
)

forward = make_pipeline(
    preprocess, SequentialFeatureSelector(LinearRegression(), n_features_to_select=6,
                                          direction="forward", cv=cv,
                                          scoring="neg_root_mean_squared_error"),
    LinearRegression())
forward.fit(X, y)
```

The first block of code does one hot encoding on the categorical variables, which is fine, but when it creates the pipeline, it applies the StandardScaler to all features, numeric and categorical alike. 

Why does this matter? Well, it depends who you ask and what method you are using. 

Scaling doesn't really matter for Forward and Backward Selection, since features are added or removed based on predictive accuracy (via AIC, BIC, or cross-validation) and the fitted values don't change. 

In the in-class notebook, one of the methods that we explored was recursive feature elimination. This method works by ranking the coefficients by magnitude, and dropping the ones with the lowest values. Whether or not we scale the categorical variables will impact which features are chosen. In fact, if we rerun that notebook and scale the numeric variables only, we get completely different results! 

As another point of comparison, when we learned about regularization in Week 8, the dataset demonstrated in class had a categorical variable (sex) that was scaled so the values were not 0 and 1. In addition, the [glmnet library](https://glmnet.stanford.edu/reference/glmnet.html) that was used scales all variables by default. Standardization is normally done in regularization methods because the penalty depends on coefficient magnitudes, and without standardization, features on different scales are penalized unequally, which can impact both the choice of lambda and which features are selected. 

While scaling features might make sense for determining which features to include in the model, it can make interpretability a challenge. One way to approach this is by scaling regression inputs by two standard deviations, as [proposed by Gelman](https://sites.stat.columbia.edu/gelman/research/unpublished/standardizing.pdf), and leaving the categorical variables alone. If you care only about prediction and not inference, this might not matter as much. 

So what should you do? As with everything, it depends. A very reasonable approach is to scale categorical variables when doing feature selection, particularly with regularization, and take care when interpreting the results. There is some good discussion about this online, links below:

* Paper by Tibshirani on [variable selection in the Cox model](https://www.ms.uky.edu/~mai/sta635/CoxLASSO.pdf). The discussion section has a particularly relevant paragraph:
    * The lasso method requires initial standardization of the regressors, so that the penalization
scheme is fair to all regressors. For categorical regressors, one codes the regressor with dummy
variables and then standardizes the dummy variables. As pointed out by a referee, however, the
relative scaling between continuous and categorical variables in this scheme can be somewhat
arbitrary.
* [Stats Exchange Post #1](https://stats.stackexchange.com/questions/359015/ridge-lasso-standardization-of-dummy-indicators) The most upvoted answer has a nice summary of considerations. 
* [Stats Exchange Post #2](https://stats.stackexchange.com/questions/69568/whether-to-rescale-indicator-binary-dummy-predictors-for-lasso)


### Including Categorical Variables in Feature Selection

The bigger question is how to include categorical variables in feature selection methodologies in the first place. After OneHotEncoding, whether or not we scale the variables, each encoded column is included independently in the feature selection process. This is true for Forward Selection, Backward Selection, RFE, RFECV, and Lasso. 

Consider the following regarding independent inclusion: In the bike data example, there are four seasons, and one hot encoding categorizes them as ```season_2, season_3, season_4``` That is all well and good, but what does it mean if one of the feature selection algorithms removes season_3, but keeps season_2 and season_4. How do we interpret that, knowing that season_1 (the reference level, which was dropped), also doesn't show up? Effectively, season 3 is merged with the reference level in this instance, and the choice of reference level is often arbitrary and can change the results. 

We definitely did NOT take this into consideration in our in class notebook. We just left the variables as they were. While mathematically this is possible, the features we selected would be a real challenge to interpret. 

How do we solve this problem? One way is to keep or reject variables as a group. In the example above, this means if we kept ```season_2``` we would be required to keep ```season_3``` and ```season_4```. This addresses interpretability challenges introduced by only selecting one or two of the three variables. 

There is a method called [Group Lasso](https://academic.oup.com/jrsssb/article/68/1/49/7110631?login=false) that does exactly this. It is implemented in a Python library called [skglm](https://contrib.scikit-learn.org/skglm/generated/skglm.GroupLasso.html). Sebastian Raschka has written a library called [mlxtend](https://rasbt.github.io/mlxtend/api_subpackages/mlxtend.feature_selection/#sequentialfeatureselector) that allows feature groups as well. It is possible to do this in sklearn, but it would require some custom coding that is beyond the scope of this course. 

## Summary

There are two considerations for including categorical variables in feature selection: scaling, and keeping or rejecting them as a group. As you can see from the above discussion, determining what to do is not always simple. 

Going forward in class, I will try to call this out more precisely when it occurs, or choose examples where we don't have to worry about it. 

