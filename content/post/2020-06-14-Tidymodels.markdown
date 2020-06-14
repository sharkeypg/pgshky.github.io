---
title: Machine Learning in R with Tidymodels
author: Paul Sharkey
date: '2020-06-14'
slug: machine-learning-tidymodels
categories: ['Tidymodels', 'Machine Learning', 'R']
description: 'A basic template for using Tidymodels for all your machine learning needs'
summary: 'A basic template for using Tidymodels for all your machine learning needs'
math: true
markup: mmark

---



# Background

There are few machine learning frameworks that compete with *scikit-learn*, but as a native R user by trade, I was keen to explore an approach for building a pipeline in R for the various stages of the machine working workflow. Luckily, around the same time, *Tidymodels* had just come into maturity, with a very nice online [vignette](https://www.tidymodels.org/) containing the nicest documentation a programmer could ask for. Also around the same time, I was working with a colleague to design an online course for introductory R users, and we decided that it would be nice to provide a gentle introduction to machine learning through the lense of Tidymodels. This post is adapted from that session, so forgive me if the mathematical and algorithmic detail is a little light. The aim of this post is to provide a basic template for Tidymodels use that can be adapted to more complex and challenging problems going forward.

# What is Tidymodels?

Tidymodels is a relatively new suite of packages for machine learning that follow the principles of the [Tidyverse](https://www.tidymodels.org/). Tidymodels has a *lot* of functionality but in this post we'll cover the main components of the machine learning workflow:

* Data preprocessing
* Training a model
* Evaluating model performance
* Model tuning

Tidymodels is an umbrella package, like Tidyverse, containing packages that contribute to individual stages of this process; for example, the `recipes` package is a neat mechanism for preprocessing data that can be loaded for isolated tasks and doesn't necessarily need to feed into training a model.

Here are the libraries this post will use:




```r
library(tidymodels)
library(data.table) #because fread() is the best!
library(tidyverse)
library(GGally)
```

We're going to use the [Wine Quality dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality) (because wine is great). Here we're trying to predict, given the chemical properties of wine, whether its quality will be good or bad. This is known as a *classification* problem, where we want to predict whether something falls into a group, as opposed to a *regression* problem, which trys to predict the value of a continuous numerical quantity, e.g. house prices.

The wine dataset is available from the UCI machine learning repository. I've made some alterations to the dataset to a) turn it into a classification problem and b) explore how we can handle not just numerical but also categorical features.


```r
red_url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
red_raw <- fread(red_url, header = TRUE, sep = ";") %>% 
  mutate(quality = ifelse(quality >= 6, 'good', 'bad'),
         alcohol = ifelse(alcohol >= 10, ifelse(alcohol > 12, 'high', 'medium'), 'low')) %>%
         mutate_if(is.character, as.factor)
```

The two classes are finely balanced, which is good!


```r
red_raw %>%
  count(quality) %>% 
  mutate(prop = n/sum(n))
```

```
## # A tibble: 2 x 3
##   quality     n  prop
##   <fct>   <int> <dbl>
## 1 bad       744 0.465
## 2 good      855 0.535
```

We can use the `glimpse()` function to see what our data looks like - all are numeric with the exception of the response variable and the level of alcohol content.


```r
glimpse(red_raw)
```

```
## Rows: 1,599
## Columns: 12
## $ `fixed acidity`        <dbl> 7.4, 7.8, 7.8, 11.2, 7.4, 7.4, 7.9, 7.3, 7.8, …
## $ `volatile acidity`     <dbl> 0.700, 0.880, 0.760, 0.280, 0.700, 0.660, 0.60…
## $ `citric acid`          <dbl> 0.00, 0.00, 0.04, 0.56, 0.00, 0.00, 0.06, 0.00…
## $ `residual sugar`       <dbl> 1.9, 2.6, 2.3, 1.9, 1.9, 1.8, 1.6, 1.2, 2.0, 6…
## $ chlorides              <dbl> 0.076, 0.098, 0.092, 0.075, 0.076, 0.075, 0.06…
## $ `free sulfur dioxide`  <dbl> 11, 25, 15, 17, 11, 13, 15, 15, 9, 17, 15, 17,…
## $ `total sulfur dioxide` <dbl> 34, 67, 54, 60, 34, 40, 59, 21, 18, 102, 65, 1…
## $ density                <dbl> 0.9978, 0.9968, 0.9970, 0.9980, 0.9978, 0.9978…
## $ pH                     <dbl> 3.51, 3.20, 3.26, 3.16, 3.51, 3.51, 3.30, 3.39…
## $ sulphates              <dbl> 0.56, 0.68, 0.65, 0.58, 0.56, 0.56, 0.46, 0.47…
## $ alcohol                <fct> low, low, low, low, low, low, low, medium, low…
## $ quality                <fct> bad, bad, bad, good, bad, bad, bad, good, good…
```

First we want to split our data into a training set and a test set. We will build our predictive model on the training set and use the test set to evaluate the model's performance. The `rsample` package is useful here and contains functions to make this process really simple. If two classes are imbalanced the `strata = ` argument makes sure that the dataset is split roughly evenly by class.



```r
set.seed(123)

#split data into 75% and 25% partitions
data_split <- initial_split(red_raw, prop = 3/4)

#gather training and test data
train_data <- training(data_split)
test_data <- testing(data_split)
```

# Exploratory analysis

The first part of any machine learning workflow should be to visualise your data. This is useful to identify:

* The distribution of each feature
* Relationships between your response variables and features
* Relationships between features


```r
ggpairs(train_data)  +
  theme(axis.text.x = element_blank(), axis.ticks = element_blank())
```

<img src="/post/2020-06-14-Tidymodels_files/figure-html/unnamed-chunk-6-1.png" width="1920" />

This reveals some interesting insights, such as:

* Bad wine is more likely to contain higher levels of sulfur dioxide and volatile acidity.
* Fixed acidity is strong positively correlated with density, for example.
* Many of the features are right-skewed.

# Preprocessing

A key component of a machine workflow is the *data preprocessing* step. This can include tasks like:

* Cleaning your data
* Feature engineering
* Transforming your data

The wine dataset is tidy and we're generally happy with the features that we have. However, there are a couple of things we could do to make our data more palatable to the machine learning algorithm we decide to use.

The first task is to consider the skewed nature of some of the predictor variables. Machine learning algorithms are thought to perform best when the input distribution is symmetric and unimodal.

The second task is to scale our data. The features in the wine dataset all vary in their ranges and units. Unfortunately many algorithms calculate distances between data points when optimising the model and predictions can be dominated by the features with the highest numerical distances. If we transform each feature to the same scale, distances can be measured on the same playing field. There are many types of scaling available, but here we'll use the standard scaler:

$$
x_{NEW} = \frac{x_{OLD} - \mu}{\sigma}
$$

The third task is to convert our features to numerical values that the algorithm requires to process. We have a categorical variable that represents high, medium or low alcohol content. To include this, we need to create some dummy variables that encode this information in our feature set in a format that our algorithm will recognise.

A *recipe* in `tidymodels` contains all the steps needed to transform our dataset before the model training step. 


```r
wine_recipe <-
  recipe(quality ~ ., data = train_data) %>% #specify your training set, response and predictor variables
  step_log(all_numeric(), -density, -pH, offset = 1) %>% #log transform with an offset
  step_normalize(all_numeric()) %>%  #normalise all numeric variables
  step_dummy(alcohol) #select column to create dummy variables
```

This recipe drops our `alcohol` variable and replaces it with two new variables, `alcohol_low` and `alcohol_medium`. The combinations below correspond to levels of low, medium and high alcohol respectively in a way that our algorithm will understand.

`prep()` calculates what operations need to be applied to the training set. `juice()` applies these operations to the training set and displays the transformed dataset.


```r
wine_recipe %>% 
  prep(training = train_data) %>% 
  juice(starts_with('alcohol')) %>%
  distinct()
```

```
## # A tibble: 3 x 2
##   alcohol_low alcohol_medium
##         <dbl>          <dbl>
## 1           1              0
## 2           0              1
## 3           0              0
```

The `recipes` package contains a *vast* range of step functions that you can use to preprocess your data, enabling you to do things like imputing missing data, type conversion and extracting principal components.

# Model Training

There are numerous classification models we could use here. The `parsnip` package features many models including decision trees, random forests and neural networks. Here we're going to use a simple logistic regression model. The probability of belonging to class $$ i \in \{\text{good}, \text{bad}\} $$ is given by

$$
\text{logit}(p_i) = \sum_{k=1}^{n} \beta_k x_k
$$

where $$ \beta_k $$ is the coefficient corresponding to feature $$ x_k $$. The logit function constrains $$ p_i $$ to fall within $$ [0,1] $$.

We define a model, first by specify the functional form of the model (see `parsnip` for a list of these). We also need a method of fitting the model, which is included as an 'engine'. The `glm` package is a standard tool for fitting logistic regression models, so we use this.


```r
log_reg_model <-
  logistic_reg() %>% 
  set_engine('glm')
```

It's quite simple, really!

We can now use the `workflow()` function to build a workflow that includes our preprocessing and training steps in one object. It gives a summary of what data transformations we have used, what modelling approach we have taken and what computational engine we have chosen.


```r
wine_workflow <-
  workflow() %>% 
  add_model(log_reg_model) %>% 
  add_recipe(wine_recipe)

wine_workflow
```

```
## ══ Workflow ═══════════════════════════════════════════════════════════════════════════════════
## Preprocessor: Recipe
## Model: logistic_reg()
## 
## ── Preprocessor ───────────────────────────────────────────────────────────────────────────────
## 3 Recipe Steps
## 
## ● step_log()
## ● step_normalize()
## ● step_dummy()
## 
## ── Model ──────────────────────────────────────────────────────────────────────────────────────
## Logistic Regression Model Specification (classification)
## 
## Computational engine: glm
```

We can now use the `fit()` function to fit the model on the training data. This gives us some summary information about the model, including the coefficients of the logistic regression. 



```r
wine_fit <- wine_workflow %>% fit(data = train_data)
wine_fit
```

```
## ══ Workflow [trained] ═════════════════════════════════════════════════════════════════════════
## Preprocessor: Recipe
## Model: logistic_reg()
## 
## ── Preprocessor ───────────────────────────────────────────────────────────────────────────────
## 3 Recipe Steps
## 
## ● step_log()
## ● step_normalize()
## ● step_dummy()
## 
## ── Model ──────────────────────────────────────────────────────────────────────────────────────
## 
## Call:  stats::glm(formula = formula, family = stats::binomial, data = data)
## 
## Coefficients:
##            (Intercept)         `fixed acidity`      `volatile acidity`  
##                1.91060                 0.52278                -0.62282  
##          `citric acid`        `residual sugar`               chlorides  
##               -0.32791                 0.06512                -0.17609  
##  `free sulfur dioxide`  `total sulfur dioxide`                 density  
##                0.35616                -0.54590                -0.35653  
##                     pH               sulphates             alcohol_low  
##                0.05225                 0.56132                -2.38205  
##         alcohol_medium  
##               -1.20433  
## 
## Degrees of Freedom: 1199 Total (i.e. Null);  1187 Residual
## Null Deviance:	    1652 
## Residual Deviance: 1231 	AIC: 1257
```

Higher values of the features that have positive coefficients are more likely to lead to good wines, whereas higher values of features that have negative coefficients are more likely to lead to bad wines. Here we can see that high levels of volatile acidity and free sulfur dioxide are more likely to lead to bad wines, something we pick up in our exploratory analysis. The shows the usefulness of the data visualisation step - it gives us a way to sense check our model.

# Evaluating our model

Most of the functions we need for model evaluation come from the `yardstick` package. We use two measures of performance:

* **accuracy** - the proportion of observations that were classed correctly
* **recall** - the proportion of good wines that were predicted correctly.

There are numerous metrics to evaluate a machine learning model based on different measures of performance; it's up to user to decide what is most important to them and to ensure that their model is optimised to their metric of choice.

But what dataset to we use to evaluate the model? If we evaluate on our training set, we run the risk that our model captures both the signal and the noise contained in this dataset and unables to generalise very well to data that hasn't been seen before. This is known as **overfitting**. Luckily, we have a test set that we split out at the beginning that hasn't been exposed to the model yet. We use this to evaluate the performance of the model.

In some scenarios, it might be difficult to separate out a test set because it might not be large enough to evaluate the model with a degree of confidence. In this case we would use the whole dataset to train the model and instead use a **cross-validation** strategy as a proxy. We won't cover that here but later in this post we explore how we might use cross-validation for tuning hyperparameters of more complex models.



```r
wine_test_pred <-
  predict(wine_fit, test_data) %>% 
  bind_cols(test_data %>%  select(quality))
```



```r
wine_test_pred %>%                
  recall(truth = quality, .pred_class)
```

```
## # A tibble: 1 x 3
##   .metric .estimator .estimate
##   <chr>   <chr>          <dbl>
## 1 recall  binary         0.665
```


```r
wine_test_pred %>% 
  accuracy(truth = quality, .pred_class)
```

```
## # A tibble: 1 x 3
##   .metric  .estimator .estimate
##   <chr>    <chr>          <dbl>
## 1 accuracy binary         0.714
```

So 71% of our wines were classes correctly by our model, and 66% of truly good wines were predicted to be good. That's not bad - let's try a more complicated model to get these metrics up.

# Tuning hyperparameters

Let's see if we can improve on this performance by using a different model. Let's instead use a random forest model. Without going into the mathematical detail, random forests are rule-based ensemble approaches that make predictions from multiple decision trees constructed in parallel. Decision trees are prone to overfitting but random forests by introducing an extra level of randomness to how trees are constructed.

We can fit random forests as easily as we did the logistic regression model. This time, there are some parameters that are key to determining the overall performance of the model, which we might like to optimise. These are:

* `mtry` - the number of predictors to be randomly sampled at each split
* `min_n` - the minimum number of data points at a node needed for a node to be split further.

There is another parameter, giving the number of trees, but we set that as default 500 for the purpose of this exercise.

The model is set up as before, except we use the function `tune()` as a placeholder for each of these parameters to show that these are going to be tuned.


```r
rf_model <-
  rand_forest(mode = 'classification', mtry = tune(), min_n= tune()) %>% 
  set_engine('randomForest')
```

We can update our earlier workflow quite easily using `update_model()`.


```r
wine_workflow <- wine_workflow %>% update_model(rf_model)
wine_workflow
```

```
## ══ Workflow ═══════════════════════════════════════════════════════════════════════════════════
## Preprocessor: Recipe
## Model: rand_forest()
## 
## ── Preprocessor ───────────────────────────────────────────────────────────────────────────────
## 3 Recipe Steps
## 
## ● step_log()
## ● step_normalize()
## ● step_dummy()
## 
## ── Model ──────────────────────────────────────────────────────────────────────────────────────
## Random Forest Model Specification (classification)
## 
## Main Arguments:
##   mtry = tune()
##   min_n = tune()
## 
## Computational engine: randomForest
```

So how do we tune the hyperparameters? We could evaluate the model for multiple combinations of hyperparameters. We can't use the test set because we using that to evaluate the overall model, but we can't use the training set either because evaluations aren't reliable and could ignore overfitting.

What we could do is to split the training set down further, say, into $$ k $$ folds. We could hold out one of these folds, train the model on the other $$ k-1 $$ folds, and evaluate the model on the other fold. To ensure that performance is not sensitive to this choice of fold, we evaluate the model $$ k $$ times, where each fold is left out once. We average the resulting metrics to get the overall score.

First, we define our folds.


```r
cell_folds <- vfold_cv(train_data,v = 5)
cell_folds
```

```
## #  5-fold cross-validation 
## # A tibble: 5 x 2
##   splits            id   
##   <named list>      <chr>
## 1 <split [960/240]> Fold1
## 2 <split [960/240]> Fold2
## 3 <split [960/240]> Fold3
## 4 <split [960/240]> Fold4
## 5 <split [960/240]> Fold5
```

There are many clever ways we can choose the set of hyperparameters to tune - here we can just choose an integer and the backend will choose a set automatically.


```r
set.seed(345)
wine_tune <- 
  wine_workflow %>% 
  tune_grid(cell_folds,
            grid = 50,
            control = control_grid(save_pred = TRUE),
            metrics = metric_set(accuracy,recall))
```

From this object we can extract the metrics we need for each combination of hyperparameters.


```r
wine_tune %>% collect_metrics()
```

```
## # A tibble: 98 x 7
##     mtry min_n .metric  .estimator  mean     n std_err
##    <int> <int> <chr>    <chr>      <dbl> <int>   <dbl>
##  1     1    24 accuracy binary     0.753     5 0.00705
##  2     1    24 recall   binary     0.698     5 0.00675
##  3     1    31 accuracy binary     0.747     5 0.00610
##  4     1    31 recall   binary     0.689     5 0.00603
##  5     2     2 accuracy binary     0.81      5 0.0114 
##  6     2     2 recall   binary     0.776     5 0.0186 
##  7     2    15 accuracy binary     0.779     5 0.00801
##  8     2    15 recall   binary     0.752     5 0.0139 
##  9     2    24 accuracy binary     0.775     5 0.00685
## 10     2    24 recall   binary     0.742     5 0.00819
## # … with 88 more rows
```

We can then plot our hyperparameters quite easily against the performance metrics we chose.


```r
autoplot(wine_tune)
```

<img src="/post/2020-06-14-Tidymodels_files/figure-html/unnamed-chunk-20-1.png" width="672" />


We see that low values of `\(min_n\)` are associated with better performance. `mtry` in contrast shows no obvious trend, but the best performance occurs when `mtry = 7`. We can use `select_best()` to extract the optimal hyperparameters as judged by our performance metric of choice.


```r
wine_best <- 
  wine_tune %>% 
  select_best(metric = "accuracy")
wine_best
```

```
## # A tibble: 1 x 2
##    mtry min_n
##   <int> <int>
## 1     7     4
```

# Final model

We go back and update our workflow with the optimal model that we have found from the above exercise. The nice function `last_fit()` carries out the final fit on our training set and evaluates the model on the test set. We just need to input the splitted data object we made at the beginning.


```r
final_model <- rand_forest(mode = 'classification', mtry = 7, min_n = 4) %>% 
  set_engine('randomForest')

final_workflow <- wine_workflow %>% update_model(final_model)

final_fit <- final_workflow %>% last_fit(data_split, metrics = metric_set(accuracy,recall))
```



```r
final_fit %>% collect_metrics()
```

```
## # A tibble: 2 x 3
##   .metric  .estimator .estimate
##   <chr>    <chr>          <dbl>
## 1 accuracy binary         0.782
## 2 recall   binary         0.759
```

We have now boosted our accuracy metric by 7 percentage points up to 79% and our recall metric up by 10 points to 76%, which suggests that our new and more complex model is performing very well indeed.

# Resources

Tidymodels provides an ideal starting point for R users familiar with the code structure of the Tidyverse who are eager to construct a machine learning workflow. It's comparable to `scikit-learn` when it comes to ease-of-use and flexibility, and is easily adaptable to more complex problems.

The official [website](https://www.tidymodels.org/) is probably the best place to go for further information. For those interested in a more mathematical treatment of basic machine learning principles, the [Introduction to Statistical Learning](http://faculty.marshall.usc.edu/gareth-james/ISL/) book is a great start.

