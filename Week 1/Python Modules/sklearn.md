## Installing Scikit Learn
```
pip isntall scikit-learn
```
## Importing SciKit Learn
```
import sklearn  # General import (rarely used alone)

# Import a classifier
from sklearn.linear_model import LogisticRegression

# For data splitting
from sklearn.model_selection import train_test_split

# For preprocessing
from sklearn.preprocessing import StandardScaler

# For metrics
from sklearn.metrics import accuracy_score

```
## Video Recommendation
[Watch this video till the metrics part start(Highly Recommended)](https://youtu.be/0B5eIE_1vpU?si=05uDADMHLsuFzWHH)
```
## Basic SciKit Learn Example
# Step 1: Import necessary modules
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Step 2: Sample data (feature and target)
X = [[1], [2], [3], [4], [5]]      # Features (independent variable)
y = [2, 4, 6, 8, 10]               # Target (dependent variable)

# Step 3: Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Create and train (fit) the model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Make predictions
predictions = model.predict(X_test)

# Step 6: Evaluate the model
mse = mean_squared_error(y_test, predictions)
print("Predictions:", predictions)
print("Mean Squared Error:", mse)
```
### Output
Predictions: [10.]<br>
Mean Squared Error: 0.0

## Explanation
 ### 1. Importing the required modules
```
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
```
- `LinearRegression`: The machine learning model we’ll use to predict a continuous value.


- `train_test_split`: Function to divide your data into training and testing sets.

- `mean_squared_error`
: A metric to measure how good the model’s predictions are
### 2. Sample Data
```
X = [[1], [2], [3], [4], [5]]  # Independent variable (features)
y = [2, 4, 6, 8, 10]       # Dependent variable (target)
```
- `X` contains your input features. Each inner list is one sample (e.g., [[1]], [[2]], etc.).

- `y` is the output or label you want the model to learn (in this case, it’s simply 2 times the input).

 ### 3. Splitting the dataset
```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
- `Divides the data`: 80% for training and 20% for testing.

- `random_state=42` ensures the same split every time (for consistency).

- `X_train, y_train`: Used to train the model.

- `X_test, y_test`: Used to test the model after training.

### 4. Creating and Training the Model
```
model = LinearRegression()
model.fit(X_train, y_train)
```
- `model = LinearRegression()` creates a Linear Regression model object.

- `.fit(X_train, y_train)` trains the model using the training data so it learns the relationship between X and y.

### 5. Making Predictions
```
predictions = model.predict(X_test)
```
- `.predict(X_test)` uses the trained model to predict values for the unseen test data.

### 6. Evaluating The Model
```
mse = mean_squared_error(y_test, predictions)
print("Predictions:", predictions)
print("Mean Squared Error:", mse)
```
- `mean_squared_error()` calculates the average squared difference between predicted and actual values.

- A `lower MSE` means the model is performing well.

# Preprocessing
Preprocessing in Scikit-learn involves transforming raw data into a format suitable for machine learning models. It helps improve model accuracy and performance.

## Feature Scaling
Feature Scaling is the process of normalizing or standardizing the range of independent variables (features). It ensures that no feature dominates others due to its scale.

### 1. Standardization (Z-score scaling)
Formula:

$$
z = \frac{x - \mu}{\sigma}
$$

​
 
`Result`: Mean = 0, Std = 1

### Tool:
```
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```
### 2. Min-Max Scaling
Formula:
$$
x_{\text{scaled}} = \frac{x - x_{\text{min}}}{x_{\text{max}} - x_{\text{min}}}
$$



 
`Result`: Scales features to a [0, 1] range.

### Tool:

```
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
```

## Quantile Transformer
Quantile Transformer is used to transform features so that they follow a uniform or normal (Gaussian) distribution. It works by ranking the data and mapping it to a specified distribution.

- Ranks the data.

- Maps the ranks to a desired distribution:

  - Uniform (default)

   - Normal (useful for linear models)

```
from sklearn.preprocessing import QuantileTransformer
import numpy as np

# Example data
X = np.array([[1.0], [2.0], [2.5], [3.0], [5.0]])

# Create the transformer
qt = QuantileTransformer(output_distribution='normal')  # or 'uniform'

# Fit and transform
X_trans = qt.fit_transform(X)

print(X_trans)
```
Parameters
- `output_distribution`=`'uniform'` (default) or `'normal'`

- `n_quantiles`: Number of quantiles to compute (controls granularity and Smoothness of Data)

## One Hot Encoder
One Hot Encoding is a technique to convert categorical variables (like colors, labels) into a binary numeric format suitable for machine learning models.

### Why use One Hot Encoding?
- Many ML algorithms require numerical input.

- It prevents the model from assuming any order or priority in categorical values.

- Converts each category into a separate binary (0/1) feature.

### Example
```
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Sample categorical data
data = np.array([['red'], ['blue'], ['green'], ['blue']])

# Create OneHotEncoder object
encoder = OneHotEncoder(sparse=False)  # sparse=False returns a dense array

# Fit and transform the data
encoded_data = encoder.fit_transform(data)

print(encoded_data)
```
### Output
```
[[0. 0. 1.]   # 'red'
 [1. 0. 0.]   # 'blue'
 [0. 1. 0.]   # 'green'
 [1. 0. 0.]]  # 'blue'
```
## Polynomial Features
Polynomial Features allow machine learning models to capture non-linear relationships by adding polynomial terms and interaction terms to the original features.
Ex:
 $$
x^2,\ x^3,\ x_1.x_2$$ 

```
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

# Sample input with two features
X = np.array([[2, 3]])

# Create polynomial features of degree 2
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

print(X_poly)
```
### Output:
[[2. 3. 4. 6. 9.]]

## Pipeline in SciKit Learn
A Pipeline in scikit-learn lets you chain multiple processing steps together (like preprocessing + modeling) into one streamlined object.

### Why Use a Pipeline?
- Clean and readable code
- Prevents data leakage

- Easily reproduce and tune the entire process

- Useful with cross-validation and grid search
```
from sklearn.pipeline import Pipeline

pipe = Pipeline(
    steps=[
        ('step1_name', transformer1),
        ('step2_name', transformer2),
        # ...
        ('model_name', estimator)
    ],
    memory=None,          # Optional: caching (e.g., for grid search)
    verbose=False         # Optional: set True to show logs during fitting
)

```
Example:
```
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Create a pipeline with scaling and logistic regression
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])

# Example data
X = [[1, 2], [3, 4], [5, 6]]
y = [0, 1, 0]

# Fit the pipeline
pipe.fit(X, y)

# Predict
predictions = pipe.predict([[2, 3]])
print(predictions)
```

##  GridSearchCV
`GridSearchCV` stands for Grid Search with Cross-Validation.

It helps you:

- Find the best combination of hyperparameters

- Automatically test all combinations of parameters you define

- Evaluate each using cross-validation

### How It Works
- Define a model

- Set up a dictionary of hyperparameters to try

- Use cross-validation to evaluate each combination

- Choose the best one based on a scoring metric (e.g., accuracy)

``` 
from sklearn.model_selection import GridSearchCV

grid_search = GridSearchCV(
    estimator,            # model (e.g., SVC(), RandomForestClassifier(), etc.)
    param_grid,           # dict or list of dicts of hyperparameters
    scoring=None,         # metric to optimize (e.g., 'accuracy', 'neg_mean_squared_error')
    n_jobs=None,          # number of jobs to run in parallel (-1 = use all CPUs)
    iid='deprecated',     # ignored (was used in older versions)
    refit=True,           # refit the best model on the full dataset
    cv=None,              # number of cross-validation folds or CV splitter (e.g., 5 or StratifiedKFold)
    verbose=0,            # control output (0 = silent, higher = more output)
    pre_dispatch='2*n_jobs', # controls how many jobs are dispatched during parallel execution
    error_score=np.nan,   # value to assign if an error occurs in fitting
    return_train_score=False # include training scores in cv_results_
)
```

- `grid.best_params_`: Best hyperparameter combination

- `grid.best_estimator_`: Best model with those parameters

- `grid.best_score_`: Best cross-validation score

- `grid.cv_results_`: Full result dictionary

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
```
# Model
model = SVC()

# Parameter grid to search
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf']
}

# Grid search with 5-fold cross-validation
grid = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')

# Fit on data
grid.fit(X_train, y_train)

# Best model
print("Best parameters:", grid.best_params_)
print("Best score:", grid.best_score_)
```
### Output
```
Best parameters: {'C': 1, 'kernel': 'rbf'}
Best score: 0.92
```
