# 📈 Linear Regression from Scratch
This is a simple Python implementation of Linear Regression from scratch using NumPy. The results of the project can be seen below.

Given `X` (independent variable) and `Y` (dependent variable):

![data](https://github.com/raj-pulapakura/lin-reg-from-scratch/assets/87762282/95f99a07-088d-4cf1-acb4-abe8e671e719)

Linear Regression finds a 'best-fit' regression line to model the relationship between `X` and `Y`:

![data_with_regression_line](https://github.com/raj-pulapakura/lin-reg-from-scratch/assets/87762282/49145ebc-4c5b-43be-a5b2-4e0f02fd0268)

## 🤔 What is Linear Regression?
Linear Regression is a popular and widely used statistical method for modeling the relationship between a dependent variable (target) and one or more independent variables (features). It assumes that there is a linear relationship between the features and the target variable. The goal of Linear Regression is to find the best-fit line that represents this relationship and can be used to predict the target variable for new input data.

In simple terms, Linear Regression aims to find a straight line equation of the form:

```
y = mx + b
```

Where:

- `y` is the target variable (dependent variable).
- `x` is the input feature (independent variable).
- `m` is the slope of the line (coefficient).
- `b` is the y-intercept (bias).

The line's slope (`m`) represents the change in the target variable for a unit change in the input feature, and the y-intercept (`b`) represents the value of the target variable when the input feature is zero.

## 💪 How does Linear Regression work?

Linear Regression works in the following steps:

**1. Data Preparation**: We start by collecting and organizing our data. We need pairs of input features `X` and corresponding target values `y`.

**2. Initializing Coefficients**: We initialize the slope `m` and the y-intercept `b` with random or zero values. These will be the parameters of our linear model.

**3. Making Predictions**: Using the equation `y = mx + b`, we make predictions for each input feature in `X`. This gives us a set of predicted target values `y_pred`.

**4. Calculating the Error**: We calculate the error between the predicted target values `y_pred` and the actual target values `y`. A common way to measure error is using Mean Squared Error (MSE):

```
MSE = (1/n) * Σ(y - y_pred)^2
```

Where `n` is the number of data points.

**5. Updating Coefficients**: The goal of Linear Regression is to minimize the error (MSE). We do this by adjusting the coefficients `m` and `b` using an optimization technique known as Gradient Descent. It involves calulating the derivative of the loss with respect to each parameters of the model (`m` and `b`) and using these values to update the parameters. 

**6. Iterative Process**: Steps 3 to 5 are repeated iteratively until the error converges to a minimum.

**7. Making Predictions**: After finding the optimal coefficients `m` and `b`, we can use the equation `y = mx + b` to make predictions for new input data.
