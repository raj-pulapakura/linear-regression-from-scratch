import numpy as np

class LinearRegression:
    """
    Linear Regression is a popular and widely used statistical method for modeling the relationship between a dependent variable (target) and one or more independent variables (features). It assumes that there is a linear relationship between the features and the target variable. The goal of Linear Regression is to find the best-fit line that represents this relationship and can be used to predict the target variable for new input data.
    """
    
    def __init__(self):
        self.coefficients = np.random.uniform()
        self.bias = np.random.uniform()

    def calculate_loss(self, y_pred: np.ndarray, y_true: np.ndarray):
        """
        Calculates the Mean Squared Error
        """
        return (y_pred - y_true) ** 2
    
    def calculate_loss_derivative(self, y_pred: np.ndarray, y_true: np.ndarray):
        """
        Calculates the derivative of the loss with respect to y_pred
        """
        return 2 * (y_pred - y_true)

    def forward(self, x: np.ndarray):
        """
        Calculates y_pred
        """
        return self.coefficients * x + self.bias

    def backward(self, x: np.ndarray, y_pred: np.ndarray, y_true: np.ndarray, learning_rate: float):
        """
        Calculates the partial derivatives of the loss with respect to the coefficient and bias, which are used to update the parameters.
        """        
        dloss_dypred = self.calculate_loss_derivative(y_pred, y_true)

        dloss_dc = np.average(dloss_dypred * x)
        dloss_db = np.average(dloss_dypred)

        self.coefficients -= learning_rate * dloss_dc
        self.bias -= learning_rate * dloss_db

    def fit(self, x: np.ndarray, y: np.ndarray, epochs: int = 10, learning_rate: float = 0.01, verbose: int = 1):
        """
        Runs forward and backward propagation for a certain number of epochs
        """
        for e in range(1, epochs+1):
            if verbose: print(f"Epoch: {e}")
            # calculate y_hat using the formula m*x+b
            y_pred = self.forward(x)
            # calculate the loss
            loss = np.average(self.calculate_loss(y_pred, y))
            if verbose: print(f"Loss: {loss}\n")
            # update the slope 'm' and y-intercept 'b' using gradient descent
            self.backward(x, y_pred, y, learning_rate)
