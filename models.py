import numpy as np

class RegressionModel:
    def __init__(self, number_of_parameters):
        self.coefficients = np.random.uniform(size=(number_of_parameters))
        self.bias = np.random.uniform()

    def calculate_loss(self, y_pred: np.ndarray, y_true: np.ndarray):
        return (y_pred - y_true) ** 2
    
    def calculate_loss_derivative(self, y_pred: np.ndarray, y_true: np.ndarray):
        return 2 * (y_pred - y_true)
    
    def forward(self, x: np.ndarray):
        pass

    def backward(self):
        pass

    def fit(self, epochs: int, learning_rate: float):
        pass

class LinearRegression(RegressionModel):
    def __init__(self):
        super().__init__(number_of_parameters=1)

    def forward(self, x: np.ndarray):
        return self.coefficients * x + self.bias

    def backward(self, x: np.ndarray, y_pred: np.ndarray, y_true: np.ndarray, learning_rate: float):
        dloss_dypred = self.calculate_loss_derivative(y_pred, y_true)

        dloss_dc = np.average(dloss_dypred * x)

        dloss_db = np.average(dloss_dypred)

        self.coefficients -= learning_rate * dloss_dc
        self.bias -= learning_rate * dloss_db

    def fit(self, x: np.ndarray, y: np.ndarray, epochs: int = 10, learning_rate: float = 0.01, verbose: int = 1):
        for e in range(1, epochs+1):
            if verbose: print(f"Epoch: {e}")
            # calculate y_hat using the formula m*x+b
            y_pred = self.forward(x)
            # calculate the loss
            loss = np.average(self.calculate_loss(y_pred, y))
            if verbose: print(f"Loss: {loss}\n")
            # update the slope 'm' and y-intercept 'b' using gradient descent
            self.backward(x, y_pred, y, learning_rate)

class MultipleLinearRegression(RegressionModel):
    def __init__(self, number_of_parameters):
        super().__init__(number_of_parameters=number_of_parameters)

    def forward(self, x: np.ndarray):
        return np.dot(self.coefficients, x) + self.bias
    
    def backward(self, x: np.ndarray, y_pred: np.ndarray, y_true: np.ndarray, learning_rate: float):
        dloss_dypred = self.calculate_loss_derivative(y_pred, y_true)

        dloss_dc = np.average(dloss_dypred * x.T)

        dloss_db = np.average(dloss_dypred)

        self.coefficients -= learning_rate * dloss_dc
        self.bias -= learning_rate * dloss_db

    def fit(self, x: np.ndarray, y: np.ndarray, epochs: int = 10, learning_rate: float = 0.01, verbose: int = 1):
        for e in range(1, epochs+1):
            if verbose: print(f"Epoch: {e}")
            # calculate y_hat using the formula m*x+b
            y_pred = self.forward(x)
            # calculate the loss
            loss = np.average(self.calculate_loss(y_pred, y))
            if verbose: print(f"Loss: {loss}\n")
            # update the slope 'm' and y-intercept 'b' using gradient descent
            self.backward(x, y_pred, y, learning_rate)
