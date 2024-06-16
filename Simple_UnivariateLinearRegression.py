import numpy as np


class UnivariateLinearRegression:

    def __init__(self, learning_rate=.01, epoch=10000):
        self.learning_rate = learning_rate
        self.epochs = epoch
        self.theta0 = 0
        self.theta1 = 0

    def hypothesis(self, x):
        return self.theta0 + self.theta1 * x

    def compute_cost(self, x, y):
        m = len(y)
        y_predict = self.hypothesis(x)
        cost = (1 / (2 * m) * np.sum(y_predict - y) ** 2)
        return cost

    def fit(self, x, y):
        m = len(y)
        for epoch in range(self.epochs):
            y_predict = self.hypothesis(x)
            d_theta0 = (1 / m) * np.sum(y_predict - y)
            d_theta1 = (1 / m) * np.sum((y_predict - y) * x)
            self.theta0 -= self.learning_rate * d_theta0
            self.theta1 -= self.learning_rate * d_theta1

            if epoch % 100 == 0:
                cost = self.compute_cost(x, y)
                print(f'{epoch = }, {cost = }, {self.theta0 = }, {self.theta1 = }')

    def predict(self, x):
        return self.hypothesis(x)


if __name__ == '__main__':

    x = np.array([1, 2, 3, 4, 5])
    y = np.array([5, 7, 9, 11, 13])

    model = UnivariateLinearRegression()
    model.fit(x, y)

    print(f'Final theta0 : {model.theta0}, Final theta1 : {model.theta1}')
    print(f'Slope of the line : {model.theta0} + {model.theta1} * x')

    x_new = np.array([6, 7])
    y_new = model.predict(x_new)
    print('Predicted value : ', y_new)