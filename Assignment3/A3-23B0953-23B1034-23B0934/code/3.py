import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

class PolynomialOLS:
    def __init__(self, degree: int):
        self.degree = degree
        self.beta = None
        
    def __polynomial_matrix(self, X):
        """
        Creates the design matrix with polynomial features up to the specified degree.
        X is a vector of shape (n, 1), where n is the number of data points.
        Returns a matrix of shape (n, degree+1).
        """
        return np.hstack([X**i for i in range(self.degree + 1)])
        
    def fit(self, X, Y):
        """
        Fits the polynomial OLS regression model.
        X is the independent variable vector of shape (n, 1)
        Y is the dependent variable vector of shape (n, 1)
        """       
        X_poly = self.__polynomial_matrix(X)
        self.beta = np.linalg.inv(X_poly.T @ X_poly) @ (X_poly.T @ Y)
    
    def predict(self, X):
        """
        Predicts the values using the fitted model.
        X is the independent variable vector of shape (n, 1).
        Returns predicted values of shape (n, 1).
        """
        X_poly = self.__polynomial_matrix(X)
        return X_poly @ self.beta

def calculate_ssr(y_true, y_pred):
    """
    Calculates the sum of squared residuals (SSR)
    y_true and y_pred are vectors of shape (n, 1).
    """
    ssr = np.sum((y_true - y_pred) ** 2)
    return ssr

def calculate_r2(y_true, y_pred):
    """
    Calculates coefficient of determination (R2) value.
    y_true and y_pred are vectors of shape (n, 1).
    """
    ssr = np.sum((y_true - y_pred) ** 2)
    ssy = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ssr / ssy
    return r2

def k_fold_SSR(polyOLS, x_data, y_data, degree, k:int = 10):
    
    data_length = int(len(x_data)/k)
    risk = 0
    for i in range(k):
        x_train = np.concatenate([x_data[:i*data_length], x_data[(i+1)*data_length:]])
        y_train = np.concatenate([y_data[:i*data_length], y_data[(i+1)*data_length:]])
        
        x_test = x_data[i*data_length:(i+1)*data_length]
        y_test = y_data[i*data_length:(i+1)*data_length]
        
        OLS = polyOLS(degree)
        OLS.fit(x_train, y_train)
        y_pred = OLS.predict(x_test)
        
        risk = risk + calculate_ssr(y_pred, y_test)

    return risk


train_data = pd.read_csv("train.csv")
X_train = train_data['x'].values.reshape(-1, 1)
Y_train = train_data['y'].values.reshape(-1, 1)

test_data = pd.read_csv("test.csv")
X_test = test_data['x'].values.reshape(-1, 1)
ids = test_data['id'].values


degrees_to_explore = range(1, 30)    #taking degrees in this range for wide exploration
SSR_at_degree = []

# Explore polynomials of various degrees
for degree in degrees_to_explore: 
    SSR_val = k_fold_SSR(PolynomialOLS, X_train, Y_train, degree)
    SSR_at_degree.append(SSR_val)

best_degree = np.argmin(np.array(SSR_at_degree))+1   # degree is index+1
print("best degree ", best_degree)

def plot_results(X_data, Y_data, degree, fit_type = "Correct Fit"):
    plt.figure(figsize=(10, 6))
    plt.scatter(X_data, Y_data, color='blue', label='Train data', alpha=0.5)

    polyFit = PolynomialOLS(degree)
    polyFit.fit(X_data, Y_data)
    Y_pred = polyFit.predict(X_test)

    X_range = np.linspace(X_data.min(), X_data.max(), 100).reshape(-1, 1)
    y_range = polyFit.predict(X_range)

    # Line plot for the predicted data using the fitted model
    plt.plot(X_range, y_range, color='red', label=f'Degree {degree} - {fit_type}', linewidth=2)
    
    plt.title(f'Polynomial Regression (Degree {degree})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"images/3_{fit_type}.png")
    plt.close()

# for degrees [1,2,3] the model is underfit, for [4,5,6] the model is best fit, while higher degrees give overfit, one each relevant plot is as follows: 

underfit_degree = 2
overfit_degree = 20

plot_results(X_train, Y_train, underfit_degree, "underfit")
plot_results(X_train, Y_train, best_degree, "correctfit")
plot_results(X_train, Y_train, overfit_degree, "overfit")


underFit = PolynomialOLS(underfit_degree)
underFit.fit(X_train, Y_train)
underfit_y_est = underFit.predict(X_train)
underFit_SSR = calculate_ssr(underfit_y_est, Y_train)
underfit_R2 = calculate_r2(underfit_y_est, Y_train)

correctFit = PolynomialOLS(best_degree)
correctFit.fit(X_train, Y_train)
correctFit_y_est = correctFit.predict(X_train)
correctFit_SSR = calculate_ssr(correctFit_y_est, Y_train)
correctFit_R2 = calculate_r2(correctFit_y_est, Y_train)

overFit = PolynomialOLS(overfit_degree)
overFit.fit(X_train, Y_train)
overfit_y_est = overFit.predict(X_train)
overFit_SSR = calculate_ssr(overfit_y_est, Y_train)
overfit_R2 = calculate_r2(overfit_y_est, Y_train)

print(f"For a fit with polynomial of degree {underfit_degree} (underfit), the calculated SSR is: {underFit_SSR:.0f}, and the calculated R2 is: {underfit_R2:.4f}")
print(f"For a fit with polynomial of degree {best_degree} (correct fit), the calculated SSR is: {correctFit_SSR:.0f}, and the calculated R2 is: {correctFit_R2:.4f}")
print(f"For a fit with polynomial of degree {overfit_degree} (over fit), the calculated SSR is: {overFit_SSR:.0f}, and the calculated R2 is: {overfit_R2:.4f}")

bestFit_beta = correctFit.beta
with open("3_weights.pkl", "wb") as f:
    pickle.dump(bestFit_beta, f)
    
test_predictions = correctFit.predict(X_test)

# Create a DataFrame to store test IDs and predicted values
predictions_df = pd.DataFrame({
    'id': ids,
    'x': X_test.flatten(),
    'y': test_predictions.flatten()
})
predictions_df.to_csv("3_predictions.csv", index=False)