import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

"""
Welcome to the Wild Blueberry Yield Prediction Challenge! In this Kaggle competition, you will work
with the Wild Blueberry Yield Prediction Dataset to explore the relationships between multiple variables
and predict yield using multiple linear regression. Your task is to implement regression from scratch. Ex-
plore multi-variate OLS and kernel regression techniques. Use code implemented in previous questions.
You can extend kernel regression to n-dimensions based on material covered in class. Uncover hidden
patterns among the covariates, and, if necessary, choose the most relevant subset of features for final
modeling
"""

class PolynomialOLS:
    def __init__(self,  degree: list[int]):
        self.degree = degree
        self.beta = None
        
    def __polynomial_matrix(self, X):
        """
        Generate polynomial features for multivariate data with individual degrees per feature.
        X: An array of shape (n_samples, n_features).
        Returns: A design matrix of shape (n_samples, n_polynomial_features) based on the degree list.
        """
        n_samples, n_features = X.shape
        X_poly = np.ones((n_samples, 1))
        
        if len(self.degree) != n_features:
            raise ValueError("Length of 'degree' must match the number of features in X.")

        # For each feature, generate polynomial features up to the specified degree for that feature
        for i in range(n_features):
            for d in range(1, self.degree[i] + 1):
                X_poly = np.hstack((X_poly, (X[:, i:i+1] ** d)))
        
        return X_poly
        
    def fit(self, X, Y):
        """
        Fit the model to the data using Ordinary Least Squares.
        X: An array of shape (n_samples, n_features).
        Y: The target values of shape (n_samples, 1).
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

def calculate_mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

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


data = pd.read_csv('data/train.csv')

X_train = data.drop(columns=['id', 'yield']).values
Y_train = data['yield'].values

no_of_features = X_train.shape[1]

def generate_next_degree_combinations_replace_1_with_0(current_best_degree):
    """
    Generate new degree combinations by introducing more 0s into the current best degree.
    Only place additional 0s where there are 1s in the current degree list.
    """
    next_combinations = []
    for i in range(len(current_best_degree)):
        if current_best_degree[i] == 1:
            new_combination = current_best_degree.copy()
            new_combination[i] = 0
            next_combinations.append(new_combination)
    return next_combinations

def generate_next_degree_combinations_replace_1_with_2(current_best_degree):
    """
    Generate new degree combinations by replacing 1s with 2s in the current best degree list.
    Only change degrees that are currently 1.
    """
    next_combinations = []
    for i in range(len(current_best_degree)):
        if current_best_degree[i] == 1:
            new_combination = current_best_degree.copy()
            new_combination[i] = 2
            next_combinations.append(new_combination)
    return next_combinations


def iterative_degree_search_replace_1_with_0(X_train, Y_train, no_of_features, current_best_degree):
    # Start with the best combination from the first run
    best_ssr = np.inf
    converged = False
    
    while not converged:
        converged = True
        next_degree_combinations = generate_next_degree_combinations_replace_1_with_0(current_best_degree)
        SSR_at_degree = []
        
        # Evaluate SSR for each new combination
        for degree_list in next_degree_combinations:
            SSR_val = k_fold_SSR(PolynomialOLS, X_train, Y_train, degree_list)
            SSR_at_degree.append(SSR_val)
        
        # Find the combination with the lowest SSR in this round
        min_ssr = min(SSR_at_degree)
        best_degree_idx = np.argmin(np.array(SSR_at_degree))
        best_combination = next_degree_combinations[best_degree_idx]
        
        # Check if the new combination improves the SSR
        if min_ssr < best_ssr:
            best_ssr = min_ssr
            current_best_degree = best_combination
            converged = False  # Continue refining if SSR improves
            print(f"Found better degree combination: {current_best_degree} with SSR: {best_ssr}")
        else:
            print(f"No further improvement. Current best combination: {current_best_degree} with SSR: {best_ssr}")
    
    return current_best_degree

def iterative_degree_search_replace_1_with_2(X_train, Y_train, no_of_features, current_best_degree):
    best_ssr = np.inf
    converged = False
    
    while not converged:
        converged = True
        next_degree_combinations = generate_next_degree_combinations_replace_1_with_2(current_best_degree)
        SSR_at_degree = []
        
        # Evaluate SSR for each new combination
        for degree_list in next_degree_combinations:
            SSR_val = k_fold_SSR(PolynomialOLS, X_train, Y_train, degree_list)
            SSR_at_degree.append(SSR_val)
        
        # Find the combination with the lowest SSR in this round
        min_ssr = min(SSR_at_degree)
        best_degree_idx = np.argmin(np.array(SSR_at_degree))
        best_combination = next_degree_combinations[best_degree_idx]
        
        # Check if the new combination improves the SSR
        if min_ssr < best_ssr:
            best_ssr = min_ssr
            current_best_degree = best_combination
            converged = False  # Continue refining if SSR improves
            print(f"Found better degree combination: {current_best_degree} with SSR: {best_ssr}")
        else:
            print(f"No further improvement. Current best combination: {current_best_degree} with SSR: {best_ssr}")
    
    return current_best_degree

current_best_degree = [1] * no_of_features

current_best_degree = iterative_degree_search_replace_1_with_0(X_train, Y_train, no_of_features, current_best_degree)
# current_best_degree = iterative_degree_search_replace_1_with_2(X_train, Y_train, no_of_features, current_best_degree)

print("Final best degree combination:", current_best_degree)

OLS = PolynomialOLS(current_best_degree)
OLS.fit(X_train, Y_train)
y_pred = OLS.predict(X_train)

MAE = calculate_mae(y_pred, Y_train)
print(MAE)

data = pd.read_csv('data/test.csv')
x_final = data.drop(columns=['id']).values[:]
ids=data['id']

y_pred=OLS.predict(x_final)

predictions_df = pd.DataFrame({
    'id': ids,
    'y': y_pred.flatten()
})
predictions_df.to_csv("submission.csv", index=False)