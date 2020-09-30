#!/usr/bin/env python

"""Basic code for assignment 1."""

import numpy as np
import pandas as pd
import scipy.stats as stats

def load_unicef_data():
    """Loads Unicef data from CSV file.

    Retrieves a matrix of all rows and columns from Unicef child mortality
    dataset.

    Args:
      none

    Returns:
      Country names, feature names, and matrix of values as a tuple (countries, features, values).

      countries: vector of N country names
      features: vector of F feature names
      values: matrix N-by-F
    """
    #fname = '/Users/prat/Desktop/Assignments/Assignment1/assignment1-datacode/SOWC_combined_simple.csv'
    fname = 'SOWC_combined_simple.csv'
    # Uses pandas to help with string-NaN-numeric data.
    data = pd.read_csv(fname, na_values='_')

    # Strip countries title from feature names.
    features = data.axes[1][1:]
    # Separate country names from feature values.
    countries = data.values[:,0]
    values = data.values[:,1:]
    # Convert to numpy matrix for real.
    values = np.asmatrix(values,dtype='float64')

    # Modify NaN values (missing values).
    mean_vals = np.nanmean(values, axis=0)
    inds = np.where(np.isnan(values))
    values[inds] = np.take(mean_vals, inds[1])
    return (countries, features, values)


def normalize_data(x):
    """Normalize each column of x to have mean 0 and variance 1.
    Note that a better way to normalize the data is to whiten the data (decorrelate dimensions).  This can be done using PCA.

    Args:
      input matrix of data to be normalized

    Returns:
      normalized version of input matrix with each column with 0 mean and unit variance

    """
    mvec = x.mean(0)
    stdvec = x.std(axis=0)
    
    return (x - mvec)/stdvec
    


def linear_regression(x, t, basis, reg_lambda=0, degree=0):
    """Perform linear regression on a training set with specified regularizer lambda and basis

    Args:
      x is training inputs
      t is training targets
      reg_lambda is lambda to use for regularization tradeoff hyperparameter
      basis is string, name of basis to use
      degree is degree of polynomial to use (only for polynomial basis)

    Returns:
      w vector of learned coefficients
      train_err RMS error on training set
      """
    phi = design_matrix(x,basis,degree)
    w = np.linalg.pinv(phi) *t
    y_train = (phi*w)
    # Measure root mean squared error on training data.
    train_err = root_mean_squared_err(np.asarray(y_train),np.asarray(t))
    return (w, train_err)

def polynomial_regression(lambda_function, x_train_data, y_train_data, x_test_data, y_test_data, degree=1, constant=True):
    if constant:
        x_training = add_constant(polynomial_attach(x_train_data, degree)) 
        x_testing = add_constant(polynomial_attach(x_test_data, degree))
    else:
        x_training = polynomial_attach(x_train_data, degree)
        x_testing = polynomial_attach(x_test_data, degree)
    
    symmetric_training_data = x_training.T * x_training
    symmetric_training_data += float(lambda_function) * np.identity(symmetric_training_data.shape[0])
    w = np.linalg.inv(symmetric_training_data) * x_training.T * y_train_data

    error_train = np.sqrt(np.mean(np.square(x_training * w - y_train_data)))
    error_test = np.sqrt(np.mean(np.square(x_testing * w - y_test_data)))
    return w, error_train, error_test

def design_matrix(x, basis, degree=0):
    """ Compute a design matrix Phi from given input datapoints and basis.
	Args:
      x matrix of input datapoints
      basis string name of basis

    Returns:
      phi design matrix
    """
    if basis == 'polynomial':
        phi = lambda x,d : x**d
        result = get_design_matrix(x,phi,degree)
        return result
    elif basis == 'ReLU':
      # degree is ignored here
      phi = lambda x, d: map(lambda val: max(0, -val+5000), x)
      return get_design_matrix(x,phi)
    else: 
      assert(False), 'Unknown basis %s' 
    return None


def get_design_matrix(values,func,degree = 1) :
  result = []
  for row in values.tolist():
    row_result = []
    row_result.append(1)
    for d in range(1, degree+1):
      row_result.extend(np.apply_along_axis(lambda x : func(x,d), axis = 0 , arr = row).tolist())
    result.append(row_result)
  return result   

def root_mean_squared_err(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def evaluate_regression(x, t, w, basis, degree=0):
    """Evaluate linear regression on a dataset.
	Args:
      x is evaluation (e.g. test) inputs
      w vector of learned coefficients
      basis is string, name of basis to use
      degree is degree of polynomial to use (only for polynomial basis)
      t is evaluation (e.g. test) targets

    Returns:
      t_est values of regression on inputs
      err RMS error on the input dataset 
      """
    phi = design_matrix(x,basis,degree)
    t_est = phi * w
    err = root_mean_squared_err(np.asarray(t_est),np.asarray(t))
    return (t_est, err)

def array(val) :
  result = []
  for i in val :
    result.append(i)
  return result    

def add_constant(X):
    return np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)

def polynomial_attach(graph_data, n=1):
    graph_data = np.array(graph_data)
    result_data = graph_data
    for range_index in range(2, n+1):
        result_data = np.concatenate((graph_data ** range_index, result_data), axis=1)
    return np.matrix(result_data)
