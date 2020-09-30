#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as mat_plot

(countries, features, values) = a1.load_unicef_data()

targets = values[:,1]
x = values[:,7:]
normalized_x = a1.normalize_data(x)

N_TRAINING = 100
x_training = normalized_x[0:N_TRAINING, :]
t_training = targets[0:N_TRAINING]

lambdaa = (0.01, 0.1, 1, 10, 100, 1000, 10000)

test_err_count = []
train_err_count = []

def counting_errors():

    for lambdaa_val in lambdaa:

        start = [i * 10 for i in range(10)]
        end = [i * 10 for i in range(11)][1:]

        err_test_array = np.array([0] * 10)
        err_train_array = np.array([0] * 10)

        for index in range(10):
            s2 = start[(index+1) % 10]
            e2 = end[(index+1) % 10]
            
            x_training_one = np.concatenate((x_training[:s2], x_training[e2:]), axis=0)
            t_training_one = np.concatenate((t_training[:s2], t_training[e2:]), axis=0)
            x_test_one = x_training[s2:e2]
            t_test_one = t_training[s2:e2]
            
            w, err_train, err_test = a1.polynomial_regression(lambdaa_val, x_training_one, t_training_one, x_test_one, t_test_one, degree=2)
            err_test_array[index] = err_test
            err_train_array[index] = err_train
            '''
            finding traning error and testing error from unregulaized regression 
            Sending the value of lambda as 0 
            '''
            w, error_train, error_test = a1.polynomial_regression(0,x_training_one, t_training_one, x_test_one, t_test_one,2)
            
        test_err_count.append(np.average(err_test_array))
        train_err_count.append(np.average(err_train_array))
        
    return (test_err_count,error_test)
        
def plot_average_valset_error(error_count_test,err_test):
    '''
    plotting regulaired and unregularized error
    '''
    p, axis = mat_plot.subplots()
    mat_plot.semilogx(lambdaa, error_count_test, color='r')
    axis.hlines(err_test, lambdaa[0], lambdaa[6], color='g', linestyles='dashed')
    mat_plot.title('Testing error versus lambda regularization')
    mat_plot.xlabel('Regularizer value')
    mat_plot.ylabel('Average Validation set RMS Error')
    mat_plot.legend(['with regularization','without regularization'],loc=5)
    mat_plot.show()

(error_count_test, err_test) = counting_errors()
plot_average_valset_error(error_count_test,err_test)


