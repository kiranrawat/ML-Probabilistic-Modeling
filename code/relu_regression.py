#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as mat_plot

(countries, features, values) = a1.load_unicef_data()

targets = values[:,1]
x = values[:,10]

N_TRAINING = 100
x_train = x[0:N_TRAINING,:]
x_test = x[N_TRAINING:,:]
t_train = targets[0:N_TRAINING]
t_test = targets[N_TRAINING:]

(w,train_error) = a1.linear_regression(x_train,t_train,'ReLU')
(t_est, test_error) = a1.evaluate_regression(x_test,t_test,w,'ReLU')

#plots
def generate_relu_plots():
    """
    Implementing regression using a modified version of ReLU basis function for a single input feature.
    """
    x_ev = np.linspace(np.asscalar(min(x_train)), np.asscalar(max(x_train)), num=500)
    x_ev = np.reshape(x_ev,(x_ev.shape[0],1))
    phi_ev = a1.design_matrix(x_ev,'ReLU')
    w, _ = a1.linear_regression(x_train, t_train, basis = 'ReLU',)
    y_ev = phi_ev * w

    mat_plot.plot(x_train, t_train,'go', color='orange')
    mat_plot.plot(x_test, t_test, 'bo', color=(0.3, 0.5, 0.7, 0.7))
    mat_plot.plot(x_ev, y_ev, 'r.-')
    mat_plot.legend(['Training data','Test data','Learned Polynomial'])
    mat_plot.title('ReLU visualization: Training data, Test data and Learned polynomial')
    mat_plot.show()

def generate_relu_error_plots():
    X = ['Training Error', 'Test Error']
    Y = [train_error, test_error]
    index = np.arange(len(X))
    bar_width = 0.20
    opacity = 0.8
    mat_plot.bar(index, Y, bar_width, alpha=opacity, color='orange')
    mat_plot.xticks(index, X)
    mat_plot.show()

'''
Generate ReLU plots and ReLU error plots 
'''
generate_relu_plots()
generate_relu_error_plots()
