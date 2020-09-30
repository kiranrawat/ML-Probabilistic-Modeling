#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as mat_plot

'''
Loading the unicef data and storing the countries, features and values in tuple
'''
(countries, features, values) = a1.load_unicef_data()

targets = values[:,1]
x = values[:,7:]

N_TRAINING = 100
x_training = x[0:N_TRAINING,:] #training data
x_testing = x[N_TRAINING:,:] #testing data
t_training = targets[0:N_TRAINING] #target training data
t_testing = targets[N_TRAINING:] # target testing data 
training_error = {}
testing_error = {}

def plot_bar_chart():
	'''
	Performing regression using just a single input feature 
	Trying features 8-15. For each (un-normalized) feature fitting a degree 3 polynomial (unregularized).
	'''
	for column in range(0,8) :
		(w, train_error) = a1.linear_regression(x_training[:,column],t_training,'polynomial',0,3)
		(_, test_error) = a1.evaluate_regression(x_testing[:,column],t_testing,w,'polynomial',3)
		training_error[column+7] = train_error
		testing_error[column+7] = test_error	
		
	index = np.arange(7, 15) + 1
	bar_size = 0.35
	opacity = 0.8

	mat_plot.bar(index, a1.array(testing_error.values()), bar_size, alpha=opacity, color=(0.3, 0.5, 0.7, 0.7))
	mat_plot.bar(index + bar_size, a1.array(training_error.values()), bar_size, alpha=opacity, color=(0.9, 0.6, 0.1, 0.7))

	mat_plot.ylabel('RMSE')
	mat_plot.legend(['Test error','Training error'])
	mat_plot.title('RMSE for single input feature, no regularization')
	mat_plot.xlabel('Feature index')
	mat_plot.show()

plot_bar_chart()
