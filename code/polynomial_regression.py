#!/usr/bin/env python

import assignment1 as a1
import matplotlib.pyplot as mat_plot

(countries, features, values) = a1.load_unicef_data()

targets = values[:,1]
x = values[:,7:]
# Uncomment for normalized data
#x = a1.normalize_data(x)

N_TRAINING = 100
x_train = x[0:N_TRAINING,:]
x_test = x[N_TRAINING:,:]
t_train = targets[0:N_TRAINING]
t_test = targets[N_TRAINING:]
train_err = {}
test_err = {}
degree = 6

for degree in range(1,degree+1) :
	(w,train_error) = a1.linear_regression(x_train,t_train,'polynomial',0,degree)
	(t_est, test_error) = a1.evaluate_regression(x_test,t_test,w,'polynomial',degree)
	train_err[degree] = train_error
	test_err[degree] = test_error	
	
# Produce a plot of results.
mat_plot.plot(a1.array(train_err.keys()), a1.array(train_err.values()),color='r')
mat_plot.plot(a1.array(test_err.keys()), a1.array(test_err.values()),color='g')
mat_plot.ylabel('RMS')
mat_plot.legend(['Test error','Training error'])
mat_plot.title('Fit with polynomials, no regularization, unnormalized input data')
mat_plot.xlabel('Polynomial degree')
mat_plot.show()
