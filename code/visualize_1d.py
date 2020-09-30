#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as mat_plot


(countries, features, values) = a1.load_unicef_data()

targets = values[:,1]
x = values[:,:]
#x = a1.normalize_data(x)

N_TRAINING = 100

for feature in (10, 11 ,12):
   
    x_train = x[0:N_TRAINING, feature]
    t_train = targets[0:N_TRAINING]
    x_test = x[N_TRAINING:, feature]
    t_test = targets[N_TRAINING:]
    
    x_ev = np.linspace(np.asscalar(min(x_train)), np.asscalar(max(x_train)), num=500)
    x_ev = np.reshape(x_ev,(x_ev.shape[0],1))
    phi_ev = a1.design_matrix(x_ev,'polynomial',degree=3)
    w, _ = a1.linear_regression(x_train, t_train, basis = 'polynomial', degree=3)
    y_ev = phi_ev * w

    mat_plot.plot(x_ev, y_ev, 'g.-')
    mat_plot.plot(x_test, t_test, 'bo')
    mat_plot.legend(['fitting curve','testing points'])
    mat_plot.title('A visualization of a testing points and fitting curve of feature {0}'.format(feature + 1))
    mat_plot.show()
