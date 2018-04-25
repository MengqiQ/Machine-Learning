import os
import argparse
import sys
import pickle
import math
import numpy as np

class AdaboostAlg:

    def __init__(self,X,y):
        # TODO
        self.X = X.toarray()
        y[y==0] = -1
        self.y = y
        self.m, self.n = X.shape

        self.D = np.ones(self.m) / self.m

    def get_yhat(self,j,c):
        y_hat = np.ones(self.m)
        y_hat[self.X[:,j] <= c] = -1

        if np.sum(y_hat == self.y) >= np.sum(y_hat != self.y):
            return y_hat,1
        else:
            return -y_hat, -1

    def get_ht(self):
        j_update = 0
        c_update = 0
        y_hat_update_t = 0
        y_hat_dim_t = 0
        error = 1

        for j in range(self.n):
            j_sorted = np.sort(self.X[:,j])

            c_sets = np.unique((j_sorted[:self.m - 1] + j_sorted[1:])/2)

            for c in c_sets:
                y_hat_update, y_hat_dim = self.get_yhat(j, c)
                error_arr = np.zeros(self.m)
                error_arr[y_hat_update != self.y] = 1

                error_update = np.dot(self.D, error_arr)
                # print(error_update)
                # print(j, c, y_hat_dim_t, error_update)
                if error_update < error:
                    j_update = j
                    c_update = c
                    y_hat_update_t = y_hat_update
                    y_hat_dim_t = y_hat_dim
                    error = error_update

        return j_update, c_update, y_hat_update_t, y_hat_dim_t, error

    def get_alpha(self,error):
        return 0.5 * math.log((1/error) - 1)


    def update_D(self, alpha, y_hat):
        self.D = np.multiply(self.D, np.exp(np.multiply(-alpha * self.y, y_hat)))
        z = self.D.sum()
        self.D /= z


