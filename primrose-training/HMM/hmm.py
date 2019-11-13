# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 14:52:10 2019

@author: royru
"""

# -*- coding: utf-8 -*-
"""
Created on Tue sep 24 12:56:50 2019

@author: royru
"""

import numpy as np
class viterbi:
    def __init__(self, pi, emission_matrix, transition_matrix):
        self.pi = pi
        self.emission_matrix = emission_matrix
        self.transition_matrix = transition_matrix

    def decode(self,observations):
        self.t1 = np.zeros((transition_matrix.shape[0], len(observations)))
        self.t2 = np.zeros((transition_matrix.shape[0], len(observations)))
        self.t1[:, 0] = self.pi*self.emission_matrix[:, observations[0]]
        self.t2[:, 0] = np.zeros((self.transition_matrix.shape[0]))
        # update t1 ,t2 tables step by step
        for i in range(1, len(observations)):
            step_t1_matrix = self.calc_t1_matrix ( self.t1[:, i - 1],
                             self.transition_matrix, self.emission_matrix[:, observations[i]])
            step_t2_matrix = self.calc_t2_matrix ( self.t1[:, i - 1], self.transition_matrix)
            self.t1[:, i] = np.amax(step_t1_matrix, axis=0)
            self.t2[:, i] = np.argmax(step_t2_matrix, axis=0)
        # create predictions according to observations
        self.predictions = np.zeros ( len ( observations ) ) - 1
        # last prediction is argmax of t1 in last step
        pred = np.argmax ( self.t1[:, len ( observations ) - 1] )
        self.predictions[(len ( observations ) - 1)] = pred
        prev_pred = pred
        # iterate backwards on t2 and fill with predictions
        for i in range ( len ( observations ), 1, -1 ):
            pred = int (self.t2[prev_pred, i - 1] )
            self.predictions[i - 2] = pred
            prev_pred = pred
        print (self.predictions)

    def calc_t1_matrix(self, prev_step, transition_matrix, emissions): # t1[:,i-1]*transition_matrix*emission_matrix[:, observations[i]]
        one = transition_matrix*emissions
        return np.diag(prev_step) @ one
    def calc_t2_matrix(self,prev_step,transition_matrix): # t1[:,i-1]@transition_matrix
        return np.diag(prev_step) @ transition_matrix

if __name__ == "__main__":
    transition_matrix = np.array([ \
    [0.08, 0.02, 0.10, 0.05, 0.07, 0.08, 0.07, 0.04, 0.08, 0.10, 0.07, 0.02, 0.01, 0.10, 0.09, 0.01], \
    [0.06, 0.10, 0.11, 0.01, 0.04, 0.11, 0.04, 0.07, 0.08, 0.10, 0.08, 0.02, 0.09, 0.05, 0.02, 0.02], \
    [0.08, 0.07, 0.08, 0.07, 0.01, 0.03, 0.10, 0.02, 0.07, 0.03, 0.06, 0.08, 0.03, 0.10, 0.10, 0.08], \
    [0.08, 0.04, 0.04, 0.05, 0.07, 0.08, 0.01, 0.08, 0.10, 0.07, 0.11, 0.01, 0.05, 0.04, 0.11, 0.06], \
    [0.03, 0.03, 0.08, 0.10, 0.11, 0.04, 0.06, 0.03, 0.03, 0.08, 0.03, 0.07, 0.10, 0.11, 0.07, 0.03], \
    [0.02, 0.05, 0.01, 0.09, 0.05, 0.09, 0.05, 0.12, 0.09, 0.07, 0.01, 0.07, 0.05, 0.05, 0.11, 0.06], \
    [0.11, 0.05, 0.10, 0.07, 0.01, 0.08, 0.05, 0.03, 0.03, 0.10, 0.01, 0.10, 0.08, 0.09, 0.07, 0.02], \
    [0.03, 0.02, 0.16, 0.01, 0.05, 0.01, 0.14, 0.14, 0.02, 0.05, 0.01, 0.09, 0.07, 0.14, 0.03, 0.01], \
    [0.01, 0.09, 0.13, 0.01, 0.02, 0.04, 0.05, 0.03, 0.10, 0.05, 0.06, 0.06, 0.11, 0.06, 0.03, 0.14], \
    [0.09, 0.03, 0.04, 0.05, 0.04, 0.03, 0.12, 0.04, 0.07, 0.02, 0.07, 0.10, 0.11, 0.03, 0.06, 0.09], \
    [0.09, 0.04, 0.06, 0.06, 0.05, 0.07, 0.05, 0.01, 0.05, 0.10, 0.04, 0.08, 0.05, 0.08, 0.08, 0.10], \
    [0.07, 0.06, 0.01, 0.07, 0.06, 0.09, 0.01, 0.06, 0.07, 0.07, 0.08, 0.06, 0.01, 0.11, 0.09, 0.05], \
    [0.03, 0.04, 0.06, 0.06, 0.06, 0.05, 0.02, 0.10, 0.11, 0.07, 0.09, 0.05, 0.05, 0.05, 0.11, 0.08], \
    [0.04, 0.03, 0.04, 0.09, 0.10, 0.09, 0.08, 0.06, 0.04, 0.07, 0.09, 0.02, 0.05, 0.08, 0.04, 0.09], \
    [0.05, 0.07, 0.02, 0.08, 0.06, 0.08, 0.05, 0.05, 0.07, 0.06, 0.10, 0.07, 0.03, 0.05, 0.06, 0.10], \
    [0.11, 0.03, 0.02, 0.11, 0.11, 0.01, 0.02, 0.08, 0.05, 0.08, 0.11, 0.03, 0.02, 0.10, 0.01, 0.11]])
    emission_matrix = np.array([
                [0.01,0.99], \
                [0.58,0.42], \
                [0.48,0.52], \
                [0.58,0.42], \
                [0.37,0.63], \
                [0.33,0.67], \
                [0.51,0.49], \
                [0.28,0.72], \
                [0.35,0.65], \
                [0.61,0.39], \
                [0.97,0.03], \
                [0.87,0.13], \
                [0.46,0.54], \
                [0.55,0.45], \
                [0.23,0.77], \
                [0.76,0.24]])

    pi = np.array([[0.04, 0.02, 0.06, 0.04, 0.11, 0.11, 0.01, 0.09, 0.03, 0.05, 0.06, 0.11, 0.05, 0.11, 0.03, 0.08]])
    observations2 = ([1, 1, 0, 0, 1, 1, 1, 0, 0, 0])
    observations1 = ([0, 0, 0, 0, 0, 0, 1, 0, 1, 1])
    vit = viterbi(pi,emission_matrix,transition_matrix)
    vit.decode(observations1)
    vit.decode(observations2)


