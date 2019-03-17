from __future__ import division, print_function
import os
import numpy as np
import random as rnd
from scipy.linalg import norm
import time

def kernel(x, y, gamma=0.02):
    return np.exp(- (gamma * norm(x - y) ** 2))

class SVMSMO():
    def __init__(self, max_iter=10000, epsilon=0.001, C=1.0, gamma=0.02):
        self.max_iter = max_iter
        self.C = C
        self.epsilon = epsilon
        self.gamma = gamma

        self.support_vectors = None
        self.weights = None
        self.intercepts = None
        self.optimization_time = None

    def fit(self, X, y):
        n, d = X.shape[0], X.shape[1]
        alpha = np.zeros((n))
        count = 0
        start = time.time()
        while True:
            count += 1
            alpha_prev = np.copy(alpha)

            for j in range(0, n):
                i = self.get_rnd_int(0, n-1, j)
                x_i, x_j, y_i, y_j = X[i,:], X[j,:], y[i], y[j]
                k_ij = kernel(x_i, x_i, self.gamma) + kernel(x_j, x_j, self.gamma) - 2 * kernel(x_i, x_j, self.gamma)
                if k_ij == 0:
                    continue

                alpha_prime_j, alpha_prime_i = alpha[j], alpha[i]

                (L, H) = self.compute_L_H(self.C, alpha_prime_j, alpha_prime_i, y_j, y_i)

                self.weights = self.calculate_weights(alpha, y, X)
                self.intercepts = self.calculate_intercepts(X, y, self.weights)

                E_i = self.prediction_error(x_i, y_i, self.weights, self.intercepts)
                E_j = self.prediction_error(x_j, y_j, self.weights, self.intercepts)

                alpha[j] = alpha_prime_j + float(y_j * (E_i - E_j))/k_ij
                alpha[j] = max(alpha[j], L)
                alpha[j] = min(alpha[j], H)

                alpha[i] = alpha_prime_i + y_i*y_j * (alpha_prime_j - alpha[j])

            diff = np.linalg.norm(alpha - alpha_prev)
            if diff < self.epsilon:
                break

            if count >= self.max_iter:
                print("Iteration number exceeded:", str(self.max_iter))
                return

        end = time.time()
        self.optimization_time = end - start
        self.intercepts = self.calculate_intercepts(X, y, self.weights)

        alpha_idx = np.where(alpha > 0)[0]
        support_vectors = X[alpha_idx, :]
        self.support_vectors = support_vectors
        self.count = count

        return self

    def predict(self, X):
        return self.project(X, self.weights, self.intercepts)

    def calculate_intercepts(self, X, y, w):
        b_tmp = y - np.dot(w.T, X.T)

        return np.mean(b_tmp)

    def calculate_weights(self, alpha, y, X):
        return np.dot(X.T, np.multiply(alpha,y))

    # Prediction
    def project(self, X, w, b):
        return np.sign(np.dot(w.T, X.T) + b).astype(int)

    # Prediction error
    def prediction_error(self, x_k, y_k, w, b):
        return self.project(x_k, w, b) - y_k

    def compute_L_H(self, C, alpha_prime_j, alpha_prime_i, y_j, y_i):
        if(y_i != y_j):
            return (max(0, alpha_prime_j - alpha_prime_i), min(C, C - alpha_prime_i + alpha_prime_j))
        else:
            return (max(0, alpha_prime_i + alpha_prime_j - C), min(C, alpha_prime_i + alpha_prime_j))

    def get_rnd_int(self, a,b,z):
        i = z
        cnt=0
        while i == z and cnt<1000:
            i = rnd.randint(a,b)
            cnt=cnt+1
        return i

    def statistics(self):
        return [
            ('Numbers to classify:', '1 and 8'),
            ('Optimization time:', self.optimization_time),
            ('Value of γ:', self.gamma),
            ('Optimization solver chose:', 'SMO'),
            ('Total number of iterations:', self.count)
        ]