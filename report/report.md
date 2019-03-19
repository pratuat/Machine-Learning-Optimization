# Optimization Methods for Machine Learning
## Fall 2018 FINAL PROJECT

Submitted by: Pratuat Amatya (Matricola ID: 1800063)</br>
Date: March 18, 2019</br>


## Introduction and Setup

The training data provided are 16 x 16 grayscale image for numeric classes '1', '2' and '8'. The first two data class are used as training dataset since higher number of data points favors the accuracy of the model. The train and test data splits are loaded into python data frames directly from respective csv files. Afterwards data is standard-nomralized, which at times will help avoid the pitfall of vanishing gradient in certian backpropagation algorithms.


## 1. RBF Neural Network (2 class classification problem)

The RBF neural network model was implemented in python using `numpy` and `scipy` for optimazation problem and `scikit-learn` for hyper parameter tuning with k-fold cross-validation with cv factor of 5. The classification problem was modelled as regression problem for binary number of 0 and 1 and optimization was performed by minimizing sum of squared error and regularization error. Hyper parameter grid search for numerous minimization algorithms were performed but ones except 'L-BFGS-B' were computatinally expensive as complexity of the model grew with other parameters.

Gridsearch for hyperparamter tuning was performed across given values.

```python
# @paramters
#   'noc' : number of centers or weights
#   'solver' : solver algorithm
#   'sigma' : spread of gaussian kernel function
#   'rho' : regularization parameter
param_grid = {
    'noc' : list(range(2, 5, 1)),
    'solver' : ['L-BFGS-B', 'BFGS', 'TNC', 'CG', 'Nelder-Mead'],
    'sigma' : list(np.arange(1, 3.1, 0.5)),
    'rho' : [1e-6, 1e-7, 1e-8]
}
```


### 1.1 Optimization algorithm for minimization with respect to both $(w, c)$

A Rbf neural network model was implemented with optimization algorithm to minimize total training error w.r.t. both centers and weights parameters. The hyper-parameters were picked from the best estimators determined using grid-search.

The parameters and performance stastics for resulting model are given below.

```
Number of neurons N:                            4
Initial Training Error:                         0.1397415304806623
Final Training Error:                           0.005964984367506203
Final Test Error:                               0.010977546424625208
Norm of the gradient at the final point:        1.4142135616287155e-08
Optimization solver chosen:                     L-BFGS-B
Total Number of function/gradient evaluations:  15435
Time for optimizing the network (seconds):      17.497059106826782
Value of σ:                                     2
Value of ρ:                                     1e-07
```


### 1.2 Two block decomposition method for alternationg minimization with respect to $w$ and $c$

In second section we implemented a Rbf neural network model with two block decomposition algorithm to alternatively minimize total training error w.r.t. centers and weights parameters.

The parameters and performance stastics for resulting model are given below.

```
Number of neurons N:                            2
Initial Training Error:                         0.16248521207627464
Final Training Error:                           0.012056263195476091
Final Test Error:                               0.01564292933818273
Norm of the gradient at the final point:        1.4142126745264763e-08
Optimization solver chosen:                     L-BFGS-B
Total Number of function/gradient evaluations:  18027
Time for optimizing the network (seconds):      19.084325075149536
Value of σ:                                     1
Value of ρ:                                     1e-6
Value of accuracy convergence threshold:        1e-6
```


## 2. Support Vector Machine (2 class classification problem)

In order to implement the QP, we consider an SVM with gaussian kernel function k(,) and implement the non-linear decision function as

$$y(x) =  sign(\sum_{p=0}^{P} \alpha_p y^pK(x,x^p) + b)$$

where $\alpha$ and $b$ are derived from the optimal solution of the dual non-linear SVM problem and
the kernel function is given as

$$K(x,y)=\exp(-\gamma||x-y||^2)$$
, where $\gamma > 0$


### 2.1 Standard QP algorithm for the solution of dual QP problem

Here we implemented standard QP algorithm using CVXOPT pacakge to solve the QP dual problem. We performed grid search across gamma values within range 0.01 to 0.06 with step-size of 0.01. The choice of dataset were for class **1** and **8**.

Model parameters for best etimator are

```
Numbers to classify:                    1 and 8
Optimization time:                      1.4546749591827393
Value of γ:                             0.02
Optimization solver chosen:             cvxopt
Total number of iterations:             13
Misclassification rate on training set: 0.0
Misclassification rate on test set:     0.018604651162790753
```


### 2.2 Decomposition method for the dual quadriatic problem

In the decomposed QP SVM with the dual problem q=2, we solve a single quadratic sub-problems using the optimization routine similarly to the previous probem. First we iteratively selected **alpha_prime_i** and **alpha_prime_j** and fitted those values to minimize the prediction error. The iteration continued till the value of difference in consecutive **alphas** converged below a threshold value or the maximum number of iteration is reached.

Model parameters for best etimator are

```
Numbers to classify:                    1 and 8
Optimization time:                      1.7014095783233643
Value of γ:                             0.02
Optimization solver chose:              SMO
Total number of iterations:             4
Misclassification rate on training set: 0.0025856496444731647
Misclassification rate on test set:     0.016279069767441867
```


## 3. Multi-class classification

A multi-class classification algorithm was modelled through **one-against-all** approach using previous **Rbf** neural network model developed in previous section. It involved training a single classifier per class, with the samples of that class as positive samples and all other samples as negatives. Hence 3 composite models were trained for each class using hot-encoded label components. The composite model produces real-valued number resembling confidence score for its decision for that particular  class and the prediction class is inferred from the score with highest score value.

The parameter generated from section 1.1 were used for one-against-all sub-models

```python

model = Rbf(noc = 4, solver = 'L-BFGS-B', sigma = 2, rho = 1e-7)

```

The model parameters and predictions scores our hence developed model is given below.

```
Algorithm used:                             one-against-all
Optimization solver:                        L-BFGS-B
Misclassification rate on training set:     0.10623353819139592
Misclassification rate on test set:         0.1560509554140127
Optimization time:                          80.91520094871521
```
