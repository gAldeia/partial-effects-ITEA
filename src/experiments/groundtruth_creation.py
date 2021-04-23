import os
import sys
import shap

import pandas as pd
import numpy  as np

sys.path.append('../../docs/')

# Importing the strings, lambdas and names of the datasets
from feynman_equations import original_expressions, original_lambdas, datasets
from autograd          import grad, jacobian
from sklearn.metrics   import mean_squared_error
from scipy.stats       import rankdata
from _auxiliary        import *


datasets_folder    = '../../datasets/'
filelock_name      = './locks/GT_lock.txt.lock'
groundtruth_folder = f'../../results/ground_truth/'


def gradient_groundtruth(dataset, predict_f, Xtrain, Xtest, nvars, nobs):

    # Automatic differentiation
    gradient_f = jacobian(original_lambdas[dataset])

    gradients = np.zeros_like( Xtest )
    for i in range(nobs):
        gradients[i, :] = gradient_f(Xtest[i, :])
        
    return gradients


def SHAP_groundtruth(dataset, predict_f, Xtrain, Xtest, nvars, nobs):

    # calculating the shapley values
    explainer = shap.KernelExplainer(predict_f, Xtrain, silent=True)
    shapley   = explainer.shap_values(Xtest, silent=True)

    return shapley


if __name__ == "__main__":
    for groundtruth in [gradient_groundtruth, SHAP_groundtruth]:
        for ds in datasets:
            print(ds)

            Xtrain, ytrain, Xtest, ytest = None, None, None, None
            
            try:
                dataset_data = np.loadtxt(f'{datasets_folder}{ds}.dat', delimiter=',', skiprows=1)
                LHS_data     = np.loadtxt(f'{datasets_folder}{ds}_LHS.dat', delimiter=',', skiprows=1)
                label        = np.loadtxt(f'{datasets_folder}{ds}.dat', delimiter=',', max_rows=1, dtype=str)

                Xtrain = dataset_data[:, :-1]
                ytrain = dataset_data[:, -1]

                nobs  = Xtrain.shape[0]
                nvars = Xtrain.shape[1]

                Xtest = LHS_data[:, :-1]
                ytest = LHS_data[:, -1]
            except:
                print(f'Could not open data set {ds}.')
                sys.exit()

            # Transforming the lambda function to a numpy vectorized function, so it can work with SHAP
            # X is a numpy array with observations x variables
            original_vectorized = lambda X: np.array([original_lambdas[ds](X[i, :]) for i in range(len(X))])

            save_explanations(
                explanations  = groundtruth(
                    dataset   = ds,
                    predict_f = original_vectorized,
                    Xtrain    = Xtrain,
                    Xtest     = Xtrain,
                    nvars     = nvars,
                    nobs      = nobs
                ),
                dataset       = ds,
                rep           = 0,
                explainer     = f'Global-{groundtruth.__name__}',
                fname         = f'{groundtruth_folder}/{groundtruth.__name__}-globalexplanations.csv',
                filelock_name = filelock_name 
            )

            save_explanations(
                explanations  = groundtruth(
                    dataset   = ds,
                    predict_f = original_vectorized,
                    Xtrain    = Xtrain,
                    Xtest     = Xtest,
                    nvars     = nvars,
                    nobs      = nobs
                ),
                dataset       = ds,
                rep           = 0,
                explainer     = f'Local-{groundtruth.__name__}',
                fname         = f'{groundtruth_folder}/{groundtruth.__name__}-localexplanations.csv',
                filelock_name = filelock_name 
            )