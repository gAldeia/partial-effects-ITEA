import os
import sys
import shap
import lime
import lime.lime_tabular

import pandas         as pd
import numpy          as np
import autograd.numpy as npgrad

from filelock             import FileLock
from autograd             import grad, jacobian
from scipy.stats          import rankdata

#from eli5.permutation_importance import get_score_importances
from sklearn.metrics             import mean_squared_error

sys.path.append('../../docs/')
from feynman_equations import original_expressions, original_lambdas, datasets


# Takes an ITExpression and check the derivatives with autograd
def verify_derivative(itexpr, X, nvars, nobs):

    def ITExpr_predict(itexpr):

        # We need to use the autograd versions of numpy functions.
        grad_funcList = {
            'id'      : lambda x: x,
            'sin'     : npgrad.sin,
            'cos'     : npgrad.cos,
            'tanh'    : npgrad.tanh,
            'pexp'    : lambda x: (lambda exp_x: np.ma.array(exp_x, mask=(exp_x>=npgrad.exp(300)), fill_value=npgrad.exp(300) ).filled())(npgrad.exp(x)),
            'pexpn'   : lambda x: (lambda exp_x: np.ma.array(exp_x, mask=(exp_x>=npgrad.exp(300)), fill_value=npgrad.exp(300) ).filled())(npgrad.exp(-x)),
            'plog'    : lambda x: (lambda log_x: np.ma.array(log_x, mask=(log_x==np.nan), fill_value=0.0 ).filled())(npgrad.log(x)),
            'sqrtabs' : lambda x: npgrad.sqrt(npgrad.abs(x)),
            'sqrt'    : npgrad.sqrt,
            'arcsin'  : npgrad.arcsin,
        }

        # args should be an array representing a single observation [x1, x2, ...]        
        def predict(*args):

            Z = []
            for i, (ci, ni, fi) in enumerate( zip(itexpr.coeffs, itexpr.terms, itexpr.funcs) ):
                Z.append( ci*grad_funcList[fi](np.prod(np.power(args, ni))) )

            return np.sum(Z) + itexpr.intercept
        
        return predict

    # Performing the exhaustive check
    for i in range(nvars):
        dITExprdx = grad(ITExpr_predict(itexpr), i)

        for j in range(nobs):
            assert np.allclose( [dITExprdx(*X[j])], [np.sum(itexpr.dx(i, X[j]))] ), \
                "\n".join([
                    "Autograd and analytical derivatives are different.",
                    f"Expression: {itexpr}",
                    f"x: {X[j]}",
                    f"Partial derivative w.r.t {i}",
                    f"autograd value: {dITExprdx(*X[j])}",
                    f"IT dx value: {np.sum(itexpr.dx(i, X[j]))}",
                ])
            
    print(f"Successfully checked {nvars*nobs} derivatives!")


# Function to save explanation ranks instead of saving only the average.
def save_explanations(explanations, dataset, rep, explainer, fname, filelock_name):

    # Save importance for EACH observation (ranks can be calculated on post processing if desired)
    columns = ['Dataset', 'Explainer', 'Rep'] + [f'ImportanceObs{i}' for i in range(100)]
    data    = {c:[] for c in columns}
    df      = pd.DataFrame(columns=columns)

    with FileLock(filelock_name):

        # When the process have permission to write, then it must read the
        # values again to ensure that the most actual version is bein considered
        if os.path.isfile(fname):
            df = pd.read_csv(fname)
            data   = df.to_dict('list')

        # If save_explanations has been called, and exists already a line with
        # the new importances to write, then it is outdated. This happens because
        # we only call data explanations if the global explanations are not
        # calculated yet, but the script can be interrupted before the global
        # explanations are calculated and after some data explanations.
        if len( df[(df['Dataset']==dataset) & (df['Rep']==rep) & (df['Explainer']==explainer)] )>=1:
            print(f'Invalid data found for {dataset}-{rep}. They will be deleted')

            # Let's erase them
            df = df.drop(df[(df['Dataset']==dataset) & (df['Rep']==rep) & (df['Explainer']==explainer)].index)

        data   = df.to_dict('list')
        data['Dataset'  ].append(dataset)
        data['Explainer'].append(explainer)
        data['Rep'      ].append(rep)

        # Expecting that explanations is a numpy array, we'll convert it to string
        # and clean linebreaks
        for i in range(len(explanations)): 
            data[f'ImportanceObs{i}'].append(str(explanations[i]).replace('\n', ''))

        # There are explanators that gives only a single explanation
        # filling remaining columns if applicable
        for i in range(len(explanations), 100):
            data[f'ImportanceObs{i}'].append(np.nan)

        df = pd.DataFrame(data)
        
        df.to_csv(fname, index=False)


def SHAP_explainer(dataset, regressor, predict_f, Xtrain, Xtest, nvars, nobs):
    # calculating the shapley values
    explainer = shap.KernelExplainer(predict_f, Xtrain, silent=True)
    shapley   = explainer.shap_values(Xtest, silent=True)

    return shapley


def SHAP_adj_explainer(dataset, regressor, predict_f, Xtrain, Xtest, nvars, nobs):
    def signal(x):
        z = np.sign(x)
        z[z==0] =  1
    
        return z
        
    # calculating the shapley values
    shapley = SHAP_explainer(dataset, regressor, predict_f, Xtrain, Xtest, nvars, nobs)
    
    # Adjusting to make SHAP approximate PE
    for i in range(nvars):

        # sign will be used to prevent loosing sign information
        mean_diff = (Xtest[:, i] - Xtrain[:, i].mean())
        shapley[:, i] = signal(mean_diff) * shapley[:, i] / np.sqrt(1 + np.power(mean_diff, 2))

        # To avoid division by numbers close to zero when x is approx x.mean(), 
        # we use the analytic quotient 

    return shapley


# Model specific - explain ITEA expressions with PE
def PE_explainer(dataset, regressor, predict_f, Xtrain, Xtest, nvars, nobs):
    PE = np.zeros_like( Xtest )
    for i in range(nvars):
        PE[:, i] = np.sum(regressor.dx(dydx=i, x=Xtest), axis=1)
        
    return PE


def PE_adj_explainer(dataset, regressor, predict_f, Xtrain, Xtest, nvars, nobs):
    # Saving the gradients
    gradients = PE_explainer(dataset, regressor, predict_f, Xtrain, Xtest, nvars, nobs)

    # Adaptation to make PE correspond to SHAP
    importances = np.zeros_like( Xtest )
    for i in range(nvars):
        importances[:, i] = gradients[:, i].mean() * (Xtest[:, i] - Xtrain[:, i].mean())

    return importances


# Model agnostic - LIME explainer. predict_f should be a prediction function
def LIME_explainer(dataset, regressor, predict_f, Xtrain, Xtest, nvars, nobs):
    # LIME does not handle NaN/inf. We must create a protected predict function
    def predict_f_protected(x):
        pred = predict_f(x)
        
        if (np.isinf(pred).any() or np.isnan(pred).any() or np.any(pred > 1e+300) or np.any(pred < -1e+300)):
            return np.zeros( (x.shape[0], 1) )

        return pred

    explainer = lime.lime_tabular.LimeTabularExplainer(Xtrain, verbose=False, mode='regression')

    importances = np.zeros_like( Xtest )
    for i in range(nobs):
        explanation = explainer.explain_instance(Xtest[i], predict_f_protected, num_features=nvars, num_samples=100)

        for (feature_id, weight) in explanation.as_map()[0]:
            importances[i, feature_id] = weight

    return importances


# Model specific - Scikit regressor explanations with GINI
def GINI_explainer(dataset, regressor, predict_f, Xtrain, Xtest, nvars, nobs):

    # Converting to numpy array and making it an matrix of a single explanation
    return np.array([regressor.feature_importances_])


# Specific to original lambda equations
def gradient_explainer(dataset, regressor, predict_f, Xtrain, Xtest, nvars, nobs):

    # Automatic differentiation
    gradient_f = jacobian(original_lambdas[dataset])

    gradients = np.zeros_like( Xtest )
    for i in range(nobs):
        gradients[i, :] = gradient_f(Xtest[i, :])
        
    return gradients


def gradient_adj_explainer(dataset, regressor, predict_f, Xtrain, Xtest, nvars, nobs):

    gradients = gradient_explainer(dataset, regressor, predict_f, Xtrain, Xtest, nvars, nobs)

    # Adaptation to make PE correspond to SHAP
    importances = np.zeros_like( Xtest )
    for i in range(nvars):
        importances[:, i] = gradients[:, i].mean() * (Xtest[:, i] - Xtrain[:, i].mean())

    return importances


explainer_functions = {
    # ITEA specific
    'PE'            : PE_explainer,
    'PE_adj'        : PE_adj_explainer,

    # original lambdas specific
    'gradient'     : gradient_explainer,
    'gradient_adj' : gradient_adj_explainer,
    
    # tree ensemble specific
    'GINI'          : GINI_explainer,

    # Agnostic
    'SHAP'          : SHAP_explainer,
    'SHAP_adj'      : SHAP_adj_explainer,
    'LIME'          : LIME_explainer,
}