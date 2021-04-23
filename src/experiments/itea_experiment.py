import os
import sys
import time

import pandas as pd
import numpy  as np

from filelock             import FileLock
from sklearn.linear_model import LinearRegression

# Importing my files
sys.path.append('../itea/')
sys.path.append('../../docs/')

import transf_funcs

import itea as sr

from feynman_equations import original_expressions, original_lambdas, datasets
from _auxiliary        import *


# ------------------------------------------------------------------------------
# Experiment control
datasets_folder     = '../../datasets/'
filelock_name       = './locks/itea_lock.txt.lock'
fname_globalexpl    = f'../../results/tabular_raw/itea-globalexplanations.csv'
fname_localexpl     = f'../../results/tabular_raw/itea-localexplanations.csv'
fname_regressionres = f'../../results/tabular_raw/itea-regressionresults.csv'

check_derivatives   = False

n_runs              = 30

configuration = {
    'popsize'             : 300,
    'gens'                : 300, 
    'minterms'            : 1,
    'model'               : LinearRegression(),
    'expolim'             : (-5, 5),
    'maxterms'            : 15,
    'check_fit'           : False,
    'check_vif'           : False,
    'vif_threshold'       : 5,
    'simplify'            : True,
    'simplify_threshold'  : 1e-2,
    'funs'                : {k:v for k, v in transf_funcs.transf_funcs.items()}, # if k in ['id', 'sin', 'cos', 'tanh', 'sqrt', 'pexp', 'plog', 'pexpn', 'arcsin']}, 
}

explainers = ['PE', 'PE_adj', 'SHAP', 'SHAP_adj', 'LIME']
# ------------------------------------------------------------------------------


if __name__ == '__main__':        
    assert len(sys.argv) > 1, \
    f"""
    Please inform one or more datasets (separated by whitespace) to run.
    It is possible to run parallel processes without any complication ONLY
    IF none of the processess are running over the same dataset.
    Possible datasets are: {" ".join(datasets)}
    """

    for ds in sys.argv[1:]:
        ds = str(ds)

        assert ds in datasets, \
            f'dataset {ds} is not listed for the experiments.'

        Xtrain, ytrain, Xtest, ytest = None, None, None, None
        
        try:
            dataset_data = np.loadtxt(f'{datasets_folder}{ds}.dat', delimiter=',', skiprows=1)
            LHS_data     = np.loadtxt(f'{datasets_folder}{ds}_LHS.dat', delimiter=',', skiprows=1)
            label        = np.loadtxt(f'{datasets_folder}{ds}.dat', delimiter=',', max_rows=1, dtype=str)

            Xtrain = dataset_data[:, :-1]
            ytrain = dataset_data[:, -1]
            
            nobs   = Xtrain.shape[0]
            nvars  = Xtrain.shape[1]

            Xtest = LHS_data[:, :-1]
            ytest = LHS_data[:, -1]
        except:
            print(f'Could not open data set {ds}.')
            sys.exit()

        # Only performance informations. Explanations will be stored on separated files
        columns = [ # All column names should be capitalized
            # Dataset and rep are used as keys to merge different results files
            'Dataset',            # Dataset name 
            'Rep',                # Number of the repetition 
            
            'RMSE-train-ITEA',    # Final RMSE error on training data
            'RMSE-test-ITEA',     # Final RMSE error on training data
            'Tottime-train-ITEA', # Training time

            'NumberOfTerms',      # (itea exclusive) number of IT terms
            'TreeLength',         # (itea exclusive) tree length for ITExpr
            'Expression',         # (itea exclusive) Final ITEA expression
        ] + [
            f'Tottime-{explainer}-ITEA' 
            for explainer in explainers
        ]


        # Creating or recovering the results file
        results   = {c:[] for c in columns}
        resultsDF = pd.DataFrame(columns=columns)

        if os.path.isfile(fname_regressionres):
            resultsDF = pd.read_csv(fname_regressionres)
            results   = resultsDF.to_dict('list')    


        print(f'Executing now for dataset {ds}')
        for rep in range(n_runs):

            # Restart from checkpoint
            if len( resultsDF[(resultsDF['Dataset']==ds) & (resultsDF['Rep']==rep)] )==1:
                print(f'already evaluated {ds}-{rep}')
                continue

            # You SHOULD NOT RUN PARALLEL PROCESSES OVER THE SAME DATASET
            if len( resultsDF[(resultsDF['Dataset']==ds) & (resultsDF['Rep']==rep)] )>1:
                print(f'There is more experiments than {n_runs}')
                continue

            print(f'evaluating {ds}-{rep}')

                
            # Running the regression
            TimeTrain = time.time()
            itea       = sr.ITEA(label=label, **configuration).run(Xtrain, ytrain)
            bestsol    = itea.symbolic_f
            TimeTrain  = time.time() - TimeTrain


            # Optional step for ITEA
            if check_derivatives:
                verify_derivative(bestsol, Xtrain, nvars, nobs)

            
            # Evaluating global and local explanations for all explainers
            tottimes = []
            for explainer in explainers:
                print(f'Explaining with {explainer}...')

                starttime = time.time()
                
                # Saving every explanation for global data
                save_explanations(
                    explanations  = explainer_functions[explainer](
                        dataset   = ds,
                        regressor = bestsol,
                        predict_f = bestsol.predict,
                        Xtrain    = Xtrain,
                        Xtest     = Xtrain,
                        nvars     = nvars,
                        nobs      = nobs
                    ),
                    dataset       = ds,
                    rep           = rep,
                    explainer     = f'Global-{explainer}-ITEA',
                    fname         = fname_globalexpl,
                    filelock_name = filelock_name
                )

                # Saving local data
                save_explanations(
                    explanations  = explainer_functions[explainer](
                        dataset   = ds,
                        regressor = bestsol,
                        predict_f = bestsol.predict,
                        Xtrain    = Xtrain,
                        Xtest     = Xtest,
                        nvars     = nvars,
                        nobs      = nobs
                    ),
                    dataset       = ds,
                    rep           = rep,
                    explainer     = f'Local-{explainer}-ITEA',
                    fname         = fname_localexpl,
                    filelock_name = filelock_name
                )

                tottimes.append(time.time() - starttime)


            # Locking write on file and saving the results
            with FileLock(filelock_name):
                if os.path.isfile(fname_regressionres):
                    resultsDF = pd.read_csv(fname_regressionres)
                    results   = resultsDF.to_dict('list')

                # Saving regression informations.
                results['Dataset'                ].append(ds)
                results['Rep'                    ].append(rep)
                results['RMSE-train-ITEA'        ].append(bestsol.fitness)
                results['RMSE-test-ITEA'         ].append(mean_squared_error(bestsol.predict(Xtest), ytest, squared=False))                
                results['NumberOfTerms'          ].append(bestsol.len)
                results['TreeLength'             ].append(bestsol.tree_len())
                results['Tottime-train-ITEA'     ].append(TimeTrain)
                results['Expression'             ].append(str(bestsol))

                for i, explainer in enumerate(explainers):
                    results[f'Tottime-{explainer}-ITEA'].append(tottimes[i])

                df = pd.DataFrame(results)
                df.to_csv(fname_regressionres, index=False)

        print('done')