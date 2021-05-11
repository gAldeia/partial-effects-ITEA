# experiment to evaluate the mse error between PE and SHAP over the original
# equations.
# for each dataset we'll calculate the local and global explanations using
# the governing equation, and compare PE_adj against SHAP, and PE against
# SHAP_adj

import os
import sys
import time

import pandas as pd
import numpy  as np

from filelock             import FileLock

# Importing my files
sys.path.append('../../docs/')

from feynman_equations import original_expressions, original_lambdas, datasets
from _auxiliary        import *


# ------------------------------------------------------------------------------
# Experiment control
datasets_folder     = '../../datasets/'
filelock_name       = './locks/pevsshap_lock.txt.lock'
fname_globalexpl    = f'../../results/tabular_raw/PEvsSHAP-globalexplanations.csv'
fname_localexpl     = f'../../results/tabular_raw/PEvsSHAP-localexplanations.csv'
fname_regressionres = f'../../results/tabular_raw/PEvsSHAP-regressionresults.csv'

n_runs = 1
explainers = ['gradient', 'gradient_adj', 'SHAP', 'SHAP_adj']
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
            
            'RMSE-train-PEvsSHAP',    # Final RMSE error on training data
            'RMSE-test-PEvsSHAP',     # Final RMSE error on training data
            'Tottime-train-PEvsSHAP', # Training time
        ] + [
            f'Tottime-{explainer}-PEvsSHAP' 
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
            TimeTrain   = time.time()
            original_eq = lambda X: np.array([original_lambdas[ds](X[i, :]) for i in range(len(X))])
            TimeTrain   = time.time() - TimeTrain
            
            # Evaluating global and local explanations for all explainers
            tottimes = []
            for explainer in explainers:
                print(f'Explaining with {explainer}...')

                starttime = time.time()
                
                # Saving every explanation for global data
                save_explanations(
                    explanations  = explainer_functions[explainer](
                        dataset   = ds,
                        regressor = original_eq, #only matters for ITEA
                        predict_f = original_eq,
                        Xtrain    = Xtrain,
                        Xtest     = Xtrain,
                        nvars     = nvars,
                        nobs      = nobs
                    ),
                    dataset       = ds,
                    rep           = rep,
                    explainer     = f'Global-{explainer}-PEvsSHAP',
                    fname         = fname_globalexpl,
                    filelock_name = filelock_name
                )

                # Saving local data
                save_explanations(
                    explanations  = explainer_functions[explainer](
                        dataset   = ds,
                        regressor = original_eq,
                        predict_f = original_eq,
                        Xtrain    = Xtrain,
                        Xtest     = Xtest,
                        nvars     = nvars,
                        nobs      = nobs
                    ),
                    dataset       = ds,
                    rep           = rep,
                    explainer     = f'Local-{explainer}-PEvsSHAP',
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
                results['Dataset'               ].append(ds)
                results['Rep'                   ].append(rep)
                results['RMSE-train-PEvsSHAP'   ].append(mean_squared_error(original_eq(Xtrain), ytrain, squared=False))
                results['RMSE-test-PEvsSHAP'    ].append(mean_squared_error(original_eq(Xtest), ytest, squared=False))                
                results['Tottime-train-PEvsSHAP'].append(TimeTrain)

                for i, explainer in enumerate(explainers):
                    results[f'Tottime-{explainer}-PEvsSHAP'].append(tottimes[i])

                df = pd.DataFrame(results)
                df.to_csv(fname_regressionres, index=False)

        print('done')