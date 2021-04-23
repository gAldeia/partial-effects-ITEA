import numpy  as np
import pandas as pd

import time

from sklearn.linear_model                 import LinearRegression 
from statsmodels.stats.outliers_influence import variance_inflation_factor

from typing import Dict, Callable, List, Tuple, Union

# Simple type definition for clearer code
# (type check is just for clearer code. Numpy lacks deeper compatibility)

# TODO: improve typing
FuncsList = Dict[str, Callable] # Dict with name of the function as key, transf. function as value
MutatList = Dict[str, Callable] # Dict with name of mutation as key, mutation function as value
Funcs     = List[str]           # List of transformation function keys
Terms     = List[List[int]]     # Matrix of integers representing exponents for an IT expression
Coeffs    = List[float]         # List of coefficients for each IT term
IT        = Tuple[Terms, Funcs] # IT expression
Vars      = List[float]         # List with variabce of each IT Term
    
np.seterr(all='ignore')
    

# IT expression class
class ITExpr:

    # Memorization: dict shared between instances. Store fit() results
    _memory = dict()


    def __init__(self, ITs: IT, funcList: FuncsList, label: List[str] = []) -> None:
        
        assert len(ITs[0])>0 and len(ITs[1])>0, 'Expression created without terms'

        self.terms: Terms
        self.funcs: Funcs

        self.terms, self.funcs = ITs

        self.label    : List[str] = label
        self.funcList : FuncsList = funcList
        self.len      : int       = len(self.terms)

        # Variables below are modified when fit() is called
        self.intercept : float  = 0.0
        self.coeffs    : Coeffs = np.ones(self.len)
        self._fitness  : float  = -1.0
        self._vars     : Vars   = np.ones(self.len)/self.len
        
        # The fitness is always "private" (_fitness) because it is only
        # calculated when the expression participates on a tournament. Only the
        # final expression returned by the ITEA will have a "public" (fitness) fitness.


    def __str__(self) -> str:
        terms_str = [] 

        for c, f, t in zip(self.coeffs, self.funcs, self.terms):
            c_str = "" if round(c, 3) == 1.0 else f'{round(c, 3)}*'
            f_str = "" if f == "id" else f
            t_str = ' * '.join([
                "placeholder_x" + str(i) + ("^"+str(ti) if ti!=1 else "")
                for i, ti in enumerate(t) if ti!=0
            ])

            terms_str.append(f'{c_str}{f_str}({t_str})')

        expr_str = ' + '.join(terms_str)

        if len(self.label)>0:
            for i, l in enumerate(self.label):
                expr_str = expr_str.replace(f'placeholder_x{i}', l)
        else:
            expr_str = expr_str.replace(f'placeholder', '')

        return expr_str  + ("" if self.intercept == 0.0 else f' + {round(self.intercept, 3)}')


    # Inner eval to create the matrix for fitting the linear regression method
    def _eval(self, X: List[List[float]]) -> List[List[float]]:

        Z = np.zeros( (len(X), self.len) )

        for i, (ni, fi) in enumerate( zip(self.terms, self.funcs) ):
            #Z[:, i] = [self.funcList[fi](z) for z in Z[:, i]]

            non_zeros = np.where(ni != 0)[0]
            Z[:, i] = self.funcList[fi](np.prod(np.power(X[:, non_zeros], ni[non_zeros]), axis=1))

        return Z


    # Adjusts the weights and intercept of the expression and then evaluates the fitness
    def fit(self, model, X: List[List[float]], y: List[float], refit=False) -> Union[float, None]:
        
        # Already has been fitted before (remember, _fitness is only calculated
        # on tournament)
        if self._fitness != -1.0:
            return self._fitness

        # Memorization of previous expressions
        key_t = b''.join([t.tobytes() for t in self.terms])
        key_f = b''.join([f.encode()  for f in self.funcs])

        key = (key_t, key_f)

        if (key not in ITExpr._memory) or (refit):
            Z = self._eval(X)
            
            # fitness of expressions with terms evaluating to NaN is 1e+300
            if (np.isinf(Z).any() or np.isnan(Z).any() or np.any(Z > 1e+300) or np.any(Z < -1e+300)):                
                ITExpr._memory[key] = (
                    np.ones(self.len),            # Coeffs
                    np.ones(self.len)/self.len,   # Variances
                    0.0,                          # Intercept
                    1e+300                        # Fitness
                )
            else:
                model.fit(Z, y)

                ITExpr._memory[key] = (
                    model.coef_.tolist(),
                    np.var( (Z * model.coef_), axis=0 ),
                    model.intercept_,
                    np.sqrt(np.square(model.predict(Z) - y).mean())
                )

        self.coeffs, self._vars, self.intercept, self._fitness = ITExpr._memory[key] 

        return self._fitness


    # Takes a matrix of observations and return a array of predictions.
    # For predicting single observations, use reshape(1, -1)
    def predict(self, X: List[List[float]]) -> float:

        return np.dot(self._eval(X), self.coeffs) + self.intercept


    # Calculates the number of nodes that the expression would have if it was
    # a symbolic tree
    def tree_len(self):
        tlen = 0
        for c, f, t in zip(self.coeffs, self.funcs, self.terms):
            if c==0.0:
                continue
            elif c != 1.0:
                tlen += 3 # product between term and coeff
            
            tlen += 1 # transf function
            tlen += len(t)-1 # products between variables
            tlen += len(t==1) # exponents equal to 1
            tlen += 3*len(t[(t!=0) & (t!=1)]) #exponents != 0 and != 1
            tlen -= len(t==0) # Ignoring variable associated with 0 exponents
        
        return tlen + self.len - 1


    # Calculates the partial derivative for variable dydx and the given
    # observations x (should be numpy matrix of observations), and return
    # the derivative of each term in an array (final derivative is the sum(array)).
    # representative should be a function that takes multiple values and returns
    # one representative value (i.e. mean, median)
    def dx(self, dydx, x, representative=None):

        if len(x.shape) == 1:
            x = x.reshape(1, -1)

        # Plus one column to represent the derivative of the intercept
        res = np.zeros( (len(x), self.len+1) )

        for i, (ni, fi, ci) in enumerate(zip(self.terms, self.funcs, self.coeffs)):

            intermed_ni = ni.copy() 

            intermed_ni[dydx] = 0

            if representative != None:
                pi_d = np.power(x[:, dydx], ni[dydx]-1) * representative(np.prod(x**intermed_ni, axis=1))
                res[:, i] = ci*ni[dydx]*self.funcList[fi].dx(pi_d * x[:, dydx]) * pi_d
            else:
                pi_d = np.power(x[:, dydx], ni[dydx]-1) * np.prod(x**intermed_ni, axis=1)
                res[:, i] = ci*ni[dydx]*self.funcList[fi].dx(pi_d * x[:, dydx]) * pi_d

        return res


# Mutation handler class, instance takes an expression and applies random mutation
class MutationIT:
    def __init__(self, minterms: int, maxterms: int, nvars: int, expolim: int, funs: FuncsList) -> None:
        self.minterms = minterms
        self.maxterms = maxterms
        self.nvars    = nvars
        self.expolim  = expolim
        self.funs     = funs

        self.singleITGenerator = _randITBuilder(1, 1, nvars, expolim, funs)

        # Mutations do never modify the passed arguments. Mutation functions
        # expects tuple with (terms, funcs, variances)

    
    def _mut_drop(self, ITs) -> IT:
        terms, funcs = ITs

        index = np.random.choice(len(terms))

        mask = [True if i is not index else False for i in range(len(terms))]

        return (terms[mask], funcs[mask]), index


    def _mut_add(self, ITs) -> IT:
        terms, funcs = ITs
        newt,  newf  = next(self.singleITGenerator)
        
        return ( np.concatenate((terms, newt)), np.concatenate((funcs, newf)) ), len(terms)+1


    def _mut_term(self, ITs) -> IT:
        terms, funcs = ITs

        index = np.random.choice(len(terms))

        newt, _ = next(self.singleITGenerator)
        newf = [funcs[index]]

        mask = [True if i is not index else False for i in range(len(terms))]

        return ( np.concatenate((terms[mask], newt)), np.concatenate((funcs[mask], newf)) ), index


    def _mut_func(self, ITs) -> IT:
        terms, funcs = ITs

        index = np.random.choice(len(terms))

        _, newf = next(self.singleITGenerator)
        newt = [terms[index]]

        mask = [True if i is not index else False for i in range(len(terms))]

        return ( np.concatenate((terms[mask], newt)), np.concatenate((funcs[mask], newf)) ), index


    def _mut_interp(self, ITs) -> IT:
        terms, funcs = ITs

        term1_index, term2_index = np.random.choice(len(terms), size=2)

        newt = terms[term1_index] + terms[term2_index]

        newt[newt < self.expolim[0]] = self.expolim[0]
        newt[newt > self.expolim[1]] = self.expolim[1]
        
        return ( np.concatenate((terms, [newt])), np.concatenate((funcs, [funcs[term1_index]])) ), len(terms)+1


    def _mut_intern(self, ITs) -> IT:
        terms, funcs = ITs

        term1_index, term2_index = np.random.choice(len(terms), size=2)

        newt = terms[term1_index] - terms[term2_index]

        newt[newt < self.expolim[0]] = self.expolim[0]
        newt[newt > self.expolim[1]]  = self.expolim[1]
        
        return ( np.concatenate((terms, [newt])), np.concatenate((funcs, [funcs[term1_index]])) ), len(terms)+1


    def mutate(self, ITs) -> IT:
        mutations = {
            'term' : self._mut_term,
            #'func' : self._mut_func
        }

        if len(ITs[0]) > self.minterms:
            mutations['drop'] = self._mut_drop
        
        if len(ITs[0]) < self.maxterms:
            mutations['add']  = self._mut_add

            mutations['interpos'] = self._mut_interp
            mutations['interneg'] = self._mut_intern

        mut_ = np.random.choice(list(mutations.keys()))

        # Mutation returns the term added/removed/modified separated
        (newt, newf), index = mutations[mut_](ITs)

        return (newt, newf), (mut_, index)


# Random term infinite list
def _randITBuilder(minterms: int, maxterms: int, nvars: int, expolim: Tuple[int], funs: FuncsList) -> IT:
    while True:
        nterms = np.random.randint(minterms, maxterms + 1)

        terms: Terms = np.random.randint(expolim[0], expolim[1] + 1, size=(nterms, nvars)) 
        funcs: Funcs = np.random.choice(list(funs.keys()), size=nterms)

        yield (terms, funcs)


# Class to perform the symbolic regression
class ITEA:
    def __init__(self, funs, minterms, maxterms, model, expolim, popsize, gens, check_fit=False, check_vif=False, vif_threshold=5, simplify=False, simplify_threshold=1e-8, label=[]):

        # Required parameters:
        self.funs     = funs
        self.minterms = minterms
        self.maxterms = maxterms
        self.model    = model
        self.expolim  = expolim
        self.popsize  = popsize
        self.gens     = gens

        # Optional. Quality check of solutions during evolution:
        self.check_fit          = check_fit     # Should discard terms that eval to NaN?
        self.check_vif          = check_vif     # Should discard terms with VIF > vif_threshold?
        self.simplify           = simplify      # Should discard terms with prediction variance smaller than simplify_threshold?
        
        # Thresholds. Not considered if check_vif/simplify is False.
        self.vif_threshold      = vif_threshold
        self.simplify_threshold = simplify_threshold

        # Optional. Save execution info on csv:
        self.label         = label
        self.log_dict      = None
        self.log_df        = None
        
        # Memorizing invalid terms. MUST be cleaned before finishing run()
        # Dicionário para memorizar termos inválidos (DEVE ser resetado sempre que mudar a base de dados)
        self._memory     = dict()


    # Applies the quality checks before using the terms into an expression.
    # This is not done in the constructor to guarantee that if a expression is
    # created, it will follow the quality control. The cleaning goes from the
    # least computationally expensive to the most.
    def _sanitizeIT(self, ITs: IT, funcsList, X: List[List[float]], y) -> Union[IT, None]:

        terms, funcs = ITs[0], ITs[1]

        # This mask will be manipulated trhough verifications. This is the core
        # of the cleaning process
        mask = np.full( len(terms), False )

        # Selecting only unique terms
        _, unique_ids = np.unique(np.column_stack((terms, funcs)), return_index=True, axis=0)

        # Matrix where each column in a term evaluated over all X
        Z = np.zeros( (len(X), len(terms)) )

        # We actually fill Z only if is needed. If all conditions below are false,
        # then the user don't want expressions to be cleaned. Then, we'll focus only
        # on removing repeated terms and terms with all exponents equal to zero.
        if self.check_fit or self.check_vif or self.simplify:
            for (unique_id, t, f) in zip(unique_ids, terms[unique_ids], funcs[unique_ids]):
                non_zeros = np.where(t != 0)[0]
                Z[:, unique_id] = funcsList[f](np.prod(np.power(X[:, non_zeros], t[non_zeros]), axis=1))

        # Focusing only on unique terms
        for unique_id in unique_ids:
            assert funcs[unique_id] in funcsList.keys(), f'{funcs[unique_id]} is not a valid function'

            # Check fit if allowed and if any exponent is not zero
            if np.any(terms[unique_id]!=0):
                # OBS: the fit() method of ITExpr already discard the WHOLE expression
                # if fails to fit one term. This check is computationally expensive,
                # but could lead to less waste of expressions in special cases
                if self.check_fit:
                    mask[unique_id] = not (
                        np.isinf(Z[:, unique_id]).any() or np.isnan(Z[:, unique_id]).any() or
                        np.any(Z[:, unique_id] > 1e+300) or np.any(Z[:, unique_id] < -1e+300))
                else:
                    mask[unique_id] = True

        # Simplifying by the variance of the term in relation to total variance of the expression
        if self.simplify:

            # Try block because it is not guaranteed that evey term will fit
            # (simplify is independent of check_fit, so we need to do this)
            try:
                self.model.fit(Z[:, mask], y)

                pred_std = np.std( (Z[:, mask] * self.model.coef_), axis=0 )
                mask[np.where(mask)] = pred_std/np.sum(pred_std) > self.simplify_threshold
            except:
                pass

        # Removing terms with high multicolinearity
        if self.check_vif:
            try:
                multicolinearities = np.array([variance_inflation_factor(Z[:, mask], i) 
                    for i in range(np.sum(mask))])
                mask[np.where(mask)] = multicolinearities < self.vif_threshold
            except:
                pass
        
        # Return either a expression or None. Must be handled by the calling method.
        return (terms[mask], funcs[mask]) if np.any(mask) else None


    # Generates initial population (with guarantee that there will be the specified
    # number of expressions), respecting bounds for expressions. Since crossover
    # and mutation can create invalid terms, we start with the correct number of
    # individuals
    def _generate_random_pop(self) -> List[IT]:
        randITGenerator = _randITBuilder(self.minterms, self.maxterms, self.nvars, self.expolim, self.funs)
        
        pop = []

        while len(pop) < self.popsize:
            itxClean = self._sanitizeIT(next(randITGenerator), self.funs, self.Xtrain, self.ytrain)

            if itxClean:
                itexpr = ITExpr(itxClean, self.funs)
                
                #itexpr.fit(self.model, self.Xtrain, self.ytrain)
                pop.append(itexpr)
            
        return pop


    # "private" mutation, to allow vectorizing the function with numpy
    def _mutate(self, ind) -> List[IT]:

        mt, trace = self.mutate.mutate( (ind.terms, ind.funcs) )
        
        itxClean = self._sanitizeIT(mt, self.funs, self.Xtrain, self.ytrain)
        
        if itxClean:
            itexpr = ITExpr(itxClean, self.funs)
        
            #itexpr.fit(self.model, self.Xtrain, self.ytrain)
            return itexpr
        
        return None


    # Performs the regression. Verbose lets the user specifies the number og
    # generations before each new progress notification on terminal
    def run(self, Xtrain, ytrain, log=False, verbose=False):
        
        ITExpr._memory = dict()
        self._memory   = dict()

        self.Xtrain  = Xtrain
        self.ytrain  = ytrain
        self.nvars   = len(Xtrain[0])

        self.mutate    = MutationIT(self.minterms, self.maxterms, self.nvars, self.expolim, self.funs)

        # Columns for log file
        columns = ['lowstfit', 'mean', 'std']

        if log or verbose:
            self.log_dict = {c:[] for c in columns}
        if verbose:
            print('gen \t best_fitness \t p_mean_fitness \t p_std_fitness \t remaining_t')

        # Fitness tie: wins the expression with smaller number of terms.
        # Tournament uses fit() function of ITExpr, that will calculate _fitness.
        ftournament = np.vectorize(
            lambda x, y: x if (x.fit(self.model, self.Xtrain, self.ytrain), x.len) < (y.fit(self.model, self.Xtrain, self.ytrain), y.len) else y)

        fmutation = np.vectorize(self._mutate)

        pop = self._generate_random_pop()

        # Circular list to give simple estimations of remaining time
        last_5_times    = np.zeros(5)
        last_5_times[:] = np.nan 

        for g in range(self.gens):

            t = time.time()

            child_aux = fmutation(pop)
            child = child_aux[child_aux != None]
            
            # tournament is done with parents + child, because at one given
            # iteration if mutation fails completely there still have enough
            # solutions to compete
            pop = np.concatenate( (pop, child) )

            pop = ftournament(*np.random.choice(pop, size=(2, self.popsize))) 
                    
            if log or verbose: 
                pop_fitness  = np.array([itexpr.fit(self.model, self.Xtrain, self.ytrain) for itexpr in pop])
                lowstfit_idx = np.argmin(pop_fitness) 
                    
                mean, std   = np.mean(pop_fitness), np.std(pop_fitness)

                gen_data = [pop_fitness[lowstfit_idx], mean, std]

                for column, data in zip(columns, gen_data):
                    self.log_dict[column].append(data)
                    
                # Estimating remaining time and printing info
                if ((verbose) and (g%verbose==0)):
                    last_5_times[0] = time.time() - t
                    last_5_times    = np.roll(last_5_times, 1)

                    remain_s_tot = int(np.ceil(np.nanmean(last_5_times) * (self.gens - g - 1)))

                    remaining = f"{remain_s_tot // 60}min{remain_s_tot % 60}seg"
                            
                    print(f'{g}/{self.gens}\t{self.log_dict["lowstfit"][g]}\t{self.log_dict["mean"][g]}\t{self.log_dict["std"][g]}\t{remaining}')


        # Creating the final fitness variable (public) for every equation
        for itexpr in pop:
            itexpr.fitness = itexpr.fit(self.model, self.Xtrain, self.ytrain)

        # Getting the best solution of the final population
        self.symbolic_f = min(pop, key= lambda itexpr: (itexpr.fitness, itexpr.len))
        self.symbolic_f.label = self.label

        if log:
            self.log_df = pd.DataFrame(
                data    = [self.log_dict[c] for c in columns],
                columns = [f'gen {i}' for i in range(self.gens)],
                index   = columns
            )
            if isinstance(log, str):
                self.log_df.to_csv(log)
        
        ITExpr._memory = dict()
        self._memory   = dict()

        return self


# ---------------------------------------------------------------------------------
if __name__ == '__main__':

    from sklearn.model_selection import train_test_split
    from transf_funcs import *

    df = pd.read_csv('../../datasets/I.10.7.dat')

    label = df.columns.values[:-1]

    X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42
    )

    params = {
        # Required
        'funs'          : transf_funcs,
        'minterms'      : 1,
        'maxterms'      : 10,
        'model'         : LinearRegression(),
        'expolim'       : (-3, 3),
        'popsize'       : 100,
        'gens'          : 100,

        # Optional (has default value)
        'check_fit'           : False,
        'check_vif'           : False,
        'vif_threshold'       : 5,
        'simplify'            : False,
        'simplify_threshold'  : 1e-8,
        'label'               : label,
    }

    print('Running ITEA...')

    symbreg = ITEA(**params)
    best    = symbreg.run(X_train, y_train, log='itea_exemple.csv', verbose=10).symbolic_f
    print(best)

    print(f'Train fitness: {best.fitness}')
    print(f'Test fitness:  {np.sqrt(np.square(best.predict(X_test) - y_test).mean())}')