How to use the ITEA implementation with Partial Effects
=====

In this markdown, I will talk about some specific observations to have in mind if you are going to run my python implementation of the ITEA algorithm.

To run the ITEA, first, you need to create an instance of the class, then calls the fit method. File ```./src/itea/itea.py``` have a simple example at the end:

```python
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

itea = ITEA(**params)

# run() returns self
itea.run(X_train, y_train, log='itea_exemple.csv', verbose=10)

# symbolic_f is the final function
best = itea.symbolic_f

print(best)

# Or you can do it in one line:
#best = ITEA(**params).run(X_train, y_train, log='itea_exemple.csv', verbose=10).symbolic_f

print(f'Train fitness: {best.fitness}')
print(f'Test fitness:  {np.sqrt(np.square(best.predict(X_test) - y_test).mean())}')
```

Transformation functions
-----

There are some particularities with the transformation functions for this implementation.

Originally, the transformation functions should be a dictionary where the keys are the names of the functions, and the values can be any unary function (compatible with NumPy vectorization, which means that it should apply the unary function element-wise when the passed value is a vector. The simpler way is to use NumPy functions).

This way, for the code above, this example below should work:

```python
transf_funcs = {
    'id' : lambda x: x, 
    'sin': np.sin,
    'tanh'
}

# We can call a function by accessing the dict with the key. This is
# how ITEA will use the transformation functions.
print( transf_funcs['sin'](0.0) )
```

During the implementation of the Partial Effect study, I decided that the transformation functions should be a little more elaborate than that. 

I wanted to still have the same behavior showed in the example, but as I would work with the derivatives, I also wanted to have an easy way to evaluate the derivative of the function (which must be given) without having a whole new list to store the transformation function derivatives.

So I decided that the transformation functions should be a class that implements the ```__call__``` method (so it will work the same way as the previous approach), but also have the ```.dx()``` method that can be used to evaluate the derivative.

Now, I create the function as classes, and the transformation function uses as value one instance of the class:

```python
class func_id:
    class dx:
        def __call__(self, x)       : return np.ones_like(x)

    def __init__(self)          : self.dx = self.dx()
    def __call__(self, x)       : return x

transf_funcs = {
    'id'  : func_id()
}
```

Summarizing:
1. ```transf_funcs``` should be a dict
2. the keys of the dict are the name of the transformation functions
3. the values should be callable
4. the callable function should work as NumPy functions (vectorization)
5. the ITEA will use the function named 'func' by accessing the callable: ```transf_funcs['func'](x)```
6. optionally, if you want to calculate the partial effect, the callable should have a ```.dx()``` method. In other words: ```transf_funcs['func'].dx(x)``` must work.