import numpy as np


class func_id:
    class dx:
        def __call__(self, x): return np.ones_like(x)

    def __init__(self): self.dx = self.dx()
    def __call__(self, x): return x


class func_sin:
    class dx:
        def __call__(self, x): return np.cos(x)

    def __init__(self): self.dx = self.dx()
    def __call__(self, x): return np.sin(x)


class func_cos:
    class dx:
        def __call__(self, x): return -np.sin(x)

    def __init__(self): self.dx = self.dx()
    def __call__(self, x): return np.cos(x)


class func_tanh:
    class dx:
        def __call__(self, x): return np.power(1/np.cosh(x), 2) # (Hyperbolic sec) ^ 2

    def __init__(self): self.dx = self.dx()
    def __call__(self, x): return np.tanh(x)


class func_arcsin:
    class dx:
        def __call__(self, x): return 1/np.sqrt(1 - x**2)

    def __init__(self): self.dx = self.dx()
    def __call__(self, x): return np.arcsin(x)


class func_sqrt:
    class dx:
        def __call__(self, x): return 1/(2*np.sqrt( x ))

    def __init__(self): self.dx = self.dx()
    def __call__(self, x): return np.sqrt(x)


class func_sqrtabs:
    class dx:
        def __call__(self, x): return x/(2*(np.abs( x )**(3/2)))

    def __init__(self): self.dx = self.dx()
    def __call__(self, x): return np.sqrt(np.abs( x ))


class func_pexp:
    class dx:
        def __call__(self, x): return (lambda exp_x: np.ma.array(exp_x, mask=(exp_x>=np.exp(300)), fill_value=np.exp(300) ).filled())(np.exp(x))

    def __init__(self): self.dx = self.dx()
    def __call__(self, x): return (lambda exp_x: np.ma.array(exp_x, mask=(exp_x>=np.exp(300)), fill_value=np.exp(300) ).filled())(np.exp(x))


class func_pexpn:
    class dx:
        def __call__(self, x): return (lambda exp_x: np.ma.array(exp_x, mask=(exp_x>=np.exp(300)), fill_value=np.exp(300) ).filled())(-np.exp(-x))

    def __init__(self): self.dx = self.dx()
    def __call__(self, x): return (lambda exp_x: np.ma.array(exp_x, mask=(exp_x>=np.exp(300)), fill_value=np.exp(300) ).filled())(np.exp(-x))


class func_plog:
    class dx:
        def __call__(self, x): return (lambda x: np.ma.array(x, mask=(np.abs(x)<=1e-8), fill_value=0.0).filled())(1/x)

    def __init__(self): self.dx = self.dx()
    def __call__(self, x): return (lambda log_x: np.ma.array(log_x, mask=(log_x==np.nan), fill_value=0.0 ).filled())(np.log(x))


transf_funcs = {
    # Identity
    'id'  : func_id(),
    
    # Trigonometric
    'sin'    : func_sin(),
    'cos'    : func_cos(),
    'tanh'   : func_tanh(),
    'arcsin' : func_arcsin(),
    
    # tan?
    # gauss?
    # sin(x)/x?

    # Convex
    'sqrt'    : func_sqrt(),
    'sqrtabs' : func_sqrtabs(),
    
    # abs?
    # relu?

    # Logarithimic/exponential
    'pexp'  : func_pexp(),
    'pexpn' : func_pexpn(),
    'plog'  : func_plog(),
}