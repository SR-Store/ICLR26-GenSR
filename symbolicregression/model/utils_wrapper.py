from abc import ABC, abstractmethod
import sklearn
from scipy.optimize import minimize
import numpy as np
import time
import torch
from functools import partial
import traceback

class TimedFun:
    def __init__(self, fun, verbose=False, stop_after=3):
        self.fun_in = fun
        self.started = False
        self.stop_after = stop_after
        self.best_fun_value = np.inf
        self.best_x = None
        self.loss_history=[]
        self.verbose = verbose

    def fun(self, x, *args):
        if self.started is False:
            self.started = time.time()
        elif abs(time.time() - self.started) >= self.stop_after:
            self.loss_history.append(self.best_fun_value)
            raise ValueError("Time is over.")
        self.fun_value = self.fun_in(x, *args)
        self.loss_history.append(self.fun_value)
        if self.best_x is None:
            self.best_x=x
        elif self.fun_value < self.best_fun_value:
            self.best_fun_value=self.fun_value
            self.best_x=x
        self.x = x
        return self.fun_value

class Scaler(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def fit(self, X):
        pass

    @abstractmethod
    def fit_transform(self, X):
        pass

    @abstractmethod
    def transform(self, X):
        pass

    @abstractmethod
    def get_params(self):
        pass

    def rescale_function(self, env, tree, a, b):
        prefix = tree.prefix().split(",")
        idx = 0
        while idx < len(prefix):
            if prefix[idx].startswith("x_"):
                k = int(prefix[idx][-1])
                if k>=len(a): 
                    continue
                a_k, b_k = str(a[k]), str(b[k])
                prefix_to_add = ["add", b_k, "mul", a_k, prefix[idx]]
                prefix = prefix[:idx] + prefix_to_add + prefix[min(idx + 1, len(prefix)):]
                idx += len(prefix_to_add)
            else:
                idx+=1
                continue
        rescaled_tree = env.word_to_infix(prefix, is_float=False, str_array=False)
        return rescaled_tree

class StandardScaler(Scaler):
    def __init__(self):
        self.scaler = sklearn.preprocessing.StandardScaler()

    def fit(self, X):
        self.scaler.fit(X)

    def fit_transform(self, X):
        scaled_X = self.scaler.fit_transform(X)
        return scaled_X
    
    def transform(self, X):
        m, s = self.scaler.mean_, np.sqrt(self.scaler.var_)
        return (X-m)/s

    def get_params(self):
        m, s = self.scaler.mean_, np.sqrt(self.scaler.var_)
        a, b = 1/s, -m/s
        return (a, b)
    
class MinMaxScaler(Scaler):
    def __init__(self):
        self.scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1,1))

    def fit(self, X):
        self.scaler.fit(X)

    def fit_transform(self, X):
        scaled_X = self.scaler.fit_transform(X)
        return scaled_X

    def transform(self, X):
        val_min, val_max = self.scaler.data_min_, self.scaler.data_max_
        return 2*(X-val_min)/(val_max-val_min)-1.

    def get_params(self):
        val_min, val_max = self.scaler.data_min_, self.scaler.data_max_
        a, b = 2./(val_max-val_min), -1.-2.*val_min/(val_max-val_min)
        return (a, b)

class BFGSRefinement():

    def __init__(self):
        super().__init__()

    def go(
        self, env, tree, coeffs0, X, y, downsample=-1, stop_after=10
    ):
        _X = X[:downsample] if downsample > 0 else X
        _y = y[:downsample] if downsample > 0 else y
        _y_flat = _y.reshape(-1)

        infix = tree.infix()
        numexpr_equivalence = {"add": "+", "sub": "-", "mul": "*", "pow": "**", "inv": "1/"}
        for old, new in numexpr_equivalence.items():
            infix = infix.replace(old, new)

        # Build variable dict once (handles out-of-range variables by filling zeros)
        var_dict = {}
        max_dim = getattr(env.simplifier.params, 'max_input_dimension', 10)
        for d in range(max_dim):
            key = f"x_{d}"
            if key in infix:
                var_dict[key] = _X[:, d] if d < _X.shape[1] else np.zeros(_X.shape[0])

        import numexpr as ne

        def objective(coeffs):
            local_dict = dict(var_dict)
            for d in range(len(coeffs)):
                local_dict[f"CONSTANT_{d}"] = np.full(_X.shape[0], coeffs[d])
            try:
                y_tilde = ne.evaluate(infix, local_dict=local_dict)
                if y_tilde.ndim == 0:
                    y_tilde = np.full_like(_y_flat, y_tilde.item())
                mse = np.mean((_y_flat - y_tilde.reshape(-1)) ** 2) / 2.0
                if not np.isfinite(mse):
                    return 1e30
                return mse
            except Exception:
                return 1e30

        objective_timed = TimedFun(objective, stop_after=stop_after)

        try:
            minimize(
                objective_timed.fun,
                coeffs0,
                method="BFGS",
                jac="2-point",
                options={"disp": False},
            )
        except ValueError:
            pass  # TimedFun timeout

        best_constants = objective_timed.best_x
        if best_constants is None:
            return None

        return env.wrap_equation_floats(tree, best_constants)
