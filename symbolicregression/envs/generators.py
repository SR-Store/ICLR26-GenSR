
from abc import ABC, abstractmethod
from ast import parse
from operator import length_hint, xor

import numpy as np
import math
import scipy.special
import copy
from logging import getLogger
import time
from numpy.compat.py3k import npy_load_module
from sympy import Min
from symbolicregression.envs import encoders
from collections import defaultdict
from scipy.stats import special_ortho_group

logger = getLogger()
import random
import re

operators_real = {
    "add": 2,
    "sub": 2,
    "mul": 2,
    "div": 2,
    "abs": 1,
    "inv": 1,
    "sqrt": 1,
    "log": 1,
    "exp": 1,
    "sin": 1,
    "arcsin": 1,
    "cos": 1,
    "arccos": 1,
    "tan": 1,
    "arctan": 1,
    "pow2": 1,
    "pow3": 1,
}

operators_extra = {"pow": 2}

math_constants = ["e", "pi", "euler_gamma", "CONSTANT"]
all_operators = {**operators_real, **operators_extra}


class Node:
    
    def __init__(self, value, params, children=None):
        self.value = value
        self.children = children if children else []
        self.params = params

    def push_child(self, child):
        self.children.append(child)

    def prefix(self):
        s = str(self.value)
        for c in self.children:
            s += "," + c.prefix()
        return s

    def qtree_prefix(self):
        s = "[.$" + str(self.value) + "$ "
        for c in self.children:
            s += c.qtree_prefix()
        s += "]"
        return s

    def infix(self):
        nb_children = len(self.children)

        if nb_children == 0:
            if self.value.lstrip("-").isdigit():
                return str(self.value)
            else:
                s = str(self.value)
                return s

        if nb_children == 1:
            s = str(self.value)
            if s == "pow2":
                s = "(" + self.children[0].infix() + ")**2"
            elif s == "pow3":
                s = "(" + self.children[0].infix() + ")**3"
            else:
                s = s + "(" + self.children[0].infix() + ")"
            return s

        s = "(" + self.children[0].infix()
        for c in self.children[1:]:
            s = s + " " + str(self.value) + " " + c.infix()
        return s + ")"

    def __len__(self):
        lenc = 1
        for c in self.children:
            lenc += len(c)
        return lenc

    def __str__(self):
        return self.infix()

    def __repr__(self):
        return str(self)

    def val(self, x, deterministic=True):
        if len(self.children) == 0:
            if str(self.value).startswith("x_"):
                _, dim = self.value.split("_")
                dim = int(dim)
                return x[:, dim]
            elif str(self.value) == "rand":
                if deterministic:
                    return np.zeros((x.shape[0],))
                return np.random.randn(x.shape[0])
            elif str(self.value) in math_constants:
                return getattr(np, str(self.value)) * np.ones((x.shape[0],))
            else:
                return float(self.value) * np.ones((x.shape[0],))

        if self.value == "add":
            return self.children[0].val(x) + self.children[1].val(x)
        if self.value == "sub":
            return self.children[0].val(x) - self.children[1].val(x)
        if self.value == "mul":
            m1, m2 = self.children[0].val(x), self.children[1].val(x)
            try:
                return m1 * m2
            except Exception as e:
                nans = np.empty((m1.shape[0],))
                nans[:] = np.nan
                return nans
        if self.value == "pow":
            m1, m2 = self.children[0].val(x), self.children[1].val(x)
            try:
                return np.power(m1, m2)
            except Exception as e:
                nans = np.empty((m1.shape[0],))
                nans[:] = np.nan
                return nans
        if self.value == "max":
            return np.maximum(self.children[0].val(x), self.children[1].val(x))
        if self.value == "min":
            return np.minimum(self.children[0].val(x), self.children[1].val(x))

        if self.value == "div":
            denominator = self.children[1].val(x)
            denominator[denominator == 0.0] = np.nan
            try:
                return self.children[0].val(x) / denominator
            except Exception as e:
                nans = np.empty((denominator.shape[0],))
                nans[:] = np.nan
                return nans
        if self.value == "inv":
            denominator = self.children[0].val(x)
            denominator[denominator == 0.0] = np.nan
            try:
                return 1 / denominator
            except Exception as e:
                nans = np.empty((denominator.shape[0],))
                nans[:] = np.nan
                return nans
        if self.value == "log":
            numerator = self.children[0].val(x)
            if self.params.use_abs:
                numerator[numerator <= 0.0] *= -1
            else:
                numerator[numerator <= 0.0] = np.nan
            try:
                return np.log(numerator)
            except Exception as e:
                nans = np.empty((numerator.shape[0],))
                nans[:] = np.nan
                return nans

        if self.value == "sqrt":
            numerator = self.children[0].val(x)
            if self.params.use_abs:
                numerator[numerator <= 0.0] *= -1
            else:
                numerator[numerator < 0.0] = np.nan
            try:
                return np.sqrt(numerator)
            except Exception as e:
                nans = np.empty((numerator.shape[0],))
                nans[:] = np.nan
                return nans
        if self.value == "pow2":
            numerator = self.children[0].val(x)
            try:
                return numerator ** 2
            except Exception as e:
                nans = np.empty((numerator.shape[0],))
                nans[:] = np.nan
                return nans
        if self.value == "pow3":
            numerator = self.children[0].val(x)
            try:
                return numerator ** 3
            except Exception as e:
                nans = np.empty((numerator.shape[0],))
                nans[:] = np.nan
                return nans
        if self.value == "abs":
            return np.abs(self.children[0].val(x))
        if self.value == "sign":
            return (self.children[0].val(x) >= 0) * 2.0 - 1.0
        if self.value == "step":
            x = self.children[0].val(x)
            return x if x > 0 else 0
        if self.value == "id":
            return self.children[0].val(x)
        if self.value == "fresnel":
            return scipy.special.fresnel(self.children[0].val(x))[0]
        if self.value.startswith("eval"):
            n = self.value[-1]
            return getattr(scipy.special, self.value[:-1])(n, self.children[0].val(x))[
                0
            ]
        else:
            fn = getattr(np, self.value, None)
            if fn is not None:
                try:
                    return fn(self.children[0].val(x))
                except Exception as e:
                    nans = np.empty((x.shape[0],))
                    nans[:] = np.nan
                    return nans
            fn = getattr(scipy.special, self.value, None)
            if fn is not None:
                return fn(self.children[0].val(x))
            assert False, "Could not find function"

    def get_recurrence_degree(self):
        recurrence_degree = 0
        if len(self.children) == 0:
            if str(self.value).startswith("x_"):
                _, _, offset = self.value.split("_")
                offset = int(offset)
                if offset > recurrence_degree:
                    recurrence_degree = offset
            return recurrence_degree
        return max([child.get_recurrence_degree() for child in self.children])

    def replace_node_value(self, old_value, new_value):
        if self.value == old_value:
            self.value = new_value
        for child in self.children:
            child.replace_node_value(old_value, new_value)


class NodeList:
    def __init__(self, nodes):
        self.nodes = []
        for node in nodes:
            self.nodes.append(node)
        self.params = nodes[0].params

    def infix(self):
        return " | ".join([node.infix() for node in self.nodes])

    def __len__(self):
        return sum([len(node) for node in self.nodes])

    def prefix(self):
        return ",|,".join([node.prefix() for node in self.nodes])

    def __str__(self):
        return self.infix()

    def __repr__(self):
        return str(self)

    def val(self, xs, deterministic=True):
        batch_vals = [
            np.expand_dims(node.val(np.copy(xs), deterministic=deterministic), -1)
            for node in self.nodes
        ]
        return np.concatenate(batch_vals, -1)

    def replace_node_value(self, old_value, new_value):
        for node in self.nodes:
            node.replace_node_value(old_value, new_value)


class Generator(ABC):
    def __init__(self, params):
        pass

    @abstractmethod
    def generate_datapoints(self, rng):
        pass


class RandomFunctions(Generator):
    def __init__(self, params, special_words):
        super().__init__(params)
        self.params = params
        self.prob_const = params.prob_const
        self.prob_rand = params.prob_rand
        self.max_int = params.max_int
        self.min_binary_ops_per_dim = params.min_binary_ops_per_dim
        self.max_binary_ops_per_dim = params.max_binary_ops_per_dim
        self.min_unary_ops = params.min_unary_ops
        self.max_unary_ops = params.max_unary_ops
        self.min_output_dimension = params.min_output_dimension
        self.min_input_dimension = params.min_input_dimension
        self.max_input_dimension = params.max_input_dimension
        self.max_output_dimension = params.max_output_dimension
        self.max_number = 10 ** (params.max_exponent + params.float_precision)
        self.operators = copy.deepcopy(operators_real)

        self.operators_dowsample_ratio = defaultdict(float)
        if params.operators_to_downsample != "":
            for operator in self.params.operators_to_downsample.split(","):
                operator, ratio = operator.split("_")
                ratio = float(ratio)
                self.operators_dowsample_ratio[operator] = ratio

        if params.required_operators != "":
            self.required_operators = self.params.required_operators.split(",")
        else:
            self.required_operators = []

        if params.extra_binary_operators != "":
            self.extra_binary_operators = self.params.extra_binary_operators.split(",")
        else:
            self.extra_binary_operators = []
        if params.extra_unary_operators != "":
            self.extra_unary_operators = self.params.extra_unary_operators.split(",")
        else:
            self.extra_unary_operators = []

        self.unaries = [
            o for o in self.operators.keys() if np.abs(self.operators[o]) == 1
        ] + self.extra_unary_operators

        self.binaries = [
            o for o in self.operators.keys() if np.abs(self.operators[o]) == 2
        ] + self.extra_binary_operators

        unaries_probabilities = []
        for op in self.unaries:
            if op not in self.operators_dowsample_ratio:
                unaries_probabilities.append(1.0)
            else:
                ratio = self.operators_dowsample_ratio[op]
                unaries_probabilities.append(ratio)
        self.unaries_probabilities = np.array(unaries_probabilities)
        self.unaries_probabilities /= self.unaries_probabilities.sum()

        self.original_unaries_probabilities = self.unaries_probabilities.copy()
        
        self.category_unary_mappings = {
            'abs': ['abs'],
            'inv': ['inv'], 
            'sqrt': ['sqrt'],
            'log': ['log'],
            'exp': ['exp'],
            'sin': ['sin'],
            'arcsin': ['arcsin'],
            'cos': ['cos'],
            'arccos': ['arccos'],
            'tan': ['tan'],
            'arctan': ['arctan'],
            'pow2': ['pow2'],
            'pow3': ['pow3'],
        }
        
        self.current_target_category = None

        binaries_probabilities = []
        for op in self.binaries:
            if op not in self.operators_dowsample_ratio:
                binaries_probabilities.append(1.0)
            else:
                ratio = self.operators_dowsample_ratio[op]
                binaries_probabilities.append(ratio)
        self.binaries_probabilities = np.array(binaries_probabilities)
        self.binaries_probabilities /= self.binaries_probabilities.sum()

        self.unary = False
        self.distrib = self.generate_dist(
            2 * self.max_binary_ops_per_dim * self.max_input_dimension
        )

        self.constants = [
            str(i) for i in range(-self.max_int, self.max_int + 1) if i != 0
        ]
        self.constants += math_constants
        self.variables = ["rand"] + [f"x_{i}" for i in range(self.max_input_dimension)]
        self.symbols = (
            list(self.operators)
            + self.constants
            + self.variables
            + ["|", "INT+", "INT-", "FLOAT+", "FLOAT-", "pow", "0"]
        )

        self.constants.remove("CONSTANT")

        if self.params.extra_constants is not None:
            self.extra_constants = self.params.extra_constants.split(",")
        else:
            self.extra_constants = []

        self.general_encoder = encoders.GeneralEncoder(
            params, self.symbols, all_operators
        )
        self.float_encoder = self.general_encoder.float_encoder
        self.float_words = special_words + sorted(list(set(self.float_encoder.symbols)))
        self.equation_encoder = self.general_encoder.equation_encoder
        self.equation_words = sorted(list(set(self.symbols)))
        self.equation_words = special_words + self.equation_words
        print('finish init')

    def generate_dist(self, max_ops):
        p1 = 1 if self.unary else 0

        D = []

        D.append([0] + ([1 for i in range(1, 2 * max_ops + 1)]))

        for n in range(1, 2 * max_ops + 1):
            s = [0]

            for e in range(1, 2 * max_ops - n + 1):
                value = (s[e - 1]
                        + p1 * D[n - 1][e]
                        + D[n - 1][e + 1])
                s.append(value)

            D.append(s)

        assert all(
            len(D[i]) >= len(D[i + 1]) for i in range(len(D) - 1)
        ), "Distribution matrix generation failed"
        
        return D

    def generate_float(self, rng, exponent=None):
        sign = rng.choice([-1, 1])

        mantissa = float(rng.choice(range(1, 10 ** self.params.float_precision)))

        min_power = (
            -self.params.max_exponent_prefactor - (self.params.float_precision + 1) // 2
        )
        max_power = (
            self.params.max_exponent_prefactor - (self.params.float_precision + 1) // 2
        )
        
        if not exponent:
            exponent = rng.randint(min_power, max_power + 1)

        constant = sign * (mantissa * 10 ** exponent)
        return str(constant)

    def generate_int(self, rng):
        return str(rng.choice(self.constants + self.extra_constants))

    def generate_leaf(self, rng, input_dimension):
        if rng.rand() < self.prob_rand:
            return "rand"
        else:
            if self.n_used_dims < input_dimension:
                dimension = self.n_used_dims
                self.n_used_dims += 1
                return f"x_{dimension}"
            else:
                draw = rng.rand()
                if draw < self.prob_const:
                    return self.generate_int(rng)
                else:
                    dimension = rng.randint(0, input_dimension)
                    return f"x_{dimension}"

    def generate_ops(self, rng, arity):
        if arity == 1:
            ops = self.unaries
            probas = self.unaries_probabilities
        else:
            ops = self.binaries
            probas = self.binaries_probabilities

        return rng.choice(ops, p=probas)

    def set_category_probabilities(self, category):
        if category is None:
            self.unaries_probabilities = self.original_unaries_probabilities.copy()
            self.current_target_category = None
            return
            
        if category not in self.category_unary_mappings:
            logger.warning(f"Unknown category: {category}. Using original probabilities.")
            self.unaries_probabilities = self.original_unaries_probabilities.copy()
            self.current_target_category = None
            return
        
        target_ops = self.category_unary_mappings[category]
        new_probabilities = np.zeros(len(self.unaries))
        
        for i, op in enumerate(self.unaries):
            if op in target_ops:
                new_probabilities[i] = 1.0
            else:
                new_probabilities[i] = 0.0
        
        if new_probabilities.sum() > 0:
            new_probabilities /= new_probabilities.sum()
            self.unaries_probabilities = new_probabilities
            self.current_target_category = category
        else:
            logger.warning(f"No valid operators found for category {category}. Using original probabilities.")
            self.unaries_probabilities = self.original_unaries_probabilities.copy()
            self.current_target_category = None

    def reset_probabilities(self):
        self.set_category_probabilities(None)

    def sample_next_pos(self, rng, nb_empty, nb_ops):
        assert nb_empty > 0, "Number of empty nodes must be greater than 0"
        assert nb_ops > 0, "Remaining operator count must be greater than 0"

        probs = []

        if self.unary:
            for i in range(nb_empty):
                probs.append(self.distrib[nb_ops - 1][nb_empty - i])

        for i in range(nb_empty):
            probs.append(self.distrib[nb_ops - 1][nb_empty - i + 1])

        probs = [p / self.distrib[nb_ops][nb_empty] for p in probs]

        probs = np.array(probs, dtype=np.float64)

        e = rng.choice(len(probs), p=probs)

        arity = 1 if self.unary and e < nb_empty else 2

        e %= nb_empty
        
        return e, arity

    def generate_tree(self, rng, nb_binary_ops, input_dimension):
        self.n_used_dims = 0

        tree = Node(0, self.params)

        empty_nodes = [tree]

        next_en = 0

        nb_empty = 1

        while nb_binary_ops > 0:
            next_pos, arity = self.sample_next_pos(rng, nb_empty, nb_binary_ops)

            next_en += next_pos

            op = self.generate_ops(rng, arity)

            empty_nodes[next_en].value = op

            for _ in range(arity):
                e = Node(0, self.params)
                empty_nodes[next_en].push_child(e)
                empty_nodes.append(e)

            next_en += 1

            nb_empty += arity - 1 - next_pos

            nb_binary_ops -= 1

        rng.shuffle(empty_nodes)

        for n in empty_nodes:
            if len(n.children) == 0:
                n.value = self.generate_leaf(rng, input_dimension)
        
        return tree

    def generate_multi_dimensional_tree(
        self,
        rng,
        input_dimension=None,
        output_dimension=None,
        nb_unary_ops=None,
        nb_binary_ops=None,
    ):
        trees = []

        if input_dimension is None:
            input_dimension = rng.randint(
                self.min_input_dimension, self.max_input_dimension + 1
            )
        if output_dimension is None:
            output_dimension = rng.randint(
                self.min_output_dimension, self.max_output_dimension + 1
            )
        if nb_binary_ops is None:
            min_binary_ops = self.min_binary_ops_per_dim * input_dimension
            max_binary_ops = self.max_binary_ops_per_dim * input_dimension
            nb_binary_ops_to_use = [
                rng.randint(
                    min_binary_ops, self.params.max_binary_ops_offset + max_binary_ops
                )
                for dim in range(output_dimension)
            ]
        elif isinstance(nb_binary_ops, int):
            nb_binary_ops_to_use = [nb_binary_ops for _ in range(output_dimension)]
        else:
            nb_binary_ops_to_use = nb_binary_ops
        if nb_unary_ops is None:
            nb_unary_ops_to_use = [
                rng.randint(self.min_unary_ops, self.max_unary_ops + 1)
                for dim in range(output_dimension)
            ]
        elif isinstance(nb_unary_ops, int):
            nb_unary_ops_to_use = [nb_unary_ops for _ in range(output_dimension)]
        else:
            nb_unary_ops_to_use = nb_unary_ops

        for i in range(output_dimension):
            tree = self.generate_tree(rng, nb_binary_ops_to_use[i], input_dimension)
            tree = self.add_unaries(rng, tree, nb_unary_ops_to_use[i])
            trees.append(tree)
        tree = NodeList(trees)

        nb_unary_ops_to_use = [
            len([x for x in tree_i.prefix().split(",") if x in self.unaries])
            for tree_i in tree.nodes
        ]
        nb_binary_ops_to_use = [
            len([x for x in tree_i.prefix().split(",") if x in self.binaries])
            for tree_i in tree.nodes
        ]

        for op in self.required_operators:
            if op not in tree.infix():
                return self.generate_multi_dimensional_tree(
                    rng, input_dimension, output_dimension, nb_unary_ops, nb_binary_ops
                )

        return (
            tree,
            input_dimension,
            output_dimension,
            nb_unary_ops_to_use,
            nb_binary_ops_to_use,
        )

    def add_unaries(self, rng, tree, nb_unaries):
        prefix = self._add_unaries(rng, tree)

        prefix = prefix.split(",")

        indices = []
        for i, x in enumerate(prefix):
            if x in self.unaries:
                indices.append(i)

        rng.shuffle(indices)

        if len(indices) > nb_unaries:
            to_remove = indices[: len(indices) - nb_unaries]

            for index in sorted(to_remove, reverse=True):
                del prefix[index]

        tree = self.equation_encoder.decode(prefix).nodes[0]
        return tree

    def _add_unaries(self, rng, tree):
        s = str(tree.value)

        for c in tree.children:
            if len(c.prefix().split(",")) < self.params.max_unary_depth:
                unary = rng.choice(self.unaries, p=self.unaries_probabilities)

                s += f",{unary}," + self._add_unaries(rng, c)
            else:
                s += f"," + self._add_unaries(rng, c)
        
        return s

    def add_prefactors(self, rng, tree):
        transformed_prefix = self._add_prefactors(rng, tree)

        if transformed_prefix == tree.prefix():
            a = self.generate_float(rng)
            transformed_prefix = f"mul,{a}," + transformed_prefix

        a = self.generate_float(rng)
        transformed_prefix = f"add,{a}," + transformed_prefix

        tree = self.equation_encoder.decode(transformed_prefix.split(",")).nodes[0]
        return tree

    def _add_prefactors(self, rng, tree):
        s = str(tree.value)

        a, b = self.generate_float(rng), self.generate_float(rng)

        if s in ["add", "sub"]:
            if tree.children[0].value in ["add", "sub"]:
                s += "," + self._add_prefactors(rng, tree.children[0])
            else:
                s += f",mul,{a}," + self._add_prefactors(rng, tree.children[0])

            if tree.children[1].value in ["add", "sub"]:
                s += "," + self._add_prefactors(rng, tree.children[1])
            else:
                s += f",mul,{b}," + self._add_prefactors(rng, tree.children[1])

        elif s in self.unaries and tree.children[0].value not in ["add", "sub"]:
            s += f",add,{a},mul,{b}," + self._add_prefactors(rng, tree.children[0])

        else:
            for c in tree.children:
                s += f"," + self._add_prefactors(rng, c)
        
        return s

    def add_linear_transformations(self, rng, tree, target, add_after=False):

        prefix = tree.prefix().split(",")
        indices = []

        for i, x in enumerate(prefix):
            if x in target:
                indices.append(i)

        offset = 0
        for idx in indices:
            a, b = self.generate_float(rng), self.generate_float(rng)
            if add_after:
                prefix = (
                    prefix[: idx + offset + 1]
                    + ["add", a, "mul", b]
                    + prefix[idx + offset + 1 :]
                )
            else:
                prefix = (
                    prefix[: idx + offset]
                    + ["add", a, "mul", b]
                    + prefix[idx + offset :]
                )
            offset += 4
        tree = self.equation_encoder.decode(prefix).nodes[0]

        return tree

    def relabel_variables(self, tree):
        active_variables = []
        for elem in tree.prefix().split(","):
            if elem.startswith("x_"):
                active_variables.append(elem)
        active_variables = list(set(active_variables))
        input_dimension = len(active_variables)
        if input_dimension == 0:
            return 0
        active_variables.sort(key=lambda x: int(x[2:]))
        for j, xi in enumerate(active_variables):
            tree.replace_node_value(xi, "x_{}".format(j))
        return input_dimension

    def function_to_skeleton(
        self, tree, skeletonize_integers=False, constants_with_idx=False
    ):
        constants = []
        prefix = tree.prefix().split(",")
        j = 0
        for i, pre in enumerate(prefix):
            try:
                float(pre)
                is_float = True
                if pre.lstrip("-").isdigit():
                    is_float = False
            except ValueError:
                is_float = False

            if pre.startswith("CONSTANT"):
                constants.append("CONSTANT")
                if constants_with_idx:
                    prefix[i] = "CONSTANT_{}".format(j)
                j += 1
            elif is_float or (pre is self.constants and skeletonize_integers):
                if constants_with_idx:
                    prefix[i] = "CONSTANT_{}".format(j)
                else:
                    prefix[i] = "CONSTANT"
                while i > 0 and prefix[i - 1] in self.unaries:
                    del prefix[i - 1]
                try:
                    value = float(pre)
                except:
                    value = getattr(np, pre)
                constants.append(value)
                j += 1
            else:
                continue

        new_tree = self.equation_encoder.decode(prefix)
        return new_tree, constants

    def wrap_equation_floats(self, tree, constants):
        tree = self.tree
        env = self.env
        prefix = tree.prefix().split(",")
        j = 0
        for i, elem in enumerate(prefix):
            if elem.startswith("CONSTANT"):
                prefix[i] = str(constants[j])
                j += 1
        assert j == len(constants), "all constants were not fitted"
        assert "CONSTANT" not in prefix, "tree {} got constant after wrapper {}".format(
            tree, constants
        )
        tree_with_constants = env.word_to_infix(prefix, is_float=False, str_array=False)
        return tree_with_constants

    def order_datapoints(self, inputs, outputs):
        mean_input = inputs.mean(0)
        distance_to_mean = np.linalg.norm(inputs - mean_input, axis=-1)
        order_by_distance = np.argsort(distance_to_mean)
        return inputs[order_by_distance], outputs[order_by_distance]

    def _generate_datapoints(
        self,
        tree,
        n_points,
        scale,
        rng,
        input_dimension,
        input_distribution_type,
        n_centroids,
        max_trials,
        rotate=True,
        offset=None,
    ):
        
        inputs, outputs = [], []
        remaining_points = n_points
        trials = 0

        means = rng.randn(n_centroids, input_dimension,)

        covariances = rng.uniform(0, 1, size=(n_centroids, input_dimension))

        if rotate:
            rotations = [
                special_ortho_group.rvs(input_dimension)
                if input_dimension > 1
                else np.identity(1)
                for i in range(n_centroids)
            ]
        else:
            rotations = [np.identity(input_dimension) for i in range(n_centroids)]

        weights = rng.uniform(0, 1, size=(n_centroids,))
        weights /= np.sum(weights)

        n_points_comp = rng.multinomial(n_points, weights)
        
        while remaining_points > 0 and trials < max_trials:

            if input_distribution_type == "gaussian":
                input = np.vstack([
                    rng.multivariate_normal(mean, np.diag(covariance), int(sample))
                    @ rotation
                    for (mean, covariance, rotation, sample) in zip(
                        means, covariances, rotations, n_points_comp
                    )
                ])

            elif input_distribution_type == "uniform":
                input = np.vstack([
                    (
                        mean
                        + rng.uniform(-1, 1, size=(sample, input_dimension))
                        * np.sqrt(covariance)
                    )
                    @ rotation
                    for (mean, covariance, rotation, sample) in zip(
                        means, covariances, rotations, n_points_comp
                    )
                ])
            
            input = (input - np.mean(input, axis=0, keepdims=True)) / np.std(
                input, axis=0, keepdims=True
            )

            input *= scale

            if offset is not None:
                mean, std = offset
                input *= std
                input += mean

            output = tree.val(input)


            is_nan_idx = np.any(np.isnan(output), -1)
            input = input[~is_nan_idx, :]
            output = output[~is_nan_idx, :]

            output[np.abs(output) >= self.max_number] = np.nan
            output[np.abs(output) == np.inf] = np.nan

            is_nan_idx = np.any(np.isnan(output), -1)
            input = input[~is_nan_idx, :]
            output = output[~is_nan_idx, :]
            
            valid_points = output.shape[0]
            trials += 1
            remaining_points -= valid_points

            if valid_points == 0:
                continue

            inputs.append(input)
            outputs.append(output)
        
        if remaining_points > 0:
            return None, None

        inputs = np.concatenate(inputs, 0)[:n_points]
        outputs = np.concatenate(outputs, 0)[:n_points]

        if self.params.normalize_y:
            eps = 1e-10
            outputs = np.divide(
                np.subtract(outputs, min(outputs)),
                max(outputs) - min(outputs) + eps
            )
        
        return inputs, outputs


    def generate_datapoints(
        self,
        tree,
        n_input_points,
        n_prediction_points,
        prediction_sigmas,
        rotate=True,
        offset=None,
        **kwargs,
    ):
        inputs, outputs = self._generate_datapoints(
            tree=tree,
            n_points=n_input_points,
            scale=1,
            rotate=rotate,
            offset=offset,
            **kwargs,
        )

        if inputs is None:
            return None, None
        datapoints = {"fit": (inputs, outputs)}

        inputs, outputs = self._generate_datapoints(
            tree=tree,
            n_points=n_input_points,
            scale=1,
            rotate=rotate,
            offset=offset,
            **kwargs,
        )
        datapoints["predict"] = (inputs, outputs)


        return tree, datapoints

def string_to_node(expr_str, params):
    expr_str = expr_str.strip()
    
    var_pattern = r'[a-zA-Z_][a-zA-Z0-9_]*'
    variables = set(re.findall(var_pattern, expr_str))
    functions = {'sqrt', 'sin', 'cos', 'tan', 'exp', 'log', 'abs', 'tanh'}
    variables = variables - functions
    
    var_map = {}
    for i, var in enumerate(variables):
        var_map[var] = f'x_{i}'

    
    return _parse_expression(expr_str, params, var_map, level=0)

def _tokenize_expression(expr):
    for op in ['(', ')', '+', '-', '*', '/', '**']:
        if op == '**':
            expr = expr.replace(op, ' ** ')
        else:
            expr = expr.replace(op, f' {op} ')
    
    tokens = [token for token in expr.split() if token]
    return tokens

def _parse_expression(expr, params, var_map, level=0):
    indent = "  " * level
    
    expr = expr.strip()
    
    if not expr:
        return Node("0", params)
    
    if expr.startswith('(') and expr.endswith(')'):
        if _is_balanced(expr[1:-1]):
            return _parse_expression(expr[1:-1], params, var_map, level=level+1)
    
    if expr in var_map:
        return Node(var_map[expr], params)
    
    lowest_op = _find_lowest_precedence_op(expr)
    
    if lowest_op:
        op_idx, op = lowest_op
        if op in ['+', '-'] and (op_idx > 0 or (op == '+' and op_idx == 0)):
            left = expr[:op_idx]
            right = expr[op_idx+1:]
            
            node = Node('add' if op == '+' else 'sub', params)
            node.push_child(_parse_expression(left, params, var_map, level=level+1))
            node.push_child(_parse_expression(right, params, var_map, level=level+1))
            return node
    
    lowest_op = None
    lowest_precedence = float('inf')
    paren_level = 0
    i = 0
    while i < len(expr):
        char = expr[i]
        
        if i < len(expr) - 1 and expr[i:i+2] == '**':
            i += 2
            continue
            
        if char == '(':
            paren_level += 1
        elif char == ')':
            paren_level -= 1
        
        if paren_level == 0:
            if char in ['*', '/']:
                op = char
                if i > 0 and i < len(expr) - 1:
                    lowest_op = (i, op)
                    break
        
        i += 1
    
    if lowest_op:
        op_idx, op = lowest_op
        left = expr[:op_idx]
        right = expr[op_idx+1:]
        
        node = Node('mul' if op == '*' else 'div', params)
        node.push_child(_parse_expression(left, params, var_map, level=level+1))
        node.push_child(_parse_expression(right, params, var_map, level=level+1))
        return node
    
    
    if expr.startswith('-') and len(expr) > 1:
        node = Node('mul', params)
        node.push_child(Node('-1', params))
        node.push_child(_parse_expression(expr[1:], params, var_map, level=level+1))
        return node
    
    if '**' in expr:
        
        pos = -1
        paren_level = 0
        i = 0
        while i < len(expr) - 1:
            if expr[i] == '(':
                paren_level += 1
            elif expr[i] == ')':
                paren_level -= 1
            elif paren_level == 0 and expr[i:i+2] == '**':
                pos = i
            i += 1
                
        if pos != -1:
            base = expr[:pos].strip()
            exponent = expr[pos+2:].strip()
            
            
            if exponent == '2':
                node = Node('pow2', params)
                if base in var_map:
                    node.push_child(Node(var_map[base], params))
                else:
                    base_node = _parse_expression(base, params, var_map, level=level+1)
                    node.push_child(base_node)
                return node
            elif exponent == '3':
                node = Node('pow3', params)
                if base in var_map:
                    node.push_child(Node(var_map[base], params))
                else:
                    base_node = _parse_expression(base, params, var_map, level=level+1)
                    node.push_child(base_node)
                return node
            else:
                node = Node('pow', params)
                if base in var_map:
                    node.push_child(Node(var_map[base], params))
                else:
                    node.push_child(_parse_expression(base, params, var_map, level=level+1))
                
                if exponent.isdigit() or (exponent.startswith('-') and exponent[1:].isdigit()):
                    node.push_child(Node(exponent, params))
                elif exponent in var_map:
                    node.push_child(Node(var_map[exponent], params))
                else:
                    node.push_child(_parse_expression(exponent, params, var_map, level=level+1))
                return node
    
    for func in ['sqrt', 'sin', 'cos', 'tan', 'exp', 'log', 'abs', 'tanh']:
        if expr.startswith(func + '(') and expr.endswith(')'):
            arg = expr[len(func)+1:-1]
            
            node = Node(func, params)
            node.push_child(_parse_expression(arg, params, var_map, level=level+1))
            return node
    
    try:
        value = float(expr)
        return Node(str(value), params)
    except ValueError:
        for var, mapped_var in var_map.items():
            if expr == var:
                return Node(mapped_var, params)
        
        return Node(expr, params)

def _is_balanced(expr):
    count = 0
    for char in expr:
        if char == '(':
            count += 1
        elif char == ')':
            count -= 1
            if count < 0:
                return False
    return count == 0

def _find_lowest_precedence_op(expr):
    paren_level = 0
    lowest_op = None
    lowest_precedence = float('inf')
    
    precedence = {'+': 1, '-': 1, '*': 2, '/': 2}
    
    i = 0
    while i < len(expr):
        char = expr[i]
        
        if i < len(expr) - 1 and expr[i:i+2] == '**':
            i += 2
            continue
            
        if char == '(':
            paren_level += 1
        elif char == ')':
            paren_level -= 1
        
        if paren_level == 0:
            if char in precedence:
                if char == '-' and (i == 0 or expr[i-1] in '+*/('):
                    i += 1
                    continue
                    
                op = char
                if precedence[op] <= lowest_precedence:
                    lowest_precedence = precedence[op]
                    lowest_op = (i, op)
        
        i += 1
    
    return lowest_op

def string_to_nodelist(expr_str, params):
    node = string_to_node(expr_str, params)
    return NodeList([node])


if __name__ == "__main__":

    from parsers import get_parser
    from symbolicregression.envs.environment import SPECIAL_WORDS

    parser = get_parser()
    params = parser.parse_args()
    generator = RandomFunctions(params, SPECIAL_WORDS)
    rng = np.random.RandomState(0)
    tree, _, _, _, _ = generator.generate_multi_dimensional_tree(
        np.random.RandomState(0), input_dimension=1
    )
    print(tree)
    x, y = generator.generate_datapoints(rng, tree, "gaussian", 10, 200, 200)
    generator.order_datapoints(x, y)

