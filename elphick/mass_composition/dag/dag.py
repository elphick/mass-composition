from collections import UserDict
from inspect import signature
from typing import Iterable

import networkx as nx
import numpy as np
from joblib import Parallel, delayed
from scipy.sparse import issparse, dok_matrix
from sklearn.utils import _print_elapsed_time, _safe_indexing, Bunch
from sklearn.utils.metaestimators import _BaseComposition
from sklearn.utils.validation import check_memory
from sklearn.base import clone

from elphick.mass_composition import MassComposition
from elphick.mass_composition.dag.utils import (
    _format_output,
    _in_notebook,
    _is_pandas,
    _is_passthrough,
    _is_predictor,
    _is_transformer,
    _stack,
)


def _get_columns(X, dep, cols, is_root, dep_is_passthrough, axis=1):
    if callable(cols):
        # sklearn.compose.make_column_selector
        cols = cols(X)

    if not is_root and not dep_is_passthrough:
        # The DAG will prepend output columns with the step name, so add this in to any
        # dep columns if missing. This helps keep user-provided deps readable.
        if isinstance(cols, str):
            cols = cols if cols.startswith(f"{dep}__") else f"{dep}__{cols}"
        elif isinstance(cols, Iterable):
            orig = cols
            cols = []
            for col in orig:
                if isinstance(col, str):
                    cols.append(col if col.startswith(f"{dep}__") else f"{dep}__{col}")
                else:
                    cols.append(col)

    return _safe_indexing(X, cols, axis=axis)


def _stack_inputs(dag, X, node):
    # For root nodes, the dependency is just the node name itself.
    deps = {node.name: None} if node.is_root else node.deps

    cols = [
        _get_columns(
            X[dep],
            dep,
            cols,
            node.is_root,
            _is_passthrough(dag.graph_.nodes[dep]["step"].transformer),
            axis=1,
        )
        for dep, cols in deps.items()
    ]

    to_stack = [
        # If we sliced a single column from an input, reshape it to a 2d array.
        col.reshape(-1, 1)
        if col is not None and deps[dep] is not None and col.ndim < 2
        else col
        for col, dep in zip(cols, deps)
    ]

    X_stacked = _stack(to_stack)

    return X_stacked


def _fit_transform_one(
        transformer,
        X,
        y,
        weight,
        message_clsname="",
        message=None,
        allow_predictor=True,
        **fit_params,
):
    """
    Fits ``transformer`` to ``X`` and ``y``. The transformed result is returned
    with the fitted transformer. If ``weight`` is not ``None``, the result will
    be multiplied by ``weight``.
    """
    with _print_elapsed_time(message_clsname, message):
        failed = False
        if _is_passthrough(transformer):
            res = X
        elif hasattr(transformer, "fit_transform"):
            res = transformer.fit_transform(X, y, **fit_params)
        elif hasattr(transformer, "transform"):
            res = transformer.fit(X, y, **fit_params).transform(X)
        elif allow_predictor:
            for fn in ["predict_proba", "decision_function", "predict"]:
                if hasattr(transformer, fn):
                    res = getattr(transformer.fit(X, y, **fit_params), fn)(X)
                    if res.ndim < 2:
                        res = res.reshape(-1, 1)
                    break
            else:
                failed = True
                res = None

            if res is not None and res.ndim < 2:
                res = res.reshape(-1, 1)
        else:
            failed = True

        if failed:
            raise AttributeError(
                f"'{type(transformer).__name__}' object has no attribute 'transform'"
            )

    if weight is not None:
        res = res * weight

    return res, transformer


def _parallel_fit(dag, step, Xin, Xs, y, fit_transform_fn, memory, **fit_params):
    transformer = step.estimator

    if step.deps:
        X = _stack_inputs(dag, Xs, step)
    else:
        # For root nodes, the destination rather than the source is
        # specified.
        # X = Xin[step.name]
        X = _stack_inputs(dag, Xin, step)

    clsname = type(dag).__name__
    with _print_elapsed_time(clsname, dag._log_message(step)):
        if transformer is None or transformer == "passthrough":
            Xt, fitted_transformer = X, transformer
        else:
            if hasattr(memory, "location") and memory.location is None:
                # we do not clone when caching is disabled to
                # preserve backward compatibility
                cloned_transformer = transformer
            else:
                cloned_transformer = clone(transformer)

            # Fit or load from cache the current transformer
            Xt, fitted_transformer = fit_transform_fn(
                cloned_transformer,
                X,
                y,
                None,
                message_clsname=clsname,
                message=dag._log_message(step),
                **fit_params,
            )

    Xt = _format_output(Xt, X, step)

    return Xt, fitted_transformer


def _parallel_transform(dag, step, Xin, Xs, transform_fn, **fn_params):
    transformer = step.estimator
    if step.deps:
        X = _stack_inputs(dag, Xs, step)
    else:
        # For root nodes, the destination rather than the source is
        # specified.
        X = _stack_inputs(dag, Xin, step)
        # X = Xin[step.name]

    clsname = type(dag).__name__
    with _print_elapsed_time(clsname, dag._log_message(step)):
        if transformer is None or transformer == "passthrough":
            Xt = X
        else:
            # Fit or load from cache the current transformer
            Xt = transform_fn(
                transformer,
                X,
                None,
                message_clsname=clsname,
                message=dag._log_message(step),
                **fn_params,
            )

    Xt = _format_output(Xt, X, step)

    return Xt


def _parallel_fit_leaf(dag, leaf, Xts, y, **fit_params):
    with _print_elapsed_time(type(dag).__name__, dag._log_message(leaf)):
        if leaf.transformer == "passthrough":
            fitted_estimator = leaf.transformer
        else:
            Xt = _stack_inputs(dag, Xts, leaf)
            fitted_estimator = leaf.transformer.fit(Xt, y, **fit_params)

    return fitted_estimator


def _parallel_execute(
        dag, leaf, fn, Xts, y=None, fit_first=False, fit_params=None, fn_params=None
):
    with _print_elapsed_time("DAG", dag._log_message(leaf)):
        Xt = _stack_inputs(dag, Xts, leaf)
        fit_params = fit_params or {}
        fn_params = fn_params or {}
        if leaf.estimator == "passthrough":
            Xout = Xt
        elif fit_first and hasattr(leaf.estimator, f"fit_{fn}"):
            Xout = getattr(leaf.estimator, f"fit_{fn}")(Xt, y, **fit_params)
        else:
            if fit_first:
                leaf.estimator.fit(Xt, y, **fit_params)

            est_fn = getattr(leaf.estimator, fn)
            if "y" in signature(est_fn).parameters:
                Xout = est_fn(Xt, y=y, **fn_params)
            else:
                Xout = est_fn(Xt, **fn_params)

        Xout = _format_output(Xout, Xt, leaf)

    fitted_estimator = leaf.estimator

    return Xout, fitted_estimator


class DAGStep:
    """
    A single node/step in a DAG.

    Parameters
    ----------
    name : str
        The reference name for this step.
    estimator : estimator-like
        The estimator (transformer or predictor) that will be executed by this step.
    deps : dict
        A map of dependency names to columns. If columns is ``None``, then all input
        columns will be selected.
    """

    def __init__(self, name, mc_transformer, deps):
        self.name = name
        self.transformer = mc_transformer
        self.deps = deps
        self.is_root = False
        self.is_leaf = False

    def __repr__(self):
        return f"{type(self).__name__}({repr(self.name)}, {repr(self.estimator)})"


class DAG(_BaseComposition):
    """
    A Directed Acyclic Graph (DAG) of MassComposition methods

    A DAG may consist of a simple chain of steps or a more complex path of dependencies.
    But as the name suggests, it may not contain any cyclic dependencies and data may only flow
    from one or more start points (roots) to one or more endpoints (leaves).

    Parameters
    ----------

    graph : :class:`networkx.DiGraph`
        A directed graph with string node IDs indicating the step name. Each node must
        have a ``step`` attribute, which contains a :class:`skdag.dag.DAGStep`.

    Attributes
    ----------

    graph_ : :class:`networkx.DiGraph`
        A read-only view of the workflow.
    """

    def __init__(self, graph, *, memory=None, n_jobs=None, verbose=False):
        self.graph = graph
        self.memory = memory
        self.verbose = verbose
        self.n_jobs = n_jobs

    def fit(self, x: [None, MassComposition] = None, y=None, **fit_params) -> 'DAG':

        """
        Fit the model.

        Fit all the transformers one after the other and transform the
        data.

        """
        self._validate_graph()
        fit_params_steps = self._check_fit_params(**fit_params)
        Xts = self._fit(x, y, **fit_params_steps)
        fitted_estimators = Parallel(n_jobs=self.n_jobs)(
            [
                delayed(_parallel_fit_leaf)(
                    self, leaf, Xts, y, **fit_params_steps[leaf.name]
                )
                for leaf in self.leaves_
            ]
        )
        for est, leaf in zip(fitted_estimators, self.leaves_):
            leaf.estimator = est
            leaf.is_fitted = True

        # If we have a single root, mirror certain attributes in the DAG.
        if len(self.roots_) == 1:
            root = self.roots_[0].estimator
            for attr in ["n_features_in_", "feature_names_in_"]:
                if hasattr(root, attr):
                    setattr(self, attr, getattr(root, attr))

        return self

    def _fit(self, X, y=None, **fit_params_steps):
        # Setup the memory
        memory = check_memory(self.memory)

        fit_transform_one_cached = memory.cache(_fit_transform_one)

        root_names = set([root.name for root in self.roots_])
        Xin = self._resolve_inputs(X)
        Xs = {}
        with Parallel(n_jobs=self.n_jobs) as parallel:
            for stage in self._iter(with_leaves=False, filter_passthrough=False):
                stage_names = [step.name for step in stage]
                outputs, fitted_transformers = zip(
                    *parallel(
                        delayed(_parallel_fit)(
                            self,
                            step,
                            Xin,
                            Xs,
                            y,
                            fit_transform_one_cached,
                            memory,
                            **fit_params_steps[step.name],
                        )
                        for step in stage
                    )
                )

                for step, fitted_transformer in zip(stage, fitted_transformers):
                    # Replace the transformer of the step with the fitted
                    # transformer. This is necessary when loading the transformer
                    # from the cache.
                    step.estimator = fitted_transformer
                    step.is_fitted = True

                Xs.update(dict(zip(stage_names, outputs)))

                # If all of a dep's dependents are now complete, we can free up some
                # memory.
                root_names = root_names - set(stage_names)
                for dep in {dep for step in stage for dep in step.deps}:
                    dependents = self.graph_.successors(dep)
                    if all(d in Xs and d not in root_names for d in dependents):
                        del Xs[dep]

        # If a root node is also a leaf, it hasn't been fit yet and we need to pass on
        # its input for later.
        Xs.update({name: Xin[name] for name in root_names})
        return Xs

    def _log_message(self, step):
        if not self.verbose:
            return None

        return f"(step {step.name}: {step.index} of {len(self.graph_)}) Processing {step.name}"

    def _iter(self, with_leaves=True, filter_passthrough=True):
        """
        Generate stage lists from self.graph_.
        When filter_passthrough is True, 'passthrough' and None transformers
        are filtered out.
        """
        for stage in nx.topological_generations(self.graph_):
            stage = [self.graph_.nodes[step]["step"] for step in stage]
            if not with_leaves:
                stage = [step for step in stage if not step.is_leaf]

            if filter_passthrough:
                stage = [
                    step
                    for step in stage
                    if step.estimator is not None and step.estimator != "passthough"
                ]

            if len(stage) == 0:
                continue

            yield stage

    def __len__(self):
        """
        Returns the size of the DAG
        """
        return len(self.graph_)

    def __getitem__(self, name):
        """
        Retrieve a named estimator.
        """
        return self.graph_.nodes[name]["step"].estimator

    def _resolve_inputs(self, X):
        if isinstance(X, (dict, Bunch, UserDict)) and not isinstance(X, dok_matrix):
            inputs = sorted(X.keys())
            if inputs != sorted(root.name for root in self.roots_):
                raise ValueError(
                    "Input dicts must contain one key per entry node. "
                    f"Entry nodes are {self.roots_}, got {inputs}."
                )
        else:
            if len(self.roots_) != 1:
                raise ValueError(
                    "Must provide a dictionary of inputs for a DAG with multiple entry "
                    "points."
                )
            X = {self.roots_[0].name: X}

        # X = {
        #     step: x if issparse(x) or _is_pandas(x) else np.asarray(x)
        #     for step, x in X.items()
        # }

        return X

    def _validate_graph(self):
        if len(self.graph_) == 0:
            raise ValueError("DAG has no nodes.")

        for i, (name, est) in enumerate(self.steps_):
            step: DAGStep = self.graph_.nodes[name]["step"]
            step.index = i

        # validate names
        # TODO: locate this method?
        # self._validate_names([name for (name, step) in self.steps_])

        # validate transformers
        for step in self.roots_ + self.branches_:
            if step in self.leaves_:
                # This will get validated later
                continue

            est = step.transformer
            # Unlike pipelines we also allow predictors to be used as a transformer, to support
            # model stacking.
            if (
                    not _is_passthrough(est)
                    and not _is_transformer(est)
                    and not _is_predictor(est)
            ):
                raise TypeError(
                    "All intermediate steps should be "
                    "transformers and implement fit and transform "
                    "or be the string 'passthrough' "
                    f"'{est}' (type {type(est)}) doesn't"
                )

        # Validate final estimator(s)
        for step in self.leaves_:
            est = step.transformer
            if not _is_passthrough(est) and not hasattr(est, "fit"):
                raise TypeError(
                    "Leaf nodes of a DAG should implement fit "
                    "or be the string 'passthrough'. "
                    f"'{est}' (type {type(est)}) doesn't"
                )

    @property
    def graph_(self):
        if not hasattr(self, "_graph"):
            # Read-only view of the graph. We should not modify
            # the original graph.
            self._graph = self.graph.copy(as_view=True)

        return self._graph

    @property
    def leaves_(self):
        if not hasattr(self, "_leaves"):
            self._leaves = [node for node in self.nodes_ if node.is_leaf]

        return self._leaves

    @property
    def branches_(self):
        if not hasattr(self, "_branches"):
            self._branches = [
                node for node in self.nodes_ if not node.is_leaf and not node.is_root
            ]

        return self._branches

    @property
    def roots_(self):
        if not hasattr(self, "_roots"):
            self._roots = [node for node in self.nodes_ if node.is_root]

        return self._roots

    @property
    def nodes_(self):
        if not hasattr(self, "_nodes"):
            self._nodes = []
            for name, estimator in self.steps_:
                step = self.graph_.nodes[name]["step"]
                if self.graph_.out_degree(name) == 0:
                    step.is_leaf = True
                if self.graph_.in_degree(name) == 0:
                    step.is_root = True
                self._nodes.append(step)

        return self._nodes

    @property
    def steps_(self):
        "return list of (name, estimator) tuples to conform with Pipeline interface."
        if not hasattr(self, "_steps"):
            self._steps = [
                (node, self.graph_.nodes[node]["step"].transformer)
                for node in nx.lexicographical_topological_sort(self.graph_)
            ]

        return self._steps

    def _check_fit_params(self, **fit_params):
        fit_params_steps = {
            name: {} for (name, step) in self.steps_ if step is not None
        }
        for pname, pval in fit_params.items():
            if pval is None:
                continue

            if "__" not in pname:
                raise ValueError(
                    f"DAG.fit does not accept the {pname} parameter. "
                    "You can pass parameters to specific steps of your "
                    "DAG using the stepname__parameter format, e.g. "
                    "`DAG.fit(X, y, logisticregression__sample_weight"
                    "=sample_weight)`."
                )
            step, param = pname.split("__", 1)
            fit_params_steps[step][param] = pval
        return fit_params_steps
