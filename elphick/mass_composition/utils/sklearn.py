import logging

import pandas as pd

try:
    from sklearn.pipeline import Pipeline
    from sklearn.base import BaseEstimator, RegressorMixin
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestRegressor

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def extract_feature_names(pipeline):
    for name, step in pipeline.named_steps.items():
        if hasattr(step, 'get_feature_names_out'):
            # This step has a get_feature_names_out method, so we use it
            return step.get_feature_names_out()
        elif hasattr(step, 'get_feature_names'):
            # This step has a get_feature_names method
            return step.get_feature_names()
        elif hasattr(step, 'get_params'):
            # This step doesn't have a method to get feature names directly, but it might have transformer(s) that do
            params = step.get_params()
            for param_name, param_value in params.items():
                if hasattr(param_value, 'get_feature_names_out'):
                    return param_value.get_feature_names_out()
                elif hasattr(param_value, 'get_feature_names'):
                    return param_value.get_feature_names()
    return []


if SKLEARN_AVAILABLE:
    class PandasPipeline(Pipeline, RegressorMixin):
        def __init__(self, steps, memory=None, verbose=False):
            super().__init__(steps, memory=memory, verbose=verbose)
            self._logger = logging.getLogger(__class__.__name__)
            self.feature_names_in__ = None
            self.feature_names_out_ = None
            self.set_output(transform='pandas')

        def fit(self, X, y=None, **fit_params):
            if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.DataFrame):
                raise ValueError("Input X and y must be pandas DataFrame")
            self.feature_names_in__ = X.columns.to_list()
            self.feature_names_out_ = y.columns.tolist()
            super().fit(X, y, **fit_params)
            return self

        def transform(self, X: pd.DataFrame, **kwargs) -> pd.DataFrame:
            if not isinstance(X, pd.DataFrame):
                raise ValueError("Input X must be pandas DataFrame")
            # ignore any features that the model was not fitted on and log
            if self.feature_names_in__ is not None and any([col not in self.feature_names_in__ for col in X.columns]):
                missing_features = [col for col in X.columns if col not in self.feature_names_in__]
                self._logger.info(f"Features {missing_features} were passed but are ignored since they"
                                  f" are not required by the model")
                X = X.copy().drop(columns=missing_features)
            Xt: pd.DataFrame = Pipeline.transform(self, X)
            return Xt

        def predict(self, X: pd.DataFrame) -> pd.DataFrame:
            if not isinstance(X, pd.DataFrame):
                raise ValueError("Input X must be pandas DataFrame")
            # ignore any features that the model was not fitted on and log
            if self.feature_names_in__ is not None and any([col not in self.feature_names_in__ for col in X.columns]):
                missing_features = [col for col in X.columns if col not in self.feature_names_in__]
                self._logger.info(f"Features {missing_features} were passed but are ignored since they"
                                  f" are not required by the model")
                X = X.copy().drop(columns=missing_features)
            predictions = super().predict(X)
            return pd.DataFrame(predictions, columns=self.feature_names_out_, index=X.index)

        def score(self, X: pd.DataFrame, y: pd.DataFrame) -> float:
            # ignore any features that the model was not fitted on and log
            if self.feature_names_in__ is not None and any([col not in self.feature_names_in__ for col in X.columns]):
                missing_features = [col for col in X.columns if col not in self.feature_names_in__]
                self._logger.info(f"Features {missing_features} were passed but are ignored since they"
                                  f" are not required by the model")
                X = X.copy().drop(columns=missing_features)

            # Call the parent class's score method
            return super().score(X, y)

        def get_feature_names_out(self):
            return self.feature_names_out_

        @classmethod
        def from_pipeline(cls, pipeline):
            return PandasPipeline(pipeline.steps)


else:
    class PandasPipeline:
        def __init__(self, *args, **kwargs):
            raise ImportError("sklearn is not installed but is required for PandasPipeline. Please install it.")
