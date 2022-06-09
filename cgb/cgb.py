import numbers
import numpy as np
from sklearn.tree import _tree
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelBinarizer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble._gb import BaseGradientBoosting, VerboseReporter
from sklearn.ensemble import _gradient_boosting, _gb_losses, GradientBoostingClassifier, GradientBoostingRegressor

from scipy.special import logsumexp
from scipy.sparse.base import issparse
from sklearn.utils.multiclass import type_of_target
from sklearn.base import BaseEstimator, is_classifier
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection._split import train_test_split
from sklearn.ensemble._gradient_boosting import predict_stages
from sklearn.utils.validation import check_array, check_random_state, column_or_1d, _check_sample_weight
from sklearn.utils import check_scalar

TREE_LEAF = _tree.TREE_LEAF
DTYPE = _tree.DTYPE


class CondensedDeviance(_gb_losses.ClassificationLossFunction):
    def __init__(self, n_classes_):
        self.K = 1
        self.n_classes_ = n_classes_
        self.is_multi_class = False

    def init_estimator(self):
        return DummyClassifier(strategy="prior")

    def __call__(self, y, raw_predictions, sample_weight=None):

        return np.average(-1 * (y * raw_predictions).sum(axis=1) +
                          logsumexp(raw_predictions, axis=1),
                          weights=sample_weight)

    def update_terminal_regions(self,
                                tree,
                                X,
                                y,
                                residual,
                                raw_predictions,
                                sample_weight,
                                sample_mask,
                                learning_rate,
                                k=0):

        terminal_regions = tree.apply(X)

        masked_terminal_regions = terminal_regions.copy()
        masked_terminal_regions[~sample_mask] = -1

        for leaf in np.where(tree.children_left == TREE_LEAF)[0]:
            self._update_terminal_region(tree, masked_terminal_regions, leaf,
                                         X, y, residual, raw_predictions,
                                         sample_weight)

        raw_predictions[:, :] += \
            (learning_rate * tree.value[:, :, 0]
             ).take(terminal_regions, axis=0)

    def negative_gradient(self, y, raw_predictions, k=0, **kwargs):

        return y - np.nan_to_num(
            np.exp(raw_predictions -
                   logsumexp(raw_predictions, axis=1, keepdims=True)))

    def _update_terminal_region(
        self,
        tree,
        terminal_regions,
        leaf,
        X,
        y,
        residual,
        raw_predictions,
        sample_weight,
    ):
        n_classes = self.n_classes_
        terminal_region = np.where(terminal_regions == leaf)[0]
        residual = residual.take(terminal_region, axis=0)
        y = y.take(terminal_region, axis=0)
        sample_weight = sample_weight.take(terminal_region, axis=0)
        sample_weight = sample_weight[:, np.newaxis]
        numerator = np.sum(sample_weight * residual, axis=0)
        numerator *= (n_classes - 1) / n_classes
        denominator = np.sum(sample_weight * (y - residual) *
                             (1 - y + residual),
                             axis=0)
        tree.value[leaf, :, 0] = np.where(
            abs(denominator) < 1e-150, 0.0, numerator / denominator)

    def _raw_prediction_to_proba(self, raw_predictions):
        return np.nan_to_num(
            np.exp(raw_predictions -
                   (logsumexp(raw_predictions, axis=1)[:, np.newaxis])))

    def _raw_prediction_to_decision(self, raw_predictions):
        proba = self._raw_prediction_to_proba(raw_predictions)
        return np.argmax(proba, axis=1)

    def get_init_raw_predictions(self, X, estimator):
        probas = estimator.predict_proba(X)
        eps = np.finfo(np.float32).eps
        probas = np.clip(probas, eps, 1 - eps)
        raw_predictions = np.log(probas).astype(np.float64)
        return raw_predictions


class MultiOutputLeastSquaresError(_gb_losses.RegressionLossFunction):
    def init_estimator(self):
        return DummyRegressor(strategy='mean')

    def get_init_raw_predictions(self, X, estimator):
        predictions = estimator.predict(X)
        if type_of_target(predictions) == 'continuous-multioutput' or 'multiclass-multioutput':
            predictions = predictions.reshape(-1, predictions.shape[1]).astype(
                np.float64)
        else:
            predictions = predictions.reshape(-1, 1).astype(np.float64)
        return predictions

    def __call__(self, y, raw_predictions, sample_weight=None):

        if sample_weight is None:
            init = np.mean((y - raw_predictions.ravel())**2)
        else:
            if type_of_target(raw_predictions) == 'continuous-multioutput' or 'multiclass-multioutput':
                init = (1 / sample_weight.sum() *
                        np.sum(sample_weight[:, None] *
                               ((y - raw_predictions)**2)))
            else:
                init = (1 / sample_weight.sum() *
                        np.sum(sample_weight *
                               ((y - raw_predictions.ravel())**2)))

        return init

    def negative_gradient(self, y, raw_predictions, **kargs):
        if type_of_target(y) == 'continuous-multioutput' or 'multiclass-multioutput':
            negative_gradient = np.squeeze(y) - raw_predictions
        else:
            negative_gradient = np.squeeze(y) - raw_predictions.ravel()
        return negative_gradient

    def update_terminal_regions(self,
                                tree,
                                X,
                                y,
                                residual,
                                raw_predictions,
                                sample_weight,
                                sample_mask,
                                learning_rate=0.1,
                                k=0):
        if type_of_target(y) == 'continuous-multioutput' or 'multiclass-multioutput':
            for i in range(y.shape[1]):
                raw_predictions[:, i] += learning_rate * \
                    tree.predict(X)[:, i, 0]
        else:
            raw_predictions[:, k] += learning_rate * tree.predict(X).ravel()

    def _update_terminal_region(self, tree, terminal_regions, leaf, X, y,
                                residual, raw_predictions, sample_weight):
        pass


class CondensedGradientBoosting(BaseGradientBoosting):

    def __init__(self,
                 n_estimators=100,
                 learning_rate=0.1,
                 loss="log_loss",
                 criterion="squared_error",
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0,
                 subsample=1.0,
                 max_features=None,
                 max_depth=5,
                 min_impurity_decrease=0.0,
                 ccp_alpha=0.0,
                 alpha=0.9,
                 verbose=0,
                 max_leaf_nodes=None,
                 warm_start=False,
                 validation_fraction=0.1,
                 n_iter_no_change=None,
                 tol=0.0001,
                 init=None,
                 random_state=None):

        super().__init__(n_estimators=n_estimators,
                         learning_rate=learning_rate,
                         loss=loss,
                         criterion=criterion,
                         min_samples_split=min_samples_split,
                         min_samples_leaf=min_samples_leaf,
                         min_weight_fraction_leaf=min_weight_fraction_leaf,
                         subsample=subsample,
                         max_features=max_features,
                         max_depth=max_depth,
                         min_impurity_decrease=min_impurity_decrease,
                         ccp_alpha=ccp_alpha,
                         alpha=alpha,
                         verbose=verbose,
                         max_leaf_nodes=max_leaf_nodes,
                         warm_start=warm_start,
                         validation_fraction=validation_fraction,
                         n_iter_no_change=n_iter_no_change,
                         tol=tol,
                         init=init,
                         random_state=random_state)

    def _fit_stage(self,
                   i,
                   X,
                   y,
                   raw_predictions,
                   sample_weight,
                   sample_mask,
                   random_state,
                   X_csc=None,
                   X_csr=None):

        assert sample_mask.dtype == np.bool
        loss = self._loss

        original_y = y

        raw_predictions_copy = raw_predictions.copy()

        for k in range(loss.K):
            if loss.is_multi_class:
                y = np.array(original_y == k, dtype=np.float64)

            residual = loss.negative_gradient(y,
                                              raw_predictions_copy,
                                              k=k,
                                              sample_weight=sample_weight)

            tree = DecisionTreeRegressor(
                criterion=self.criterion,
                splitter='best',
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                min_impurity_decrease=self.min_impurity_decrease,
                max_features=self.max_features,
                max_leaf_nodes=self.max_leaf_nodes,
                random_state=random_state,
                ccp_alpha=self.ccp_alpha)

            if self.subsample < 1.0:

                sample_weight = sample_weight * sample_mask.astype(np.float64)

            X = X_csr if X_csr is not None else X
            tree.fit(X,
                     residual,
                     sample_weight=sample_weight,
                     check_input=False)

            loss.update_terminal_regions(tree.tree_,
                                         X,
                                         y,
                                         residual,
                                         raw_predictions,
                                         sample_weight,
                                         sample_mask,
                                         learning_rate=self.learning_rate,
                                         k=k)

            self.estimators_[i, k] = tree

        return raw_predictions

    def _fit_stages(self,
                    X,
                    y,
                    raw_predictions,
                    sample_weight,
                    random_state,
                    X_val,
                    y_val,
                    sample_weight_val,
                    begin_at_stage=0,
                    monitor=None):
        """Iteratively fits the stages.
        For each stage it computes the progress (OOB, train score)
        and delegates to ``_fit_stage``.
        Returns the number of stages fit; might differ from ``n_estimators``
        due to early stopping.
        """
        n_samples = X.shape[0]
        do_oob = self.subsample < 1.0
        sample_mask = np.ones((n_samples, ), dtype=np.bool)
        n_inbag = max(1, int(self.subsample * n_samples))
        loss_ = self._loss

        if self.verbose:
            verbose_reporter = VerboseReporter(verbose=self.verbose)
            verbose_reporter.init(self, begin_at_stage)

        X_csc = csc_matrix(X) if issparse(X) else None
        X_csr = csr_matrix(X) if issparse(X) else None

        if self.n_iter_no_change is not None:
            loss_history = np.full(self.n_iter_no_change, np.inf)
            # We create a generator to get the predictions for X_val after
            # the addition of each successive stage
            y_val_pred_iter = self._staged_raw_predict(X_val)

        # Transform the y shape, if the problem is classification
        if type_of_target(y) == 'multiclass':
            y = LabelBinarizer().fit_transform(y)
        elif type_of_target(y) == 'binary':
            Y = np.zeros((y.shape[0], 2), dtype=np.float64)
            for k in range(2):
                Y[:, k] = y == k
            y = Y

        # perform boosting iterations
        i = begin_at_stage
        _random_sample_mask = _gradient_boosting._random_sample_mask
        for i in range(begin_at_stage, self.n_estimators):

            # subsampling
            if do_oob:
                sample_mask = _random_sample_mask(n_samples, n_inbag,
                                                  random_state)
                # OOB score before adding this stage
                old_oob_score = loss_(y[~sample_mask],
                                      raw_predictions[~sample_mask],
                                      sample_weight[~sample_mask])

            # fit next stage of trees
            raw_predictions = self._fit_stage(i, X, y, raw_predictions,
                                              sample_weight, sample_mask,
                                              random_state, X_csc, X_csr)

            # track loss
            if do_oob:
                self.train_score_[i] = loss_(y[sample_mask],
                                             raw_predictions[sample_mask],
                                             sample_weight[sample_mask])
                self.oob_improvement_[i] = (
                    old_oob_score -
                    loss_(y[~sample_mask], raw_predictions[~sample_mask],
                          sample_weight[~sample_mask]))
            else:
                # no need to fancy index w/ no subsampling
                self.train_score_[i] = loss_(y, raw_predictions, sample_weight)

            if self.verbose > 0:
                verbose_reporter.update(i, self)

            if monitor is not None:
                early_stopping = monitor(i, self, locals())
                if early_stopping:
                    break

            # We also provide an early stopping based on the score from
            # validation set (X_val, y_val), if n_iter_no_change is set
            if self.n_iter_no_change is not None:
                # By calling next(y_val_pred_iter), we get the predictions
                # for X_val after the addition of the current stage
                validation_loss = loss_(y_val, next(y_val_pred_iter),
                                        sample_weight_val)

                # Require validation_score to be better (less) than at least
                # one of the last n_iter_no_change evaluations
                if np.any(validation_loss + self.tol < loss_history):
                    loss_history[i % len(loss_history)] = validation_loss
                else:
                    break

        return i + 1

    def _check_params(self):
        """Check validity of parameters and raise ValueError if not valid."""
        check_scalar(
            self.learning_rate,
            name="learning_rate",
            target_type=numbers.Real,
            min_val=0.0,
            include_boundaries="neither",
        )

        check_scalar(
            self.n_estimators,
            name="n_estimators",
            target_type=numbers.Integral,
            min_val=1,
            include_boundaries="left",
        )

        if (
            self.loss not in self._SUPPORTED_LOSS
            or self.loss not in _gb_losses.LOSS_FUNCTIONS
        ):
            raise ValueError(f"Loss {self.loss!r} not supported. ")

        if self.loss == 'log_loss':
            loss_class = CondensedDeviance
        else:
            loss_class = MultiOutputLeastSquaresError

        if self.loss == 'log_loss':
            self._loss = loss_class(self.n_classes_)
        elif self.loss in ("huber", "quantile"):
            self._loss = loss_class(self.alpha)
        else:
            self._loss = loss_class()

        check_scalar(
            self.subsample,
            name="subsample",
            target_type=numbers.Real,
            min_val=0.0,
            max_val=1.0,
            include_boundaries="right",
        )

        if self.init is not None:
            # init must be an estimator or 'zero'
            if isinstance(self.init, BaseEstimator):
                self._loss.check_init_estimator(self.init)
            elif not (isinstance(self.init, str) and self.init == "zero"):
                raise ValueError(
                    "The init parameter must be an estimator or 'zero'. "
                    f"Got init={self.init!r}"
                )

        check_scalar(
            self.alpha,
            name="alpha",
            target_type=numbers.Real,
            min_val=0.0,
            max_val=1.0,
            include_boundaries="neither",
        )

        if isinstance(self.max_features, str):
            if self.max_features == "auto":
                if is_classifier(self):
                    max_features = max(1, int(np.sqrt(self.n_features_in_)))
                else:
                    max_features = self.n_features_in_
            elif self.max_features == "sqrt":
                max_features = max(1, int(np.sqrt(self.n_features_in_)))
            elif self.max_features == "log2":
                max_features = max(1, int(np.log2(self.n_features_in_)))
            else:
                raise ValueError(
                    f"Invalid value for max_features: {self.max_features!r}. "
                    "Allowed string values are 'auto', 'sqrt' or 'log2'."
                )

        elif self.max_features is None:
            max_features = self.n_features_in_
        elif isinstance(self.max_features, numbers.Integral):
            check_scalar(
                self.max_features,
                name="max_features",
                target_type=numbers.Integral,
                min_val=1,
                include_boundaries="left",
            )
            max_features = self.max_features
        else:  # float
            check_scalar(
                self.max_features,
                name="max_features",
                target_type=numbers.Real,
                min_val=0.0,
                max_val=1.0,
                include_boundaries="right",
            )
            max_features = max(1, int(self.max_features * self.n_features_in_))

        self.max_features_ = max_features

        check_scalar(
            self.verbose,
            name="verbose",
            target_type=(numbers.Integral, np.bool_),
            min_val=0,
        )

        check_scalar(
            self.validation_fraction,
            name="validation_fraction",
            target_type=numbers.Real,
            min_val=0.0,
            max_val=1.0,
            include_boundaries="neither",
        )

        if self.n_iter_no_change is not None:
            check_scalar(
                self.n_iter_no_change,
                name="n_iter_no_change",
                target_type=numbers.Integral,
                min_val=1,
                include_boundaries="left",
            )

        check_scalar(
            self.tol,
            name="tol",
            target_type=numbers.Real,
            min_val=0.0,
            include_boundaries="neither",
        )

    def fit(self, X, y, sample_weight=None, monitor=None):

        if not self.warm_start:
            self._clear_state()

        X, y = self._validate_data(X,
                                   y,
                                   accept_sparse=['csr', 'csc', 'coo'],
                                   dtype=DTYPE,
                                   multi_output=True)

        sample_weight_is_none = sample_weight is None

        sample_weight = _check_sample_weight(sample_weight, X)

        if is_classifier(self):
            y = column_or_1d(y, warn=True)
            y = self._validate_y(y, sample_weight)
        else:
            y = self._validate_y(y)

        if self.n_iter_no_change is not None:
            stratify = y if is_classifier(self) else None
            X, X_val, y, y_val, sample_weight, sample_weight_val = (
                train_test_split(X,
                                 y,
                                 sample_weight,
                                 random_state=self.random_state,
                                 test_size=self.validation_fraction,
                                 stratify=stratify))
            if is_classifier(self):
                if self.n_classes_ != np.unique(y).shape[0]:
                    raise ValueError(
                        'The training data after the early stopping split '
                        'is missing some classes. Try using another random '
                        'seed.')
        else:
            X_val = y_val = sample_weight_val = None

        self._check_params()

        if not self._is_initialized():

            self._init_state()

            if self.init_ == 'zero':
                raw_predictions = np.zeros(shape=(X.shape[0], self._loss.K),
                                           dtype=np.float64)
            else:
                if sample_weight_is_none:
                    self.init_.fit(X, y)
                else:
                    msg = ("The initial estimator {} does not support sample "
                           "weights.".format(self.init_.__class__.__name__))
                    try:
                        self.init_.fit(X, y, sample_weight=sample_weight)
                    except TypeError:  # regular estimator without SW support
                        raise ValueError(msg)
                    except ValueError as e:
                        if "pass parameters to specific steps of "\
                            "your pipeline using the "\
                                "stepname__parameter" in str(e):  # pipeline
                            raise ValueError(msg) from e
                        else:  # regular estimator whose input checking failed
                            raise

                raw_predictions = \
                    self._loss.get_init_raw_predictions(X, self.init_)

            begin_at_stage = 0

            # The rng state must be preserved if warm_start is True
            self._rng = check_random_state(self.random_state)

        else:

            if self.n_estimators < self.estimators_.shape[0]:
                raise ValueError(
                    'n_estimators=%d must be larger or equal to '
                    'estimators_.shape[0]=%d when '
                    'warm_start==True' %
                    (self.n_estimators, self.estimators_.shape[0]))
            begin_at_stage = self.estimators_.shape[0]
            X = check_array(X, dtype=DTYPE, order="C", accept_sparse='csr')
            raw_predictions = self._raw_predict(X)
            self._resize_state()

        # fit the boosting stages
        n_stages = self._fit_stages(X, y, raw_predictions, sample_weight,
                                    self._rng, X_val, y_val, sample_weight_val,
                                    begin_at_stage, monitor)

        # changne shape of arrays after fit (early-stopping or additional ests)
        if n_stages != self.estimators_.shape[0]:
            self.estimators_ = self.estimators_[:n_stages]
            self.train_score_ = self.train_score_[:n_stages]
            if hasattr(self, 'oob_improvement_'):
                self.oob_improvement_ = self.oob_improvement_[:n_stages]

        self.n_estimators_ = n_stages
        return self

    def _raw_predict_init(self, X):
        """Check input and compute raw predictions of the init estimator."""
        self._check_initialized()
        X = self.estimators_[0, 0]._validate_X_predict(X, check_input=True)
        if self.init_ == 'zero':
            raw_predictions = np.zeros(shape=(X.shape[0], self._loss.K),
                                       dtype=np.float64)
        else:
            raw_predictions = self._loss.get_init_raw_predictions(
                X, self.init_).astype(np.float64)
        return raw_predictions

    def _raw_predict(self, X):
        """Return the sum of the trees raw predictions (+ init estimator)."""
        raw_predictions = self._raw_predict_init(X)
        if raw_predictions.shape[1] == 1:
            raw_predictions = np.squeeze(raw_predictions)
        for i in range(self.n_estimators):
            tree = self.estimators_[i, 0]
            raw_predictions += (self.learning_rate * tree.predict(X))
        return raw_predictions

    def _staged_raw_predict(self, X):
        X = check_array(X, dtype=DTYPE, order="C", accept_sparse='csr')
        raw_predictions = self._raw_predict_init(X)
        if raw_predictions.shape[1] == 1:
            raw_predictions = np.squeeze(raw_predictions)
        for i in range(self.n_estimators):
            tree = self.estimators_[i, 0]
            raw_predictions += (self.learning_rate * tree.predict(X))
            yield raw_predictions.copy()


class cgb_clf(GradientBoostingClassifier, CondensedGradientBoosting):

    def __init__(self,
                 *,
                 loss='log_loss',
                 learning_rate=0.1,
                 n_estimators=100,
                 subsample=1.0,
                 criterion='squared_error',
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_depth=3,
                 min_impurity_decrease=0.,
                 init=None,
                 random_state=None,
                 max_features=None,
                 verbose=0,
                 max_leaf_nodes=None,
                 warm_start=False,
                 validation_fraction=0.1,
                 n_iter_no_change=None,
                 tol=1e-4,
                 ccp_alpha=0.0):

        super().__init__(loss=loss,
                         learning_rate=learning_rate,
                         n_estimators=n_estimators,
                         criterion=criterion,
                         min_samples_split=min_samples_split,
                         min_samples_leaf=min_samples_leaf,
                         min_weight_fraction_leaf=min_weight_fraction_leaf,
                         max_depth=max_depth,
                         init=init,
                         subsample=subsample,
                         max_features=max_features,
                         random_state=random_state,
                         verbose=verbose,
                         max_leaf_nodes=max_leaf_nodes,
                         min_impurity_decrease=min_impurity_decrease,
                         warm_start=warm_start,
                         validation_fraction=validation_fraction,
                         n_iter_no_change=n_iter_no_change,
                         tol=tol,
                         ccp_alpha=ccp_alpha)

    def predict(self, X):
        raw_predictions = self.decision_function(X)
        encoded_labels = \
            self._loss._raw_prediction_to_decision(raw_predictions)
        return self.classes_.take(encoded_labels, axis=0)

    def staged_predict(self, X):
        for raw_predictions in self._staged_raw_predict(X):
            encoded_labels = \
                self._loss._raw_prediction_to_decision(raw_predictions)
            yield self.classes_.take(encoded_labels, axis=0)


class cgb_reg(GradientBoostingRegressor, CondensedGradientBoosting):

    def __init__(self,
                 *,
                 loss='ls',
                 learning_rate=0.1,
                 n_estimators=100,
                 subsample=1.0,
                 criterion='squared_error',
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_depth=3,
                 min_impurity_decrease=0.,
                 init=None,
                 random_state=None,
                 max_features=None,
                 alpha=0.9,
                 verbose=0,
                 max_leaf_nodes=None,
                 warm_start=False,
                 validation_fraction=0.1,
                 n_iter_no_change=None,
                 tol=1e-4,
                 ccp_alpha=0.0,
                 metric='RMSE'):

        super().__init__(loss=loss,
                         learning_rate=learning_rate,
                         n_estimators=n_estimators,
                         criterion=criterion,
                         min_samples_split=min_samples_split,
                         min_samples_leaf=min_samples_leaf,
                         min_weight_fraction_leaf=min_weight_fraction_leaf,
                         max_depth=max_depth,
                         init=init,
                         subsample=subsample,
                         max_features=max_features,
                         min_impurity_decrease=min_impurity_decrease,
                         random_state=random_state,
                         verbose=verbose,
                         alpha=alpha,
                         max_leaf_nodes=max_leaf_nodes,
                         warm_start=warm_start,
                         validation_fraction=validation_fraction,
                         n_iter_no_change=n_iter_no_change,
                         tol=tol,
                         ccp_alpha=ccp_alpha)
        self.metric = metric

    def predict(self, X):
        X = check_array(X, dtype=DTYPE, order="C", accept_sparse='csr')
        return self._raw_predict(X)

    def score(self, X, y):
        pred = self.predict(X)
        if self.metric == 'RMSE':
            err = mean_squared_error(y_true=y, y_pred=pred,
                                     squared=False, multioutput='raw_values')
        else:
            err = r2_score(y_true=y, y_pred=pred)
        return err
