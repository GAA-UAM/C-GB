""" Condensed Gradient Boosting Decision Tree """

# Author: Seyedsaman Emami
# Author: Gonzalo Martínez-Muñoz

# Licence: GNU Lesser General Public License v2.1 (LGPL-2.1)


from Base._Base import CondensedGradientBoosting
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils.validation import check_array
from sklearn.tree import _tree

DTYPE = _tree.DTYPE


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
                 metric='rmse'):

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
        if self.metric == 'rmse':
            err = mean_squared_error(y_true=y, y_pred=pred,
                                     squared=False, multioutput='raw_values')
        else:
            err = r2_score(y_true=y, y_pred=pred)
        return err
