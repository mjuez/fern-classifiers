#!/usr/bin/env python
# Copyright (C) 2019  Mario Juez-Gil <mariojg@ubu.es>
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# Aknowledgements
# ---------------
# This work was partially supported by the Consejería de Educación of the 
# Junta de Castilla y León and by the European Social Fund with the 
# EDU/1100/2017 pre-doctoral grants; by the project TIN2015-67534-P 
# (MINECO/FEDER, UE) of the Ministerio de Economía Competitividad of the 
# Spanish Government and the project BU085P17 (JCyL/FEDER, UE) of the Junta de 
# Castilla y León both cofinanced from European Union FEDER funds.

import numpy as np
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import check_classification_targets
from sklearn.base import BaseEstimator, ClassifierMixin
from ._ferns import c_comp_leaf, c_populate_leafs
from ._ferns import c_comp_proj_leaf, c_populate_proj_leafs

__author__ = "Mario Juez-Gil"
__copyright__ = "Copyright 2019, Mario Juez-Gil"
__credits__ = ["Mario Juez-Gil", "Álvar Arnaiz-González", 
               "Cesar Garcia-Osorio", "Carlos López-Nozal",
               "Juan J. Rodriguez"]
__license__ = "GPLv3"
__version__ = "1.0"
__maintainer__ = "Mario Juez-Gil"
__email__ = "mariojg@ubu.es"

class FernClassifier(BaseEstimator, ClassifierMixin):
    """
    A fern classifier.
    Random Ferns is a bagging-like ensemble of fern classifiers.

    Parameters
    ----------
    depth : int, optional (default=5)
        Depth of the fern.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    See also
    --------
    BaggingClassifier

    References
    ----------
    .. [1] M. B. Kursa, “rFerns: An Implementation of the Random Ferns Method 
            for General-Purpose Machine Learning,” J. Stat. Softw., vol. 61, 
            no. 10, pp. 1–13, Nov. 2014.
    """

    def __init__(self, depth=5, random_state=None):
        """
        Fern initialization.
        """
        self.depth = depth
        self.random_state = random_state

    def fit(self, X, y):
        """
        Build a Fern classifier from the training set (X, y).

        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
          The training input samples. Internally, it will be converted to
          ``dtype=np.float32`` and if a sparse matrix is provided
          to a sparse ``csc_matrix``.
        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
          The target values (class labels) as integers or strings.

        Returns
        -------
        self : object
        """
        random_state = check_random_state(self.random_state)
        self.num_instances = X.shape[0]
        self.num_features = X.shape[1]
        self.rnd_features = random_state.randint(self.num_features,
                                                 size=self.depth)
        self.rnd_thresholds = self._compute_thresholds(X, random_state)

        check_classification_targets(y)
        y = self._encode_y(y)
        assert self.n_outputs_ == 1, \
            "Multilabel datasets are currently unsupported"

        if self.n_outputs_ == 1:
            self.n_classes_ = self.n_classes_[0]
            self.classes_ = self.classes_[0]
            
        leafs = self._populate_leafs(X, y)

        self.leafs_proba = self._estimate_probabilities(leafs)

        return self

    def predict(self, X):
        """
        Predict class value for X.
        The predicted class for each sample in X is returned.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        y : array of shape = [n_samples] or [n_samples, n_outputs]
            The predicted classes, or the predict values.
        """
        predicted_probabilitiy = self.predict_proba(X)
        return self.classes_.take((np.argmax(predicted_probabilitiy, axis=1)),
                                  axis=0)

    def predict_proba(self, X):
        """
        Predict class probabilities of the input samples X.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        p : array of shape = [n_samples, n_classes], or a list of n_outputs
            such arrays if n_outputs > 1.
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.
        """
        check_is_fitted(self, "leafs_proba")

        leafs = [self._compute_leaf(inst) for inst in X]
        return np.array([self.leafs_proba[idx] for idx in leafs])

    def predict_log_proba(self, X):
        """
        Predict class log-probabilities of the input samples X.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        p : array of shape = [n_samples, n_classes], or a list of n_outputs
            such arrays if n_outputs > 1.
            The class log-probabilities of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.
        """
        return np.log(self.predict_proba(X))

    def _encode_y(self, y):
        """
        Encodes the target classes with consecutive numbers beginning from 
        zero.

        Parameters
        ----------
        y : original target classes.

        Returns
        -------
        y_encoded : numpy array.
        """
        self.classes_ = []
        self.n_classes_ = []

        if y.ndim == 1:
            y = np.reshape(y, (-1, 1))

        self.n_outputs_ = y.shape[1]

        y_encoded = np.zeros(y.shape, dtype=np.int)
        for k in range(self.n_outputs_):
            classes_k, y_encoded[:, k], self.class_counts = np.unique(
                y[:, k], return_inverse=True, return_counts=True)
            self.classes_.append(classes_k)
            self.n_classes_.append(classes_k.shape[0])

        self.n_classes_ = np.array(self.n_classes_, dtype=np.intp)

        return y_encoded

    def _compute_thresholds(self, X, random_state):
        """
        Computes thresholds for each depth of the fern. A threshold is 
        calculated as the mean value of the feature selected for that depth of
        two random instances of the dataset.

        Parameters
        ----------
        X : the set of training instances.
        random_state : a random state for choosing the random instances.

        Returns
        -------
        thresholds : an array of the same size as fern depth with computed
            thresholds.
        """
        rnd_instances = random_state.choice(self.num_instances, 
                                     (self.depth, 2), replace=False)
        thresholds = np.empty(self.depth)
        for i in range(self.depth):
            x1, x2 = rnd_instances[i]
            f = self.rnd_features[i]
            thresholds[i] = (X[x1][f] + X[x2][f]) / 2
        return thresholds

    def _estimate_probabilities(self, leafs):
        """
        Estimates probabilities for each class in each leaf of the fern.
        This refers to formulas 5 and 6 of [1].

        Parameters
        ----------
        leafs : a matrix containing all leaf counts of the fern.

        Returns
        -------
        leafs_proba : a matrix containing all probabilities for each leaf of 
            the fern.
        """
        _bal_nume = ((self.num_instances + self.n_classes_))
        _bal_deno = (self.class_counts + 1)
        bal_term = _bal_nume / _bal_deno

        denominator = (np.sum(leafs, axis=1) + self.n_classes_)
        return np.log(((leafs + 1) / denominator[:, np.newaxis]) * bal_term)

    def _compute_leaf(self, inst):
        """
        Computes the leaf index of the fern where given instance falls in.

        Parameters
        ----------
        inst : given instance. An instance is an array of features.

        Returns
        -------
        leaf : leaf index where given instance falls in.
        """
        return c_comp_leaf(inst, self.rnd_features, self.rnd_thresholds, 
                           self.depth)
    
    def _populate_leafs(self, X, y):
        """
        Populates fern leafs using a set of training instances. The population
        process consists in counting the number of instances of each class that
        falls in each leaf. The counts are stored in a matrix of integers with
        2^depth (leafs) rows and and number of classes columns.

        Parameters
        ----------
        X : the set of training instances.
        y : target values, are the class of each instance.

        Returns
        -------
        leafs : a matrix containing the number of examples of each class in
            each leaf of the fern.
        """
        return c_populate_leafs(X, y, self.n_classes_, self.num_instances,
                                self.rnd_features, self.rnd_thresholds, 
                                self.depth)

class ProjectionFernClassifier(BaseEstimator, ClassifierMixin):
    """
    A projection fern classifier.
    Random Projection Ferns is a bagging-like ensemble of projection fern 
    classifiers.

    Parameters
    ----------
    depth : int, optional (default=5)
        Depth of the fern.
    num_pf : int, optional (default=3)
        Number of features to be projected.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    See also
    --------
    BaggingClassifier

    References
    ----------
    .. [1] M. B. Kursa, “rFerns: An Implementation of the Random Ferns Method 
            for General-Purpose Machine Learning,” J. Stat. Softw., vol. 61, 
            no. 10, pp. 1–13, Nov. 2014.
    """

    def __init__(self, depth=5, num_pf=3, random_state=None):
        """
        Fern initialization.
        """
        self.depth = depth
        self.num_pf = num_pf
        self.random_state = random_state

    def fit(self, X, y):
        """
        Build a Fern classifier from the training set (X, y).

        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
          The training input samples. Internally, it will be converted to
          ``dtype=np.float32`` and if a sparse matrix is provided
          to a sparse ``csc_matrix``.
        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
          The target values (class labels) as integers or strings.

        Returns
        -------
        self : object
        """
        random_state = check_random_state(self.random_state)
        self.num_instances = X.shape[0]
        self.num_features = X.shape[1]
        self.rnd_features = random_state.randint(self.num_features,
                                                size=(self.depth, self.num_pf))
        self.rnd_proj = random_state.uniform(-1, 1, (self.num_pf, self.depth))
        X = self._compute_projections(X)
        self.rnd_thresholds = self._compute_thresholds(X, random_state)

        check_classification_targets(y)
        y = self._encode_y(y)
        assert self.n_outputs_ == 1, \
            "Multilabel datasets are currently unsupported"

        if self.n_outputs_ == 1:
            self.n_classes_ = self.n_classes_[0]
            self.classes_ = self.classes_[0]
            
        leafs = self._populate_leafs(X, y)

        self.leafs_proba = self._estimate_probabilities(leafs)

        return self

    def predict(self, X):
        """
        Predict class value for X.
        The predicted class for each sample in X is returned.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        y : array of shape = [n_samples] or [n_samples, n_outputs]
            The predicted classes, or the predict values.
        """
        predicted_probabilitiy = self.predict_proba(X)
        return self.classes_.take((np.argmax(predicted_probabilitiy, axis=1)),
                                  axis=0)

    def predict_proba(self, X):
        """
        Predict class probabilities of the input samples X.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        p : array of shape = [n_samples, n_classes], or a list of n_outputs
            such arrays if n_outputs > 1.
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.
        """
        check_is_fitted(self, "leafs_proba")

        X = self._compute_projections(X)
        leafs = [self._compute_leaf(inst) for inst in X]
        return np.array([self.leafs_proba[idx] for idx in leafs])

    def predict_log_proba(self, X):
        """
        Predict class log-probabilities of the input samples X.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        p : array of shape = [n_samples, n_classes], or a list of n_outputs
            such arrays if n_outputs > 1.
            The class log-probabilities of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.
        """
        return np.log(self.predict_proba(X))

    def _encode_y(self, y):
        """
        Encodes the target classes with consecutive numbers beginning from 
        zero.

        Parameters
        ----------
        y : original target classes.

        Returns
        -------
        y_encoded : numpy array.
        """
        self.classes_ = []
        self.n_classes_ = []

        if y.ndim == 1:
            y = np.reshape(y, (-1, 1))

        self.n_outputs_ = y.shape[1]

        y_encoded = np.zeros(y.shape, dtype=np.int)
        for k in range(self.n_outputs_):
            classes_k, y_encoded[:, k], self.class_counts = np.unique(
                y[:, k], return_inverse=True, return_counts=True)
            self.classes_.append(classes_k)
            self.n_classes_.append(classes_k.shape[0])

        self.n_classes_ = np.array(self.n_classes_, dtype=np.intp)

        return y_encoded

    def _compute_projections(self, X):
        """
        Calculates random projections for all depth levels.
        Each column contains the projected values of the random chosen 
        attributes for all instances for each depth level in the fern.

        Parameters
        ----------
        X : the set of training instances.
        
        Returns
        -------
        projected_training_set : array of shape (num_instances, depth)
        """
        return np.squeeze(np.array([X[:,self.rnd_features[i]] @ \
            self.rnd_proj[:,i,np.newaxis] for i in range(self.depth)])).T

    def _compute_thresholds(self, X, random_state):
        """
        Computes thresholds for each depth of the fern. A threshold is 
        calculated as the mean value of the feature selected for that depth of
        two random instances of the dataset.

        Parameters
        ----------
        X : the set of training instances.
        random_state : a random state for choosing the random instances.

        Returns
        -------
        thresholds : an array of the same size as fern depth with computed
            thresholds.
        """
        rnd_instances = random_state.choice(self.num_instances, 
                                     (self.depth, 2), replace=False)
        thresholds = np.empty(self.depth)
        for i in range(self.depth):
            x1, x2 = rnd_instances[i]
            thresholds[i] = (X[x1,i] + X[x2,i]) / 2
        return thresholds

    def _estimate_probabilities(self, leafs):
        """
        Estimates probabilities for each class in each leaf of the fern.
        This refers to formulas 5 and 6 of [1].

        Parameters
        ----------
        leafs : a matrix containing all leaf counts of the fern.

        Returns
        -------
        leafs_proba : a matrix containing all probabilities for each leaf of 
            the fern.
        """
        _bal_nume = ((self.num_instances + self.n_classes_))
        _bal_deno = (self.class_counts + 1)
        bal_term = _bal_nume / _bal_deno

        denominator = (np.sum(leafs, axis=1) + self.n_classes_)
        return np.log(((leafs + 1) / denominator[:, np.newaxis]) * bal_term)

    def _compute_leaf(self, inst):
        """
        Computes the leaf index of the fern where given instance falls in.

        Parameters
        ----------
        inst : given instance. An instance is an array of features.

        Returns
        -------
        leaf : leaf index where given instance falls in.
        """
        return c_comp_proj_leaf(inst, self.rnd_thresholds, self.depth)
    
    def _populate_leafs(self, X, y):
        """
        Populates fern leafs using a set of training instances. The population
        process consists in counting the number of instances of each class that
        falls in each leaf. The counts are stored in a matrix of integers with
        2^depth (leafs) rows and and number of classes columns.

        Parameters
        ----------
        X : the set of training instances.
        y : target values, are the class of each instance.

        Returns
        -------
        leafs : a matrix containing the number of examples of each class in
            each leaf of the fern.
        """
        return c_populate_proj_leafs(X, y, self.n_classes_, self.num_instances,
                                self.rnd_thresholds, self.depth)