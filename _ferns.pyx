# cython: boundscheck=False

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

cimport numpy as np
import numpy as np

np.import_array()
ctypedef fused numeric:
    double
    long

cpdef int c_comp_leaf(numeric[:] inst, long[:] rnd_features, 
    double[:] rnd_thresholds, int depth):
    """
    Computes the leaf index of the fern where given instance falls in.

    Parameters
    ----------
    inst : given instance. An instance is an array of features (float values).
    rnd_features : an array of random feature indexes of size the fern depth.
    rnd_thresholds : an array of random thresholds for making the decisions.
    depth : fern depth.

    Returns
    -------
    leaf : leaf index where given instance falls in.
    """
    
    cdef int leaf = 0
    cdef int d
    for d in range(depth):
        if inst[rnd_features[d]] >= rnd_thresholds[d]:
            leaf += 2 ** d
    return leaf

cpdef int c_comp_proj_leaf(double[:] inst, double[:] rnd_thresholds, 
    int depth):
    """
    Computes the leaf index of the fern where given instance falls in.

    Parameters
    ----------
    inst : given instance. An instance is an array of features (float values).
    rnd_thresholds : an array of random thresholds for making the decisions.
    depth : fern depth.

    Returns
    -------
    leaf : leaf index where given instance falls in.
    """
    
    cdef int leaf = 0
    cdef int d
    for d in range(depth):
        if inst[d] >= rnd_thresholds[d]:
            leaf += 2 ** d
    return leaf

cpdef np.ndarray[long, ndim=2] c_populate_leafs(numeric[:,:] X, np.ndarray y, 
    long n_classes, long n_instances, long[:] rnd_features, 
    double[:] rnd_thresholds, int depth):
    """
    Populates fern leafs using a set of training instances. The population
    process consists in counting the number of instances of each class that
    falls in each leaf. The counts are stored in a matrix of integers with
    2^depth (leafs) rows and and number of classes columns.

    Parameters
    ----------
    X : the set of training instances.
    y : target values, are the class of each instance.
    n_classes : number of different classes.
    n_instances : number of instances.
    rnd_features : an array of random feature indexes of size the fern depth.
    rnd_thresholds : an array of random thresholds for making the decisions.
    depth : fern depth.

    Returns
    -------
    leafs : a matrix containing the number of examples of each class in each
            leaf of the fern.
    """

    cdef int n_leafs = 2 ** depth
    cdef np.ndarray[long, ndim=2] leafs = np.zeros([n_leafs, n_classes], 
                                                    dtype=np.int)
    cdef int i
    for i in range(n_instances):
        leaf = c_comp_leaf(X[i], rnd_features, rnd_thresholds, depth)
        leafs[leaf][y[i]] = leafs[leaf][y[i]] + 1
    return leafs

cpdef np.ndarray[long, ndim=2] c_populate_proj_leafs(double[:,:] X, 
    np.ndarray y, long n_classes, long n_instances, double[:] rnd_thresholds, 
    int depth):
    """
    Populates fern leafs using a set of training instances. The population
    process consists in counting the number of instances of each class that
    falls in each leaf. The counts are stored in a matrix of integers with
    2^depth (leafs) rows and and number of classes columns.

    Parameters
    ----------
    X : the set of training instances.
    y : target values, are the class of each instance.
    n_classes : number of different classes.
    n_instances : number of instances.
    rnd_thresholds : an array of random thresholds for making the decisions.
    depth : fern depth.

    Returns
    -------
    leafs : a matrix containing the number of examples of each class in each
            leaf of the fern.
    """

    cdef int n_leafs = 2 ** depth
    cdef np.ndarray[long, ndim=2] leafs = np.zeros([n_leafs, n_classes], 
                                                    dtype=np.int)
    cdef int i
    for i in range(n_instances):
        leaf = c_comp_proj_leaf(X[i], rnd_thresholds, depth)
        leafs[leaf][y[i]] = leafs[leaf][y[i]] + 1
    return leafs

cpdef np.ndarray[double, ndim=2] c_comp_projs(numeric[:,:] X, 
    long[:,:] rnd_features, double[:,:] rnd_proj, long n_instances, int depth):
    """
    Computes the random projections of each instance given a set of features
    for each depth level. The resulting matrix will have n_instances rows and
    depth columns. The first column will be the projection for the first depth
    level, and so on.

    Parameters
    ----------
    X : the set of instances
    rnd_features : a matrix of random feature indexes of size the fern depth.
    rnd_proj : a matrix of random projections where to project each instance.
    n_instances : the number of instances to project.
    depth : fern depth.

    Returns
    -------
    projs : a matrix containing the projections of the instances.
    """
    
    cdef np.ndarray[double, ndim=2] projs = np.zeros([n_instances, depth],  
                                                        dtype=np.float)
    cdef int d, i, f
    for d in range(depth):
        for i in range(n_instances):
            for f in range(len(rnd_features[d])):
                projs[i, d] += X[i, rnd_features[d, f]] * rnd_proj[d, f]
    return projs