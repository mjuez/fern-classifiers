# Fern-based classifiers

In a nutshell, a fern is a decision tree where all nodes of each depth level use the same feature and split criteria. Although it is simpler than a tree, when used as an ensemble of ferns, it turns fast and powerful!

Random Ferns were first proposed in [[1]](#ref_1) for image keypoint recognition tasks. Later, in [[2]](#ref_2) a general purpose machine learning method was developed in R, and is the base line of this study.

Our implementations follow the python [scikit-learn](https://scikit-learn.org/stable/) API.

## Authors

This implementation was developed and maintained by [Mario Juez-Gil](mailto:mariojg@ubu.es) from [ADMIRABLE](https://www.admirable-ubu.es) research group of the University of Burgos, with the help and useful advice from [Álvar Arnaiz-González](https://scholar.google.es/citations?user=_9C0tpMAAAAJ&hl=es), [Juan J. Rodriguez](https://scholar.google.es/citations?user=p4m8t6oAAAAJ&hl=es), and [Ludmila Kuncheva](https://lucykuncheva.co.uk/). The work was supervised by [César García-Osorio](https://scholar.google.es/citations?user=X08I-_4AAAAJ&hl=es) and [Carlos López-Nozal](https://scholar.google.es/citations?user=JAS4N-oAAAAJ&hl=es).

---

## Usage

Cython was used for speeding the algorithms. Because of that, the code must be compiled as follows:

```bash
python setup.py build_ext --inplace
```

Once compiled, it can be used like any scikit-learn classifier. A basic example could be:

`example.py`

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from fc.ferns import FernClassifier

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target,
                                                    test_size=0.5)
fern = FernClassifier(depth=5, random_state=1)
fern.fit(X_train, y_train)
y_pred = fern.predict(X_test)
print(confusion_matrix(y_test, y_pred))
```

`output:`

```
[[24  1  0]
 [ 3 19  1]
 [ 0  7 20]]
```

---

## References

<a name="ref_1"></a>[1] M. Ozuysal, M. Calonder, V. Lepetit, and P. Fua, “Fast Keypoint Recognition Using Random Ferns,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 32, no. 3, pp. 448–461, Mar. 2010. DOI: [10.1109/TPAMI.2009.23](https://doi.org/10.1109/TPAMI.2009.23)

<a name="ref_2"></a>[2] M. B. Kursa, “rFerns: An Implementation of the Random Ferns Method for General-Purpose Machine Learning,” J. Stat. Softw., vol. 61, no. 10, pp. 1–13, Nov. 2014. DOI: [10.18637/jss.v061.i10](https://doi.org/10.18637/jss.v061.i10)

---

## License

Licensed under the [GNU GPLv3](https://opensource.org/licenses/GPL-3.0), please see the [LICENSE](LICENSE) file for more details.

---

## Acknowledgements

This work was partially supported by the Consejería de Educación of the 
Junta de Castilla y León and by the European Social Fund with the 
EDU/1100/2017 pre-doctoral grants; by the project TIN2015-67534-P 
(MINECO/FEDER, UE) of the Ministerio de Economía Competitividad of the 
Spanish Government and the project BU085P17 (JCyL/FEDER, UE) of the Junta de 
Castilla y León both cofinanced from European Union FEDER funds.