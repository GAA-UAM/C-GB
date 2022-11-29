# Condensed-Gradient Boosting 

# C-GB
Condensed-Gradient Boosting

# Table of contents
* [Introduction](#Introduction)
* [Installation](#Installation)
* [Usage](#Usage)
* [Requirements](#Requirements)
* [Citation](#Citation)
* [Contributions](#Contributions)
* [Version](#Version)

# Introduction
Gradient Boosting Machine is a machine learning model for classification and regression problems. In the following, we present a Condensed Gradient Boosting model that works well for multiclass multi-output regression with high precision and speed. 


# Installation
inbuilt Python package management system, pip. You can can install, update, or delete the cgb.


`pip install cgb`


# Usage
To train the CGB model for both multiclass classification and multioutput regression, first, it should be installed.
After importing the class, define the model with [hyperparameters](https://github.com/samanemami/C-GB/blob/main/docs/parameters.rst) or use the default values for it.
Models run on both Windows and Linux.

To access more examples, plots, and related codes, please refer to [C_GB-EX](https://github.com/samanemami/C_GB-EX).

In the following, the implementation of the algorithm for two problems (classification and regression) is described;

### Classification
```Python
from cgb import cgb_clf
import sklearn.datasets as dts
from sklearn.model_selection import train_test_split

X, y = dts.make_classification(
    n_samples=2000, n_classes=3, n_clusters_per_class=2,
    random_state=1, n_informative=4)

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1)

model = cgb_clf(max_depth=20,
                subsample=1,
                loss='log_loss',
                max_features='sqrt',
                learning_rate=0.1,
                random_state=1,
                criterion="squared_error",
                n_estimators=100)

model.fit(x_train, y_train)
print(model.score(x_test, y_test))
confusion_matrix(y_test, model.predict(x_test))
```

```output
Return the mean accuracy regarding the n_classes.
>>> 79.67%
array([[151,  16,  24],
       [ 14, 166,  23],
       [ 29,  16, 161]], dtype=int64)
```

<hr>

### Regression/ Multi-label classification
```Python
from cgb import cgb_reg
import sklearn.datasets as dts
from sklearn.model_selection import train_test_split

seed = 1

X, y = dts.make_regression(n_samples=100,
                           n_features=100,
                           n_targets=3,
                           random_state=seed)
                           
                           
x_train, x_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3, 
                                                    random_state=seed)


model = cgb_reg(learning_rate=0.1,
                subsample=1,
                loss='ls',
                max_features="sqrt",
                n_estimators=100,
                max_depth=3,
                random_state=seed)

model.fit(x_train, y_train)
model.predict(x_test)
model.score(x_test, y_test)
```

```output
Return the RMSE for n_outputs
>>> array([164.36658903, 101.0311495 , 166.13994623])
```

# Requirements
This package uses the following libraries, which we already defined the dependencies;

<ul>
  <li>scipy</li>
  <li>numbers</li>
  <li>numpy - Numerical Python</li>
  <li>scikit-learn - Machine learning in Python</li>
</ul>

# Citation
[Cite](CITATION.cff) this package as below.

```yaml
References:
    Type: article
    Authors:
      - Seyedsaman Emami
      - Gonzalo Martínez-Muñoz
    Arxiv:
      - https://arxiv.org/pdf/2211.14599.pdf
    Keywords:
      - "Gradient Boosting"
      - "multi-output regression"
      - "multiclass-classification"
```
## How to cite C-GB
If you are using this package in your paper, please cite our work as follows;

```yaml
Emami, Seyedsaman, and Gonzalo Martínez-Muñoz. "Condensed Gradient Boosting." arXiv preprint arXiv:2211.14599 (2022).
```



## Keywords
`Gradient Boosting`, `multi-output regression`, `multiclass-classification regression`


# Contributions
You may improve this project by creating an issue, reporting an improvement or a bug, forking, and of course, sending a pull request to the development branch. 
The authors and developers involved in this package can be found in the [contributor's file](contributors.txt).

In the following, you will find the different approaches to contribute;
<ul>
    <li> Code reviews. </li>
    <li> Create an issue. </li>
    <li> GitHub pull request </li>
    <li> Contribute to producing different experiments. </li>
    <li> Write posts on the blog, Linkedin, your websites about `C-GB`. </li>
</ul>
 
## Key members of C-GB
* [Gonzalo Martínez-Muñoz](https://github.com/gmarmu)
* [Seyedsaman Emami](https://github.com/samanemami)

# Version
0.0.3

## Updated
08.Jun.2022

## Date-released
01.Oct.2021

# Related links
* Examples, codes to reproduce the results, and additional experiments. Refer [C_GB-EX](https://github.com/samanemami/C_GB-EX).
* For the condensed model and analysis features, refer to our [paper](https://arxiv.org/abs/2211.14599)
* For instructions, please refer to the [documentation](https://github.com/samanemami/C-GB/tree/main/docs).

