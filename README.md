# Condensed-Gradient Boosting (C-GB)


# Table of contents

* [Introduction](#Introduction)
* [Usage](#Usage)
* [Citation](#Citation)
* [Contributions](#Contributions)
* [Version](#Version)

# Introduction

Gradient Boosting Machine is a machine learning model for classification and regression problems. In the following, we present a Condensed Gradient Boosting model that works well for multiclass multi-output regression with high precision and speed.


# Usage

To train the CGB model for both multiclass classification and multioutput regression, first, it should be installed using pip.
`pip install cgb`

After importing the class, define the model with [hyperparameters](https://github.com/samanemami/C-GB/blob/main/docs/parameters.rst) or use the default values for it.
Models run on both Windows and Linux.

To access more examples, plots, and related codes, please refer to [C_GB-EX](https://github.com/samanemami/C_GB-EX).

On the [wiki](https://github.com/GAA-UAM/C-GB/wiki) page, the implementation of the algorithm for two problems (classification and regression) is described.

# Citation

[Cite](CITATION.cff) this package as below.

```log
@article{emami2022condensed,
  title={Condensed Gradient Boosting},
  author={Emami, Seyedsaman and Mart{\'\i}nez-Mu{\~n}oz, Gonzalo},
  journal={arXiv preprint arXiv:2211.14599},
  year={2022}
}
```

## Keywords

`Gradient Boosting`, `multi-output regression`, `multiclass-classification regression`

# Contributions

You may improve this project by creating an issue, reporting an improvement or a bug, forking, and of course, sending a pull request to the development branch.
The authors and developers involved in this package can be found in the [contributor&#39;s file](contributors.txt).

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

09.Jul.2023

## Date-released

01.Oct.2021

# Related links
* wiki, the model introduction along with complete examples, API, hyperparameters. Refer [wiki](https://github.com/GAA-UAM/C-GB/wiki)
* Examples, codes to reproduce the results, and additional experiments. Refer [C_GB-EX](https://github.com/samanemami/C_GB-EX).
* For the condensed model and analysis features, refer to our [paper](https://arxiv.org/abs/2211.14599)
* For instructions, please refer to the [documentation](https://github.com/samanemami/C-GB/tree/main/docs).
