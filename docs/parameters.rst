.. Parameter documentation master file.

Parameters
==========

Specifications of the C-GB parameters.

Common
----
    
  - ``n_estimators`` : int, default = **100**
  
    - Number of Decision Regressor Tree to build an ensemble.
 
  - ``subsample`` : float, default = **1.0**
  
    - The division of samples for fitting the base learners. 

  - ``max_features`` : {‘auto’, ‘sqrt’, ‘log2’}, int or float, default= **None**
  
    - The number of points for splitting the tree.

        - ``auto`` , ``sqrt`` >> sqrt(n_features)
        - ``log2`` >> log2(n_features)
        - ``None`` >> n_features


C_GradientBoostingClassifier
-----

- ``loss``: {`deviance`, `ls`}, default = **deviance**
  
    - The loss function for optimization. For the Multi-class/Binary classification, it should be ``deviance``.



C_GradientBoostingRegressor
-----
- ``loss``: {`deviance`, `ls`}, default = **ls**
  
    - The loss function for optimization. For the regression it should set to ``ls``.

- ``metric`` : {`RMSE`, ``euclidean``}, default = **RMSE**

    - It returns the error of the model. ``RMSE`` will return the average error on the euclidean space, where ``euclidean`` returns the distance between the real and predicted of each point.
