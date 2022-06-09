.. Parameter documentation master file.

Parameters
==========

Specifications of the C-GB parameters.

Base model
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


cgb_clf
-----

  - ``loss``: {`log_loss`, `ls`}, default = **log_loss**
  
    - The loss function for optimization. For the Multi-class/Binary classification, it should be ``deviance``.



cgb_reg
-----
  - ``loss``: {`log_loss`, `ls`}, default = **ls**
  
    - The loss function for optimization. For the regression it should set to ``ls``.

  - ``metric`` : {`rmse`, `r2_score`}, default = **rmse**

    - It returns the error of the model. ``rmse`` will return the average R2 score.
