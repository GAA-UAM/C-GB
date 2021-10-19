.. Parameter documentation master file.

Parameters
==========

Specifications of the C-GB parameters.

Parameters
----
  - ``loss``: {`deviance`, None}, default=**deviance**
  
    - The loss function for optimization. For the Multi-class/Binary classification, it should be deviance and else for regression.
    
  - ``n_estimators`` : int, default=**100**
  
    - Number of Decision Regressor Tree to build an ensemble.
 
  - ``subsample`` : float, default=**1.0**
  
    - The division of samples for fitting the base learners. 
