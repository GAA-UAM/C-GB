.. Examples documentation master file.

Examples
========

Multi-class classification
-------------

import the ``cgb``


  >>> import cgb
  >>> import sklearn.datasets as dts
  >>> from sklearn.model_selection import train_test_split

You may fit the model for a dataset with n>2 class labels. Although it works perfectly for binary problems as well.

  >>> X, y = dts.make_classification(n_samples=100, n_classes=3,
                               n_clusters_per_class=2,
                               random_state=1, n_informative=4)
 
Split the data. Here to show the instruction of the model we consider a simple split. We suggest choosing the proper splitting method with regard to your dataset.

  >>> x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1)

      
import `C_GradientBoostingClassifier` fro the ``cgb``. For classification, you should use ``deviance`` as the loss function. You may leave other hyperparameters with default values.

  >>> >> > model = C_GradientBoostingClassifier(max_depth=5, subsample=1, max_features='sqrt', learning_rate=0.1, random_state=1, criterion="mse", loss="deviance", n_estimators=100)

                                          
 


Fit the model with 100 trees

  >>> model.fit(x_train, y_train)
  
  
Multi-output regression
-------------
