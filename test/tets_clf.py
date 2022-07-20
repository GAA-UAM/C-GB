from sklearn.model_selection import train_test_split
from cgb import cgb_clf, cgb_reg
import sklearn.datasets as dt
import warnings

warnings.simplefilter("ignore")

def model(clf=True):

    if clf:
        X, y = dt.load_iris(return_X_y=True)

        x_train, x_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=0.2)

        model_ = cgb_clf(max_depth=5,
                         subsample=0.5,
                         max_features='sqrt',
                         learning_rate=0.05,
                         random_state=1,
                         criterion="mse",
                         loss="log_loss",
                         n_estimators=100)

    else:
        X, y = dt.make_regression(n_targets=3)

        x_train, x_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=0.2)
        model_ = cgb_reg(learning_rate=0.1,
                         subsample=1,
                         max_features="sqrt",
                         loss='ls',
                         n_estimators=100,
                         max_depth=3,
                         random_state=2)

    model_.fit(x_train, y_train)
    print(model_.score(x_test, y_test))


if __name__ == "__main__":
    print('clf')
    model(clf=True)
    print('-----')
    print('reg')
    model(clf=False)
