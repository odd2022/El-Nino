import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA



############# DATA PREPROCESSING ###############

def get_data(Path):
    """
    Path: Path of the preprocessed data you want to get
    """
    data = np.load(Path)
    # Get the variables
    #print(data.files)
    X = data["X"]
    y = data["y"]

    # Check dimensions
    print('==== GET THE DATA ====')
    print("Shape of X:", X.shape)
    print("Shape of y:", y.shape)

    print("First 25 elements of y:", y[:25])
    return(X,y)

def split_scale_data(X, y, RANDOM_SPLIT=False, train_size=374, verbose=True, scale=True):
    """
    X, y: Inputs and outputs
    RANDOM_SPLIT: Boolean, if False splits for train and test are done in in time order, if True, splits are random
    train_size: number of data in the train sample (test_size=474 - train_size)
    verbose: if True, the function prints information about the shape of the outputs data sets
    scale: if True, data are scaled
    """
    # Split the data
    if RANDOM_SPLIT:
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=42)
    else:
        X_train, X_test, y_train, y_test = X[:train_size], X[train_size:], y[:train_size], y[train_size:]
    if scale:
        # Introduce a scaler
        scaler = StandardScaler()
        scaler.fit(X_train)
        # Standardize the train and test data
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

    # Check dimensions
    if verbose:
        print("==== SPLIT & SCALE THE DATA ====")
        print("Shape of X_train:", X_train.shape)
        print("Shape of y_train:", y_train.shape)
        print("Shape of X_test:", X_test.shape)
        print("Shape of y_test:", y_test.shape)

    return(X_train, X_test, y_train, y_test)


############# VALIDATION FUNCTION FOR LASSO OR RIDGE ###############

def validation(X_train, y_train, X_validation, y_validation, low_power, high_power, nb_points, Ridge=True):
    
    # Hyperparameter tuning: Evaluate different lambda (regularization strength) values
    parameter_values = np.logspace(low_power, high_power, nb_points)


    accuracy_train_cv_list = []
    accuracy_valid_list = []
    coefficients = []

    for param in parameter_values:
        if Ridge:
            classifier = RidgeClassifier(alpha = param)
            classifier.fit(X_train, y_train)
        else:
            classifier = LogisticRegression(penalty = 'l1', solver = 'liblinear', C = param)
            classifier.fit(X_train, y_train)

        # Store accuracies for training and validation data
        accuracy_train_cv_list.append(classifier.score(X_train, y_train))
        accuracy_valid_list.append(classifier.score(X_validation, y_validation))

        # Store coefficients
        coefficients.append(classifier.coef_)

    coefficients = np.array(coefficients)

    # Find the lambda value that maximizes validation accuracy
    max_accuracy_index = np.argmax(accuracy_valid_list)
    max_param = parameter_values[max_accuracy_index]
    max_accuracy_validation = accuracy_valid_list[max_accuracy_index]
    train_accuracy = accuracy_train_cv_list[max_accuracy_index]

    print(f"Best parameter value after validation: {max_param:.1f}")
    print(f"Accuracy for training: {train_accuracy:.3f}")
    print(f"Accuracy for validation: {max_accuracy_validation:.3f}")

    return max_param, max_accuracy_validation, train_accuracy, parameter_values, coefficients, accuracy_train_cv_list, accuracy_valid_list

