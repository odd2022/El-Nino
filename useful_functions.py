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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay




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

def test_model(model_to_test, X_train, X_test, y_train, y_test, verbose=True):
    
    # Define the classifier
    model = model_to_test

    # Fit the model on the data
    model.fit(X_train,y_train)

    # Get the prediction
    y_test_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)

    if str(type(model)) == "<class 'sklearn.linear_model._ridge.RidgeClassifier'>" or str(type(model)) == "<class 'sklearn.linear_model._logistic.LogisticRegression'>":
        # Decision values for AUC computation
        train_decision_values = model.decision_function(X_train)
        test_decision_values = model.decision_function(X_test)
        # Compute scores
        accuracy_score_train = accuracy_score(y_train, y_train_pred)
        roc_auc_score_train = roc_auc_score(y_train, train_decision_values)
        accuracy_score_test = accuracy_score(y_test, y_test_pred)
        roc_auc_score_test = roc_auc_score(y_test, test_decision_values)

    else:
        # Computation of the AUC and accuracy
        roc_auc_score_train = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
        accuracy_score_train = accuracy_score(y_train, y_train_pred)

        roc_auc_score_test = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        accuracy_score_test = accuracy_score(y_test, y_test_pred)

    if verbose:
        # Print the results
        print("==== TRAIN ====")
        print(f"Accuracy for TRAIN data: {accuracy_score_train:.3f}")
        print(f"     AUC for TRAIN data: {roc_auc_score_train:.3f}")

        print("==== TEST ====")
        print(f"Accuracy for TEST data: {accuracy_score_test:.3f}")
        print(f"     AUC for TEST data: {roc_auc_score_test:.3f}")

        # Compute the confusion matrix
        conf_matrix = confusion_matrix(y_test, y_test_pred)

        disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=model.classes_)
        disp.plot(cmap='Blues')
        plt.title(f"Confusion Matrix - Model: {model}")

    return accuracy_score_test, roc_auc_score_test


def validation_PCA(X_train_cv, y_train_cv, X_validation, y_validation, model, dim_max, verbose = True): 
    """ Pipeline for validation of the parameter p of PCA for dimensionality reduction """
    
    p_values = np.linspace(1, dim_max, dim_max).astype(int)
    p_values = p_values.astype(int)
    test_score_accuracy = []
    train_score_accuracy = []
    
    for p in p_values:
        
        # PCA with p principal components
        pca = PCA(n_components = p)
        pca.fit(X_train_cv)

        # Project data
        X_train_projected = pca.transform(X_train_cv)
        X_validation_projected = pca.transform(X_validation)

        model.fit(X_train_projected, y_train_cv)

        # Get the prediction
        y_val_pred = model.predict(X_validation_projected)
        y_train_pred = model.predict(X_train_projected)

        # Computation of the accuracy
        test_score_accuracy.append(accuracy_score(y_validation, y_val_pred))
        train_score_accuracy.append(accuracy_score(y_train_cv, y_train_pred))

    # Get the best score
    best_index = np.argsort(-np.array(test_score_accuracy))[0]
    best_p_value = p_values[best_index]
    best_test_score_pca = test_score_accuracy[best_index]
    best_train_score_pca = train_score_accuracy[best_index]

    if verbose: 
        print(f"Best value of p - PCA : {best_p_value}")
        print(f"Train Score for the best p: {best_train_score_pca}")
        print(f"Test Score for the best p: {best_test_score_pca}")
        
        # Plot the validation curve
        plt.figure(figsize = [9,6])
        plt.plot(p_values, test_score_accuracy, 'o', markersize=4, ls='--', label="Test")
        plt.plot(p_values, train_score_accuracy, 'o', markersize=4, ls='--', label="Train")
        plt.axvline(x = best_p_value, color = 'red', linestyle = '-', label = f'Best p = {best_p_value}', alpha = 0.45)
        plt.title("Evolution of the accuracy according to the value p of Principal Components for PCA")
        plt.xlabel("Number of Principal Components")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid()

    return best_p_value, best_test_score_pca
