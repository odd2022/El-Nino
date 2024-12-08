import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LinearRegression, LogisticRegression
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

def split_scale_data(X, y, RANDOM_SPLIT=False, train_size=374, verbose=True):
    """
    X, y: Inputs and outputs
    RANDOM_SPLIT: Boolean, if False splits for train and test are done in in time order, if True, splits are random
    train_size: number of data in the train sample (test_size=474 - train_size)
    verbose: if True, the function prints information about the shape of the outputs data sets
    """
    # Split the data
    if RANDOM_SPLIT:
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=42)
    else:
        X_train, X_test, y_train, y_test = X[:train_size], X[train_size:], y[:train_size], y[train_size:]

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


########## MODEL EVALUATION #############

#Let's define a model for all the evaluations 
def evaluate_model(model, model_name, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    #let's copmute the scores
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    y_proba = model.predict_proba(X_test)[:, 1] #keep the proba of the class 1
    auc = roc_auc_score(y_test, y_proba)

    print(f"{model_name} - Accuracy: {accuracy}, AUC: {auc}")
    return model, accuracy, auc



#Let's define a model for all the cross-validations
def cross_validation(model, param_grid, X_train, y_train, X_test, y_test, model_name):

    #perform cross validation
    grid_search = GridSearchCV(model, param_grid, cv = 5, scoring = 'accuracy') 
    grid_search.fit(X_train, y_train)
    
    #find the model with the best parameters
    best_estimator = grid_search.best_estimator_ 
    best_params = grid_search.best_params_ #find the best parameters
    print(f"{model_name} - Best Parameters: {best_params}")

    #compute the scores
    y_pred =  best_estimator.predict(X_test) 
    best_accuracy = accuracy_score(y_test, y_pred) #compute the accuracy

    y_proba =  best_estimator.predict_proba(X_test)[:, 1] #compute the auc
    best_auc = roc_auc_score(y_test, y_proba)

    print(f"{model_name} - Tuned Accuracy: {best_accuracy}, Tuned AUC: {best_auc}")

    return best_estimator, best_params, best_accuracy, best_auc


#we define a function to plot histograms 
def plot_score(model_names, auc_scores, accuracy_scores):
    x = np.arange(len(model_names))
    fig, ax = plt.subplots(figsize = (12, 7))

    #plot the histograms of accuracy and auc 
    auc_bars = ax.bar(x - 0.175, auc_scores, 0.35, label = 'AUC', color = 'skyblue')
    accuracy_bars = ax.bar(x + 0.175, accuracy_scores, 0.35, label = 'Accuracy', color = 'lightcoral')

    #plot horizontal lines for the highest accuracy and auc
    max_auc = max(auc_scores)
    max_accuracy = max(accuracy_scores)
    ax.hlines(y = max_auc, xmin = -0.5, xmax = len(model_names)-0.5, colors = 'skyblue', linestyles = '--', linewidth = 1.5) 
    ax.hlines(y = max_accuracy, xmin = -0.5, xmax = len(model_names)-0.5, colors = 'lightcoral', linestyles = '--', linewidth = 1.5)  
    ax.text(len(model_names)-0.5, max_auc + 0.02, f'Best AUC: {max_auc:.3f}', color='skyblue', ha='right')
    ax.text(len(model_names)-0.5, max_accuracy + 0.02, f'Best Accuracy: {max_accuracy:.3f}', color='lightcoral', ha='right')

    ax.set_xlabel('Model')
    ax.set_ylabel('Scores')
    ax.set_ylim(0, 1)
    ax.set_title('AUC and Accuracy Scores of Different Models')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation = 45, ha = 'right')
    ax.set_xticklabels(model_names)
    ax.legend()

    #annotations for the value of each bar
    for bar in auc_bars + accuracy_bars:
        ax.annotate(f'{bar.get_height():.3f}', 
                    xy = (bar.get_x() + bar.get_width() / 2, bar.get_height()), 
                    xytext = (0, 3), 
                    textcoords = "offset points",
                    ha = 'center', va = 'bottom')

    plt.show()
