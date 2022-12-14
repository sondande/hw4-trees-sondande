
import pandas as pd
import sys
import numpy as np
import math
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype

def data_preprocessing(dataset):
    # Determine whether a column contains numerical or nominial values
    # Create a new Pandas dataframe to maintain order of columns when doing One-Hot Coding on Nominial values
    new_dataframe = pd.DataFrame()
    # Iterate through all the columns of the training_set
    for x in dataset.columns:
        # Determine if the column 'x' in training set is a Nominial Data or Numerical
        if is_string_dtype(dataset[x]) and not is_numeric_dtype(dataset[x]):
            # Apply One-Hot Encoding onto Pandas Series at column 'x'
            dummies = pd.get_dummies(dataset[x], prefix=x, prefix_sep='.', drop_first=True)
            # Combine the One-Hot Encoding Dataframe to our new dataframe to the new_dataframe
            new_dataframe = pd.concat([new_dataframe, dummies],axis=1)
        else:
            # If the column 'x' is a Numerical Data, then just add it to the new_dataframe
            new_dataframe = pd.concat([new_dataframe, dataset[x]],axis=1)
    return new_dataframe

"""
Function to run an input dataset on Decision Tree Classifier
"""
def run_decision_tree_classifier(X_train, X_test, y_train, y_test):

    # Create a decision tree classifier
    clf = tree.DecisionTreeClassifier(criterion='gini',random_state=randomSeed)

    # Train the classifier
    clf = clf.fit(X_train, y_train)

    # Predict the labels of the testing set
    y_pred = clf.predict(X_test)

    # Calculate the accuracy of the classifier
    accuracy = accuracy_score(y_test, y_pred)

    # Print the confusion matrix of the classifier
    cm = confusion_matrix(y_test, y_pred)

    # Make confusion matrix pandas dataframe
    cm_df = pd.DataFrame(cm)

    # Calculate the recall of the classifier
    recall = list(recall_score(y_test, y_pred, labels=label_list, average=None))

    return cm_df, accuracy, recall

"""
Calculate confidence interval for a given accuracy
"""
def calculate_confidence_interval(accuracy, test_set_size, number_of_comparisons):
    # Assign Z value using Bonferroni Correction
    if number_of_comparisons < 3:
        z_value = 1.96
    elif number_of_comparisons == 3:
        z_value = 2.24
    else:
        z_value = 2.39
    # Calculate the confidence interval
    internal = (z_value * math.sqrt((accuracy * (1-accuracy))/(test_set_size)))
    confidence_interval_array =  [accuracy - internal, accuracy + internal]
    return confidence_interval_array

"""
Main part of the program
"""
try:
    # a.The path to a file containing a data set (e.g., monks1.csv)
    file_path = sys.argv[1]

    # b. The percentage of instances to use for a training set
    training_set_percent = float(sys.argv[2])

    # Ensure training set percent is a valid percent that can be used
    if 0 >= training_set_percent or training_set_percent > 1:
        print("Invalid percent. Please choose a value between 0 and 1.\n Input can not be 0 or 1 as well")
        exit(1)

    # Store the size of the testing set
    testing_set_percent = (1 - training_set_percent)

    # e. A random seed as an integer
    randomSeed = int(sys.argv[3])

    # f. The threshold to use for deciding if the predicted label is a 0 or 1
    handle_numeric = str(sys.argv[4])

    # Print all input values given for user to see
    print(f"Inputs:\nFile: {file_path}\nTraining set percent: {training_set_percent}")
    print(f"Random Seed: {randomSeed}\nHandle numeric as numeric: {handle_numeric}")

    # Run decision tree classifier for each dataset in the list
    # Read dataset into a pandas dataframe
    df = pd.read_csv(file_path)
        # Get the target column
    label_col = df.iloc[:,:1]

    # Get the features
    features = df.iloc[:,1:]

    # Preprocess the dataset with one hot encoding
    cleaned_features = data_preprocessing(features)

    # Combine the target column with the features
    df = pd.concat([label_col, cleaned_features], axis=1)

    # Shuffle the dataset and split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(cleaned_features, label_col, test_size=testing_set_percent, random_state=randomSeed)

    # Flatten the target columns into a 1D array. Prevents a DataConversionWarning warning from being thrown
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()

    # Get the unique labels in the target column
    label_list = np.unique(y_test)

# Decision Tree Classifier
    # Run decision tree classifier
    cm_df, accuracy, recall = run_decision_tree_classifier(X_train, X_test, y_train, y_test)
    print(f"accuracy:{accuracy} and recall:{recall}")
    confidence_interval = calculate_confidence_interval(accuracy, len(y_test), 4)
    print(f"confidence interval:{confidence_interval}")


except IndexError as e:
    print(f"Error. Message below:\n{e}\nPlease correct and try again.")
    exit(1)
except ValueError as e:
    print(f"Error. Message below:\n{e}\nPlease correct and try again.")
    exit(1)
