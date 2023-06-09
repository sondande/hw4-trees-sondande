"""
a. The path to a file containing a data set (e.g., opticalDigit.csv)
b. The percentage of instances to use for a training set
c. A random seed as an integer
d. Either True or False indicating whether we should handle numeric attributes as
numeric (if False, then we treat them as categorical values)
"""

# TODO: I tihnk the problem is how the tree is doing the binary splitting and then the creation of the trees
# Could also be the part of the find the threshold to use accordingly
import sys
import numpy as np
import pandas as pd
import csv
from copy import copy
from collections import Counter
import math

class Node:
    def __init__(self, data):
        self.data = data
        self.children = {}
        # Only is 1 if it is a leaf.
        self.leaf = 0
        # Contains a threshold if the dataset being handled is numerical
        self.threshold = None

    def __str__(self, level = 0):
        ret = "\t" * level + repr(self.data) + "=> Threshold: " + repr(self.threshold) + "\n"
        for child in self.children:
            ret += "\t" * (level + 1) + "Branch: " + child + "\n" + self.children[child].__str__(level + 1)
        return ret

"""
Calculate total entropy of the whole dataset
"""
def totalEntropy(data_set, output_labels):
    # num_rows = data_set.shape[0]
    entropy = 0
    total_entropy = 0
    # for every label in possible unique labels, i.e. yes/no for play tennis (the decision)
    for label in output_labels:
        count_l = 0
        # for every instance using num_rows as the number of instances
        for instance in data_set:
            # if the label of the instance matches the possible label we're looking at
            if instance[0] == label:
                # increment count by 1
                count_l += 1
        # if there are occurrences of this label
        if count_l != 0:
            # calculate the probability of it occurring: number of occurance / size of input set
            probability = count_l / len(data_set)
            # get entropy of the label
            entropy = (probability * np.log2(probability))
        #print(f"count of category:{count_l} and category is{label}")
        #print("class entropy is: ", class_entropy)
        # Add to the total entropy
        total_entropy += entropy
        #print("total entropy is: ", total_entropy)
    if total_entropy == 0:
        return total_entropy
    return -1 * total_entropy

"""
Create a Sv that is using an attribute of the dataset as the root node and partitioning through the dataset for values that. Used for information gain 
"""
def createSv(value_list, training_set, attribute, attribute_list):
    # create a dictionary of the subsets for S with it's attributes
    # Ex: if the attribute is Outlook, create Ssunny, Srainy, Swindy with attributes and labels
    Sv = {}
    if attribute in attribute_list:
        attribute_index_in_training = attributes.index(attribute)
    else:
        # If the attribute doesn't exist, return an empty dictionary to maintain data structure format
        return {}
    for value in value_list:
        listofvalues = []
        # print("value:",value)
        for instance in training_set:
            # print("instance:",instance)
            if value == instance[attribute_index_in_training + 1]:
                # print("listInstance:",instance)
                listofvalues.append(instance)
        Sv[value] = listofvalues
    return Sv

def createSv2(S, attribute, value):
    Sv = []
    ind = attributes.index(attribute)
    for instance in S:
        if value in instance[ind + 1]:
            Sv.append(instance)
    return Sv

# calculate information gain
def informationGain(data_set, attribute, list_A):
    num_instances = len(data_set)
    att_info = 0.0
    # get unique values of a specific attribute from the dictionary, this is important Sv
    value_list = set()
    for key in myDict[attribute]:           #create a value_list for possible labels of attribute, rainy,sunny,windy for Outlook
        value_list.add(key)
    value_list = list(value_list)
    # Partition attribute into its labels
    Sv = createSv(value_list, data_set, attribute, list_A)
    # for every label in attribute:
    for key, val in Sv.items():
        # Calculate the total entropy for the Sv value set
        entropy_of_att = totalEntropy(val, uniqueLabels)
        # get the probability amount by the length of that Sv against the total number of instances in S
        probability = len(val) / num_instances
        # Get the first part of that instance to add with all the rest. The summation of all the entropy* prob
        att_info += (probability * entropy_of_att)
    # Calculate Information Gain through the total entropy of the whole dataset against the summation of breaking that down
    new_entropy = totalEntropy(data_set, uniqueLabels) - att_info
    return new_entropy                                                            # return this value

"""
Find Thresholds from an input dataset
"""
def threshold_find(data_set, attribute):
    data_set = np.array(data_set)
    # Create list of Thresholds
    T = []
    # Get the index of the attribute we want in the numpy dataset
    ind = attributes.index(attribute) + 1
    # Sort the input dataset by the attribute given as the one we want to sort
    sorted_data_set = data_set[data_set[:, ind].argsort()]
    # Get the first label from sorted dataset
    last_label = sorted_data_set[0]
    # iterate through all the instances in the datset and calculate the thresholds
    for instance in sorted_data_set[1:]:
        instance_label = instance[0]
        if last_label[0] != instance_label:
            new_value = (float(instance[ind]) + float(last_label[ind])) / 2
            T.append(new_value)
        last_label = instance
    return set(T)

"""
Handling Continous Attributes
"""
def infoGain_con_att(data_set, attribute, list_A):
    ind = attributes.index(attribute) + 1
    T_list = threshold_find(data_set, attribute)
    dict_threshold = {}
    for threshold in T_list:
        left_side_dataset = []
        right_side_dataset = []
        for instance in data_set:
            if instance[ind] <= threshold:
                left_side_dataset.append(instance)
            else:
                right_side_dataset.append(instance)
        prob_left = len(left_side_dataset) / len(data_set)
        prob_right = len(right_side_dataset) / len(data_set)
        gain_result = 1 - (prob_left * totalEntropy(left_side_dataset, uniqueLabels) + prob_right * totalEntropy(right_side_dataset, uniqueLabels))
        if threshold in dict_threshold.keys():
            if dict_threshold[threshold] < gain_result:
                dict_threshold[threshold] = gain_result
        else:
            dict_threshold[threshold] = gain_result
    best_threshold_result = max(dict_threshold, key=dict_threshold.get)
    return best_threshold_result, dict_threshold[best_threshold_result]

# find threshold and info gain for numeric attributes
# TODO Fix this method to work
def infoGainNumeric(data_set, attribute, list_A):
    data = copy(data_set)
    att_list = copy(list_A)
    att_info = 0.0
    info_gain_max = -1
    best_attribute = None
    best_threshold = 0
    for attribute in att_list:
        sortedIndex = data[attribute].sort_values(ascending=True).index()
        sortedData = data[attribute][sortedIndex]
        for i in range(0, len(sortedData) - 1):
            if sortedData[i] != sortedData[i + 1]:
                threshold = (sortedData[i] + sortedData[i + 1]) / 2
                att_info = informationGain(data, sortedData[i], att_list)
            if att_info > info_gain_max:
                info_gain_max = att_info
                best_threshold = threshold
                best_attribute = sortedData[i]
    return best_attribute, best_threshold


"""
Function used to check an instance of the training set and tell you either the most common value and if it is the only only label in that instance of the training set
"""
def labels_check(S):
    count = {}
    for inst in S:
        if inst[0] in count:
            count[inst[0]] += 1
        else:
            count[inst[0]] = 1
    max_value = max(count, key=count.get)
    if len(count.keys()) == 1:
        return True, max_value
    else:
        return False, max_value

"""
Inputs: 
    A = attributes not appearing earlier in tree
    S = subset of training instances to learn from 
Output: 
    tree node N classifying S
"""
def ID3(A, S):
    # if A is empty then
    if len(A) == 0:
        # N <- leaf node with most common label y in S
        # TODO implement with leaf node
        most_common_label = labels_check(S)[1]
        N = Node(most_common_label)
        N.leaf = 1
    # else if all instances in S have the same label y then
    elif labels_check(S)[0]:
        # N <- leaf node with label y
        N = Node(labels_check(S)[1])
        N.leaf = 1
    # else
    else:
        # Iterate through all the attributes which attribute in A produces the best gain
        a_star_dict = {}
        a_star_dict_thres = {}
        for att in A:
            if handle_numeric:
                best_thres, gain  = infoGain_con_att(S, att, A)
                a_star_dict[att]= gain
                a_star_dict_thres[att]= best_thres
            else:
                a_star_dict[att] = informationGain(S, att, A)
        # a* <- argmax_(a in A) in A Gain(S, a)
        if handle_numeric:
            a_star = max(a_star_dict, key=a_star_dict.get)
            a_star_thres = a_star_dict_thres[a_star]
        else:
            a_star = max(a_star_dict, key=a_star_dict.get)
        # N <- non-leaf node with attribute a*
        N = Node(a_star)
        # Store the threshold if we are handling numeric
        if handle_numeric:
            N.threshold = a_star_thres
        if not handle_numeric:
            # for each possible value v of a* do
            for possible_v in myDict[a_star]:
                S_v = createSv2(S, a_star, possible_v)
                # if S_v is empty then
                if len(S_v) == 0:
                    # N.child_v <- leaf node with most common label y in S
                    N.children[possible_v] = Node(labels_check(S)[1])
                    N.children[possible_v].leaf = 1
                # else
                else:
                    # N.child_v <- ID3(A \ {a*}, S_v)
                    newlist = [attribute for attribute in A if attribute != a_star]
                    #N.children[possible_v] = newlist
                    #print(f"test:{N.children[possible_v]}")
                    # For all the children of the current node, recursively call ID3 with a new list of attributes not
                    # including a* and using the subsection of the data as the input data set
                    N.children[possible_v] = ID3(newlist, S_v)
        # Handling numerical part
        else:
            # Create S_v_left and S_v_right using threshold
            S_v_left = []
            S_v_right = []
            ind = attributes.index(a_star)
            for instance in S:
                # See if the instance amount is less than or equal to the threshold
                if instance[ind + 1] <= a_star_thres:
                    S_v_left.append(instance)
                else:
                    S_v_right.append(instance)
            # Left side creation
            if len(S_v_left) == 0:
                # N.child_v <- leaf node with most common label y in S
                N.children[a_star] = Node(labels_check(S)[1])
                N.children[a_star].leaf = 1
            else:
                # create children for the left side of the tree
                N.children["left"] = ID3(A, S_v_left)

            if len(S_v_right) == 0:
                # N.child_v <- leaf node with most common label y in S
                N.children[a_star] = Node(labels_check(S)[1])
                N.children[a_star].leaf = 1
            else:
                # TODO We don't remove the attribute for numeric
                # create children for the right side of the tree
                N.children["right"] = ID3(A, S_v_right)
    return N


"""
Print a tree method to help with visualization
"""
def printTree(result):
    if result.leaf == 1:
        print(f"Leaf Value: {result.data}")
    else:
        print(f"Node Value: {result.data} -> Branches: {list(result.children.keys())}", sep=" ")
        for values in result.children.keys():
            print(values, "- ", end="")
            printTree(result.children[values])

"""
Prediction Function
"""
def predict(result, test_set):
    # Key is the top
    results_dict = {x: [] for x in labels}
    current = result
    for instance in test_set:
        while current.leaf != 1:
            index_att = attributes.index(current.data) + 1
            instance_value = instance[index_att]
            current = current.children[instance_value]
        predicted_value = current.data
        results_dict[instance[0]].append(predicted_value)
        current = result

    # Write name of the results file
    outFileName = f"results-tree-{file_path}-{handle_numeric}-{randomSeed}.csv"

    # Writing Section
    outputFile = open(outFileName, 'w')
    writer = csv.writer(outputFile)

    # Write Labels Row
    writer.writerow(uniqueLabels)

    # Create a confusion matrix result
    confusion_m_result = []

    # Write Columns Row
    for label in uniqueLabels:
        confusion_row = []
        labelCounter = Counter(results_dict[label])
        for compareLabel in uniqueLabels:
            confusion_row.append(labelCounter[compareLabel])
        confusion_row.append(label)
        confusion_m_result.append(confusion_row)
        writer.writerow(confusion_row)
    outputFile.close()
    return confusion_m_result

def predict_numeric(result, test_set):
    # Key is the top
    results_dict = {x: [] for x in labels}
    current = result
    for instance in test_set:
        while current.leaf != 1:
            index_att = attributes.index(current.data) + 1
            if instance[index_att] <= current.threshold:
                current = current.children["left"]
            else:
                current = current.children["right"]
        predicted_value = current.data
        results_dict[instance[0]].append(predicted_value)
        current = result

    # Write name of the results file
    outFileName = f"results-tree-{file_path}-{handle_numeric}-{randomSeed}.csv"

    # Writing Section
    outputFile = open("DecisionTree-results/" + outFileName, 'w')
    writer = csv.writer(outputFile)

    # Write Labels Row
    writer.writerow(uniqueLabels)

    # Create a confusion matrix result
    confusion_m_result = []

    # Write Columns Row
    for label in uniqueLabels:
        confusion_row = []
        labelCounter = Counter(results_dict[label])
        for compareLabel in uniqueLabels:
            confusion_row.append(labelCounter[compareLabel])
        confusion_row.append(label)
        confusion_m_result.append(confusion_row)
        writer.writerow(confusion_row)
    outputFile.close()
    return confusion_m_result

# print out all the stats from the confusion matrix such as accuracy, recall for each label, and their confidence interval
def calculateStats(matrix):
    # calculate accuracy
    sum_diagnol = 0
    sum_of_cells = 0
    for x in range(len(matrix)):
        for y in range(len(matrix)):
            value = matrix[x][y]
            if x == y:
                sum_diagnol += matrix[x][y]
            sum_of_cells += matrix[x][y]

    accuracy = sum_diagnol / sum_of_cells
    print(f"Accuracy: {accuracy}")
    # calculate recall
    for recall_x in range(len(matrix)):
        sum_of_row_y = 0
        for recall_y in range(len(matrix)):
            if recall_y == recall_x:
                cellyy = matrix[recall_x][recall_y]
            sum_of_row_y +=  matrix[recall_y][recall_x]
        # Error catching for zero division errr
        if sum_of_row_y != 0:
            recall = cellyy / sum_of_row_y
        else:
            recall = 0
        print(f"Recall for {matrix[recall_x][-1]}: {recall}")

    print()
    # Calculate the confidence interval
    confidence_interval_positive = accuracy + (1.96 * math.sqrt((accuracy * (1-accuracy))/(len(testing_set))))
    confidence_interval_negative = accuracy - (1.96 * math.sqrt((accuracy * (1-accuracy))/(len(testing_set))))
    print(f"Confidence Interval: [{confidence_interval_negative}, {confidence_interval_positive}]")

# Beginning of code
try:
    # Get Dataset File

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

    # Read in dataset
    df = pd.read_csv(file_path)

    # Get Unique labels for prediction function and confusion matrix
    labels = df[df.columns[0]].unique()
    if handle_numeric == "True":
        handle_numeric = True
    elif handle_numeric == "False":
        handle_numeric = False
    else:
        print("Invalid Boolean. Please enter 'True' or 'False' on how you'd like to evaluate numerical values")
        exit(1)

    if not handle_numeric:
        df = df.astype(str)
    else:
        for col in df.columns[1:]:
            df[col] = pd.to_numeric(df[col], errors="ignore")

    print(df.info())
    attributes = list((df.columns[1:]).to_numpy())      #get the features aside from the label in column[0]
    print(f"features of the dataset:{attributes}")            #this is important in calculating class entropy

    # create a dictionary with attribute names as keys and associated data as values
    myDict = df.to_dict('list')

    # shuffle the dataframe. Use random seed from input and fraction 1 as we want the whole dataframe
    shuffled_df = df.sample(frac=1, random_state=randomSeed)

    print(f"Number of Instances in Dataframe: {len(df)}")

    splits_indices = [int(training_set_percent * len(df))]
    print(f"Splits indexes they begin at: {splits_indices}\n")
    training_set, testing_set = np.split(shuffled_df, splits_indices)

    # Print out the lengths of the training, validation, and testing sets
    print(f"Length of training: {len(training_set)}")
    print(f"Length of testing: {len(testing_set)}\n")

    # Set sets to numpy arrays
    training_set = training_set.to_numpy()
    testing_set = testing_set.to_numpy()

    labels = []
    for insta_count in range(len(training_set)):
        labels.append(training_set[insta_count][0])
    uniqueLabels = np.unique(labels)
    print(uniqueLabels)

    # Cast each list of values in the dictionary as a set so that we only have unique values
    # if handle_numeric:
    #     input_list = list(myDict.keys())[1:]
    #     for value in input_list:
    #         gain = infoGain_con_att(training_set, value, list(attributes))
    #         myDict[value] = [gain]
    # else:
    for value in myDict.keys():
        myDict[value] = set(myDict[value])

    if handle_numeric:
        # store the best thresholds for each attribute if we are handling numberic
        myThresholds ={}
        for att in attributes:
            myThresholds[att] = infoGain_con_att(training_set, att, list(attributes))

    result = ID3(list(attributes), training_set)

    print("attempt 2\n")
    print(str(result))
    if handle_numeric:
        matrix = predict_numeric(result,testing_set)
        calculateStats(matrix)
    else:
        matrix = predict(result, testing_set)
        calculateStats(matrix)
except IndexError as e:
    print(f"Error. Message below:\n{e}\nPlease try again.")
    exit(1)
except ValueError as e:
    print(f"Error. Message below:\n{e}\nPlease try again.")
    exit(1)
except FileNotFoundError as e:
    print(f"Error. Message below:\n{e}\nPlease try again.")
    exit(1)
