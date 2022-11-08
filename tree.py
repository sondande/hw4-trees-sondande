"""
a. The path to a file containing a data set (e.g., opticalDigit.csv)
b. The percentage of instances to use for a training set
c. A random seed as an integer
d. Either True or False indicating whether we should handle numeric attributes as
numeric (if False, then we treat them as categorical values)
"""
import sys
import numpy as np
import pandas as pd

# calculate total entropy of the whole dataset
def totalEntropy(training_set, uniqueLabels):
    #value, counts = np.unique(attribute, return_counts=True)
    #print(f"value: {value}, counts: {counts}")
    num_rows = training_set.shape[0]
    entropy = 0
    total_entropy = 0
    for label in uniqueLabels:                                  #for every label in possible unique labels, i.e. yes/no for play tennis (the decision)
        #print(label)
        count_l = 0
        for insta_count in range(num_rows):                         #for every instance
            if training_set[insta_count][0] == label:                   #if the label of the instance matches the possible label we're looking at
                count_l += 1                                                #increment count by 1
        if count_l != 0:                                            #if there are occurrences of this label
            probability = count_l / num_rows                            #calculate the probability of it occurring
            entropy = - (probability * np.log2(probability))                #get entropy of the label
        #print(f"count of category:{count_l} and category is{label}")
        #print("class entropy is: ", class_entropy)
        total_entropy += entropy                                    #add to the total entropy
        #print("total entropy is: ", total_entropy)

    return total_entropy

def classEntropy(count, totalrows):
    class_entropy = 0
    class_count = count
    totalrows = totalrows
    if class_count != 0:
        probability = class_count / totalrows
        class_entropy= - (probability * np.log2(probability))
    return class_entropy

# create Sv i.e, for attribute Outlook, create Ssunny, Srainy, Swindy with attributes and labels
# get the count of each Sv and store it for further use in entropy function that will be called within information gain
def createSv(value_list, training_set,attribute):
    Sv = {}
    #count = 0               # get the number of occurrences/count of that value
    for value in value_list:
        listofvalues = []
        print("value:",value)
        for key in myDict[attribute]:
            print("key:",key)
            if value == key:
                print("listvalue:",value)
                listofvalues.append(value)
                #do something lol my brain melted
        # listofvalues.append(value)
        Sv[value] = listofvalues
    for key, value in Sv.items():
        print(f"key:{key},values:{value}")
        #print(key, len([item for item in value if item]))
    return Sv

# calculate information gain
def informationGain(myDict, attribute, training_set, uniqueLabels):
    num_rows = training_set.shape[0]
    att_info = 0.0
    # get unique values of a specific attribute from the dictionary, this is important Sv
    value_list = set()
    for key in myDict[attribute]:           #create a value_list for possible labels of attribute, rainy,sunny,windy for Outlook
        #print(key)
        value_list.add(key)
        #for value in key:
         #   print(value)
    value_list = list(value_list)
    print(f"attribute: {attribute},unique values of attribute: {value_list}")
    Sv = createSv(value_list, training_set, attribute)              #partition Outlook into its labels
    for key, value in Sv.items():               # for every label in attribute:
        count = len([item for item in value if item])           #get the number of values of each class in Sv, i.e. Ssunny length
        print(f"count{count}")
        print(f"Sv[key]{Sv[key]}")
        shapeofSv = sum([len(Sv[x]) for x in Sv if isinstance(Sv[x], list)])            #get the number of instances/values in Sv => total rows of Outlook
        print(f"Sv shape 0: {shapeofSv}")
        att_entropy = classEntropy(Sv[key], count, shapeofSv)              # calculate class entropy for this attribute
        probability = count/num_rows                                         # probability = count/num_rows
        att_info += probability * att_entropy                                   # info gain += probability * entropy
    new_entropy = totalEntropy(training_set,uniqueLabels) - att_info                # call total entropy - info gain
    return new_entropy                                                            # return this value

# get feature max info

#id3
## label_list = training_set[0][:]
## labels_unique = training_set[0][:].unique #so that we can calculate entropy for each label
## label_count = length(label_list)

# evaluate

# Beginning of code
try:
    # Get Dataset File

    # a.The path to a file containing a data set (e.g., monks1.csv)
    file_path = sys.argv[1]

    # b. The percentage of instances to use for a training set
    training_set_percent = float(sys.argv[2])

    # Ensure training set percent is a valid percent that can be used
    if 0 >= training_set_percent or training_set_percent >= 1:
        print("Invalid percent. Please choose a value between 0 and 1.\n Input can not be 0 or 1 as well")
        exit(1)

    # Store the size of the testing set
    testing_set_percent = (1 - training_set_percent)

    # e. A random seed as an integer
    randomSeed = int(sys.argv[3])

    # f. The threshold to use for deciding if the predicted label is a 0 or 1
    handle_numeric = str(sys.argv[4])
    if handle_numeric == "True":
        handle_numeric = True
    elif handle_numeric == "False":
        handle_numeric = False
    else:
        print("Invalid Boolean. Please enter 'True' or 'False' on how you'd like to evaluate numerical values")
        exit(1)

    # Print all input values given for user to see
    print(f"Inputs:\nFile: {file_path}\nTraining set percent: {training_set_percent}")
    print(f"Random Seed: {randomSeed}\nHandle numeric as numeric: {handle_numeric}")

    # Read in dataset
    df = pd.read_csv(file_path)
    # labels = df.iloc[:, 0]

    attributes = (df.columns[1:]).to_numpy()          #get the features aside from the label in column[0]
    print(f"features of the dataset:{attributes}")            #this is important in calculating class entropy

    myDict = df.to_dict('list')           #create a dictionary with aatribute names as keys and associated data as values

    #for key,value in myDict.items():
        #print(f"key:{key},values:{value}")

    # shuffle the dataframe. Use random seed from input and fraction 1 as we want the whole dataframe
    shuffled_df = df.sample(frac=1, random_state=randomSeed)

    print(f"Number of Instances in Dataframe: {len(df)}")

    splits_indices = [int(training_set_percent * len(df))]
    print(f"Splits indexes they begin at: {splits_indices}\n")
    training_set, testing_set = np.split(shuffled_df, splits_indices)

    # Print out the lengths of the training, validation, and testing sets
    print(f"Length of training: {len(training_set)}")
    print(f"Length of testing: {len(testing_set)}\n")

    training_set = training_set.to_numpy()
    testing_set = testing_set.to_numpy()

    labels = []
    for insta_count in range(len(training_set)):
        labels.append(training_set[insta_count][0])
    uniqueLabels = np.unique(labels)
    print(uniqueLabels)

    print(f"attribute and values:{myDict[attributes[0]]}")
    totalEntropy(training_set, uniqueLabels)

    #for attribute in attributes:
    informationGain(myDict,attributes[0],training_set, uniqueLabels)


except IndexError as e:
    print(f"Error. Message below:\n{e}\nPlease try again.")
    exit(1)
except ValueError as e:
    print(f"Error. Message below:\n{e}\nPlease try again.")
    exit(1)
except FileNotFoundError as e:
    print(f"Error. Message below:\n{e}\nPlease try again.")
    exit(1)
