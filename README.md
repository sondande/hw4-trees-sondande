[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-c66648af7eb3fe8bc4f294546bfd86ef473780cde1dea487d3c4ff354943c9ae.svg)](https://classroom.github.com/online_ide?assignment_repo_id=9173666&assignment_repo_type=AssignmentRepo)
# hw4-trees
HW4: Decision Trees

For this assignment, we will learn from four pre-defined data sets:

1.	monks1.csv: A data set describing two classes of robots using all nominal attributes and a binary label.  This data set has a simple rule set for determining the label: if head_shape = body_shape OR jacket_color = red, then yes, else no. Each of the attributes in the monks1 data set are nominal.  Monks1 was one of the first machine learning challenge problems (http://www.mli.gmu.edu/papers/91-95/91-28.pdf).  This data set comes from the UCI Machine Learning Repository: http://archive.ics.uci.edu/ml/datasets/MONK%27s+Problems

2.	penguins.csv: A data set describing observed measurements of different animals belonging to three species of penguins.  The four attributes are each continuous measurements, and the label is the species of penguin.  Special thanks and credit to Professor Allison Horst at the University of California Santa Barbara for making this data set public: see this Twitter post and thread with more information (https://twitter.com/allison_horst/status/1270046399418138625) and GitHub repository (https://github.com/allisonhorst/palmerpenguins).

3.	occupancy.csv: A data set of measurements describing a room in a building for a Smart Home application.  The task in this data set is to predict whether or not the room is occupied by people.  Each of the five attributes are continuous measurements.  The label is 0 if the room is unoccupied, and a 1 if it is occupied by a person.  This data set comes the UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/Occupancy+Detection+

4.	opticalDigit.csv: A data set of optical character recognition of numeric digits from processed pixel data.  Each instance represents a different 32x32 pixel image of a handwritten numeric digit (from 0 through 9).  Unlike MNIST from Homework 1, each image was preprocessed into a smaller number of attributes.  Each image was partitioned into 64 4x4 pixel segments and the number of pixels with non-background color were counted in each segment.  These 64 counts (ranging from 0-16) are the 64 attributes in the data set, and the label is the number from 0-9 that is represented by the image.  This data set is more complex than the Monks1 data set, but still contains only nominal attributes and a nominal label.  This data set comes from the UCI Machine Learning Repository: http://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits

## 1) Ada Ates & Sagana Ondande
## 2) Research Questions

** tree2.py is the main program file to run and readme questions were answered with results from that file.
1) Pick a single random seed and a single training set percentage (document both in your
README) and run your program on each of the four data sets. You should pass in True
as the final parameter to your program to treat all numeric attributes as numeric.
a. What is the accuracy you observed on each data set?
b. Calculate a 95% confidence interval for the accuracy on each data set.

Training set percentage = 0.8, random seed = 12345

monks1.csv~
- accuracy: n/a
- confidence interval: n/a

penguins.csv~
- accuracy: 0.9420289855072463
- confidence interval: [0.8868886874614639, 0.9971692835530288]

occupancy.csv~
- accuracy: 0.9907587548638133
- confidence interval: [0.9878340723872253, 0.9936834373404012]

opticalDigit.csv~
- accuracy: 0.8959074733096085
- confidence interval: [0.8780543472853466, 0.9137605993338704]

2) Create an image of the tree that your program learned in Question 1 for the monks1.csv
data set (you can draw by hand and scan your image into a PDF, or you can use a
drawing program to create an image file). Make sure to upload your image to GitHub.

The drawing of our tree for monks1.csv is uploaded as a PDF file and can be found on GitHub. Here is the tree print of our program:

	'jacket_color'=> Threshold: None
        Branch: blue
        'body_shape'=> Threshold: None
            Branch: octagon
            'head_shape'=> Threshold: None
                Branch: octagon
                'yes'=> Threshold: None
                Branch: round
                'no'=> Threshold: None
                Branch: square
                'no'=> Threshold: None
            Branch: round
            'head_shape'=> Threshold: None
                Branch: octagon
                'no'=> Threshold: None
                Branch: round
                'yes'=> Threshold: None
                Branch: square
                'no'=> Threshold: None
            Branch: square
            'head_shape'=> Threshold: None
                Branch: octagon
                'no'=> Threshold: None
                Branch: round
                'no'=> Threshold: None
                Branch: square
                'yes'=> Threshold: None
        Branch: red
        'yes'=> Threshold: None
        Branch: yellow
        'holding'=> Threshold: None
            Branch: sword
            'body_shape'=> Threshold: None
                Branch: octagon
                'head_shape'=> Threshold: None
                    Branch: octagon
                    'yes'=> Threshold: None
                    Branch: round
                    'no'=> Threshold: None
                    Branch: square
                    'no'=> Threshold: None
                Branch: round
                'head_shape'=> Threshold: None
                    Branch: octagon
                    'no'=> Threshold: None
                    Branch: round
                    'yes'=> Threshold: None
                    Branch: square
                    'no'=> Threshold: None
                Branch: square
                'head_shape'=> Threshold: None
                    Branch: octagon
                    'no'=> Threshold: None
                    Branch: round
                    'no'=> Threshold: None
                    Branch: square
                    'yes'=> Threshold: None
            Branch: balloon
            'body_shape'=> Threshold: None
                Branch: octagon
                'head_shape'=> Threshold: None
                    Branch: octagon
                    'yes'=> Threshold: None
                    Branch: round
                    'no'=> Threshold: None
                    Branch: square
                    'no'=> Threshold: None
                Branch: round
                'head_shape'=> Threshold: None
                    Branch: octagon
                    'no'=> Threshold: None
                    Branch: round
                    'yes'=> Threshold: None
                    Branch: square
                    'no'=> Threshold: None
                Branch: square
                'head_shape'=> Threshold: None
                    Branch: octagon
                    'no'=> Threshold: None
                    Branch: round
                    'yes'=> Threshold: None
                    Branch: square
                    'yes'=> Threshold: None
            Branch: flag
            'head_shape'=> Threshold: None
                Branch: octagon
                'body_shape'=> Threshold: None
                    Branch: octagon
                    'yes'=> Threshold: None
                    Branch: round
                    'no'=> Threshold: None
                    Branch: square
                    'no'=> Threshold: None
                Branch: round
                'body_shape'=> Threshold: None
                    Branch: octagon
                    'no'=> Threshold: None
                    Branch: round
                    'yes'=> Threshold: None
                    Branch: square
                    'no'=> Threshold: None
                Branch: square
                'body_shape'=> Threshold: None
                    Branch: octagon
                    'no'=> Threshold: None
                    Branch: round
                    'no'=> Threshold: None
                    Branch: square
                    'yes'=> Threshold: None
        Branch: green
        'head_shape'=> Threshold: None
            Branch: octagon
            'body_shape'=> Threshold: None
                Branch: octagon
                'yes'=> Threshold: None
                Branch: round
                'no'=> Threshold: None
                Branch: square
                'no'=> Threshold: None
            Branch: round
            'body_shape'=> Threshold: None
                Branch: octagon
                'no'=> Threshold: None
                Branch: round
                'yes'=> Threshold: None
                Branch: square
                'no'=> Threshold: None
            Branch: square
            'body_shape'=> Threshold: None
                Branch: octagon
                'no'=> Threshold: None
                Branch: round
                'no'=> Threshold: None
                Branch: square
                'yes'=> Threshold: None

a. What are the rules learned by the algorithm?

Firstly, the rule learned by our algorithm is that the most important attribute is jacket_color as that is the root of the tree. Also, another set of rule is that if head_shape equals body_shape or jacket_color is red, then our algorithm returns the label yes. Any other scenario has returned the label no.

b. How do these rules compare to the true rules in the data set (described on page 1
of the assignment)?

Description of the monks1.csv dataset in the assignment states: "a simple rule set for determining the label: if head_shape = body_shape OR jacket_color = red, then yes, else no." This was consistent with the rules of our algorithm. Thus, our algorithm has done well in terms of comparison to true rules in the given data set.

3) Using the same seed and training set percentage from Q1, rerun your program on the
opticalDigit.csv data set and pass in False for the final parameter so that your algorithm
treats each attribute as categorical values (instead of numeric):

a. What is the accuracy you observed?

accuracy: 0.5160142348754448

b. Calculate a 95% confidence interval around that accuracy

confidence interval:[0.48679828228334027, 0.5452301874675494]

c. Compare the confidence intervals from your answer to Q1b and Q3b. What do
you observe? What does this imply?

The accuracy values are significantly different. This also means that confidence intervals do not fall within one another. When opticalDigits is run on categorical values, the accuracy is dropped to 0.51 from 0.89 when it was run on numerical values. This is a significant drop. This means that usual trees, ID3, do not work for this dataset. This implies that this dataset has continuous values that ID3 cannot handle. When it is run with numerical values parameter, i.e. C4.5 or CART unlike ID3, there is significant improvement in accuracy and confidence intervals. This shows that although ID3 fails to do well with continuous values, the improved models of C4.5 and CART fixes this problem with thresholds, creating binary classification via these thresholds, and therefore even classifying continuous attributes.

4) Choose 9 new seeds (document in your README). Rerun your program on
opticalDigit.csv using these 9 new seeds using both True and False as the final parameter
to the program.

Previous seed: 12345

Seeds chosen & accuracy of True and False respectively: 
- 50:  0.8745551601423488  &  0.5213523131672598
- 100: 0.8905693950177936   & 0.5249110320284698
- 250: 0.8879003558718861 & 0.5302491103202847
- 500: 0.8976868327402135 & 0.5088967971530249
- 750: 0.895017793594306 &  0.445729537366548
- 1000: 0.891459074733096 &  0.5284697508896797
- 2500: 0.8754448398576512 & 0.47419928825622776
- 5000: 0.9065836298932385 & 0.5409252669039146
- 10000: 0.9065836298932385 &  0.4581850533807829

a. Calculate the average accuracy across the 10 seeds when you treated the attributes
as (1) numeric and (2) categorical

Numeric average: 0.887

Categorical average: 0.498

b. Did you observe the same trends as in Q3c? That is, if one approach achieved a
statistically significantly higher accuracy in Q3c, did the same approach achieve a
higher accuracy when averaged over 10 seeds? If they were not statistically
significantly different in Q3c, are the averages very close?

The same trends as Q3c were observed, the numerical value solution performed better accuracies over 10 seeds, there is a statistical significant difference. The averages of 10 seeds between two approaches were not close to each other at all. Numeric average is 0.887 whereas categorical average is 0.498. Numeric handling produces significantly higher accuracies, which fits the implications and conclusion we came to in Q3c.

c. Did these averages fall in your confidence intervals calculated in Q1b and Q3b?

confidence interval: [0.8780543472853466, 0.9137605993338704] from Q1b

confidence interval: [0.48679828228334027, 0.5452301874675494] from Q3b

The lowest average overall was 0.445729537366548 with random seed 750 and categorical attributes. The second lowest value also comes from categorical attributes with 0.47419928825622776 when random seed was 2500. These two values were very low that they didn't fall into the confidence interval from Q3b. This wasn't the case for numerical attributes as over 10 seeds, the values produced did fall in between the confidence intervals. However, the averages did fall in our confidence intervals.

3) A short paragraph describing your experience during the assignment (what did you enjoy,
what was difficult, etc.)

Ada Ates~

I think trees in general are challenging to understand. I understand the concept, but I am still having trouble visualizing certain aspects. I truly had trouble with understanding how to code numerical attributes. Also, the values for opticalDigit.csv doesn't seem like what they should be, which confused me a bit more.

Sagana Ondande~

Trees are the hardest concept for me personally. I understand all the concepts in terms of how a decision tree makes decisions, how we utilize Gini index and everything we discussed in class. When it came down to implementation of trees, I personally struggled with how to create the recursive function specifically related to trees and that overall process. After doing this hw, I feel more confident in implementing trees in general but also the overall implementation of decision trees and how to handle numerical and nominal values.

4) An estimation of how much time you spent on the assignment, and

Too long, no one needs to keep track of that :(
For the sake of this question, we'll give an estimate of +25 hours

5) An affirmation that you adhered to the honor code 

We affirm that we have adhered to the Honor Code. Ada Ates & Sagana Ondande