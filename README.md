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

** Except for question 2, rest of the README questions were answered with the help of scklearn's implementation. For numeric attributes, we are able to build trees, however, we cannot get predictions currently.
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
- accuracy: 0.927536231884058
- confidence interval: [0.8529430292220861, 1.0021294345460299]

occupancy.csv~
- accuracy: 0.9888132295719845
- confidence interval: [0.9848932767067243, 0.9927331824372446]

opticalDigit.csv~
- accuracy: 0.8887900355871886
- confidence interval: [0.8663777480950227, 0.9112023230793546]

2) Create an image of the tree that your program learned in Question 1 for the monks1.csv
data set (you can draw by hand and scan your image into a PDF, or you can use a
drawing program to create an image file). Make sure to upload your image to GitHub.

The drawing of our tree for monks1.csv is uploaded as a PDF file and can be found on GitHub.

a. What are the rules learned by the algorithm?

Firstly, the rule learned by our algorithm is that the most important attribute is jacket_color as that is the root of the tree. Also, another set of rule is that if head_shape equals body_shape or jacket_color is red, then our algorithm returns the label yes. Any other scenario has returned the label no.

b. How do these rules compare to the true rules in the data set (described on page 1
of the assignment)?

Description of the monks1.csv dataset in the assignment states: "a simple rule set for determining the label: if head_shape = body_shape OR jacket_color = red, then yes, else no." This was consistent with the rules of our algorithm. Thus, our algorithm has done well in terms of comparison to true rules in the given data set.

3) Using the same seed and training set percentage from Q1, rerun your program on the
opticalDigit.csv data set and pass in False for the final parameter so that your algorithm
treats each attribute as categorical values (instead of numeric):

a. What is the accuracy you observed?

accuracy: 0.9039145907473309

b. Calculate a 95% confidence interval around that accuracy

confidence interval:[0.8829055044099341, 0.9249236770847278]

c. Compare the confidence intervals from your answer to Q1b and Q3b. What do
you observe? What does this imply?

The accuracy values are not significantly different. This also means that confidence intervals fall within one another. There is a slight change in the boundaries, however it is not significant enough to draw conclusions.

4) Choose 9 new seeds (document in your README). Rerun your program on
opticalDigit.csv using these 9 new seeds using both True and False as the final parameter
to the program.

Previous seed: 12345

Seeds chosen & accuracy of True and False respectively: 
- 50:  0.9110320284697508  & 0.9128113879003559
- 100: 0.8905693950177936   & 0.9065836298932385
- 250: 0.902135231316726 & 0.8843416370106761
- 500: 0.9074733096085409 & 0.9083629893238434
- 750: 0.905693950177936 &  0.905693950177936 
- 1000: 0.902135231316726 &  0.9039145907473309
- 2500: 0.8941281138790036 & 0.9039145907473309
- 5000: 0.9083629893238434 & 0.9119217081850534
- 10000: 0.8994661921708185 &  0.900355871886121

a. Calculate the average accuracy across the 10 seeds when you treated the attributes
as (1) numeric and (2) categorical

Numeric average: 0.9021

Categorical average: 0.9035

b. Did you observe the same trends as in Q3c? That is, if one approach achieved a
statistically significantly higher accuracy in Q3c, did the same approach achieve a
higher accuracy when averaged over 10 seeds? If they were not statistically
significantly different in Q3c, are the averages very close?

The same trends as Q3c were observed, there isn't any significant differences between numerical or categorical attributes when we vary the random seed. The averages are quite close to each other across 10 different random seeds. Although no significant difference, except for random seed 250, categorical attributes resulted in slightly better accuracy values, which can also be observed from 4a. 

c. Did these averages fall in your confidence intervals calculated in Q1b and Q3b?

confidence interval: [0.8663777480950227, 0.9112023230793546] from Q1b
confidence interval: [0.8829055044099341, 0.9249236770847278] from Q3b

The lowest average overall was 0.8843416370106761 with random seed 250 and categorical attributes. Both this lowest observed value and averages of numeric and categorical fall in the confidence intervals calculated in Q1b and Q3b.

3) A short paragraph describing your experience during the assignment (what did you enjoy,
what was difficult, etc.)

Ada Ates~

I think trees in general are challenging to understand. I understand the concept, but I am still having trouble visualizing certain aspects. I truly had trouble with understanding how to code numerical attributes. Also, the values for opticalDigit.csv doesn't seem like what they should be, which confused me a bit more.

Sagana Ondande~


4) An estimation of how much time you spent on the assignment, and

Too long, no one needs to keep track of that :(
For the sake of this question, we'll give an estimate of +25 hours

5) An affirmation that you adhered to the honor code 

We affirm that we have adhered to the Honor Code. Ada Ates & 