# Good or Bad Tennis day predictor - Data Science Project

## Overview
This project predicts whether the weather is fit for the users preferences if it was a good or bad day to play tennis. I utilize Sklearn[2] and panda libraries [3] to do that. I created this project, to learn more about the data science world in a more hands on approach.

## Background 
Originally I found a Data Science project through a github repo, which directed me towards this video [1].
The video covers utilizing a Decision Tree using the Sklearn/tree library [2] , to predict if a person is Male or Female, he inputed the Height, width, and Shoe size of various people into a list of lists. Which he then indicated that position 1...n is either Male or Female by using 'Fit(array x, array y)'. Now once a new set of data will be inputed, it will try to determine whether these measurements are M or F. Which will be done by the decision tree. It goes about that by implementing "The Gini impurity test" through 'DecisionTreeClassifier()'. By finding the best outcome of questions to give us the least impurity score. So in this case it will test between height, weight, and shoe size to determine what attribute should be questioned at the root of the tree, which will be the one with the lowest impurity score. 

## My project

With my own way of approaching this concept, I wanted a prediction model that not only learns from a fixed dataset, but keeps improving as a specific user interacts with it. The program has two modes: you can ask it for a prediction (which you'll be giving it today's outlook, temperature, humidity, and windiness), or you can input labeled data directly. Either way, once you tell it whether it was actually a good or bad day, that new row gets added to the dataset and the decision tree retrains itself right away, improving.

I used the pandas library to structure all of this data into a table, which made it much easier to keep the raw inputs, the encoded features, and the labels organized and consistent. I also added a step to save the growing dataset to a CSV file, so the model doesn't forget what it's learned between runs, after each time you start the program, it picks up right where it left off.

I created a test file, with the support of Claude, to check that the encoding, training, and predictions were all behaving as expected.

## What I learnt

Working through this project gave me a good grasp of how a decision tree actually works. it's a surprisingly linear and straightforward idea: at each step, the tree picks whichever question splits the data most cleanly, using Gini impurity to measure "how mixed/ how impure" the labels are after each possible split.

Beyond the algorithm itself, this was a good refresher on the rhythm of writing code.


## Sources:
[1] https://www.youtube.com/watch?v=T5pRlIbr6gg
[2] https://scikit-learn.org/stable/modules/tree.html
[3] https://pandas.pydata.org/docs/reference/frame.html
