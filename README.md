# Male or Female predictor - Data Science Project

I found a Data Science project through a github repo, which directed me towards this video [1]. I'm aiming to first learn through project based work, to get a better grasp about Data Science. 
#
The video covers utilizing a Decision Tree using the Sklearn/tree library [2] , to predict if a person is Male or Female, he inputed the Height, width, and Shoe size of various people into a list of lists. Which he then indicated that position 1...n is either Male or Female by using 'Fit(array x, array y)'. Now once a new set of data will be inputed, it will try to determine whether these measurements are M or F. Which will be done by the decision tree. It goes about that by implementing "The Gini impurity test" through 'DecisionTreeClassifier()'. By finding the best outcome of questions to give us the least impurity score. So in this case it will test between height, weight, and shoe size to determine what attribute should be questioned at the root of the tree, which will be the one with the lowest impurity score. 
#


#Sources:
[1] https://www.youtube.com/watch?v=T5pRlIbr6gg
[2] https://scikit-learn.org/stable/modules/tree.html
