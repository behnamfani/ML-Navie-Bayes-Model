# ML-Navie-Bayes-Model
A Naive Bayes model for predicting the class of IMDB's users' comments. 
This model tells us if a given comment is a positive one or a negative comment. For the learning process, I gave this model 25000 comments (each class has 12500 instances), and base on the probabilities of words observation in each class, it calculates two probabilities. One is the comment belongs to the negative class, and the other one is the comment belongs to the positive class. The higher probability is the predicted class.
Large_movie_review_dataset has train and test folders. Each folder contains the pos and neg comments. First, we train the model with data in the train set. Then we can see how it performs on the test set.
The comments in the .py file will help you to have a better intuition of how it works.
Thank you for your attention!
