# Questions from `paper_1` :robot: 

**Q: Which of the following is NOT useful to combat overfitting?**
- Utilizing cross-validation.
- Employing regularization.
- Performing statistical significance tests.
- Penalizing classifiers with less structure.
* `[ D ]`


---

**Q: Which of the following is a Machine Learning best practice?**
- Testing on the training data
- Calculating the global optimum of the objective function
- Getting a low bias/high variance estimate
- Permorming cross-validation
* `[ D ]`


---

**Q: Which of the following is true about overfitting in a classification problem?**
- The number of features do not play a role in overfitting
- Overfitting can occur in the absence of noise
- Feature selection always help with reducing overfitting
-  Overfitting is solely caused by noise
* `[ B ]`


---

**Q: We have two classifiers, A and B.  Suppose we know that, for an infinite training set, A performs better than B. Therefore, it might be reasonable to think that**
- A and B are likely to perform the same for small training sets
- A is more complex than B and therefore it might perform better also for small training sets
- A is more complex than B and therefore it might perform worse than B for small training sets
- None of the above
* `[ C ]`


---

**Q: Which of the following classifiers generally has a low bias and a high variance?**
- Linear Bayes Normal Classifier
- K-nearest neighbour classifier with a high k
- Fisher's Least Square Linear Disciminant classifier
- A decision tree classifier
* `[ D ]`


---

**Q: Which of these statements is correct?**
- Bias is the tendency to learn random things irrespective of the real signal
- Variance is a learner's tendency to consistently learn the wrong thing
- Bias is a learner's tendency to consistently learn the wrong thing
- Variance is a measure that is used to quantify the difference between the estimated error and the true error
* `[ C ]`


---

**Q: Which of the following statements does not unequivocally hold?**
- There is no direct connection between the number of parameters a model has and its tendency to overfit.
- In general, feeding a Machine Learning system more data will yield better results than making its algorithm "cleverer".
- When a classifier has a training error of 0, one can stop the learning process.
- An increase in available features sometimes leads to an increase in performance.
* `[ C ]`


---

**Q: Assume you have several algorithms trying to predict a position in the plane. You notice your predictions have a high bias but low variance, meaning that**
- your predictions are quite close to each other and they are distributed around the wrong point
- your predictions are quite close to each other and they are distributed around the right point
- your predictions are spread apart and they are distributed around the wrong point
- your predictions are spread apart and they are distributed around the right point
* `[ A ]`


---

**Q: Which of the following statements is most true:**
- Machine Learning is a form of induction, not deduction.
- Overfitting is caused by noise.
- The simpler classifier will most likely have the lower test error.
- A function that can be represented, can also be learned.
* `[ A ]`


---

**Q: Which of these statements are true:
(1) Given finite time, memory and data, standard learners can not find a true function for classification.
(2) If a problem can be represented, the true classification function can always be found in a normal setting.**
- (1) and (2) are true
- (1) is true and (2) is false
- (1) is false and (2) is true
-  (1) and (2) are false 
* `[ D ]`


---

**Q: What are some challenges of machine learning?**
- Getting enough representative data
- Choosing a right learning algorithm
- Choosing the right input dimension size
- All of the above
* `[ D ]`


---

**Q: Overfitting is a common problem in machine learning, luckily there are different methods to detect this problem. which of the following options is NOT a method to detect overfitting?**
- Curse of dimensionality test
- adding regularization term to evaluation function
- Statistical significance test
- Cross validation
* `[ A ]`


---

**Q: In case you have a lot of data about the preconditions required by each class (eg. IF ..., THEN ...), the best representation to express this data is:**
- Graphical models
- Instances
- Hyperplanes
- Sets of rules
* `[ D ]`


---

**Q: Why does the combinatorial explosion of input variables not necessarily pose a problem for learners in the real world**
- Because in high dimensions, it is easy to approximate shapes by other shapes as a tool to derive learning boundaries.
- Because gathering more features never hurts
- The functions we want to learn are not drawn uniformly from all possible functions, and we can often make simplifying assumptions
- As the number of variables goes up, variance goes down
* `[ C ]`


---

**Q: In general, which of following would increase the amount of variance of your classifier?**
- Increasing the number of training data samples
- Increasing the regularization 
- Decreasing the number of features 
- None of the above
* `[ D ]`


---

**Q: In machine learning, which of the following is NOT helpful to combat overfitting?**
- Cross-validation
- Adding a regularization term
- Boosting
- Statistical significance test
* `[ C ]`


---

**Q: Which of the following statements is false:**
- In machine learning, representation is as important as evaluation and optimization.
- Generalizing beyond the training samples of the dataset implies reducing the bias.
- The “course of dimensionality” is partly counteracted by the “blessing of non-uniformity”.
- Fixed-Sized learners cannot take advantage of all the data.
* `[ B ]`


---

**Q: What is NOT a common mistake in machine learning?**
- Validating a classifier using data that has been used for training
- Assuming there is a classifier that works well on every dataset
- Picking a classifier with the best theoretical bounds
- Tuning a promising complex classifier by trial and error
* `[ D ]`


---

**Q: Which of the following statements is false?**
- Generalization beyond the training set is the fundamental goal of machine learning.
- Scalability for machine learning is limitited by training data, time and memory.
- Features that are irrelevant in isolation, will be irrelevant in any combination of features.
- Theoretical guarantees in machine learning are a source of understanding and a driving force for algorithm design.
* `[ C ]`


---

**Q: Which of the following is true?**
- When validating a classifier we should use all the data available to train in order to get the best fitted classifier.
- When generalising, we always have access to the function we want to optimise
- A more powerful learner is not necessarily better than a less powerful one
- More features means a better classifier in most since we are using more information about the subject
* `[ C ]`


---

**Q: What is not a main components of a machine learning algorithm?**
- Representation
- Optimization
- Parallelizability
- Evaluation
* `[ C ]`


---

**Q: Learning algorithms consist of a combination of three components. These components are: **
- Description, Representation, Evaluation
- Representation, Manipulation, Optimization
- Optimization, Representation, Evaluation
- Generalization, Evaluation, Optimization
* `[ C ]`


---

**Q: What is required to have in a deep learning network**
- Representation, Evaluation, Optimization
- Learnable, Evaluation, Optimization
- Representation, Causation, Optimization
- Representation, Evaluation, Correlation
* `[ A ]`


---

**Q: Which one of the following is not true for the reason of ML is still successful?**
- Real world functions are not drawn uniformly from all possible functions
- There are very general assumptions—like smoothness, similar examples having similar classes, limited dependences, or limited complexity—are often enough to do very well.
- Machine learning can get something from nothing.
- Induction >> deduction, requiring much less input knowledge to produce useful results
* `[ C ]`


---

**Q: Why is it important to learn on a separate training set and test on a separate test set in machine learning?**
- In order to avoid the curse of dimensionality;
- In order to avoid overfitting;
- In order to reduce training time;
- In order to reduce the number of parameters that have to be estimated.
* `[ B ]`


---

**Q: Why are multiple learners used in machine learnings?**
- To try many variations of the learner and select the best one
- To combine the results of each classifier trained with a resampled training set
- To increase the variance and reducing bias
- The question is wrong. They are not used at all
* `[ B ]`


---

**Q: In which case is a linear classifier expected to perform better than a complex one?**
- Small amount of data
- High dimensional features
- Linear separable classes
- Overlapping classes
* `[ A ]`


---

**Q: What is no reason for overfitting?**
- Having a lot of noise in the train and test data.
- Having too much parameters selected for your classifiers.
- Having  too few data available.
- All of these reasons are reasons for overfitting
* `[ A ]`


---

**Q: What is a "loss function"?**
- A loss function tells you how much data has gone lost in the neural network
- A loss function tells you which parameters of the neural network have to be updated in order to receive a better result
- A loss function describes how much the actual output from the neural network differs from the expected output
- A loss function tells you exactly which input data has been classified incorrectly
* `[ C ]`


---

**Q: Which of the following is FALSE ?**
- Overfitting can be combatted using cross-validation, regularisation term or statistical significance tests (chi-square).
- Generalization being the goal, recommendation is that, test data can be used to optimally tune parameters of the machine learning model.
- Dumb Algorithm with lots and lots of data beats a clever one with a modest amount of data. 
- Variances of learning algorithms against different dataset can be combatted using bagging, boosting and stacking.
* `[ B ]`


---

**Q: Which of the following answers is true?**
- A linear learner often has a high bias and a decision tree often has a high variance
- A linear learner often has a low bias and a decision tree often has a high bias
- A linear learner often has a high variance and a decision tree often has a low variance
- A linear learner often has a high variance and a decision tree often has a high bias
* `[ A ]`


---

**Q: What will NOT help to decrease the overfitting of the data?**
- cross-validation
- regularization term
- increase training data sample size
- add more rules to the classifier function
* `[ D ]`


---

**Q: Why is cross-validation usefull when training a ML-algorithm?**
- It prevents having to hold data aside which would reduce the amount of data available for training
- It prevents overfitting
- It can be used to speed-up the algorithm
- All of the above
* `[ A ]`


---

**Q: Which combinations of three components should you look at when using machine learning? **
- Representation, Data, Optimization
-  Representation, Evaluation, Optimization
- Generalization, Data, Optimization
- None of the above answers is correct.
* `[ B ]`


---

**Q: Which sentence best describes: "the blessing of non-uniformity"?**
- Samples are not uniformly distributed in space, but centered at lower dimensional manifolds.
- Samples are uniformly distributed in space, and not centered at lower dimensional manifolds.
- In high dimensions, most of the mass of a multivariate Guassian distribution is not near the mean, but in a "shell" around it.
- In low dimensions, most of the mass of a multivariate Guassian distribution is not near the mean, but in a "shell" around it.
* `[ A ]`


---

**Q: A good (Machine) Learning algorithm should fulfill:**
- High accuracy + Generalized model + Efficiency
- Representation + Evaluation + Optimization
- Simplicity + Low Bias and Variance + Many parameters
- Statistical significance + Large amount of data + Generalized features
* `[ B ]`


---

**Q: What is NOT an ensamble learning scheme?**
- Bagging
- Boosting
- Stacking
- Roasting
* `[ D ]`


---

**Q: Which of the statements below are true about bias and variance in machine learning?**
- A classifier with low bias is always better than a classifier with low variance.
- A linear classifier has low bias.
- A more powerful learner is always better than a less powerful one.
- Cross validation can help to solve the problem of overfitting.
* `[ D ]`


---

**Q: Which of the following is a common misconception of overfitting?**
- Overfitting is mitigated by using Cross-validation
- Overfitting is avoided when noise is absent
- Adding a regularization term to the evaluation function is a method to combat overfitting
- When data is scarce, performing a statistical significance test like chi-square before adding new structure is a useful technique to combat overfitting
* `[ B ]`


---

**Q: when trying to fit data which property generally tends to lead to higher accuracy**
- strong false assumptions
- weak true assumptions
- weak false assumptions
- none of the above
* `[ B ]`


---

**Q: Which statement about bias, variance and overfitting is false?**
- Adding a regularization term to the evaluation function can prevent overfitting. 
- A linear learner has high bias.
- Decision trees have lower bias than a linear learner, but can suffer from high variance.
- Beam search has a higher bias than greedy search, but lower variance.
* `[ D ]`


---

**Q: Overfitting...**
- ... is good.
- ... is caused by noise in the data.
- ... is caused by not having enough training data.
- ... is not desired.
* `[ D ]`


---

**Q: Which of the following is false with respect to cross validation?**
- It can help in increasing the amount of data available for training
- It can be used to reduce the true classification error of a classifier.
-  Each time, It leaves a different subset of the training data to the side to be used for testing purposes. 
- It can be used to predict how well a classifier will perform on unseen data
* `[ B ]`


---

**Q: When splitting the data in disjunct training and test sets, is it possible for the local optimum of an Objective Function to be more desirable than the global optimum?**
- No. The global optimum to the Objective Function will always minimize the test error and is therefore the most desirable.
- Local optima do not exist. Since all Objective Functions are convex, a local optimum is always the global optimum.
- Yes. Since we don't have access to the actual Objective Function, a local optimum for the training error might generalize better to unseen data.
- When minimizing an Objective Function, the global optimum will always be found. Finding a local minimum never happens in practice.
* `[ C ]`


---

**Q: All of the following describe the methods to combat over fitting in the machine learning except **
- Cross validation 
- regularization term 
- increase data size 
- perform statistical significant test 
* `[ C ]`


---

**Q: During the design of a machine learning algorithm, for some reason, the classifiers you get are not accurate enough. However, you made the best possible set of features so that cannot be changed anymore. What can you do to improve the accuracy of the classifiers? Consider following two statements:

\begin{enumerate}
	\item Design a better learning algorithm.
	\item Gather more data.
\end{enumerate}

Which one(s) of the statements are true/false?**
- 1 false and 2 false
- 1 true and 2 false
- 1 false and 2 true
- 1 true and 2 true
* `[ D ]`


---

**Q: After overfitting, the biggest problem in machine learning is the curse of dimensionality. Which statement is FALSE considering the problems that occur working in high dimensions?**
- Many algorithms that work fine in low dimensions become intractable when the input is high-dimensional
- Generalizing correctly becomes exponentially harder as the dimensionality (number of features) of the examples grows
- Human intuitions, which come from a three dimensional world, can still be helpful dealing with high dimensions.
- In many applications samples are concentrated on or near a lower-dimensional manifold, that helps to reduce dimensionality and classify.
* `[ C ]`


---

**Q: Can we combine test and training data?**
- Yes, because we want to use as much available data as possible
- No, this will mean that we have already seen the examples we want to test our model on
- Yes, especially when using flexible models
- Yes, but only in the case of cross validation
* `[ D ]`


---

**Q: What is not part of the learning process? **
- Optimization 
- Representation
- Maximization
- Evaluation
* `[ C ]`


---

**Q: Overfitting in machine learning is a issue:**
- Cross validation can help combat overfitting.
- Cross validation is the same underfitting.                             
- cross validation can have underfitting problems as well as overfitting problems.
- cross validations helps not to decide the best size of learning data, but  improved speed of the machine learning.
* `[ A ]`


---

**Q: Which of the following statements is False according to the paper, "A Few Useful Things to Know about Machine Learning"?**
- A more powerful learner is not necessarily better than a less powerful one.
- Strong false assumptions can be better than weak true ones.
- Features that look irrelevant in isolation may be relevant in combination.
- Given two classifiers with the same training error, the simpler of the two will likely have the lowest test error.
* `[ D ]`


---

**Q: In deep learning, can a dumb algorithm with a lots of data beat a clever algorithm?**
- It is not possible
- Yes, it is possible
- It cannot be stated in general 
- None of the above are correct
* `[ B ]`


---

**Q: Which of the following statements is TRUE when it comes to overfitting?**
- A more powerful learner is necessarily better than a less powerful one.
- Decision tree learners do not suffer of the high bias or the high variance problem.
- It is possible that a situation like this can appear: a learner with weak true assumptions needs more data in order to combat overfitting than a learner with strong false assumptions, thus making the latter better.
- Noise (such as training examples labeled with the wrong class) is the main cause of overfitting (i.e. severe overfitting cannot occur in the absence of noise).
* `[ C ]`


---

**Q: What is a method to prevent overfitting of data ?**
- Cross-Validation
- Regularization
- Statistical Significance Test
- All of the above
* `[ D ]`


---

**Q: Which statement is false?**
- Learning many models, rather than only one, gives better results
- Simplicity does not imply accuracy
- A cleverer algorithm beats more data
- Learning = Representation + evaluation + optimization
* `[ C ]`


---

**Q: After 5 attempts, the classification of your facebook profile image decomposes your face in the correct attributes: Ears, eyes, mouth. However, your nose is never recognized. What is likely the case?**
- The classifier has high bias.
- The classifier has high variance.
- The classifier has high bias and high variance.
- Nothing can be said about bias nor variance of the classifier.
* `[ A ]`


---

**Q: Which of the following is not a key component in machine learning?**
- Feature
- Representation
- Optimization
- Evaluation
* `[ A ]`


---

**Q: Which of the following is not a method to combat overfitting?**
- chi-square significance test to see if adding a new structure changes the distribution of a class
- cross-validation on the training set
- adding a regularization term to the evaluation function
- use a subset of the training data that performs well
* `[ D ]`


---

**Q: Bagging helps to**
- reduces the variance, increases bias 
- increases the variance, increases bias
- reduces the variance, decreases bias
- increases the variance, decreases bias
* `[ A ]`


---

**Q: What of the following method is not a technique to combat overfitting when data is very scarce?**
- cross-validation
- add regularization term to the evaluation function
- perform a statistical significance test
- apply feature extraction method like PCA
* `[ D ]`


---

**Q: Randomly dividing the training data into subsets, holding out each one while training on the rest can be best described by?**
- Bagging
- Cross-validation
- Stacking
- Boosting
* `[ B ]`


---

**Q: At the end of section ``Data Alone is Not Enough", the author states: `Learning is more like farming, which lets nature do most of the work.' Why would the author claim this?\\
Within Machine Learning,**
- the point is to combine knowledge with data to `grow' programs.
- programmers tend to be lazy, so they prefer to minimise the amount of work.
- learning is all about farming as much data as possible.
- if you have the data and some Machine Learning algorithm, all you have to do is run the program.
* `[ A ]`


---

**Q: Which of the following is not used to avoid overfitting?**
- Cross Validation
- Regularization
- Evaluate more complex structures with statistical significance tests
- Information Gain
* `[ D ]`


---

**Q: Which of the following statements are supported by the paper:
1. Getting more date often beats a cleverer algorithm
2. Simplicity always leads to more accuracy**
- Both are suported
- Only 1 is supported
- Only 2 is supported
- Both are NOT supported
* `[ B ]`


---

**Q: Which statement is true?**
- Generalization of the model describes how well your model fits your training data.
- Underfitting typically corresponds to high bias.
- In general, simply adding more data will not improve classifier performance.
- Combining models will not result in lower bias and lower variance.
* `[ B ]`


---

**Q: When it comes to supervised learning it is a best practice to: **
- Ensure that the entirety of the all labeled examples are used for training, while testing should only occur on a select few reference cases.
- Ensure that the entirety of the all labeled examples are used for training, while simultaneously testing for the networks performance on the same data, to ensure an optimal fit.
- Split the entire data set of all labeled examples in a 50/50 split into a training- & test data set.
- Only start verifying on the test data set once the training accuracy has reached at least 75%.
* `[ C ]`


---

**Q: Which method does NOT help preventing overfitting?**
- Addind a regularization term to evalutiont function
- Using smaller training set
- Using cross-validation
- Performing a statistical significance test like chi-square before
adding new structure
* `[ B ]`


---

**Q: According to the author of "A Few Useful Things to Know about Machine Learning", which of the following is not given as practical advice when tuning a model.**
- Design better features
- Train the model using larger data sets 
- Learn multiple models for the data
- Good theoretical guarantees on a model directly implies generalization
* `[ D ]`


---

**Q: How does a large hypothesis space influence the approach to building a learner.**
- It does not
- It changes the chance of obtaining a well generalizable learner
- It causes learning to take longer
- A larger hypothesis space means faster results
* `[ B ]`


---

**Q: Which of the following statements about data in machine learning is not true?**
- Nowadays, time is often the bottleneck of algorithms than the amount of data available
- It is often a good choice to try simpler learners over more sophisticated ones
- To a first approximation, most algorithms do the same when talking about learning a representation
- The only way to improve the accuracy of your algorithm is to design a better learning algorithm
* `[ D ]`


---

**Q: What is advised to do when the classifier cannot get ideal result?**
- Improve or change algorithms
- Get more trainning data
- Change the starting weight of network
- Change the structure of the network
* `[ B ]`


---

**Q: According to Pedro Domingos, a learning algorithm consists of 3 components. What statement of the following is false?**
- Neural networks are a representation method
- Gradient descent is an evaluation function
- Squared error is an evaluation function
- None of the above is false.
* `[ B ]`


---

**Q: Generalization is the goal of machine learning. Often, a local optimum (of the loss function) may perform better on the test set than the global optimum. Why is this?**
- This is because the loss function may have a local minimum that is extremely close to the global minimum and by chance, it performs better on the test set.
- This is because the global optimum may be overfitting to the training data and thus won't generalize well.
- This is because it may have learned different hyperparameters, which are better suited to the problem.
- This is because high-dimensional data can the loss function to have a highly complex surface.
* `[ B ]`


---

**Q: If your hypothesis space has a limited (bounded) subset of hypothesis function it can be called short. Short code can generalize faster so it's refereing to a simple subset. Which one of the following statements is true?**
- If your classifier is simple the bias is low so the accuracy is large.
- If your classifier is simple it is not guaranteed that it will be accurate. But effectively simplicity implies accuracy
- If the algorithm is accurate it is because we take a small hypothesis space in the first place.
- There is no connection between simplicity of the classifier and the accuracy it gives
* `[ C ]`


---

**Q: A linear lerner is more likely to have..**
- high bias and low variance
- low variance and high bias
- high bias and high variance
- low bias and low variance
* `[ A ]`


---

**Q: Which of the following approaches is the least effective if used to combat overfitting?**
- Multiple hypothesis testing
- Cross-validation
- Adding a regularization term
- Chi-square statistical significance test before adding a new structure
* `[ A ]`


---

**Q: Which statement is true?**
- The error rate of a classifier on the training set is indicative of the general performance of the classifier
- Adding a regularization term to the evaluation function can combat overfitting
- The main role of theoretical guarantees in machine learning is as a criterion for practical decisions
- A cleverer algorithm almost always beats having more data
* `[ B ]`


---

**Q: Given that training a complex and a simple classifier results in the same training error, which classifier should be selected:**
- Simple classifier, because it is likely to have a lower test error
- Simple classifier, because simplicity is a goal on itself
- Complex classifier, because it is likely to have a lower test error
- Complex classifier, because it is less likely to have overfitted
* `[ B ]`


---

**Q: Are Neural Networks a general method that should be used for all classification tasks?**
- Yes, they always outperform other methods due to their high complexity.
- No, Many methods are less complex and give  higher precision depending on the data.
- Yes, a network can always be tweaked to outperform any other method.
- No, use of floats in the network leads to high imprecision for larger networks
* `[ B ]`


---

**Q: How can Machine learning algorithms figure out how to perform important tasks?**
- By learning from reality
- By generalizing from examples
- By classification
- By building up very complex and deep model
* `[ B ]`


---

**Q: What is NOT a benefit of stochastic gradient descent compared to normal gradient descent?**
- What is NOT a benefit of stochastic gradient descent compared to normal gradient descent?
- Computes the gradient more precisely
- Introduces a small regularization effect
- Converges faster to a minimum on extremely large training datasets
* `[ B ]`


---

**Q: Which of the following statement is correct:**
- A biased learner has a tendency to learn random things irrespective of the real signal
- Using cross-validation to make too many parameter choices can increase over-fitting
- The main reason for the over-fitting is the noise in the data
- Features that are individually irrelevant to the individual variable are still irrelevant if we combine them 
* `[ B ]`


---

**Q: What is no common practice in Machine Learning?**
- Cross-validation is used to overcome the problem of a limited data sample
- To improve the prediction accuracy of ML technique, the machine is feed with more features
- To improve prediction power, different learners are combined
- To find an appropriate learning algorithm, someone has to look at the three components Representation, Evaluation and Optimization
* `[ B ]`


---

**Q: What does the generalisation error look like if a model is underfitted?**
- high bias
- high variance
- low bias
- low variance
* `[ A ]`


---

**Q: If your classifier performs well on the training data, but not on separate testing data, what is likely the case?**
- You are overfitting on the training data
- You are overfitting on the testing data
- You have not used enough training data
- You have not used enough testing data
* `[ A ]`


---

**Q: What is bagging?**
- The generation of random variations of the training set by resampling, learning a classifier on each and combining the results by voting.
- To let the classifier focus on training examples it gets wrong, give them weights.
- To let the output of individual classifiers be evaluated by a higher-level learner that combines them.
- To discard sets of examples during training to prevent examples from co-adapting.
* `[ A ]`


---

**Q: Why is error of 0% in the training set NOT a good method for measure generalization?**
- Because all the classifiers can obtain this error in any scenario
- Because the training error is too general
- Because one can memorize all the examples in the training set and get a 0% error.
- There is no reason to think this
* `[ C ]`


---

**Q: Which combination of classifier representation, evaluation technique, optimization technique works best?**
- NN, KL divergence, linear programming
- NN, Information gain, gradient descent
- NN, KL divergence, gradient descent
- NN, margin, linear programming
* `[ C ]`


---

**Q: What is meant by "the curse of dimensionality"?**
- The problem that humans can only visualize 3 dimensions, making it hard for them to solve problems in higher dimensions.
- The problem that as the number of features increases, generalizing becomes exponentially harder.
- The problem that a classifier is only accurate on training data but inaccurate on test data.
- The problem that a classifier performs worse when presented with high dimensional data.
* `[ B ]`


---

**Q: Can higher dimensional input cause trouble for a machine learning algorithm?**
- No, when the appropriate learner is used the result is the same as with a lower dimensional input.
- No, the dimension of the input has no influence on the accuracy of the result
- Yes, generalizing becomes exponentially harder with higher dimensions
- Yes, the computational time increases with higher dimensions
* `[ C ]`


---

**Q: Which of the following results best describes the problem of overfitting data?**
- 25% accurate on training data and 75% accurate on test data
- 100% accurate on training data and 50% accurate on test data
- 50% accurate on training data and 100% accurate on test data
- 75% accurate on training data and 75% accurate on test data
* `[ C ]`


---

**Q: Why do our intuitions fail in high dimensions?**
- Rules of mathematics change in higher dimensions.
- We are used to thinking in only a few dimensions, higher dimension scenarios are too complex for our intuition to grasp.
- Space is 3 dimensional so any other number of dimensions is not realistic and our intuition works best in realistic scenarios.
- Our intuitions fail equally as often in lower dimensions.
* `[ B ]`


---

**Q: Which of the following would not be considered one of the three parts of an learner?**
- Evaluation
- Representation
- Validation
- Optimization
* `[ C ]`


---

**Q: In which case does overfitting happen?**
- 0% accuracy on both training and test data
- 75% accuracy on both training and test data
- 50% accuracy on training data and 100% accuracy on test data
- 100% accuracy on training data and 50% accuracy on test data
* `[ D ]`


---

**Q: Which statement about Machine Learning is not true?**
- Generalization increases the train error 
- A classifier with a high bias is sensitive for overfitting
- The problem with high-dimensional features is that you need a lot of data to train the classifier
- A theoretical bound sets the upper limits, which is usually never reached
* `[ B ]`


---

**Q: Learners can be divided into two major types: those whose
representation has a fixed size, and
those whose representation can grow with the data. Which of the following learners´ representation has a fixed size?**
- k-Nearest Neighbors
- Naive Bayes
- Decision Trees
- Support Vector Machines
* `[ B ]`


---

**Q: Which one of these is correct?**
- A linear learner(classifier) has high bias and low variance.
- Bad and noise training data can lead to overfitting. So once we obtain a good, noise-free dataset, which is really hard, we then don't need to worry about overfitting anymore.
- High-dimensional worlds are hard for us human to understand and thus we can seek intuitions from low-dimensional worlds and apply them to high-dimensional ones.
- Given two classifiers with the same training error, the simpler one will likely to have the lowest test error.
* `[ A ]`


---

**Q: Choose the correct statement:**
- A linear learner has high bias
- The hypothesis space is the set of the classifiers
- To obtain the best classifier, the global maximum of the objective function must be fully optimized
- Representable functions can be learned
* `[ A ]`


---

**Q: Usually large amount of data has as a result that:**
- Complex classifiers learn faster
- Complex classifiers perform better than simple classifiers
- Neural networks always perform better as classifiers
- A simple classifier is a better choice
* `[ D ]`


---

**Q: What's the problem of linear classification model and decision trees respectively in terms of overfitting?**
- high bias; high variance
- high variance; high bias
- low bias; high variance
- low variance; high bias
* `[ A ]`


---

**Q: Mark the false sentence.**
- The fundamental goal of machine learning is to generalize beyond the examples in the training set.
- Bias represents the difference between the expected value of an estimator and the true value being estimated.
- Variance represents a measure of the dispersion of values.
- The goal when building the model is to find the model with the lowest bias.
* `[ D ]`


---

**Q: Which of the following methods can not be used to avoid overfitting?**
- Use of cross-validation to choose algorithm's hyperparameters
- Adding a regularization term to the evaluation function
- Perform a statistical significance test like chi-square before adding new structure to decide whether the distribution of the class really is different with and without this structure
- Reduce the noise of the training data
* `[ D ]`


---

**Q: Which of the following is not ture:**
-  Noises in the training examples can result in overfitting.
-  Gathering more features doesn't hurt the build of classifiers.
-  Beam search has lower bias than greedy search, but higher variance.
- Decision trees learned on different training sets generated by the same phenomenon are often very different. 
* `[ B ]`


---

**Q: Which of the following statements about „overfitting” is FALSE?**
- Cross-validation can help to overcome issue of overfitting.
- Regularization term may penalize more complex classifiers to favour simpler and more smooth solutions.
- Statistical chi-test solves entirely problem of overfitting entirely in all possible applications.
- Linear learners have high bias and low variance.
* `[ C ]`


---

**Q: What model ensemble technique makes the new classifier focus on the examples the previous one tended to get wrong?**
- Bagging
- Boosting
- Stacking
- Bayesian Model Averaging (BMA)
* `[ B ]`


---

**Q: Which of the following statements are true?

1.Bias is a learner’s tendency to consistently learn the same wrong thing.
2.Variance is the tendency to learn random things irrespective of the real signal.**
- Statements 1 is correct, statements 2 is incorrect
- Statement 1 is incorrect, statement 2 is correct
- Both statements are correct 
- Both statements are incorrect 
* `[ C ]`


---

**Q: What are the basic components in learning field? For classification problem, when loss function obtains the optimisation, classification error can reach the lowest as well, is it necessarily correct?**
- Representation, hypothesis, optimisation; no
- Hypothesis, evaluation, optimisation; yes
- Representation, evaluation, optimisation; no
- Hypothesis, representation, evaluation; yes
* `[ C ]`


---

**Q: What is the connection between the number of parameters of a model and it's tendency to overfit?**
- With more parameters more overfitting occurs.
- With less parameters more overfitting occurs.
- Always choose a model with more parameters to reduce overfitting.
- There is no connection between the number of parameters of a model and its tendency to overfit.
* `[ D ]`


---

**Q: Which of the following statements is generally incorrect?**
- More data beats a cleverer algorithm
- It's easier to correctly classify objects in high-dimensional spaces
- It's possible to decrease both the bias and the variance
- Neural networks are generally the best choice for every ML problem
* `[ D ]`


---

**Q: In boosting, training examples have weights, and these are varied so that each new base classifier focuses on the examples the previous ones tended to get wrong. Suppose that we would use an arbitrary boosting algorithm to create a classifier consisting of a boosted base classifier. What would for most use cases be true:**
- The classifier would use a high bias, low variance base classifier
- The classifier would use a low bias, low variance base classifier
-  The classifier would use a high bias, high variance base classifier
- The classifier would use a low bias, high variance base classifier
* `[ A ]`


---

**Q: Which of the following statements is false?**
- It is impossible for a MLP network to learn an XOR function.
- Except for the input nodes, each node is a neuron that uses a nonlinear activation function. 
- A MLP consists of, at least, three layers of nodes: an input layer, a hidden layer and an output layer. Except for the input nodes, each node is a neuron that uses a nonlinear activation function. 
- Learning occurs in the perceptron by changing connection weights after each piece of data is processed, based on the amount of error in the output compared to the expected result.
* `[ A ]`


---

**Q: What is the cause of overfitting?**
- The training dataset is too small, resulting in the inability of the model to generalize.
- The minibatches are too big, resulting in inaccurate results after training.
- The training dataset is too diverse, the model cannot model the training data nor generalize to new data.
- Too many training iterations resulting in increasing error and divergence.
* `[ A ]`


---

**Q: A linear classifier is not able to induce the barrier frontier between two classes when this frontier is not a hyper-plane. Taking this into consideration, we can say about the linear learners that they have:**
- High bias
- Low bias
- High variance
- Low variance
* `[ A ]`


---

**Q: If two classifiers have the same training error you:**
- Choose the one less complex
- Choose both
- Combine them
- Choose the one with lower variance
* `[ D ]`


---

**Q: As classifiers have grown more complex and advancements in research have allowed for record-breaking accuracies the following is true:**
- A more complex classifier will always outperform a simple classifier.
- Accuracy is the best metric to measure the performance of a model.
- A more complex classifier will not always out-perform a simpler classifier.
- More complex classifiers will always perform better with more features.
* `[ C ]`


---

**Q: Many misperceptions occur when talking about the design of learners, in particular classifiers. Which of these options is not a misconception?**
- Now a days a simple model is usually not enough, it is better to combine multiple models.
- When two models are equally good, the simplest one is the best.
- Overfitting is a phenomenon linked to variance and Underfitting is linked to bias
- The working of machine learning algorithms is not dependent on the dimension.
* `[ C ]`


---

**Q: What is one of the main problems for RNNs when trying to understand for instance longer texts.**
- It cannot understand a word properly.
- It is hard to understand a sentence.
- It is difficult to store information from multiple sentences for a longer period.
- It is impossible for the RNNs to read a sentence in the correct order when presented with longer sentences.
* `[ C ]`


---

**Q: Which one of the statements is wrong?**
- Bias is a learner’s tendency to consistently learn the same wrong thing.
- Variance is the tendency to learn random things irrespective of the real signal.
- Over-fitting is caused by noise.
- Generalizing correctly becomes exponentially harder as the dimensionality (number of features) of the examples grows
* `[ C ]`


---

**Q: In machine learning, what is the curse of dimensionality?**
- Generalizing correctly becomes exponentially harder as the dimensionality (number of features) of the examples grows, because a fixed-size training
set covers a dwindling fraction of the input space.
- The fact that many algorithms that work fine in low dimensions become intractable when the input is high-dimensional
- In very high dimensions, it becomes impossible to reason about what is happening.
- When working with high dimensional feature vectors, often a lot of the features are irrelevant.
* `[ A ]`


---

**Q: What can help solving the curse of dimensionality in a practical setting?**
- Adding more features
- Adding more training samples
- Reducing features
- Nothing can
* `[ C ]`


---

**Q: What is NOT a method evaluation of data sets?**
- Maximum likelihood
- Squared Error
- K-nearest neighbors
- Cost/Utility
* `[ C ]`


---

**Q: Which statement about combating overfitting is FALSE?**
- Reducing noise in the used data set, because overfitting is a direct result of noise.
- Adding regularization term to the evaluation function, by favoring smaller structures.
- Performing a statistical significance test before adding a new structure, to decide whether the distribution of the class is really different with and without this structure.
- Cross-validation of the model, for example by choosing the best size of decision tree. 
* `[ A ]`


---

**Q: What does bias of generalization error of overfitting mean?**
- Tendency to learn random things irrespective of the real signal. 
- Tendency to diversify the learning
- Tendency to consistently learn the same wrong thing
- None of the above
* `[ C ]`


---

**Q: What statement is correct**
- All data available should be used to train a learner
- More data is better than a complexer learner
- A complexer learner is better than a simple learner
- A good learner can learn all potential hypotheses
* `[ B ]`


---

**Q: name two big problem's in machine learning.**
- overfitting and compute power
- overfitting and curse of dimensionality
- curse of dimensionality and compute power
- float errors and compute power
* `[ B ]`


---

**Q: Which network is capable of using the spatial structure of the input array to inform the architecture of the network?**
- Fully-connected neural network
- CNNs
- RRNs
- R-CNNs
* `[ B ]`


---

**Q: Which of the following is false according to "A Few Useful Things to know about Machine Learning" by Pedro Domingos?**
- Representation, evaluation and optimization are the 3 components of learning algorithms.
- The classifier's accuracy should be near 100% , even at the cost of increasing the complexity and the number of dimensions in the model.
- Simplicity of the model does not necessarily imply accuracy.
- Cross validation is a common way to avoid overfitting.
* `[ B ]`


---

**Q: What is the correct statement about bias and variance?**
- A linear learner has high variance
- A decision tree learner has high bias
- A model ensemble using bagging technique reduces variance with only a slightly increase in bias
- A model ensemble using bagging technique reduces bias with only a slightly increase in variance
* `[ C ]`


---

**Q: 3.	Identify the techniques P, Q and R described below which are used in building “model ensembles”.  P : “… we simply generate random variations of the training set by resampling, learn a classiﬁer on each, and combine the results by voting.”. Q : “… , training examples have weights, and these are varied so that each new classiﬁer focuses on the examples the previous ones tended to get wrong”. R : “… the outputs of individual classiﬁers become the inputs of a “higher-level” learner that ﬁgures out how best to combine them”.**
- P: Bagging, Q: Boosting, R: Stacking
- P: Stacking, Q: Boosting, R: Bagging
- P: Stacking, Q: Bagging, R: Boosting
- P: Boosting, Q: Bagging, R: Stacking
* `[ A ]`


---

**Q: Which is the characteristic of decision trees?**
- High bias and high variance
- High bias and low variance
- Low bias and high variance
- Low bias and low variance
* `[ C ]`


---

**Q: Overfitting can be avoided by using which of the following ?**
- Regularization
- Normalization
- Cross fitting
- None of the above
* `[ A ]`


---

**Q: Which statement is false?**
- The fundamental goal of machine learning is to generalize beyond the examples of the training set.
- Bias is a learner's tendency to consistently learn the same wrong thing.
- Variance is a learner's tendency to learn random things irrespective of the real signal.
- Overfitting is allways caused due to the influence of noise.
* `[ D ]`


---

**Q: What means Bagging?**
- Using all formerly used features at the same time.
- Using different classifier on parts of the training set and combine the result by voting.
- Dividing training data into evenly scattered sets.
- Dividing training data into unevenly scattered sets.
* `[ B ]`


---

**Q: In order to understand better the problem of overfitting we can decompose the generalization error of a model in bias and variance errors. Which of the following statement is true?**
- Bias error is related to a structural failure in the model
- Bias error and variance error have usually similar values
- Variance error is related to a structural failure in the model
- Bias error is the tendency to learn random things irrespective of the real signal
* `[ A ]`


---

**Q: Consider a system whose certain variables are of interest (states - $x$). We attach a series of sensors to the system $z^{j}$ who are capable of measuring implicitly or explicitly the the state of the system. At certain time intervals ($i$) each sensor outputs an estimate of the state along with a confidence interval $z^{j}_{i}+\sigma^{j}_{i}$. Then, a specifically designed algorithm $\Phi$ (e.g. an adaptive Filter) combines the sensor ouputs, the estimated errors and knowledge about the system and constructs an estimate of the system's state by assigning different weights to the sensor outputs and adding them  $\Phi(i)=\lambda_{1}(i)z^{1}_{i}+\dots \lambda_{m}z^{m}_{i}$ in order to extract the best possible estimate. This model ensemble makes use of which of the following techniques:**
- Bagging
- Boosting
- Stacking
- None of the above
* `[ C ]`


---

**Q: Naive Bayes can outperform a state-ofthe-art rule learner (C4.5rules) even when the true classifier is a set of rules?**
- Yes, naive Bayes can outperform a SOTA learner when given enough data.
- Yes, naive Bayes is complexer than C4.5 and follows the rules strictly.
- No, C4.5 is created as classifier for a set of rules, Bayes is not.
- No, a complex learner always outperforms a Bayes classifier.
* `[ A ]`


---

**Q: Which of the following method cannot help us overcome overfitting in machine learning?**
- Use cross validation to tune the parameters.
- Add a regularization term to the evaluation function.
- Perform a statistical significance test like chi-square before adding new structure.
- Eliminate or reduce noise samples which are labeled with the wrong class.
* `[ D ]`


---

**Q: What does overfitting mean?**
- Encoding random quirks in the data
- Doing too much iterations over the same data
- Having classifiers dependent on each other
- The problem that occurs when you enter an infinite loop
* `[ A ]`


---

**Q: Which of the following classifier does not suffer from the problem of bias?**
- K-nearest neighbor Classifier
- Decision Trees
- Naive Bayes classifier
- Support vector machines
* `[ B ]`


---

**Q: What characterizes algorithms with high variance?**
- Underfitting
- To consistently learn the same wrong thing
- Overfitting
- Noise
* `[ C ]`


---

**Q: Bagging, a form of ensemble development of multiple models, involves ..**
- assigning weights to training data, with higher weights assigned to previously incorrectly classified data
- developing classifiers on different random training sets and combining results by voting
- learning the best combination of simple individual classifiers 
- penalizing more complex models in the cost function to reduce the chance of over-fitting
* `[ B ]`


---

**Q: Branch-and-bound is being used in a learning algorithm as:**
- Evaluation
- Optimization
- Correlation
- Representation
* `[ B ]`


---

**Q: Which statement is wrong about overfitting?**
- If the accuracy on testing dataset is higher than the accuracy on training dataset, it means the classifier does not have overfit.
- Overfit is related to bias and variance. 
- Strong false assumption is better than weak true assumption.
- Cross-validation, adding regularization term, and doing statistical significance test will help to combat overfit.
* `[ A ]`


---

**Q: What is the main difference between Bayesian Model Averaging (BMA) and Model ensemble?**
- BMA assigns weights to hypothesis space and Model ensemble does not 
- Model ensemble assigns weights to hypothesis space and BMA does not 
- Model ensemble and BMA assign weights, but they do it on different levels
- Model ensemble assigns weights to hypothesis space and BMA does not and works on a different level
* `[ C ]`


---

**Q: Which general rule of thumb for machine learning (ML) practitioners, is stated incorrectly from Domingos ML "folk-wisdom" - "A few useful things to know about machine learning" and why?**
- "Learn many models, not just one.": Increased models leads to increased bias which decreases performance.
- "A cleverer algorithm beats more data.": A cleverer algorithm can generalise better with less data.
- "Theoretical guarantees are not what the seem.": Probabilistic guarantees can be formulated on the results of induction.
- "Simplicity does not imply accuracy." - Smaller hypotheses spaces allow hypotheses to be represented by shorter codes, generalising better.
* `[ B ]`


---

**Q: Which of the following is not a component of machine learning?**
- Representation
- Optimization
- Deduction
- Evaluation
* `[ C ]`


---

**Q: Which of these often made assumptions is actually true?**
- More data beats a cleverer algorithm
- Simplicity usually leads to accuracy
- If a function is representable, it is also learnable
- It is better to select one training model that works best, than to combine several lesser models.
* `[ A ]`


---

**Q: What is the fundamental goal of Machine learning?**
- To generalize beyond the examples in the training set.
- To find general solutions using predefined representations.
- To simulate the human brain.
- To find the optimal action in a given decision tree.
* `[ A ]`


---

**Q: By decomposing generalization error, which representation fit into the description of 'low bias, high variance'?**
- Linear learner
- Decision trees
- Beam search
- Rule learner
* `[ B ]`


---

**Q: If your classifier has high variance (overfitting) which of the following is NOT a way to combat it?**
- cross validation
- regularization
- get more training data
- increase the complexity of classifier
* `[ D ]`


---

**Q: Which of the following is NOT true about Bayesian Modelling Averaging (BMA) approach?**
- BMA averages the predictions of trained examples, from the individual predictions of all classifiers in the hypothesis space.
- The hypothesis space of BMA changes and this can take a wide variety of forms.
- BMA assigns weights to the hypotheses in the original space according to a fixed formula.
- The weights of BMA are extremely skewed to point thereby making BMA effectively equivalent to its choice.
* `[ A ]`


---

**Q: Overfitting is when the trained system perfectly matches the training set.

Which of the following is a solution to Overfitting?**
- Randomly dividing your training data into k-subsets, holding out each one while training on the rest, testing each learned classifier on the examples it did not see, and averaging the results.
- Stopping the training process before the learner over-fits.
- Both A and B are solutions to overfitting.
- None of the above.
* `[ C ]`


---

**Q: Machine learning has a lot of different aspects that researchers need to consider. Which of the following is correct?**
- Boosting is an ensemble model that creates random variations of the dataset using resampling
- There are two main limited resources in machine learning: memory and training data
- When two models share the same training error, taking the more simple model is always the better choice
- When a learner is not accurate enough, it is often easier/better to gather more training data than to improve the learning algorithm
* `[ D ]`


---

**Q: Which of the following functions is not used as an activation function for CNNS ?**
- Leaky ReLU
- tanh
- sine
- Maxout
* `[ C ]`


---

**Q: Which of the following cannot reduce overfit**
- Adding a regularization term to the evaluation function
- Perform a statical significance test like chi-square
- Apply cross-validation
- Exclude noise from training data
* `[ D ]`


---

**Q: Overfitting comes in many forms that are not immediately obvious. One way of understanding overfitting is by decomposing the generalization error into bias and variance. For the following learners, classify the correct decomposition:**
- A linear learner has high bias because the learner cant induce it when the frontier between two classes is a hyperplane.
- Decision trees can represent any Boolean function but suffer from high bias; when trained on different sets of the same phenomena, they often differ a lot.`
- A linear learner has low bias because the learner can induce it when the frontier between two classes is a hyperplane.
- None of the above is correct.
* `[ A ]`


---

**Q: Which of the following statements is incorrect?**
- The classifier has bias when the number of dimensions are high
- The noise due to high number of irrelevant features reduces classifier performance 
- Even if all the features (>100) are relevant, the nearest neighbor classifier has a poor performance
- One of the reasons a classifier suffers the curse of dimensionality is because we cannot visualize data in high dimensions 
* `[ A ]`


---

**Q: Which of the following is NOT a suitable way to reduce variance when your model demonstrates high variance across different training sets?**
- Increase the amount of training data in each training set
- Improve the optimisation technique for error minimisation
- Decrease model complexity
- Reduce the noise of training data
* `[ B ]`


---

**Q: When we encounter overfitting:**
- You can always add more examples to the training set to fix it.
- We should try lowering bias.
- We know there are outliers or noise in our data.
- None of the above.
* `[ D ]`


---

**Q: Variable-size learners in principle learn any function given sufficient data. What is the reason, in practice they may not?**
- limitation of the algorithm
- computational cost
- curse of dimensionality
- all of the above
* `[ D ]`


---

