# Questions from `regularisation` :robot: 

**Q: Which statement is fully true regarding the learning curve.**
- Overfitting is the difference between the true error and the apparent error. The apparent error decreases with a larger training set size.
- Overfitting is the difference between the true error and the apparent error. The true error increases with a larger training set size.
- The apparent error is the difference between the model output and the set training target output. The apparent error increases with a larger training set size.
- The true error is the difference between the model output and the set training target output. The true error decreases with a larger training set size.
* `[ C ]`


---

**Q: Which method does NOT help combat overfitting?**
- Data augmentation.
- Early stopping during training.
- Increasing the number of model features.
- L2 regularization.
* `[ C ]`


---

**Q: Which of the following is false, with respect to stopping the training of a neural network early:**
- The norm of the weights can be used to determine when to stop training.
- Early stopping is a form of regularization
- Early stopping depends on the initialization of the weights
- With larger weights the network will stop training earlier. 
* `[ D ]`


---

**Q: Which of the following is not a form of data augmentation:**
- Reducing noise
- Adding noise
- L1 regularization
- Scaling the image up or down
* `[ C ]`


---

**Q: Which of the following is not a reason a learning curve is useful in the context of classification?**
- Help us characterize the performance of a ML model.
- Let us explore the performance of the model at different iterations.
- Let us qualitatively assess the amount of overfitting of the model.
- Let us explore the performance of the model with datasets of different sizes.
* `[ B ]`


---

**Q: Which of the following can help reducing overfitting?**
- Increasing the complexity of the model.
- Removing instances that are nearly duplicated (differ by certain transformations).
- Reduce the complexity of the model.
- Increase the dimensionality of the dataset.
* `[ C ]`


---

**Q: Which statement below is wrong?**
- Weight decay discourages large weights
- Early stopping stops the gradient updates prematurely
- Early stopping is still very useful when network is not correctly initialized
- Network should not start with large weights if using early stopping
* `[ C ]`


---

**Q: Which statement below is wrong?**
- Noise added to input can be considered data augmentation
- Noise added to weights encourages stability of the model
- Noise added to the output means label smoothing
- Noise with large enough variance is equivalent to weight norm regularization
* `[ D ]`


---

**Q: What measures are useful to consider when trying to prevent overfitting (in general machine learning)?**
- Use more data.
- Reduce the number of features.
- Reduce the complexity/flexibility of the model.
- All of the above measures are useful to consider.
* `[ D ]`


---

**Q: When performing early stopping, it is especially important to:**
- Have a correctly initialized network.
- Make use of a general model which does not depend on the amount of input.
- Remove noise from the weights which can encourage the stability of the model.
- Perform dropout on the network. 
* `[ A ]`


---

**Q: Which of the following isn't a method of regularisation?**
- Implementing early stopping by making use of a statistically independent validation set
- Reducing parameters through weight sharing
- Ignoring neurons through dropout
- Adding noise to the training labels to promote robustness
* `[ D ]`


---

**Q: Consider the following two statements about noise robustness:
\begin{enumerate}
    \item Noise added to a dataset can be seen as a form of data augmentation   
    \item Noise with extremely small variance is equivalent to weight norm regularization
\end{enumerate}
Which of the statements are true?**
- Statement 1 and 2 are true
- Statement 1 is true, statement 2 is false
- Statement 2 is false, statement 1 is true
- Statement 1 and 2 are false
* `[ A ]`


---

**Q: Which of the following concerning dropout is FALSE?**
- It trains an ensemble consisting of all subnetworks that can be constructed by removing non-output units from an underlying base network.
- It is a combination of weight decay and noise injection.
- During back propagation, a randomly selected fraction of the nodes is skipped and rescaled.
- b) and c).
* `[ C ]`


---

**Q: Which of the following regarding learning curves is NOT correct?**
- Underfitting describes the difference between the Bayes error and the error at which the test error and the train error converge. 
- The expected generalization error can never increase as the number of training examples increases.
- The gap between the training error and testing error denotes overfitting.
- The expected training error of a randomly selected model is less than or equal to the expected test error of that model. 
* `[ D ]`


---

**Q: Which of the following statements is true about a classifier learning curve?**
- It allows designers to spot instances of overfitting and underfitting.
- The apparent and true errors should be averaged over multiple runs using random sampling.
- It depicts the apparent and true error over the number of samples.
- All of the above.
* `[ D ]`


---

**Q: Overfitting can be reduced by:**
- Decreasing the number of samples.
- Performing regularization.
- Using more complex models.
- Increasing the number of features.
* `[ B ]`


---

**Q: Why the function “RandomUniform(minval=0, maxval=1)” is not good to initialize the weights of a network?**
- The function could return a too small value and starting with small weights is wrong
- The function could return a too large value and starting with large weights is wrong
- The function always returns the same value
- The function could return a too small value and the network is correctly initialised only with values larger then 1
* `[ B ]`


---

**Q: When do we implement regularisation?**
- When we have a large training set
- When we have large data but a small number of features
- When our model is too simple
- When our network is at risk of overfitting
* `[ D ]`


---

**Q: Weight initialization: Is it better to initialize small or large weights and why?**
- Large weights are better, since they are already closer to the optimal values
- It does not matter: The network simply is optimized until the best weights are found, which is independent of initialization
- Small weights are better, since large weights could already be so far off, that the gradients will not be able to pull the weights towards good values
- Large weights are better: With that, the network is less prone to overfitting
* `[ C ]`


---

**Q: Which of the following answers is NOT a method to fight overfitting?**
- Early stopping
- Regularization
- Noise Robustness
- Increase test set size
* `[ D ]`


---

**Q: Which of the following can be seen as a cause of “underfitting”?**
- The classifier is too complex
- the network has been trained using too many epochs
- momentum has been employed as optimizer during backpropagation
- None of the above
* `[ D ]`


---

**Q: The goal of regularisation is**
- trying to avoid overfitting
- trying to avoid underfitting
- trying to speed up training
- none of the above
* `[ A ]`


---

**Q: Regularisation is often used to prevent overfitting. Overfitting is**
- when the model is so flexible, that it adapts too well to the training data and loses generalization power.
- when the training error is lower than $3\%$.
- when the model is too rigid to adapt to the distribution of the data.
- when the training data exceeds one million samples.
* `[ A ]`


---

**Q: Weight decay is a regularizing term that is added to the loss function we wish to minimize. If we denote by $w$ the vector of all the weights in the network, we generally include a term $\lambda ||w||_1$ or $\lambda ||w||_2$. In practice, what is an observable difference in using the two different norms?**
- using $||w||_2$ forces the vector $w$ to reduce its coordinates uniformly while using $||w||_1$ makes the smaller values cling to $0$.
- there is no observable difference as the two norms are equivalent (i.e. exist $C_1, C_2$ such that $C_1||w||_1 \leq ||w||_2 \leq C_2||w||_1$).
- using $||w||_1$ forces the vector $w$ to reduce its coordinates uniformly while using $||w||_2$ makes the smaller values cling to $0$.
- using the regularization term $\lambda ||w||_1$ prevents weights to go over $\lambda$ while $\lambda||w||_2$ prevents the weights of going over $\lambda^2$.
* `[ A ]`


---

**Q: Which of the following is true about regularization**
- For small size training set, the true error for a complex model is greater than that for a simpler model
- The Bayes error for a simple model is smaller than that for a complex model
- Regularization is useful when you underfit
- The goal of the optimization of a network is to reach the
global optimum
* `[ A ]`


---

**Q: Which sentence is true?**
- You can beat underfitting by augmenting data
- Regularization discourages overly complex models
- You can beat overfitting by increasing the flexibility of your model
- All of the above
* `[ B ]`


---

**Q: Which statement is wrong about Regularisation?**
- Regularisation avoid overfitting by discouraging learning a more complex or flexible model. 
-  Regularisation could reduce the variance of the model, without substantial increase in its bias. 
- The weights (coefficients) obtained by Regularisation L2 can be zero. 
- The weights (coefficients) obtained by Regularisation L1 can be zero. 
* `[ C ]`


---

**Q: Which following statement is wrong?**
- For network initialisation, it is always better to start with small weights. 
- Data augmentation is to increase the dimensions of samples.  
- Noise added to the output means label smoothing
- Dropout does not have to lead a decrease of complexity of the data distribution.
* `[ B ]`


---

**Q: What is the core effect by applying dropout?**
-  prevent overfitting
- doing the data augmentation
- find the optimal layer units
- delete some layer units to prevent underfitting
* `[ A ]`


---

**Q:  Which method could not help to do the data augmentation?**
- Create fake data into training set
- Create more same data sample into training set
- add stochastic noise into input layer
- add noise into hidden layer
* `[ B ]`


---

**Q: What is NOT  a way one can reduce the chance of overfitting a model**
- By using more data
- By properly selecting the data
- By reducing the number of features
- By reducing the complexity of the model
* `[ B ]`


---

**Q: Which statement is not true about earlier stopping.
1. Earlier stopping only depends initialization
2. It is good to stop earlier if the validation set does not increase in accuracy**
- Both statements are correct
- Only statement 1 is corrct
- Only statement 2 is correct
- Both statements are NOT correct
* `[ C ]`


---

**Q: Imagine you have a learning curve for which you have on the y-axis error and x-axis the size of the training set. On this plot you have the true error and the apparent error on training set. Now somewhere in this curve you will have overfitting and underfitting. Where in this curve can you find this?**
- Overfitting is the area between 0 error and the apparent error. Underfitting is area between the true error and the apparent error.
- Overfitting is area between the true error and the apparent error. Underfitting is the area between 0 error and the apparent error. 
- Overfitting is area between the true error and the apparent error. Underfitting is the entire area beneath the true error.
- Overfitting is the area between 0 error and the apparent error. Underfitting is the area above the apparent error.
* `[ B ]`


---

**Q: What are three ways to beat overfitting in your system?**
- 1: Regularisation, 2: Using more data 3: Creating a more complex/flexible model
- 1: Use more data, 2: Reduce the amount of features, 3: Reduce complexity/flexibility of the model.
- Use less data, 2: Creating a more complex/flexible model. 3: Increase the amount of features
- 1: Use less data, 2: Reduce the amount of features, 3: Reduce complexity/flexibility of the model.
* `[ B ]`


---

**Q: How can we combat overfitting?**
- By increasing the amount of training features
- By increasing the complexity/flexibility of our model
- By applying regularisation to the model
- By using less data, thus creating a more general model
* `[ C ]`


---

**Q: What is technique used for regularisation**
- We add a term to the cost function that includes the L2 norm
- We add more and more data to the network in order to regularise it
- We add complexity to the model
- We make the apparent error equal to the test error
* `[ B ]`


---

**Q: Which technique could be used to reduce overfitting?**
- Data augmentation.
- Reduce feature dimension.
- Add regularization.
- A, B and C.
* `[ D ]`


---

**Q: What is NOT the effect of noise injection?**
- Noise added to input can be considered data augmentation.
- Noise added to weights encourages stability of the model.
- Noise added to the output means label smoothing.
- Infinitesimal noise could decrease the performance of model.
* `[ D ]`


---

**Q: Looking at the learning curve, what typically happens when the size of training set increases?**
- The model is more likely to overfit
- The model is not going to overfit
- The true error increases
- The false error increases
* `[ A ]`


---

**Q: What happens to the true- and apparent error as training set size increases?**
- The apparent error approximates the true error
- The true error increases monotonically
- The apparent error increases monotonically
- Both the true and apparent error increase monotically
* `[ A ]`


---

**Q: One of the possible tecniques used to speed the training of a neural network is known as early stopping. In which of the following condition it cannot be applied safely?**
- If the initial weights in the net are large
- If the net is trained with small batch size
- If we want to train the net on large database containing a few number of classes
- If we know in advance that the loss function in the net is very smooth
* `[ A ]`


---

**Q: What is the fundamental idea of dropout during the training procedure of a net?**
- At every iteration a subset of the neurons is made inactive
- At every iteration a random layer in the net is skipped
- At every iteration the dimension of the batch is slightly changed
- At every iteration some of the weights in the net are set to zero by default
* `[ A ]`


---

**Q: Statement 1: to beat overfitting the number of features could be reduced.
Statement 2: to beat overfitting less data could be used. **
- Both statements are true
- Statement 1 is true
- Statement 2 is true
- None of the statements are true 
* `[ B ]`


---

**Q: Statement 1: Regularisation limits the complexibility and increases the flexibility of a complex model. 
Statement 2:Early stopping is useful when the network is incorrectly initialised. **
- Both statements are true
- Statement 1 is true
- Statement 2 is true
- None of the statements are true 
* `[ D ]`


---

**Q: Which of the following statements about regularisation is TRUE ?**
- Using a large value for the Regularisation parameter can sometimes lead to the hypothesis to overfitting on the data. This can be controlled by reducing the value of the  regularisation parameter.
- Using a large value for the Regularisation parameter leads the hypothesis to underfit on the data. This can be controlled by reducing the value of the  regularisation parameter.
- Using a large value for the Regularisation parameter does not lead to overfitting or underfitting of the data, but might lead to complex numerical problems to solve, hence it is advised to reduce the value.
- All of the above.
* `[ B ]`


---

**Q: Early stopping is a form of regularization used to avoid overfitting when training a learner with an iterative method, such as gradient descent. When can this go wrong ?**
- When the Loss function is not convex.
- When the batch normalisation is not applied on the network layer inputs.
- When the initialisation of weight is large.
- None of the above.
* `[ C ]`


---

**Q: What is a good way to reduce overfitting?**
- Data augmentation.
- Reducing the number of features.
- Reducing the complexity of the model.
- All of the above.
* `[ D ]`


---

**Q: What is NOT true about noise?**
-  Noise added to the input can be considered data augmentation.
- Noise added to weights reduces the stability of the model.
- Noise with infinitesimal variance is equivalent to weight norm regularisation.
- Noise added to the output means label smoothing.
* `[ B ]`


---

**Q:  Consider overfitting and regularization. What holds true?**
- Although complex models tend to converge to better classification performance, their error with small training set is large and requires regularization
- Complex models tend to converge to better classification performance, also they are usually better than less complex models when training set is small
- Complex models are theoretically better, but it is practically impossible to achieve good classification performance as they require massive datasets.
- Complexity does not influence classification performance, chosen model has to be good for certain classification task
* `[ A ]`


---

**Q: Suppose you are carrying a study for object recognition and using Deep neural network (complex model), however you do not have sufficient training data. Which of the following will NOT help to overcome the problem?**
- Do not use Deep neural network and look for models which are less complex
- Augmenting the dataset
- adding regularization parameters
- Adding noise for training weights
* `[ A ]`


---

**Q: What is NOT a good way to prevent overfitting of your model?**
- Reduce the number of features
- Make the model more complex
- Use more data / data augmentation
- Rescale input images
* `[ B ]`


---

**Q: What is the thought that justifies the dropoup method?**
-  Node failure could happen anytime, and the network should be able to deal with it
- Removing weights is allowed, as long as you add the removed weight to other nodes
- If the network is too dependent on a single node, you are probably overfitting
- It is practically the same as bagging
* `[ C ]`


---

**Q: Which of the following data augmentation techniques would not generate effective training data for digit classification?**
- Shearing
- Rotation
- Squeezing
- Translation
* `[ B ]`


---

**Q: Which of the following options would be the incorrect approach to combat overfitting?**
- Train your model for longer
- Train your model on more data
- Reduce the dimensionality of your data
- Reduce the complexity/flexibility of your model
* `[ A ]`


---

**Q: What does a learning curve plot?**
- It plots the true and apparent error on the y-axis, with the size of the training set on the x-axis
- It plots the true and apparent error on the x-axis, with the size of the training set on the y-axis
- It plots the true and apparent error on the y-axis, with the amount of training epochs on the x-axis
- It plots the true and apparent error on the x-axis, with the amount of training epochs on the y-axis
* `[ A ]`


---

**Q: Which of the following methods might NOT help prevent overfitting, instead increasing complexity?**
- Parameter norm regularisation
- Early stopping
- Weight sharing
- Dropout
* `[ D ]`


---

**Q: How to beat overfitting?**
- reduce complexity of the model
- reduce the number of features
- data augmentation
- all of the above
* `[ D ]`


---

**Q: What encourages stability of the model? **
- noise added to input
- noise added to weights
- noise added to output
- all of the above
* `[ B ]`


---

**Q: What is FALSE about a overfitted model?**
- the model has good performance in the training set
- the model has bad performance in the test set
- the model has high variance and high bias
- overfitting can be reduced if we decrease the degree d of the  model
* `[ C ]`


---

**Q: What does NOT help to reduce overfitting?**
- Use more data
- Cross validation
- Dropout
- Regularisation
* `[ B ]`


---

**Q: If we look at learning curve for a neural net. How does this curve change if we increase the complexity of the model?**
- A.	The bayes error gets lower, the test error at the left part of the graph gets higher.
- B.	The bayes error gets higher, the test error at the left part of the graph gets higher.
- C.	The bayes error gets lower, the test error at the left part of the graph gets lower.
- D.	The bayes error gets lower, the test error at the left part of the graph gets lower.
* `[ A ]`


---

**Q: What is meant with feature reduction?**
- A. Reducing the number of non-input neurons in a network
- B. Reducing the number of non-input weights in a network
- Reducing the number of inputs in a network
- None of the above
* `[ C ]`


---

**Q: What are in general good methods to reduce overfitting?**
- Use more data (augmentation), reduce the number of features, reduce the complexity/flexibility of the model
- Leaving out all hidden layers
- Increase the learning rate hyperparameter
- Increasing the sparsity of the network to obtain a fully connected network
* `[ A ]`


---

**Q: Which statement is TRUE about dropout?**
- Dropout decreases the complexity
- It is very suitable for convolutional networks
- It is a combination of weight decay and noise injection
- At each training set, a random fraction of nodes, p is selected and are excluded in the loss calculation
* `[ C ]`


---

**Q: What is a reason of overfitting?**
- Small training set
- Large number of features
- Very flexible model on training data
- All of the above
* `[ D ]`


---

**Q: What of the following does not help solve the overfitting problem?**
- Regularization
- Feature extraction 
- Boosting
- Increasing the classifier complexity 
* `[ D ]`


---

**Q: What means regularisation in Deep-learning**
- Avoid overfitting 
- Make slight modifications to the learning algorithm such that the model generalizes better
- Regularization helps to get error more to origin
- All of the above
* `[ D ]`


---

**Q: What is correct about the random weights?**
- Set the random weight value between zero and one
- Set the value of the random weight value between 0 and 0.000005
-  Random weight helps increasing learning performance
- Weights-decade almost hte same as adding noise to inputs
* `[ C ]`


---

**Q: Which of the following answers is not an option to reduce the chance of overfitting?**
- Using more data
- Reducing the number of features
- Adding more parameters to the model
- Reducing the complexity of the model
* `[ C ]`


---

**Q: Which of the following options can never result in a better performing model?**
- Adding noise to the input data
- Adding noise to the weights
- Adding noise to the output means
- All of the above can result in a better performing model
* `[ D ]`


---

**Q: What is the Adam algorithm that is used for optimising the learning rate?**
- A variant of the Momentum algorithm
- A variant of the RMSprop algorithm
- A combination of Momentum and RMSprop
- A whole different algorithm than Momentum or RMSprop
* `[ C ]`


---

**Q: What happens when a model is increased in complexity?**
- The test error will decrease when much data is available
- It will be easier to overfit the model
- Both A and B are true
- Neither A or B are true
* `[ C ]`


---

**Q: What isn't a good way to beat overfitting?**
- Data augmentation
- Reduce the number of features
- Reduce flexibility of the model
- Increase complexity of the model
* `[ D ]`


---

**Q: When is data augmentation a good procedure?**
- When the invariance is known
- When the test pool size is small
- When the training error is high
- When featrue reduction is done
* `[ A ]`


---

**Q: Considering the learning curve, which of the following statements are true?

1. Overfitting can be seen as the difference between the true error and the apparent error on the training set?
2. Underfitting can be seen as the difference between the apparent error on the training set and the case of zero error  (the x axis).**
- Statements 1 is correct, statements 2 is incorrect
- Statement 1 is incorrect, statement 2 is correct
- Both statements are correct 
- Both statements are incorrect
* `[ A ]`


---

**Q: What is not a method to combat overfitting?**
- Use more data, i.e. Data augmentation.
- Reduce the amount of features.
- Reduce complexity/flexibility of the model.
- All of the above are correct methods to combat overfitting.
* `[ D ]`


---

**Q: What is NOT a way to beat overfitting?**
- Use more data (e.g. by using data augmentation)
- Reduce the number of features
- Reduce the complexity/flexibility of the model
- Use cross-validation to estimate the performance of the model
* `[ D ]`


---

**Q: What is NOT a way of applying regularisation to a neural network?**
- Introducing a parameter that penalizes high weights
- Prematurely stopping the gradient updates
- Applying dropout: Set a randomly selected set of nodes to 0 at each training step
- Feeding the network a batch in reverse order
* `[ D ]`


---

**Q: Which of the following statements about regularization are correct:**
- Regularization aims to reduce the apparent error.
- Regularization makes the aims to make the model more flexible.
- Dropout layers are a method of regularization used in linear regression models.
- None of the above
* `[ D ]`


---

**Q: What is true about the regularization hyperparameter:**
- It affects the degree by which the regulariser affects the loss. Increasing it will make the model more flexible.
- L1 normalization will reduce the weight to 1.
- Regularization can cause underfitting when the effect of the regularizer becomes negligible. 
- None of the above.
* `[ D ]`


---

**Q: Which is no efficient way to mitigate overfitting?**
- Randomly initialize large weights and then start with SGD. However, stop gradient updates prematurely.
- Add noise to weights, input and output.
- Set weight to zero if it is not important 
- Add a parameter norm penalty to the objective function
* `[ A ]`


---

**Q: Which statement is right?**
- One should always try to find a model which is flexible enough to cross all training data
- A CNN is in itself a method of regularization. 
- The true and apparent error on the training set converge never converge
- Regularization modifies a learning algorithm in a way that it reduces both its generalization and training error.
* `[ B ]`


---

**Q: Which of the following statements about regularization does not unequivocally hold?**
- In the "early stopping" technique, one should generally start with small weights.
- The "early stopping" technique is a form of regularization.
- Parameter norm regularization adds a term to the loss function penalizing large weights.
- Dropout consistently yields better generalization results than equivalent networks without it.
* `[ D ]`


---

**Q: When implementing the dropout technique,**
- at each step of the training process, a randomly selected fraction of nodes in the input layer is disabled.
- at each step of the training process, a randomly selected fraction of the nodes is set inactive.
- any weights below a certain threshold are set to 0.
- the end of the traditional training process, a randomly selected fraction of the weights is disabled.
* `[ B ]`


---

**Q: Regularisation doesn’t do the following to beat overfitting :**
- Reduce the number of features.
- Reduce complexity of the model.
- Early stopping, before the weights have converged. 
- Penalise loss function by adding a multiple of norm of weight vector.
* `[ C ]`


---

**Q: Dropout prevents overfitting by:**
- Randomly making a fraction of nodes inactive in every training step.
- Forcing sets of parameters to be equal.
- Regularising parameters of a model based on parameters of another model.
- Several models are trained separately and all of them vote on the output.
* `[ A ]`


---

**Q: [Learning Curve] When the size of the training set increases:**
- The true error decreases and the apparent error on the training set decreases.
- The true error increases and the apparent error on the training set decreases.
- The true error decreases and the apparent error on the training set increases.
- The true error increases and the apparent error on the training set increases.
* `[ C ]`


---

**Q: Noise robustness: noise added to weights....**
- is equivalent to weight norm regularisation.
- encourages stability of the model.
- can be considered data augmentation
- none of the above
* `[ B ]`


---

**Q: In optimizing our loss function during training, why would we introduce regularization to prevent reaching the weights corresponding to the optimal loss?**
- The optimal loss is ultimately only an estimation based on the training data.
- Regularization will prevent overfitting.
- If we optimize the apparent error, generally more data will be required for the true error to converge to the apparent error.
- All of the above.
* `[ D ]`


---

**Q: Which of the following is not a form of regularization?**
- Convolutional Neural Nets' architecture in comparison to Feed Foodward Neural Networks.
- Early Stopping with random weight initialization.
- Constraining (some) weights to be exactly 0.
- All of the above.
* `[ B ]`


---

**Q: To use "Early stopping", one needs:**
- A test set
- A validation set
- A large training set
- A regularisation term
* `[ B ]`


---

**Q: Translations, scaling, squeezing and shearing can be used for:**
- Data regularisation
- Data integration
- Data augmentation
- Data normalization
* `[ C ]`


---

**Q: What is not an option to beat overfitting?**
- Data augmentation
- Feature reduction
- Reduce model complexity
- All are correct options to beat overfitting
* `[ D ]`


---

**Q: What effect can a badly choosen learning rate have?**
- Taking a long time to find a minimum loss
- Getting stuck in a local minimum of the loss
- Skipping over global minima of the loss
- All can happen
* `[ D ]`


---

**Q: Which method is used to prevent overfitting?**
- Regularisation
- Normalisation
- Optimisation
- Centralisation
* `[ A ]`


---

**Q: Which of the following description iOS correct?**
- Regularisation is a method used for modify loss function to achieve better parameters
- Ideally, we can achieve a test error lower than Bayes error if we choose a very good model
- The visual shape of regulator in norm 1 is circle, where the corner points are regarded sparsity
- The visual shape of regulator in norm 2 is square, where the corner points are regarded sparsity
* `[ A ]`


---

**Q: Find the best completion of the following statement: The purpose of regularization is...**
- To nudge models towards simpler solutions and thus not overfit.
- To find a solution that performs better on the training data.
- To initialize neural networks using better starting weights.
- None of the above.
* `[ A ]`


---

**Q: Which of the following does not reduce overfitting.**
- Using more data.
- Reducing the number of features.
- Reducing the number of possible outcomes.
- Reducing the complexity / flexibility of the model.
* `[ C ]`


---

**Q: Which of the following statements is false?**
- In a learning curve, the apparent error increases and the true error decreases if the size of training set increases.
- When a training set is small, there is more chance at overfitting.
- Regularization can be used to prevent overfitting.
- L2 performs feature selection/neuron removing during training of the network.
* `[ D ]`


---

**Q: Which of the following statements about noise robustness is false?**
- Noise added to the output means smoothing of the labels.
- Noise added to the input can be considered as data augmentation.
- Noise added to weights discourage the stability of the model.
- Noise with infinitesimal variance is equivalent to weight norm regularization.
* `[ C ]`


---

**Q: What is a goal of regularization?**
- Reduce the flexibility of a model
- Increase the complexity of a model
- Data augmentation
- Data filtering
* `[ A ]`


---

**Q: What is typical for L1 regularization?**
- All weights become very small but nonzero
- The network becomes sparse
- The non regularized optimum will still be found
- The regularized solution is a constant factor smaller than the non regularized optimum
* `[ B ]`


---

**Q: How can overfitting be avoided?**
- Data augmentation
- Reduce the number of features
- Reduce complexity/flexibility of the model
- All of the above
* `[ D ]`


---

**Q: How to regularize a neural network?**
- Reduce the number of neurons
- Drop some connections between the neurons
- Both a and b
- Reduce the number of training samples
* `[ C ]`


---

**Q: Which of the following statements is TRUE?**
- The bayes error is independent of the training set
- With a small training set one is likely to underfit
- Data augentation does not affect overfitting
- Reducing the complexity/flexibility of a model is an effective way to combat overfitting
* `[ D ]`


---

**Q: Choose the best answer.
Which of the following should NOT be primarily used as a regularization technique?**
- Early stopping
- Dropout
- Weight sharing
- Batch normalization
* `[ D ]`


---

**Q: How do we generate adversarial data in deep learning for image classification?**
- By adding flipping and rotation
- By adding specific type of noise to cause class confusion
- By adding haluccinated feature maps as stickers to images to cause class confusion
- None
* `[ B ]`


---

**Q: How do we know we dont need much regularization?**
- The train and test dataset are very similiar to each other
- The train, val and test error curves are of the same nature and closeto each other
- If we have done lots of smart data augmentation
- We can not even make a estimate of such a situation
* `[ B ]`


---

**Q: Which of the following methods can be used to beat overfitting?**
- data augmentation
- reduce the number of features 
- reduce complexity of model
- all of the above 
* `[ D ]`


---

**Q: Which of the following are true?**
- Noise added to input can be considered as data augmentation
- Noise added to weights encourages stability of the model
- Noise added to output means label smoothing
- All of the above 
* `[ D ]`


---

**Q: Regularization within Deep Learning:**
- Discourages overly complex models.
- Encourages overly complex models.
- Makes complex models less complex.
- Prevents underfitting.
* `[ A ]`


---

**Q: Early stopping is not:**
- A regularization method.
- A method to increase the performance or your model.
- A method to increase the execution speed of your model.
- Independent of the initialization.
* `[ D ]`


---

**Q: Which of the following is NOT a form of regularization?**
- Removing layers from your network architecture
- Early Stopping
- Dropout
- Weight Sharing
* `[ A ]`


---

**Q: What would be expect to happen during training if we set the regularization term for L2 regularization to be very large, when training on a large data set? Assume you have initialized the network weights with small values?**
- The effect of each of the network weights on the end outcome remains small
- The gradient updates for all weights would be small
- The computed decision boundaries will exhibit high bias
- All of the above
* `[ D ]`


---

**Q: What happens when we keep the degree of the regression model low**
- under-fitting
- over-fitting
- Wrong model
- None of the above
* `[ A ]`


---

**Q: What does L1 regularization do?**
- Randomly drops the features
- Reduces the weights of important features
- Assigns the weights randomnly
- None of the above
* `[ B ]`


---

**Q: Beating overfitting? What doesn't help**
- Duplicating the data
- Add more data via augmentation
- Early stopping
- Drop out
* `[ A ]`


---

**Q: What is not part of data augmentation?**
- Adding noise
- Rotate image
- Feature reduction
- Adding new data
* `[ D ]`


---

**Q: With the complexity increasing for the model, how does the true error change and how does the Bayes error change?**
- The true error will decrease. The Bayes error will decrease.
- The true error will decrease. The Bayes error will increase.
- The true error will  increase. The Bayes error will increase.
- The true error will increase. The Bayes error  will decrease.
* `[ C ]`


---

**Q: which of the following about Noise robustness is not true?**
- Not only variations in input data, also noisy variations of the weights and outputs.
- Noise added to input can be considered data augmentation.
- Noise added to weights discourages stability of the model.
- Noise added to the output means label smoothing. 
* `[ C ]`


---

**Q: Which of the following statements is NOT valid?**
- Normalization techniques can be beneficial to regularization
- The true error of every model \hat{M} (e.g. a neural network) always exhibits a lower bound \epsilon > 0
- Dropout techniques look upon a base network as an ensemble of independent subnetworks
- Modeling the uncertainty of the input dataset causes overfitting
* `[ C ]`


---

**Q: Which of the following statements is NOT valid?**
- A non carefully devised dataset augmentation can introduce estimation bias to a neural network
- Adversarial training can be implemented by adding white noise to the inputs
- An  infinite amount of data does not safeguard the best possible generalization capability of a network 
- Dropout can slow down the convergence rate of a network
* `[ B ]`


---

**Q: Small Training Data size leads to  ____ and the training error is _____**
- Over fitting , High
- Over fitting, Low
- Under-fitting, High
- Under-fitting, Low
* `[ A ]`


---

**Q: 1. L2 regularisation drives the weight far away from the origin
2. L1 regularisation leads the weights to become sparse
Identify the statement/s which is/are incorrect**
- 1
- 2
- 1 and 2
- None
* `[ A ]`


---

**Q: Which amongst the below mentioned statements are the elements to using early stopping?\newline
A. Monitoring model performance.\newline
B. Trigger to stop training.\newline
C. The choice of model to use.\newline**
- B, C are correct.
- A, B are correct.
- A, C are correct.
- All are correct
* `[ D ]`


---

**Q: Which of the following statements is wrong about Dropout?**
- Dropout can be used after training when predicting with a fit network.
- Dropout is not implemented on output layer.
- Dropout is implemented per-layer in a neural network.
- Before finalizing the network, the weights are first scaled by the chosen dropout rate.
* `[ A ]`


---

**Q: Which of the following techniques cannot be used to avoid over-fitting data**
- Use data augmentation techniques to increase data samples
- Use a less complex model
- Use less features of the data
- Reduce the variations in the first moments of the data
* `[ D ]`


---

**Q: Which of the following is an argument for proper weight initialization**
- Selecting the right weights allow for faster convergence and reduces the problem of vanishing gradient
- Proper weight selection removes the need for regularization on complex models
- Choosing the right weights mitigate the disadvantages of training with a small data set.
- None of the above
* `[ A ]`


---

**Q: The overfitting happens when the test data don’t  behave as good as the training data, more specifically the overfitting phenomenon happens when we trained a ML classifier to perform well only on the training set and not to a general problem. The underfitting on the other hand happens when:**
- On a learning curve the distance between the apparent error and the test error is too big.
- The classification boundaries are too complex to classify correct a more generic problem.
- A linear classifier is complex enough to make the correct classification but we use a more complex objective function to make the classification.
- The classifier in both training and test dataset achieve a very poor classification.
* `[ D ]`


---

**Q: What does the regularisation parameter offers to a deep neural network with respect to the overfittiing and underfitting phenomenon?**
- Perform a neural removal during the training of the network, something similar with the feature reduction in machine learning.
- Shrink the magnitude of every single parameter in the cost function by adding the regularisation parameter.
- None of the above
- Both a and b
* `[ D ]`


---

**Q: What is not a 'regularization method' discussed in the lecture?**
- Parameter norm regularisation
- Early stopping
- Weight sharing
- Late starting
* `[ D ]`


---

**Q: Which of the following statements is true? With increasing classifier complexity ......**
- The Bayes error becomes smaller
- The training error goes up, for small training sets
- The test error goes up, for small training sets
- The training error goes up, for large training sets
* `[ C ]`


---

**Q: Choose the correct statement:**
- Early stopping should only be used if the network is correctly initialized
- Dataset augmentation is not effective to overcome overfitting because it adds very similar samples to the training set
- Given a relatively small fixed training set, more complex models usually outperform less complex ones
- Noise robustness considers variations only in the input data
* `[ A ]`


---

**Q: Decide if the following statements are true or false:
1. Regularization limits the flexibility of a model.
2. Dropout removes nodes from the network that always produce 0 output.**
- 1: true, 2: true
- 1: true, 2: false
- 1: false, 2: true
- 1: false, 2: false
* `[ B ]`


---

**Q: Which of the following statements about noise robustness is true?**
- Adding noise to the input cannot be considered data augmentation
- Adding noise to weights encourages stability of the model
- Noise with infinitesimal variance is equivalent to weight norm regularization
- Adding noise to the output means label smoothing
* `[ A ]`


---

**Q: Which of the following is a possible way to avoid overfitting?**
- Data augmentation
- Reducing the number of features
- Reducing complexity and flexibility of the model
- All of the above
* `[ D ]`


---

**Q: How we can recognize overfitting?**
- Big diffrence between training error rate and actual error rate
- Small diffrence between training error rate and actual error rate
- When training error rate and actual error rate meet
- We cannot recognize using error rate
* `[ A ]`


---

**Q: Is image rescalling to reduce complexity still widly use?**
- It was never used for that purporese
- Yes, it is used almost always
- No, it was mosty replace by cnn network, but can be still be secretly used
- None of it is correct
* `[ C ]`


---

**Q: What kind of transformations is NOT useful to be included for data augmentation?**
- Squeezing
- Horizontal or vertical translation
- Rotation
- Scaling
* `[ C ]`


---

**Q: What is regularisation useful for, and what is a good initialisation strategy?**
- Underfitting; Small weights
- Overfitting; Small weights
- Underfitting; Big weights
- Overfitting; Big weights
* `[ B ]`


---

**Q:  Which of the following is correct about regularized regression ?**
- It can help with bias trade-off
- No help with model selection
- No help for variance trade-off
- All of above
* `[ A ]`


---

**Q: A good initialization in regular expression always helps. **
- The statement is correct 
- The statement is incorrect 
- Nothing can be said in general
- None of above are correct
* `[ A ]`


---

**Q: What is dropout in deep learning?**
- Training algorithm.
- Regularization technique to reduce overfitting.
- Algorithm to choose learning rate.
- CNN architecture.
* `[ B ]`


---

**Q: What is the Bayes error rate?**
- The rate of error decay during training.
- The fluctuations in the error during training.
- Irreducible error.
- Error mesured on the test set.
* `[ C ]`


---

**Q: How can overfitting be visualized on a graph where x is the size of the training set and y is the error?**
- As the difference between the apparent error and y = 0.
- As the difference between the true error and y = 0.
- As the sum of the true error and apparent error.
- As the difference between the true error and the apparent error.
* `[ D ]`


---

**Q: What of the following is NOT a way to beat overfitting?**
-  Early stopping.
- Data augmentation.
- Increasing model complexity.
- Feature reduction.
* `[ C ]`


---

**Q: Which of the statements below is true regarding the Bayes error?**
- Bayes error  is the minimum error a classifier can achieve.
- Bayes error is the maximum error a classifier can achieve.
- Bayes error cannot be used as a baseline for comparing error rates of classifiers.
- None of the above.
* `[ A ]`


---

**Q: Which of the following methods can be used to beat overfitting?**
- Data augmentation
- Feature reduction
- Reduction in complexity/flexibility of the model.
- All of the above.
* `[ D ]`


---

**Q: Why do we perform L1 regularization instead of L2 in some cases?**
- L1 performs feature selection during training
- L1 performs better for huge datasets
- L2 is slower than L1
- None of the above
* `[ A ]`


---

**Q: When is it necessary to perform regularization?**
- When the apparent error is too high
- When the model overfits
- When the true error is too low
- None of the above
* `[ B ]`


---

**Q: What are ways of performing "regulariztaion"?**
- Reduce the number of features
- Reduce complexity of the model
- Both are true
- Both are false
* `[ C ]`


---

**Q: What is not an appropriate method to smoothen a time-series**
- Weighted moving average
- Exponentially weighted moving average
- Moving average
- Convolution
* `[ D ]`


---

**Q: Which is NOT the reason why we use data augmentation?**
- Reduce the number of features
- Reduce complexity/flexibility of the model
- Generate more data
- Speed up convergence
* `[ D ]`


---

**Q: How is a good initialisation?**
- Starting with small weights
- Starting with large weights
- Without any initialization
- Starting with negative weights
* `[ A ]`


---

**Q: Which of the following techniques DOES NOT improve training classification models?**
- Noise injection in training dataset
- Batch Normalization
- Regularization on Underfitting classifiers
- Regularization on Overfitting classifiers
* `[ C ]`


---

**Q: Which of the following is true?**
- Adding new features to the model helps reduce overfitting on training dataset.
- Training the model multiple times on a given training dataset improves the model’s accuracy on test dataset.
- The model perceives noise-injected training dataset as new training data.
- Using additional new features for the classifier will always lead to improved performance.
* `[ C ]`


---

**Q: What is NOT a way to avoid overfitting on training data?**
- Using more data
- Reducing  the number of features
- Reducing the complexity of the models
- Regularly testing the model with the test data.
* `[ D ]`


---

**Q: What is regularization?**
- A way to penalize overly complex models
- A way to normalize the data
- wrong answer
- wrong answer
* `[ A ]`


---

**Q: What situation would likely lead to overfitting?**
- Simple model, big data-set
- Complex model, small data-set
- Complex model, big data-set
- Simple model, small data-set
* `[ B ]`


---

**Q: Why can parameter sharing be seen to have a regularizing effect?**
- Because it adds complexity to the model.
- Because, in a sense, it introduces noise into the training.
- Because it encodes prior knowledge into the model.
- Because it reduces the number of parameters in a model.
* `[ D ]`


---

**Q: Which of the following technique is not used to beat overfitting?**
- Adding some noise to the input.
- Feature reduction.
- Regularization.
- Dropout.
* `[ D ]`


---

**Q: Which of the following is WRONG?**
- Adding noise to input is a mean of data augmentation.
- Adding noise with infinitesimal variance is equivalent to weight norm regularization. 
- In early stopping, we always start with larger weights and then decrease them. 
- In early stopping, we can also start with randomly small weight.
* `[ C ]`


---

**Q: One of the challenges of machine learning is combating overfitting? What are ways to deal with overfitting?**
- Perform data augmentation, thus using more data
- Reduce the number of features
- Reduce the complexity/ flexibility of the model
- All of the above
* `[ D ]`


---

**Q: One of the regularization strategies is noise robustness. At what stage can the noise variation be applied?**
- The noise can be applied at the inputs
- The noise can be applied to the weights
- The noise can be injected at the output targets
- The noise can be applied to the input, as well as to the weights and the output
* `[ D ]`


---

**Q: What is NOT the way to overcome the issue of overfitting?**
- Data augmentation
- Reduce the number of features
- Use more complex model
- Parameter norm regularization
* `[ C ]`


---

**Q: What statement is FALSE about the regularization?**
- Regularization is expensive process
- Should be repeated few times to find the stability
- It focuses to overcome issue of overfitting
- Encourages more complex models
* `[ D ]`


---

**Q: What is NOT a form of regularization in Neural Networks**
- Adding a value measure of the weights to the loss function
- Randomly selection a fraction of the nodes of the network and ignore other during training
- Initializing the weights of the network with small values
- Stopping training of the network early based on the training set error
* `[ D ]`


---

**Q: Adding noise to the network during training can help it become more robust. To which variable do we need to add noise to have a regularization effect in the network?**
- Input
- Weights
- Output
- Adding noise will never introduce a regularization effect
* `[ B ]`


---

**Q: What is the purpose of using L2 normalisation when training a neural network?**
- to reduce the number of weights
- to reduce the complexity of a model.
- to reduce the effect of noise on the model.
- to speed up the gradient descent algorithm.
* `[ B ]`


---

**Q: which of the following is not a method to be tried when you notice that you are overfitting to a training set ?**
- Data augmentation
- Feature reduction
- Complexity of model reduction. 
- Complexity of model increase. 
* `[ D ]`


---

**Q: What happens to the learning curve when a more complex model is used in comparison with a less complex model?**
- The curve shifts down and to the right.
- The curve shifts down and to the left.
- The curve shifts up and to the right.
- The curve shifts up and to the left.
* `[ A ]`


---

**Q: Which of the following statements is true?
I Noise added to input can be considered data augmentation
II Noise added to weights encourages stability of the model**
- Only I is true
- Only II is true
- Both statements are true
- None of the statements are true
* `[ C ]`


---

**Q: Why is it discouraged to use large weights when initializing your network?**
- With large weights your network will make big steps which may skip the global minimum.
- With large weights your network might get stuck in a local minimum very fast.
- With large weights the network will become very inconsistent because the weights will barely change after each step.
- With large weights the network will become very inconsistent because the weights will change tremendously after each step.
* `[ C ]`


---

**Q: Why would you apply regularization on your network?**
- In order to limit the complexity/flexibility of the model of your network and reduce the number of features, so you reduce the amount of overfitting.
- In order to limit the complexity/flexibility of the model of your network and reduce the number of features, so you reduce the amount of underfitting.
- In order to regularize the loss of your apparent error, so you can use this to limit the loss on your estimated error by reducing the amount of overfitting.
- In order to regularize the loss of your apparent error, so you can use this to limit the loss on your estimated error by reducing the amount of underfitting.
* `[ A ]`


---

**Q: What is no algorithm suitable to tune the learning rate?**
- SGD
- Momentum
- RMSprop
- Adam
* `[ A ]`


---

**Q: What is not a good approach to reduce overfitting in neural networks?**
- More data (e.g. data augmentation)
- Reduce number of features
- Reduce complexity/flexibility of model
- Adding more edge case scenarios
* `[ D ]`


---

**Q: What is not a way to beat overfitting?**
- Use more data.
- Reduce number of features.
- Reduce complexity of the model.
- Increase flexibility of the model.
* `[ D ]`


---

**Q: In order to battle noisy data, it is important to consider the Noise Robustness. Which of the following is not a location where noise might develop?**
- Gradients
- Input data
- Weights
- Outputs
* `[ A ]`


---

**Q: Which statements are true regarding regarding Regularisation:
- It reduces the number of features
- It increases the flexibility and complexity of the model**
- Only statement 1 is true
- Only statement 2 is true
- Statements 1 & 2 are true
- None are true
* `[ A ]`


---

**Q: What statement is FALSE regarding dropout?**
- During training dropout selects one fraction of the nodes and make them inactive
- Dropout combines weight decay and noise injection
- Weights of active nodes need rescaling due to dropout
- All statements are true
* `[ A ]`


---

**Q: What is not an appropriate way to 'beat' overfitting?**
- Use more data
- Reduce number of features
- Use large weights
- Reduce complexity of the model
* `[ C ]`


---

**Q: Which statement about dropout in neural networks is not correct?**
- Dropout is a combination of weight decay and noise injection
- When using dropout, one is not able to use backpropagation anymore
- With dropout, one makes a fraction of the nodes inactive
- When using dropout, one needs to rescale the nodes that are not dropped out
* `[ B ]`


---

**Q: What point in the figure (import figure from slide 4 of DL05.Regularisation) represents overfitting a system?**
- Point A, right top in image, above true error line
- Point B, right bottom in image, below apparent error line
- Point C, left middle in image, between true error line and apparent error line
- Point D, right middle, at intersection of true error and apparent error line
* `[ C ]`


---

**Q: To avoid overfitting of a neural network on training data, what method can be used to overcome this?**
- Increase the number of features
- Regularisation
- Increase model complexity
- Normalization
* `[ B ]`


---

**Q: Regularization is needed when?**
- With tons of data
- Regularization is always good
- When the model is too complex
- When the model is too simple
* `[ C ]`


---

**Q: Initialization of the weights in an architecture...**
- Does not matter, the network will learn them either way
- Can make a huge difference in performance
- It will only change the performance slightly
- None of the above
* `[ B ]`


---

**Q: About the learning curve of a model, which of the following options is false?**
- The ideal learning curve is when the testing and training learning curves converge at similar values. 
- The gap between the training error curve and the validation error curve tends to become narrow as the training set size increases.
- When training and testing errors converge and are high the performance will improve if we have more data.
- The learning curves allow us to verify when a model is learning as much as it can about the data.
* `[ C ]`


---

**Q: How can you beat overfitting?**
- Reduce the number of features.
- Use regularization techniques.
- Add noise variations in input data, weights and/or outputs.
- All the above.
* `[ D ]`


---

**Q: Which of these statements is incorrect?**
- The higher the degree of a regression model the more prone it is to overfitting
- A small training set is prone to overfitting
- Reducing the number of features makes a model more prone to overfitting
- Reducing the complexity of the model makes it less prone to overfitting
* `[ C ]`


---

**Q: Which of these statements is incorrect?**
- Early stopping is useful when a network is correctly initialised
- Starting with large weights is wrong
- Noise added to the output will decrease the label smoothing
- Noise added to weights encourages stability of the model
* `[ C ]`


---

**Q: When the size of the training set becomes larger and larger, what happens to the apparent and true error?**
- They converge to the same value
- They diverge
- Nothing happens
- None of the above
* `[ A ]`


---

**Q: For what type of networks does 'Dropout' seem effective?**
- Deep networks
- Convolutional networks
- Slime networks
- Wide networks
* `[ D ]`


---

**Q: Which of the following is not a good strategy to combat overfitting**
- increasing the complexity in the model
- Reducing the number of features
- reducing the complexity of the model
- Augmenting the data
* `[ A ]`


---

**Q: What is the risk associated with tuning your model after many tests**
- You might be overfitting your hyperparams on the test set
- Your performance might decrease on the training set
- You need more data for testing
- You might be overfitting your hyperparams on the training set
* `[ A ]`


---

**Q: When is early stopping a good idea?**
- When the network has a tendency to always predict one class
- If random weights are chosen from a uniform distribution (0,1)
- If weights are sampled form a distribution so the sum = 1
- Always, because overfitting has a worse adverse effect than underfitting
* `[ C ]`


---

**Q: What is Dropout?**
- Decreasing all weights every n timesteps to ensure that the sum of weights doesn't grow too large.
- Randomly deleting a fraction p nodes from the network.
- Effectively removing a fraction p nodes from the network by setting the weight to 0.
- Removing a layer from the network.
* `[ C ]`


---

**Q: Which of the following is not a technique to avoid over-fitting?**
- Data augmentation
- Feature reduction
- Model complexity reduction
- The increase of the model’s complexity
* `[ D ]`


---

**Q: When it comes to the learning curve which of the following statements is false?**
- The true error is always bigger than the apparent error
- When the size of the training set is small, over-fitting can occur
- When the size of the training set is small, the true error is small
- When the size of the training set is small, the apparent error is small
* `[ C ]`


---

**Q: Which is not a valid method to beat overfitting?**
- Using data augmentation
- Using regularization
- Reducing the amount of features
- Running the deep net multiple times
* `[ D ]`


---

**Q: Which is required in order to use dataset augmentation**
- Knowing the invariances
- Using linear transformations to generate distortions
- An eigen-representation for each class
- Identical distortion functions for each object
* `[ A ]`


---

**Q: What kind of model behavior can not be observed in learning curve?**
- Underfitting
- Overfitting
- Good fit
- loss function
* `[ D ]`


---

**Q: Which following is not solution to beat overfitting?**
- Increase the training data
- Reduce the complexity of model by regularisation
- Early stopping
-  Generalization
* `[ D ]`


---

**Q: Why is it important for early stopping to start with small weights when initializing a network?**
- When starting with large initial weights the network may take too long to start improving so early stopping gets triggered when it shouldn't, or may fail to learn anything entirely.
- Starting with large weights causes a vanishing gradient
- This is wrong, starting with larger weights improves the training time needed for the network.
- Large weights make it harder for the network to learn subtle differences between training samples
* `[ A ]`


---

**Q: What is meant by parameter sharing in the case of convolutional neural networks?**
- One filter is scanned across the input using the same weights in each place, causing a huge reduction in the number of weights to learn.
- Layers can use the same weights, so the network has to optimize less of them.
- The filters in a CNN can share their weights so they don't all need to learn the same thing.
- Because CNNs use pooling layers, you only need one weight where you'd otherwise need 4.
* `[ A ]`


---

**Q: Why is initialization with large weights often a bad idea?**
- Large weights cause extreme amounts of overfitting.
- Large weights result in a very low model complexity.
- Large weights will introduce a strong bias towards outliers in the model.
- Large weights will cause the gradients to be ill-suited for the optimization procedure.
* `[ D ]`


---

**Q: What is typically not a result of regularization?**
- Reduced model complexity.
- A smaller difference between train and test error.
- A higher train error.
- A higher test error.
* `[ D ]`


---

**Q: Given that a model has no over-fitting (small to zero), what is/are consequences of increasing the complexity of the model?**
- Decrease the Bayes error and possibly overfitting
- Increase the Bayes error and possibly overfitting
- Decrease Bayes error and possibly underfitting
- Increase Bayes error and possibly underfitting
* `[ A ]`


---

**Q: Which statement is true?**
- Variations of input is the only cause of overfitting
- Stability of the model generally increases by adding noise to weights
- Noise with a high variance is approximately the same as weight norm regularization
- When the weights are sufficient small initialized, they always end in the global optimum when infinite iterations are done.
* `[ B ]`


---

**Q: How to beat overfitting?**
- Data augmentation
- Reduce the number of features
- Reduce complexity/flexibility of the model
- All of the above
* `[ D ]`


---

**Q: When will the early stopping be useful?**
- Parameter normalization
- Weight decay
- Correct initialization
- Large training sets
* `[ C ]`


---

**Q: Select the TRUE statement:**
- Regularization is used to prevent underfitting
- Reducing the number of weights by parameter tying is used to prevent overfitting
- Dropout as a regularization technique selects at each training step in the same order a fraction p of the nodes and makes them inactive
- Early stopping is an effective way of preventing overfitting being independent of the weight initialization
* `[ B ]`


---

**Q: Early stopping:**
- Helps finding the global optimum in fewer epochs
- Requires defining the patience hyperparameter (how many epochs with no improvement to wait until stopping) 
- Works even when initializing the weight to large values
- Stops the gradients updates by looking at the performance on the testing dataset
* `[ B ]`


---

**Q: what isn't a problem of a too low learning rate?**
- it requires more data to converge
- it converges too fast
- it tends to get stuck in a local optima
- it takes momentum into account
* `[ B ]`


---

**Q: what is not usually a feature reduction technique**
- applying dimensionality reduction
- rescaling the image
- compressing the dataset
- None of the above
* `[ B ]`


---

**Q: Which statement about overfitting and the learning curve (relating the true and apparent error to the training set size) is false?**
- Underfitting means that the true and apparent error converge to a higher value
- Overfitting implies that the complexity of the model is too high such that the difference between the true and apparent error is high
- More training data necessitates more complex models 
- Less training data necessitates more simple models
* `[ C ]`


---

**Q: Which statement about regularisation is false?**
- Regularisation discourages complexity with the intention of preventing overfitting
- With parameter norm regularisation, weights should be initialised at high values when stating training, since they are pulled towards the origin at every step
- Another form of regularisation is by sharing parameters (with convolution for example)
- One way to prevent overfitting is by stopping training early
* `[ B ]`


---

**Q: What is the effect of regularisation?**
- It tries to combat overfitting with introduction of negative weights in the network
- It pushes the weight vector towards the global minimum
- Considering a classifier with and without regularisation, the one WITH performs bad at infinity but better than the one WITHOUT on a finite training set
- It slows down the network to be more accurate
* `[ C ]`


---

**Q: What is the idea behind dropout?**
- There should not be nodes "too fundamental" for the network
- The weights are calculated only according to the nodes of the last layer
- The less useful nodes are definitively erased from the network
- None of the above
* `[ A ]`


---

**Q: Which one is wrong?**
- Saddle point means that the gradient is zero, but it is neither a local minima
nor a local maxima.
- The problem will result from using a learning rate that’s too high: Cost function does not converge to an optimal solution and can even diverge.
- The problem will result from using a learning rate that’s too low: Cost function may not converge to an optimal solution, or will converge after a very long time. 
- To detect the problems result from using a learning rate that’s too high: look at the costs after each iteration (plot the cost function vs the number of iterations). The cost function decreases very slowly. You could also try higher learning rates to see if the performance improves.
* `[ D ]`


---

**Q: Which one is wrong?**
- when you increase the regularization hyperparameter lambda, the weights will be pushed toward becoming smaller (closer to 0)
- With logistic loss, correctly classified points that are far away from the decision boundary have much less impact on the decision boundary
- In terms of feature selection, L2 regularization is preferred since it comes up with sparse solutions.
- The gradient of L2 loss can grow without bound whereas the L1 loss gradient is bounded, hence the influence of an outlier is limited.
* `[ C ]`


---

**Q: Noise added to the output means:**
- Label smoothing
- Weight norm regularization
- data augmentation
- Increment model stability
* `[ C ]`


---

**Q: What is not true about dropouts**
- Some proof that it works for smaller wide networks and not for slim and deep networks
- Suitable for convolutional networks
- Based on weight decade method 
- There is a need for further research
* `[ B ]`


---

**Q: Which of the following statements about the Regularization in Machine/Deep Learning are true:

statement1: is the method for pinalzaing extra features that you use in your model

statement2: reduces the risk of overfitting without increasig the bias significantly

statement3: most common way of regulirizing is to add a weight decay term to your loss function 

statement4: L2 regularization (operates on the squared amplitude of the weights) penalizes rather smaller weights than larger weights

statement5: L1 regularization (operates on weights linearly) penalizes rather larger weights than smaller weights**
- 1 and 2
- 2 and 3
- 1 2 and 3 
- all statements
* `[ C ]`


---

**Q: There is no final answer to the question whether the goal of the optimisation of a network is to reach the global optimum? 
Which of the following statements can support this claim?

statement1 Finally, we prove that recovering the global minimum becomes harder as the network size increases and that it is in practice irrelevant as global minimum often leads to overfitting

statement2 No person can not say that a funded minimum is global because it might another one find a new minimum that is lower than it. There is no way for know this matter, so we putative the lowest minimum as global minimum.

statement3 The loss landscape can be ridden with local minimum but only a true global minima will guarantee the model will generalize well.

statement4 in theory, one needs to solve very large semidefinite programs to find the global optimum, but in practice, for several instances of polynomial optimization problems, a reasonably sized semidefinite program suffices to compute the global optimum.**
- 1
- 1 and 2 
- 2 and 3
- 4
* `[ D ]`


---

**Q: Which of the following is false:**
- L1 normalization leads to sparse solutions
- Regularization increases the standard loss
- The results generated by early stopping are influenced by the weight initialization
- Data augmentation is sometimes preferred over gathering new data
* `[ D ]`


---

**Q: Overfitting:**
- Is due to using a classifier too simple
- Is signaled by a low training error
- Is signaled by a test error similar to the training error
- Leads to poor generalization.
* `[ D ]`


---

**Q: How can we beat the overfitting phenomenon?**
- Use more data.
- Reduce the number of features.
- Reduce complexity/flexibility of the model.
- All three answers A, B and C are correct.
* `[ D ]`


---

**Q: Complete following statement: "Noise added to weights ... of the model".**
- encourages stability
- discourages stability
- encourages dropout
- discourages dropout
* `[ A ]`


---

**Q: In exponentially weighted moving average (EWMA) bias correction is introduced to ...**
- ... equal out against the border.
- ... equal out against the extreme.
- ... equal out against the weight of your mother.
- ... equal out against parts where there is no slope.
* `[ A ]`


---

**Q: Momentum is...**
- ... the average of the actual gradients with EWMA.
- ... the average of the estimated gradients with EWMA. 
- ... the average of random points with EWMA.
- ... the average of variance with EWMA.
* `[ A ]`


---

**Q: How can you prevent overfitting?**
- Use more data
- Reduce number of features
- Reduce model complexity
- All of the above
* `[ D ]`


---

**Q: Why is starting with small weights good?**
- It is not good, it's better to start with larger weights
- Large weights can saturate the network
- The weight initialization doesn't matter
- Using large weights will not allow the network to converge to a minima
* `[ B ]`


---

**Q: What is \emph{not} a way to beat overfitting?**
- Data augmentation
- Feature reduction
- Reducing complexity/flexibility of the model
- Train for a longer period
* `[ D ]`


---

**Q: Noise robustness refers to **
- data augmentation by adding noise to the input data
- model stabilisation by adding noise to the weights
- label smoothing by adding noise to the outputs
- all of the above
* `[ D ]`


---

**Q: What could be a solution to overfitting in your network?**
- Increase the amount of training weights.
- Increase the training time.
- The addition of a regularizer.
- Decrease the amount of learning data.
* `[ C ]`


---

**Q: If we want to reduce the amount of weights we are using, which regularization method should we utilize?**
- L1 regularization.
- L2 regularization.
- Neither is suitable.
- Both are equally suitable.
* `[ A ]`


---

**Q: Which property does not apply to a Toeplitz matrix?**
- Sparse (many zeros).
- Local (non-zero values occur next to each other).
- Sharing parameters (same values repeat).
- Positive (no negative values).
* `[ D ]`


---

**Q: For what reason would one apply padding to CNNs?**
- To prevent shrinking.
- To increase accuracy.
- To improve computation speed.
- To reduce information loss.
* `[ A ]`


---

**Q: Which of the following statement is not an approach to beat overfitting?**
- Use more data, Data augmentation
- Reduce the number of features
- Increase complexity/flexibility of the model.
- Weight sharing
* `[ C ]`


---

**Q: Which of the following statements for initialisation is not true?**
- For bad initialisation, early stopping useful when network is correctly initialised.
- For bad initialisation, starting with large weights is good.
- For good initialisation, early stopping useful when network is correctly initialised.
- For good initialisation, starting with small weights is good.
* `[ B ]`


---

**Q: What will not be an improvement when tweaking the learning rate parameter?**
- When the learning rate is too large, it might not find a good instiantation 
- When the learning rate is too small, the algorithm might take a long time to converge
- When the variance is different along different dimensions, you might want to have different learning rates along those dimensions
- When you are close to your goal you should increase the learning rate to converge to a good instiantation quicker. 
* `[ D ]`


---

**Q: Which of these statements is true?**
- Complexity of the model reduces the bayes error regardless of the training size set
- Complexity of the model reduces the training size set regardless of the bayes error
- Complexity of the model reduces the true error regardless of the training size set
- Complexity of the model reduces the apparent error regardless of the training size set
* `[ D ]`


---

**Q: What statement about increasing a model's complexity is true?
I: A model with an increased complexity will always achieve a lower true error for the same training set
II: A model with an increased complexity can is more likely to suffer from overfitting for a small training set**
- only statement I is true
- only statement II is true
- both statements are true
- both statements are false
* `[ B ]`


---

**Q: wat is not a valid way to decrease the problem of overfitting?**
- using more data
- reducing the number of features 
- by increasing underfitting
- reducing complexity
* `[ C ]`


---

**Q: Which of the followings is not a method to face overfitting?**
- The usage of more data.
- The reduction of the feature number.
- The reduction of the complexity/flexibility of the model.
- The increasement the number of epochs (when you have to do with iteratively training of an algorithm)
* `[ D ]`


---

**Q: Which of the following statements about early stopping is correct?**
- It stops the gradient updates later than normal.
- It is guaranteed that it finds the global minimum.
- It needs extra parameters if you look at performance on validation set.
- Starting with large weights is wrong.
* `[ D ]`


---

**Q: When is regularisation needed?**
- To tune the learning rate
- To diminish outliers
- To prevent overfitting
- To generate more data
* `[ C ]`


---

**Q: Which statement about dropout is false**
- Dropout can only be applied to hidden layers
- Dropout is a combination of weight decay and noise injection
- Using dropout also means using back progagation
- In dropout we average the outcome over some or all realisations
* `[ A ]`


---

**Q: Which of the following is not true about SGD?**
- The number of updates required to reach convergence usually increases with training set size.
- As the training set size approaches inﬁnity, the model will eventually converge to its best possible test error before SGD has sampled every example in the training set.
- If a model's best possible test error error is achieved using a training set size m, further increase in m will extend the amount of training time required to reach the model's best possible test error.
- None of the above.
* `[ C ]`


---

**Q: Why is it necessary to gradually decrease the learning rate over time in SGD?**
- To achieve better convergence.
- To improve the computational time required to achieve the best test error.
- SGD introduces a source of noise that does not vanish when a minimum is reached.
- All the above
* `[ C ]`


---

**Q: Given a small training set, a very complex model will perform:**
- better than a simplier model
- almost the same as a simple model
- worse than a simplier model
- the error will almpst reach the Bayes error
* `[ C ]`


---

**Q: Why does regularizaion help to overcome overfitting?**
- It limits the weights of the model so that it can not become very complex
- It performs feature selection so that the complexity of the model also reduces
- It helps to stop the training in the proper time when the model is not yet redundantly comples
- All the variants
* `[ A ]`


---

**Q: Which type of regularisation is more likely to result in weights that are closer to the axes?**
- L1 regularisation.
- L2 regularisation.
- Both A and B are equally likely.
- Regularisation cannot influence the distribution of the weights.
* `[ A ]`


---

**Q: What is the purpose of regularisation?**
- To reduce the number of parameters in a neural network.
- To increase the representational space of the network.
- To prevent underfitting by increasing the complexity of the network.
- To prevent overfitting by decreasing the complexity of the network.
* `[ D ]`


---

**Q: Which of the following statement is not correct:**
- Increasing the complexity of the model will decrease the true error
- Rotating the images in the CNN networks is a proper way of data augmentation
- Adding a regularizer with a proper bound can reduce overfitting
-  Early stopping can result in underfitting of the DNN model
* `[ B ]`


---

**Q: Which of the following statement is not correct:**
- initializing the DNN weight with high values make the search less efficient 
- Noise added to input can be considered as data augmentation
- Noise added to the output means label smoothing
- If loss function is changed the regularization need to be changed
* `[ D ]`


---

**Q: We start training a neural network and observe that the loss keeps increasing over time. What can be said about the learning rate?**
- The learning rate is too low.
- The learning rate is too high.
- The learning rate and the loss don't have anything to do with one other.
- The learning rate is good as is - an increasing loss is desirable.
* `[ B ]`


---

**Q: What method is not used for regularization?**
- Dropout
- Data Augmentation
- Batch Normalization
- Weight sharing
* `[ C ]`


---

**Q: Which of the following is not true about dropout?**
- Dropout approximates training a large number of neural networks with different architectures in parallel
- It ignores some number of layer outputs randomly during training
- Dropout results in layer having a different view at each update 
- Dropout technique reduces noise in training process
* `[ D ]`


---

**Q: Which of the following is NOT a way to reduce overfitting in a model?**
- Reduce features
- Increase training set size
- Reduce complexity
- Increase degree of the model
* `[ D ]`


---

**Q: Which of the following is most correct? Early stopping is most useful as regularization when ...**
- the complexity of the neural network is low
- the training set size is large enough
- the initialisation is done with large weights
- the initialisation is done with small weights
* `[ D ]`


---

**Q: Which problem do regularization techniques address?**
- High dimensional data
- Overfitting
- Underfitting
- Low training set size
* `[ B ]`


---

**Q: In a sparse solution some of the weights of a neural network are set to zero using L2 regularisation. This means:**
- In stead of reaching the global minimum, the regularized minimum is confined to a limited number of dimensions.
- The use of the euclidian distance resulted in a limited solution space.
- The use of the rectilinear distance resulted in a limited solution space
- The global optimum is always found as a composition of solely non-zero weights.
* `[ A ]`


---

**Q: The function of the learning rate in a neural network with stoichastic gradient descent is to:**
- Set the stepsize of the updates of the weights during the backward pass of a neural network, based on the loss.
- Determine how fast the neural network converges.
- Determine the noise ratio of the neural network and the gradient descent.
- Determine how many parameters are tuned during the forward pass of a neural network, based on the loss.
* `[ A ]`


---

**Q: Choose the relevant option after the reading the below statements 1 and 2
1) Practical experiences have shown that dropout works well with convolutional network
2) For infinite dataset regularisation is not needed**
- Statement 1 is correct; Statement 2 is wrong
- Statement 1 is wrong; Statement 2 is correct
- Both the statements are correct
- Both the statements are wrong
* `[ B ]`


---

**Q: Overfitting cannot be reduced by which one of the following methods?**
- Reducing the complexity of the model
- Increasing the number of features
- By employing dropout
- Using more training data
* `[ B ]`


---

**Q: True or False: A more complex network architecture is always preferred over a simpler one.**
- True; because a more complex network can represent more detailed features of your data.
- False; because simplicity is a virtue in itself.
- True; because shallow networks always result in a worse classification performance.
- False; because more variable parameters, can lead to over-fitting on the test set.
* `[ D ]`


---

**Q: The key concepts behind L2 regularisation, is...**
- ...the removal of certain nodes at every epoch, to make the system more robust.
- ...enforcing a penalty on large weight norms and hence encouraging the model to find 'simpler' solutions.
- ...subtracting the mean and dividing by the standard deviation of your output, to regularize your loss.
- ...checking the difference between test and training error at every second epoch (regular intervals), to check whether the overall performance is converging.
* `[ B ]`


---

**Q: Which of the following is a way to combat underfitting in a model?**
- Increase complexity (degree) of model 
- Reduce number of features
- Dropout
- Data Augmentation
* `[ A ]`


---

**Q: Which of the following is not true about early stopping?**
- The outcome depends on the initialization of the network
- It is useful when network is correctly initialized
- It stops training when performance on validation set starts to degrade
- It is preferable to start with large weights
* `[ D ]`


---

**Q: What is NOT a method to reduce overfitting?**
- Use data augmentation
- Reduce the number of features
- Reduce the complexity of the model
- Reduce the training error to zero
* `[ D ]`


---

**Q: What is NOT a form of regularization?**
- Noise robustness
- LSTM
- Weight sharing
- Early stopping
* `[ B ]`


---

**Q: what is correct initialization while early stopping? **
- Starting with small weights since both train and test data give similar good performance as learning epochs increase 
- Starting with large weights since train and test data performance are very different. 
- Can start with any weight
- Performance doesn't depend on initialization
* `[ A ]`


---

**Q: which of this is true about regularisation? 
1.Discourages overly complex models 
2.Required when overfitting occurs 
3.Using dropout always reduces complexity **
- 3
- 1&2
- None
- All
* `[ B ]`


---

**Q: Why does deep learning worry about network performance on adverserial data?**
- Adverserial data is the introduction of laplacian noise to input data, DNNs only works as boolean-classifiers on when this type of noise is pressent.
- Introducing underlying noise to an image sometimes results in networks giving a high confidence of a wrong classification, even though humans will still see past the introduced noise (panda_image + noise ->(classifier)-> gibbon_image)
- Adversarial data distorts cost function surface space to be non-continuous, thus disabling backpropagation
- For large data sets batchNorm are a prefered optimization tool, adversarial data groups the input into small batches, which BatchNorm performs badly on.
* `[ B ]`


---

**Q: Which of the underlying methods can be used for regularization in DNN?**
- Tikhonov Regularisation
- Weight decay
- L2 Regularisation
- All of the above
* `[ D ]`


---

**Q: How does initialisation influence regularisation?**
- Starting with large weights and stop early.
- Starting with random weights and stop early.
- Starting with small weights and stop early.
- Optimization will always result in the best weights even when stopping early.
* `[ C ]`


---

**Q: Dropout will improve the performance when:**
- It's done randomly.
- When done on convolutional networks.
- It is one of the following:SpatialDropout, probabilistic weighted pooling, max-drop, cutout
- The network is deep and dropout makes it shallow, it doesn't work for wide networks.
* `[ C ]`


---

**Q: Select the correct characteristics of L2 and L1 norm regularization.**
- [L1] Robust, stable solution, always one solution. 
[L2] Not robust, unstable solution, possibly multiple solutions.
- [L1] Not robust, stable solution, always one solution.
[L2] Robust, unstable solution, possibly multiple solutions.
- [L1] Robust, unstable solution, possibly multiple solutions.
[L2] Not robust, stable solution, always one solution.
- [L1] Robust, unstable solution, always one solution.
[L2] Not robust, stable solution, possible many solutions.
* `[ C ]`


---

**Q: What is can dropout be used to avoid?**
- Over-fitting.
- Under-fitting.
- Increased complexity.
- Reduction of weights to zero.
* `[ A ]`


---

**Q: which of the following statements is wrong?**
- when overfit, the difference between apparent error and true error is high
- small training can ease overfitting
- complex model may has lower bayes error but needs more data to train
- reducing the number of features can beat overfitting
* `[ B ]`


---

**Q: which of the following is not true?**
- adding noise to pictures can be helpful to deal with overfitting
- adding a regularizer will control the complexity of model
- in initialization phase, large or small weight will not lead to any difference
- stop gradient updates early can beat overfitting
* `[ C ]`


---

**Q: What is not a regularization technique?**
- Weight sharing
- Dataset augmentation
- Early stopping
- Holdout
* `[ D ]`


---

**Q: What statement is true?**
- A complex model typically has larger weights and is typically less prone to overfitting than a simple model
- A complex model typically has smaller weights and is typically more prone to overfitting than a simple model
- A complex model typically has larger weights and is typically more prone to overfitting than a simple model
- A complex model typically has smaller weights and is typically less prone to overfitting than a simple model
* `[ C ]`


---

**Q: What is true about the learning rate? A: a low learning rate finds faster a good instantiation than a large learning rate. B: a small learning rate will find a better instantiation than a large one.**
- A is true, B is false
- A is true, B is true
- A is false, B is true
- A is false, B is false
* `[ C ]`


---

**Q: What is a bad way to beat overfitting?**
- Use more data
- Reduce the number of features
- Reduce the complexity of the model
- Increase the flexibility of the model
* `[ D ]`


---

**Q: What is the importance of deep learning**
- Learn many hidden features
- use forward pass
- use backward pass 
- none
* `[ A ]`


---

**Q: how to overcome the problem of overfitting**
- reduce number of features
- reduce complexity
- use more data
- all
* `[ D ]`


---

**Q: Which answer below would be helpful for solving overfiting**
- Data augmentation
- Reduce features
- Redure complexity
- All of the three
* `[ D ]`


---

**Q: According to the generalisation bounds, In a multiple-layer neural networks, which parameter below has the highest influence on performance**
- Input dimension
- Sum of weights
- minimun norm of data vectors
- maximum norm of data vectors
* `[ B ]`


---

**Q: Why is network initialisation with small weights important?**
- Because large weights can cause the network to get stuck at a bad result
- Because regularisation penalises large weights
- It's not, because of weight decay you should initialise with large weights.
- Because large weights are more computationally expensive
* `[ A ]`


---

**Q: What is not a part of bagging?**
- randomly select a fraction ‘p’of the nodes, and make theminactive (set to 0) 
- Injecting noise in the training data
- Perform standard backprop onremaining network
- Average outcome over some/allrealisations
* `[ B ]`


---

**Q: Which of the following statements are correct?**
- Noise added to input can be considered data Augmentation
- Noise added to weights encourages stability of the model
- Noise added to output means label smoothing
- All of the above
* `[ D ]`


---

**Q: A)Training error can be less than bayes error
B)True Error is always greater than Bayes error
Which of the following statements are correct**
- Both statements are correct
- statement B is correct ; statement A is incorrect
- statement A is correct ; statement B is incorrect
- Both the statements are incorrect
* `[ A ]`


---

