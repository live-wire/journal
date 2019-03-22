# Questions from `refresh_and_backprop` :robot: 

**Q: The sigmoid function $\sigma(x) = \frac1{1 + e^{-x}}$ is often used as an activation function in neural networks, not only because it resembles a step function, but also because there is an expression for $\sigma'(x)$ in terms of $\sigma(x)$, it being $\sigma'(x) = $**
- $(1 - \sigma(x))\sigma(x)$
- $\sigma(\sigma(x))$
- $\frac1\sigma(x)$
- $\sigma^{-1}(x)$
* `[ A ]`


---

**Q: The backpropagation algorithm is widely used when updating the weights of a neural network. Apart from being making use of mathematics because "everything is maths", the backpropagation algorithm relies heavily on the fact that**
- updating the weights requires calculating several partial derivatives, many of which can be reused in other auxiliary calculations.
- neural networks can be used to learn any function.
- everything is maths.
- deep neural networks have many hidden layers.
* `[ A ]`


---

**Q: What is the derivative of $f(g(x))'$ according to the chain rule?**
- $f'(g(x))$
- $f(g(x))g'(x)$
- $f'(g(x))g'(x)$
- None of the above
* `[ C ]`


---

**Q: Which of the following statements is true about the cross-entropy equation, $H(p_{data}, p_{model}) = H(p_{data}) + D(p_{data}||p_{model})$?**
- It has no connection with the Kullback-Leibler divergence
- The term $H(p_{data})$ can be omitted
- It is only used within the classification setting
- The aforementioned equation needs to be mazimized
* `[ B ]`


---

**Q: The derivative of a sigmoid function is non-zero between:**
- 0.5 - inf.
-  inf - inf 
- -1 - 1
-  0 - 1
* `[ B ]`


---

**Q: Backpropagation can be done efficiently, because:**
- It uses the chain rule to determine the new parameters.
- Sigmoid functions allow for efficient gradient descent
- In its derivation pre-computed values can be reused.
- The backwards pass only needs to be done once.
* `[ C ]`


---

**Q: How does Backpropagation power Deep Learning?**
- Weights are updated in an efficient random way
- Instead of just forwarding information, a perceptron is also connected to itself, ingesting its own outputs moment after moment as input
- Computed values are re-used to update weights
- Use of GPU to accelerate computing
* `[ C ]`


---

**Q: What is not true about Maximum likelihood estimation?**
- For a given training set it optimizes the parameters in a way that the observed data appears to be most plausible
- To apply a maximum likelihood estimation all data points have to be i.i.d.
- Multiplying the target function of the maximum likelihood estimation changes the desired result
- To make calculation more feasible it is appropriate to use Logarithm in the target function
* `[ C ]`


---

**Q: What is the problem with using a square loss for a classification problem?**
- The gradient of the loss w.r.t. the weights in a network will be linear due to the loss function being quadratic. This is undesirable.
- The loss function punishes predictions with high confidence.
- None of the above.
- All of the above.
* `[ B ]`


---

**Q: What is the advantage of using back-propagation?**
- It is faster than gradient descend.
- The gradient of the loss function will be larger for predictions that are very far off the real value.
- You don’t need to compute the gradient for all the layers.
- None of the above.
* `[ D ]`


---

**Q: Which of the following are equivalent?**
- Minimizing KL divergence, minimizing negative log likelihood, minimizing cross-entropy.
- Maximizing KL divergence, minimizing negative log likelihood, minimizing cross-entropy.
- Maximizing Kl divergence, minimizing negative log likelihood, maximizing cross-entropy.
- Minimizing Kl divergence, maximizing negative log likelihood, maximizing cross-entropy.
* `[ A ]`


---

**Q: Which of the following is FALSE?**
- The mean squared error is the cross-entropy between an empirical distribution and a Gaussian.
- The output units of a deep net determine the form of the cross-entropy function.
- The logistic function is S-shaped.
- The logistic function is a generalization of the softmax function.
* `[ D ]`


---

**Q: Which of the following statement is Not true about multiplying small value?**
- It has the problem of going to zero.
- There is no difference between writing as expectation wrt empirical distribution and taking logrithm, then doing maximum, turning multiplications to sum.
- It can be solved by taking logrithm.
- It can be solved by writing as expectation.
* `[ D ]`


---

**Q: Which of the following statement about backprop is Not true?**
- It relies on chain rule, but ignore the topological ordering of gragh.
- Backprop is an algorithm to compute values.
- Bar notation for computed values:y=dF(y)/dy.
- Less cluttered and emphasizes value re-use.
* `[ A ]`


---

**Q: Which of the following statements about techniques in Machine Learning is incorrect?**
- In a deep net, we aim to optimize the cross-entropy between data distribution and model distribution.
- Rewriting the Maximum Likelihood expression in terms of data samples in the conventional way assumes i.i.d. samples.
- The problem of a typical gradient descent optimizer used on a sigmoid function is that the steps it takes are too large when in the tails of the function.
- The $\max\{0, \max\{1, y\}\}$ function is generally considered unsuitable for optimization by gradient descent.
* `[ C ]`


---

**Q: In the traditional backpropagation algorithm,**
- the gradient of the nodes in a computational graph is computed as a function of its parents.
- the gradient of the nodes in a computational graph is computed as a function of its children.
- the gradient of a node with respect to itself is always 0.
- the backward pass is executed after computing the gradients of each node in the forward pass.
* `[ B ]`


---

**Q: Backpropagation … **
- is just an efficient algorithm to compute gradients
- is used to train a huge majority of deep nets
- is just a clever application of the chain rule
- all of the above
* `[ D ]`


---

**Q: What is the problem with a linear unit P(Y=1|X)=max{0, min{1,wTh+b}} = y for binary classification?**
- no loss defined outside interval [0, 1]
- y should be either 0 or 1, but can be any value in between with the linear unit
- no more gradients outside interval [0,1]
- all of the above
* `[ C ]`


---

**Q: If there is an analog output between 0 and 1, how can we binarize this output into two classes?**
- By flooring the analog output
- By convoluting the analog output
- By thresholding the analog output at 0,5
- It is not possible to binarize an analog value
* `[ C ]`


---

**Q: What is the mean squared error?**
- The cross-entropy between empirical distribution and a Gaussian
- The mean of two errors between the prediction and the actual outcome squared
- The difference between the minimum and maximum likelihood
- The dissimilarity between P_data and P_model
* `[ A ]`


---

**Q: Which of the following cost functions does not works with the gradient descent optimization method:**
- 0-1 loss.
- Squared error.
- Logistic.
- Hinge loss.
* `[ A ]`


---

**Q: Which of the following statements is not true:**
- The regularizer in a deep learning model makes the weights go towards 0 as \lambda increases.
- The backprograpagation algorithm exploits the redundancies that appear while computing the derivatives with the chain rule.
- Backpropagation is an improved alternative to gradient descent.
- Backpropagation is performed after a forward pass.
* `[ C ]`


---

**Q: Why is  backpropagation so efficient?**
- It is based on efficient reuse of previous-evaluated values
- It gives directly the gradient of the loss function with respect to any input
- It is based on efficient linear classifiers
- Allows the training of network with many layers
* `[ A ]`


---

**Q: Which is the problem of implementing gradient descent on a linear model with a logistic non linearity?**
- In some parts of the graph the derivative in zero
- In some parts of the graph the derivative does not exist
- In some parts of the graph the derivative is very small
- There is no problem at all
* `[ C ]`


---

**Q: What could be a reason for replacing a squared loss with the Bernoulli cross-entropy loss?**
- The gradient of a big error (really wrong prediction) could otherwise be too small.
- The gradient of a big error (really wrong prediction) could otherwise be too big.
- Otherwise there will be no gradient outside the [0,1] domain. 
- All of the above reasons are reasons for replacing the squared loss with the Bernoulli cross-entropy loss.
* `[ A ]`


---

**Q: When a model: L=1/2*(sigma(w*x+b)-t)^2 is used, how many unique terms will the derivatives to w and b have together? And why could this be useful?**
- 4, for efficient re-use of the derivatives in the backpropagation algorithm
- 5, for efficient re- use of the derivatives which will result in less computations and a faster program.
- 6, this way, the bar notation method can be used which results in an easier computation of the loss function.
- None of the above. 
* `[ A ]`


---

**Q: Which statements about the backpropagation are true?

statement 1: In the context of learning, backpropagation is commonly used by the gradient descent optimization algorithm to adjust the weight of neurons by calculating the gradient of the loss function.

statement 2: In the context of learning, backpropagation is commonly used by the relaxation algorithms to adjust the weight of neurons in approximation of  originally difficult problem.

statement 3: The  key motivation for backpropagation is to train a multi-layered neural network such that it can learn the appropriate internal representations.

statement 4: The key motivation for backpropagation is to allow it to learn any arbitrary mapping of input to output.**
- 1
- 2
- 1 and 4
- 2 and 3
* `[ C ]`


---

**Q: For deep neural networks, a common activation function is ReLU(x) = max(0, x). If you graph y = ReLU(x) you can see that the function is mostly differentiable.  

Which statements are true about differentitation of RELU?

statement 1: If x is greater than 0 the derivative is 1 

statement 2: If x is less than zero the derivative is 0

statement 3: the derivative of ReLU if x is not equal to zero is different at most of the points

statement 4: When x = 0 the derivative does not exist.**
- 3
- 4 and 1
- 3 and 4
- 1 2 and 4
* `[ D ]`


---

**Q: Machine learning literature focuses on optimizing Cross Entropy to improve a given model. Which of the following tasks is NOT the same as minimizing Cross Entropy?**
- Maximizing Log Likelihood
- Minimizing KL Divergence
- Minimizing Overfitting
- Minimizing Negative Log-Likelihood
* `[ C ]`


---

**Q: Which of the following information can we learn from the topological ordering of a neural network?
I: The precise order in which our nodes are activated in a forward pass of the network
II: An order which we can use for the computations in a backpropogation**
- I
- II
- I and II
- none
* `[ B ]`


---

**Q: Training a network consists of three steps: 1: present a training sample. 2: compare the results. 3: update the weights. What method is used to perform step 3?**
- Backpropagation
- Forward pass
- Idle pass
- Loss calculation
* `[ A ]`


---

**Q: Training a network consists of three steps: 1: present a training sample. 2: compare the results. 3: update the weights. What method is used to perform step 2?**
- Loss calculation
- Backpropagation
- Forward pass
- Idle pass
* `[ A ]`


---

**Q: What is the problem with using the square loss as loss function while using a sigmoidal activation function?**
- Since it maps the output to [-1, 1], it becomes less accurate
- The derivative is small when the loss is large, so the gradient takes small steps with a large error.
- The derivative is non-zero even if y=t, so the gradient still moves if all objects are classified correctly
- All of above
* `[ B ]`


---

**Q: Why is the topological ordering essential for backpropagation?**
- The gradient of a node is calculated using its parent and children nodes
- The gradient of a node is calculated using its parent nodes
- The gradient of a node is calculated using its children nodes
- It is not essential for backpropagation, but reduces redundancy 
* `[ C ]`


---

**Q: Maximizing the LogLikelihood function instead of the Likelihood to find the parameter vector Theta, we obtain:**
- same Theta and same value of the maximum
- same Theta but different value of the maximum
- different Theta but same value of the maximum
- different Theta and different value of the maximum
* `[ B ]`


---

**Q: What is the derivative of $f(g(x))$ with respect to x?**
- $f(x) * g(x) * x$
- $f(g(x)) * g(x)$
- $f’(g(x)) * g’(x)$
- None of the above
* `[ C ]`


---

**Q: "One-hot" encoding refers to:**
- Encoding the output label as follows: (0, .., 0, 1, 0, .., 0) such that only entry k is set to 1 for the k-th target
- Encoding the output label as follows: (1, ..., 1, 0, ..., 0) such that the first k entries are 1 and the rest to 0 for the k-th target
- Encoding the output label as follows: (1, ..., 1, 0, 1, ..., 1) such that only entry k is set to 0 for the k-th target
- Encoding the output label as follows: (k) such that the output is equal to the index of target k
* `[ A ]`


---

**Q: What is FALSE about backpropagation:**
- Backpropagation is an efficient algorithm to compute gradients
- Backpropagation is used to train a huge majority of deep nets
- Backpropagation is an engine that is not 'end-to-end'. It cannot optimize the feature extraction.
- Backpropagation is the application of the chain rule of calculus
* `[ C ]`


---

**Q: When executing gradient descent on a logistic loss function, What can cause problems?**
- The derivative for the logistic function is not defined on the whole domain
- The output is limited to the interval [0,1]
- The derivative of the loss function on output far away from the correct value is almost 0
- The function is log-linear
* `[ C ]`


---

**Q: We can write the maximum likelihood of an i.i.d. sequence drawn from the true but unknown distribution $p_{data}(x)$ as follows: $argmax_\theta \prod^m_{i=1}p_{model}(x^{(i)};\theta)$ Why are we allowed to write it like this?**
- We are sampling from a i.i.d. distribution which means the joint distribution is equal to the product of individual distributions.
-  We are sampling from an i.i.d. distribution which means that the value will go to 0 quickly.
-  Because we in the data we do not depend on $p_{model}$
- Because the maximum likelihood is equal to the cross-entropy
* `[ A ]`


---

**Q: In what way does the backprop algorithm gain efficiency?**
- It is re-using pre-computed values
- It employs a better optimizer
- Prunes the network
- Adds extra layers 
* `[ A ]`


---

**Q:  What to usually optimize in a deep net?**
- The number of nodes 
- Cross-entropy between data distribution and model distribution
- The memory
- The optimizing algorithm
* `[ B ]`


---

**Q: Why is the logarithm used in the Maximum likelihood estimation?**
- To simplify the computation keeping the same position of the maximum
- To reduce the value of the maximum likelihood estimation
- To simplify the computation changing the position of the maximum but keeping the same maximum value
- To increase the number of multiplications 
* `[ A ]`


---

**Q: What is the idea behind backpropagation?**
- To compute all the derivatives for each terms using repeated applications of the univariate Chain Rule
- To randomly perturb one weight and see if it improves performance
- To share the repeated computations wherever possible
- None of the above
* `[ C ]`


---

**Q: What is the preffered appproach to training sigmoid units?**
- squared error
- gradient descent 
- maximum liklihood
- none of the above
* `[ C ]`


---

**Q: Which of the following cost functions gives the best results when used with gradient-based optimization?**
- mean absolute error
- squared error
- cross-entropy
- none of the above
* `[ C ]`


---

**Q: Which of the following statements is FALSE?**
- In practice, minimizing the KL divergence is the same as maximizing the maximum likelihood
- The logistic function has a flowing gradient
- “One-of-K” or “One-hot” encoding can be used to make sure that the gradients flow
- In cross-entropy, the model, $p_{model}$, does not depend on the entropy of the data, $H(p_{data})$
* `[ C ]`


---

**Q: Which of the following statements is FALSE?**
- Backpropagation is “just” a clever application of the chain rule of calculus
- The backpropagation algorithm re-uses pre-computed to speed up the calculation
- Backpropagation powers “end-to-end” optimization and representation learning
- Backpropagation is a gradient descent optimization algorithm
* `[ D ]`


---

**Q: In an IID dispersity model, distribution is **
- dependent of each other
- independent of each other
- mean is zero 
 
- none of these are correct
* `[ B ]`


---

**Q: For gradient descent method,**
- There is a risk of missing minima if step size is large
- weights are updated till they become 0
- weights are updated till gradient is zero
- none of these are correct
* `[ C ]`


---

**Q: In a training session with back propagation, what happens in the backward pass?**
- A new training sample is used.
- Weights are updated.
- The differences of the last backward pass are compared.
- All of the above.
* `[ B ]`


---

**Q: What is a topological ordering of a directed graph?**
- A linear ordering of vertices such that for every edge from vertex $u$ to vertex $v$, $u$ comes before $v$.
- A linear ordering of vertices such that if there is a directed path from vertex $u$ to vertex $v$ and $v$ comes before $u$ in the ordering, then there is also a directed path from vertex $v$ to vertex $u$.
- A linear ordering of vertices such that for every edge from vertex $u$ to vertex $v$, $u$ comes after $v$.
- A linear ordering of vertices such that vertices that are most dependent on other vertices come first.
* `[ A ]`


---

**Q: What does maximum likelihood say?**
- find those parameters (theta) that maximizes the probability of dataset given parameters
- find those parameters that maximizes the log likelihood of dataset given parameters
- A and B
- None
* `[ C ]`


---

**Q: Which of the following does not hold?**
- minimize negative log-likelihood <=> maximize positive log-likelihood
- minimize KL divergence <=> minimize negative log likelihood
- maximize likelihood <=> maximize log likelihood
- maximize cross-entropy <=> maximize log likelihood
* `[ D ]`


---

**Q: What is the main reason why using the cross entropy as loss function is a good idea?**
- From the cross entropy, we can retrieve the correct model in just one iteration
- It gives good penalties for logistic non linearities
- It is easier to calculate than the least squares error
- This is the only loss function that also works in higher dimensions
* `[ B ]`


---

**Q: What is NOT a result of computing the loss derivatives using the chainrule for differentiation?**
- Efficient re-use
- Modularity in the nodes
- Concise notation for gradients
- We no longer need backpropagation
* `[ D ]`


---

**Q: Which of the following sentences about the backpropagation is FALSE?**
- It is a "clever" way of computing the gradients in a deep net
- It is the engine that powers "End-to-End" optimization
- It is used in the way from the input to the output of the network
- It is an application of the chain rule of derivatives
* `[ C ]`


---

**Q: Which of the following procedures is WRONG in a machine learning setting? (in other words leads to a wrong result)**
- Minimize KL divergence
- Minimize negative log likelihood
- Minimize cross-entropy
- Minimize maximum likelihood
* `[ D ]`


---

**Q: Which of the following optimal approach is not equivalent with the other three approaches?**
- Maximize KL divergence
- Minimize negative log likelihood
- Minimize cross-entropy
- Maximize maximum likelihood
* `[ A ]`


---

**Q: For multiclass classification, which of the following strategy is not used?**
- one-hot encoding
- softmax
- use cross-entropy as loss function
- use squared error as loss function
* `[ D ]`


---

**Q: Which of the following statements is true according to the lecture?**
- Maximum likelihood estimation makes no assumption on the distribution of the parametric models.
- Mean squared error is the cross-entropy between empirical distribution and a Gaussian.
- Sigmoid activation function with mean square loss is good enough for the classification problem of Bernoulli distribution because the sigmoid is squashed between 0 and 1.
- Linear activation function is not appropriate for classification on Bernoulli distribution because it's too simple to clearly reveal the underlying distribution.
* `[ B ]`


---

**Q: Why is it better to replace squared loss with Bernoulli cross-entropy loss of classification on Bernoulli distribution?**
- It penalizes very wrong predictions.
- It can speed up the optimization process.
- It penalizes small difference of really wrong predictions。
- A, B and C
* `[ D ]`


---

**Q: How is a linear model with a logistic nonlinearity called?**
- Log-nonlinear
- Log-linear
- Sigmoidal
- A logit
* `[ B ]`


---

**Q: What is NOT true about backpropagation?**
- Backpropagation uses gradient descent to compute gradients
- Backpropagation is used to train a huge majority of deep nets
- Backpropagation powers 'End-to-End' optimization and representation learning
- Backpropagation is a clever application of the chain rule of calculus
* `[ A ]`


---

**Q: What is the disadvantage of using the squared error as a loss function for gradient descent (together with a log-linear model)**
- For very wrong predictions, the squared error becomes smaller and smaller, thus making the correction of the network slower
- The gradient can't always be computed due to discontinuities
- The error gets extremely large with very wrong values, thus making gradient descent unstable
- The more wrong a prediction is, the smaller is the gradient of the loss function, thus making the correction of the network slower.
* `[ D ]`


---

**Q: What makes Backpropagation efficient?**
- The usage of partial derivatives and the storage of intermediate results for further use
- losses are stored and re-used later on to improve the network even more (memory-effect)
- Backpropagation uses purely linear models. It is much easier to take the derivative of linear models.
- The usage of partial derivatives 
* `[ A ]`


---

**Q: Why would we want a node-centered point of view when doing back-propagation?**
- We only needs to compute derivatives with regard to its arguments
- The output of the network will be squashed between 0 and 1
- The network will become topologically ordered
- Only then we can take the derivative
* `[ A ]`


---

**Q: What is the benefit of using the log-likelihood in stead of the regular maximum likelihood estimation?**
- Faster to calculate
- We can only then classify multi-class problems
- That way we can solve the XOR problem
- One avoids problems with small input values
* `[ D ]`


---

**Q: Consider the following feedforward network (it may help to draw it out). The first node has 2 inputs, x and y. This node computes the z = wx+y. This output z feeds into a second node which calculates L = -z. The output L of this node is the output for the network. What is the gradient at input x with respect to the final output L?**
- w
- -w
- 1
- -1
* `[ B ]`


---

**Q: Consider a node the first layer of any arbitrary, multilayered feedforward network with inputs x and y, that computes some function f(x,y), and has a single output L. Which derivative can be calculated during the FOWARD pass?**
- df(x,y)/dx
- dL/dx
- dL/df(x,y)
- None. All derivatives have to be calculated during the backwards pass.
* `[ A ]`


---

**Q: According to the definition of cross-entropy, which of the following statements is TRUE?**
- For a random variable X, the expectation of all possible values of information (E[I(x)]) is called entropy.
- For a random variable X, the larger the entropy, the more certain the value of the variable
- Suppose X is a discrete random variable whose set of values is χ, the probability distribution function is p(x)=Pr(X=x), x∈ χ, We define the amount of information for the event X=x0 as:
I(x0)=−log(p(x0)), it can be understood that the greater the probability that an event occurs, the greater the amount of information it carries.
- The results obtained using cross entropy are inconsistent with the results obtained by the maximum likelihood estimation method.
* `[ A ]`


---

**Q: What is the problem when we use sigmoidal(S-shaped) as the logistic function?**
- It is a nonlinear model
- The derivatives of the function is getting smaller when the prediction is really wrong.
- The large gradient occurs when the prediction is really far away its true class.
- it cannot be classified with two classes
* `[ B ]`


---

**Q: In backpropagation, what does $\bar{w1}$ mean?**
- The partial derivative of the loss function to w1;
- The partial derivative of w1 to w1;
- The partial derivative of w1 to the loss function;
- There is not enough information to determine the meaning.
* `[ A ]`


---

**Q: Consider the following forward pass network: 
$L = \frac{1}{2}(y-t)^2$, 
$ y = \sigma(z) $, 
$ z = wx + b $. 

What would $\bar{w}$ be?**
- $\bar{z}x$
- 1
- $x + b$
- $w + \bar{x}$
* `[ A ]`


---

**Q: We have an Bernoulli distribution on P(Y = 1|X), single number between [0, 1]. Which can be divided between the following two classes: P(Y = 0|X) = 1 - P(Y = 1|X) and P(Y = 1|X). From this we have the following linear unit: $ P(Y = 1|X) = max{0,min{1,w^{T}h + b}} = y$. for features h. The problem with this linear unit is that it cannot be optimized by gradient descent as the gradients outside interval [0, 1] will be all. So the question for this is how can we make the gradient flow for this again?**
-  We squash all the outputs between 0 and 1 which results in a logistic function that is sinusoidal $  \sigma{z} =  sin{z} $
- We squash all the outputs between 0 and 1 which results in a logistic function that is sigmoidal $  \sigma{z} = frac{1}{1+ e^{-z}}$
- We squash all the outputs between 0 and 1 which results in a logistic function that is e (mathematical constant) $  \sigma{z} = e^{-z} $
- We squash all the outputs between 0 and 1 which results in a logistic function that is logarithmic $ \sigma{z} = ln{z} $
* `[ B ]`


---

**Q: We have the following model for the loss L derivatives: $ L = \frac{1}{2}(\sigma(wx + b)-t)^{2} $. Which can be decomposed in $ z = wx + b, y = \sigma{z}, L = frac{1}{2}(y - t)^{2} $. For which you calculate the Derivative to w and Derivative to b. What are the number of unique terms to compute both derivatives?**
-  2. $ 1: \frac{dz}{dw} and 2: \frac{dz}{db}  $
-  4. $ 1: \frac{dL}{dY} = y- t $, $ 2: \frac{dL}{dz} = \frac{dL}{dY} \sigma^{'}(z) $, $ 3: \frac{dL}{dw} = \frac{dL}{dz}x $, $ 4: \frac{dL}{db} = \frac{dL}{dz} $
-  3. $ 1: \frac{dL}{dw}$, $ 2: \frac{dL}{db}$, $ 3: \frac{dL}{dx}$
-  0.  There are no unique terms
* `[ B ]`


---

**Q: Which rule mainly makes it possible to get derivatives from different functions in a neural network? This is used for backpropagation.**
- Quotient rule
- Chain rule
- Reciprocal rule
- Generalized power rule
* `[ B ]`


---

**Q: What is the main advantage of the node-centered point of view of neural networks?**
- Every node had a clear understandable meaning
- The total function of a neural network can be more complex because of the number of nodes
- The derivatives are easier to calculate because they only have to be computed with respect to the node's arguments
- Multiple input variables can be dealt with because of the number of nodes
* `[ C ]`


---

**Q: How can you increase the value of the output layer of a neural network?**
- Increase bias
- Increase weights
- Change activation function
- All of the above
* `[ D ]`


---

**Q: What properties must a loss function have to be used in back propagation?**
- It can be written as an average over error functions for a number of samples.
- It can be written as a function of the outputs of a neural network.
- It must have a derivative.
- Both A and B.
* `[ D ]`


---

**Q: What is one of the problems that arises when attempting to optimise the Mean Squared Error (MSE) using gradient descent in classification problems? **
- The MSE is not differentiable
- The MSE  doesn’t distinguish bad predictions from extremely bad predictions 
- Optimising the MSE is very computationally expensive 
- The gradient of MSE is zero almost everywhere, therefore Gradient Descent will often get stuck.
* `[ B ]`


---

**Q: What is the purpose of adding a regularisation parameter in the cost function of backpropagation?**
- It ensures that the gradient descent will converge to the global maximum.
- It helps speed up the forward pass in backpropagation.
- It ensures that the cost function will have a convex shape.
- It ensures that the parameters of the neural network stay smal .
* `[ D ]`


---

**Q: Minimize KL divergence corresponds to ..**
- Maximize cross-entropy
- Minimize maximum likelihood
- Minimize cross-entropy
- Maximize negative log likelihood
* `[ C ]`


---

**Q: In the Backpropagation method the gradient is calculated in which of these functions?**
- Activation function
- Loss function
- Sigmoid function
- Likelihood function
* `[ B ]`


---

**Q: Deep learning is often called an "End to End" optimization because**
- It requires a careful analysis of the inputs and output
- The term avoids a conflation with other machine learning subsections.
- It is able to generate output labels from the input labels without having to explicitly handle feature extraction.
- None of the above
* `[ C ]`


---

**Q: Which of the following best describes the difference between stochastic gradient descent and mini-batch gradient descent?**
- Stochastic gradient descent always uses one sample from the training  set when evaluating the gradient while the mini batch uses all samples
- Stochastic gradient descent always uses all samples from the training  set when evaluating the gradient while the mini batch uses one sample
- Stochastic gradient descent always uses one sample from the training  set when evaluating the gradient while the mini batch uses one or more samples
- Stochastic gradient descent always uses all samples from the training  set when evaluating the gradient while the mini batch uses one sample
* `[ C ]`


---

**Q: What sentence describes "Maximum Likelihood Estimation"?**
- Find value(s) of parameters for which the given samples are most likely to occur.
- Find the action which is estimated to be most probable to lead to the best result.
- Find estimations of samples that are most likely to occur.
- None of the above.
* `[ A ]`


---

**Q: Why is the "Chain Rule" used when doing backpropagation in Neural Networks?**
- We cannot calculate derivatives of all parameters at the same time, thus we calculate the derivates of all parameters sequentially in a chain.
- We cannot find the minimum in (Stochastic) Gradient Descent in one step, thus we use a chain of steps.
- We cannot (easily) calculate the derivative of one parameter without decomposing it into its factors.
- None of the above.
* `[ C ]`


---

**Q:  Why in back-propagation the square of the error is used?**
- penalize positive and negative error so they don't cancel each other
- Easier derivative compared to absolute error
- Increase the size of small errors to a noticeable value
- 1 & 2
* `[ D ]`


---

**Q: Which of the following statement is correct?**
- Gradient decent can get stuck in a local min/max
- A large learning rate might cause the model to not converge
- Using stochastic gradient decent we can store some of the gradient terms and reuse them for calculating other gradients
- All above
* `[ D ]`


---

**Q: What is the main problem of squared ergo loos with logistic nonlinearity for positive samples?**
- Small gradient small step
- Large gradient small step
- Small gradient large step
- Large gradient large step
* `[ A ]`


---

**Q: What is the correct sequence for training a network using backpropagation?**
- Backward pass, Compute the loss, Forward pass
- Backward pass, Forward pass, Compute the loss
- Forward pass, Compute the loss, Backward pass
- Forward pass, Backward pass, Compute the loss
* `[ C ]`


---

**Q: To calculate the backpropagation the order of calculations should be:**
- canonical
- topological
- chronological
- causal
* `[ B ]`


---

**Q: The goal of calculating the backward pass is to**
- update the weights
- calculate the loss
- none of the above
- all of the above
* `[ A ]`


---

**Q: Minimizing the KL divergence is equivalent to?**
- Minimizing the negative log likelihood
- Minimizing cross-entropy
- Maximizing the maximum likelihood
- All of the above
* `[ D ]`


---

**Q: Given the vector {a, b, c} obtained by the output of some softmax computation. What is true?**
- If a > 0.5, then b > 0
- a + b + c can be greater than 1
- a + b + c = 1
- If a = 0 then b > 0 and c > 0
* `[ C ]`


---

**Q: When training a network, while the results have not converged, which step results in the backwards pass?**
- Presenting a training sample
- Comparing the model output to the sample
- Updating the weights
- None of the above
* `[ C ]`


---

**Q: Having a deep chain function f(x) = f(g(t), h(t)), which is the correct expression for df/dt and which values can be stored?**
- df/dt = df/dg * dg/dt - df/dh * dh/dt. The terms dg/dt and dh/dt should be stored.
- df/dt = df/dg * dg/dt + df/dh * dh/dt. The terms df/dg and df/dh should be stored.
- df/dt = df/dg * dg/dt + df/dh * dh/dt. The terms dg/dt and dh/dt should be stored.
- df/dt = df/dg * dg/dt - df/dh * dh/dt. The terms dg/dt and dh/dt should be stored.
* `[ B ]`


---

**Q: Which of the following statements is false?**
- minimize KL divergence \Leftrightarrow minimize cross-entropy
- minimize KL divergence \Leftrightarrow maximize maximum likelihood
- minimize KL divergence \Leftrightarrow minimize negative log likelihood
- None of the above
* `[ D ]`


---

**Q: Which of the following statements about cross entropy as a loss function in a softmax activation layer is false?**
- The output of the function is always positive
- Minimizing the cost function results in better models
- The loss functions output lays in the range [0 \infty)
- The softmax function has input (z1, z2, ..., zn). If a z is much larger, the output approximates the arg max()
* `[ C ]`


---

**Q: Why is using the Maximum Log-Likelihood sometimes more preferable than the “normal” Maximum Likelihood when using small probability values?**
- a) the normal maximum likelihood does not always find the correct maximum
- b) the log-likelihood turns multiplications into sums for the same maximum
- c) the maximum log-likelihood is a strictly decreasing function
- d) the maximum log-likelihood is not more preferable than the Maximum Likelihood
* `[ B ]`


---

**Q: When implementing back-propagation, choose the right algorithm to implement?**
- a) evaluate for every node its function, calculate gradient of each node, calculate gradient of each child node, create topological ordering of graph.
- b) create topological ordering of graph, evaluate for every node its function, calculate gradient of each node, calculate gradient of each child node.
- c) calculate gradient of each child node, calculate gradient of each child node, create topological ordering of graph, evaluate for every node its function.
- d) none of the above are correct.
* `[ B ]`


---

**Q: When considering a binary classification, it is sufficient to know a single number between $[0,1]$ for two classes. Which of the following four equations describes this statement mathematically?**
- $P(Y=0|X)=1-P(Y=1|X)$
- $P(Y=0|X)=P(Y=1|X)$
- $P(Y=0|X)=1-P(X=1|Y)$
- $P(Y=0|X)=P(X=1|Y)$
* `[ A ]`


---

**Q: Consider a network that is being trained. At a certain moment, the weights should be updated based on the loss. This is called the backward pass. How can one describe this backward pass mathematically?**
- It is based on regression.
- It is based on Thales's theorem.
- It is based on the chain rule.
- None of the above answers.
* `[ C ]`


---

**Q: A node in the computational graph of a feedforward network can NOT be seen as a:**
- scalar
- vector
- tensor
- stochastic process
* `[ D ]`


---

**Q: Which property of backpropagation makes it such a favourable method?**
- Derivatives can be re-used in an efficient way
- By propagating backwards, the weights can be calculated more accurately
- It saves time as no forward pass is needed at all
- Backpropagation transfers the entire system to the Laplace space to solve the derivatives in an elegant and simple way
* `[ A ]`


---

**Q: Choose the correct statement**
- In the computation graph the nodes correspond to computed values
- The backpropagation algorithm propagates information from the input of the system
- In the computation graph the flow of the information is equivalent to the direction of edges
- Each node aggregates the error signal from its parent(s)
* `[ A ]`


---

**Q: The regularizer**
- is used to penalize more complex models.
- is used to limit the value of an output between two values.
- is used to preprocess data so linear net can handle more complex tasks.
- thresholds the outputs of the nodes in the computational graph.
* `[ A ]`


---

**Q: Between which distributions is cross-entropy usually applied to optimize deep networks:**
- Train data distribution and model distribution
- Test data distribution and model distribution
- Train data distribution and test data distribution
- None of the Above
* `[ A ]`


---

**Q: Which statement about Backpropagation is FALSE:**
- The forward pass is used to compute the values of the output layers from the input data
- The backward pass is used to update the weights of the model
- Each node in the network aggregates the error signal from its children
- In the backward pass, the update of a node is only dependant on the values of its children
* `[ D ]`


---

**Q: Which of the following statements is false?**
- In a deep net, we want to optimize the cross-entropy between data distribution and model distribution.
- Input units determine the form of the cross-entropy function
- Minimize KL divergence $\leftrightarrow$ minimize negative log likelihood $\leftrightarrow$ minimize cross-entropy $\leftrightarrow$ maximize maximum likelihood.
- Dissimilarity between $\hat{p}_{data}$ and $p_{model}$ can be measured by KL divergence.
* `[ B ]`


---

**Q: Given is that $z = wx + b$ and $R = \frac{1}{2}w^{2}$. Furthermore it is known that $z$ is the child of $b$ and that $w$ is a parent of $z$ and $R$. What are the values of $\Bar{w}$ and $\Bar{b}$ if backward pass is used?**
- $\Bar{w} = \Bar{z}x\Bar{R}w$ and $\Bar{b} = \Bar{z}$
- $\Bar{w} = \Bar{z}x+\Bar{R}w$ and $\Bar{b} = \Bar{z}$
-  $\Bar{w} = \Bar{z}w\Bar{R}w$ and $\Bar{b} = \Bar{z}x$
- $\Bar{w} = \Bar{z}w+\Bar{R}w$ and $\Bar{b} = \Bar{z}x$
* `[ B ]`


---

**Q: Which of the following is achieved through Back Propagation?**
- A more efficient computation of derivatives of loss function, using the chain rule.
- Convergence to global optima, by avoiding the local optima in all cases.
- A stable Reinforcement Learning algorithm, based on the a feedback on current output.
- A simple and generalized Neural Network because, by applying the delta to only input and output layers.
* `[ A ]`


---

**Q: Which of the following is not the promise of artificial neural network?**
- It can explain the results.
- It can survive the failure of some nodes
- It can handle noise.
- All of the above.
* `[ A ]`


---

**Q: What is not true about the backpropagation algorithm?**
- It is very efficient on large networks by reusing calculated values
- It powers end-to-end optimization and representation learning
- It it very efficient because it allows parallelization of node calculations
- It is a clever application of the chain rule of calculus
* `[ C ]`


---

**Q: What is the number of unique terms necessary to compute the derivatives for back propagation?**
- 2
- 4
- 5
- 7
* `[ B ]`


---

**Q: What is the formula for the Bernoulli cross entropy loss?**
- -t log y - (1-t) log(1-y)
- -t log y + (1-t) log(1-y)
- -t log y - (1-t) log(-y)
- -t log y + (1-t) log(-y)
* `[ A ]`


---

**Q: Which of the following is not a step in the back propagation algorithm?**
- Creating a topological ordering of the graph
- Evaluate all n_i using its function f^(i) (n_i)
- Setting the derivative of the last node to 0
- Setting the derivative of every node to the sum of the derivatives of its child nodes X the derivative of the child node over its parent node
* `[ C ]`


---

**Q: What should be the topological order of the directed graph of a feed-forward network ?**
- w, b, x, z, y, t, L, R, Lreg
- w, b, x, z, y, t, L, Lreg, R
- w, b, x, z, y, t, R, L, Lreg
- w, b, x, z, y, R, L, Lreg, t
* `[ A ]`


---

**Q: How many unique terms have to be calculated in backpropagation algortihm ?**
- 3
- 4
- 5
- 6
* `[ B ]`


---

**Q: How topological ordering of function graph is used in backward pass phase of Backpropagation algorithm? **
- It shows the order of how one should chain single derivatives.
- It shows the order in which each individual derivative should be calculated.
- It shows the order of functions for forward pass and then backward pass is applied
- It provides a graph of function derivatives which is then used for backward pass
* `[ A ]`


---

**Q: What property of deep neural networks makes scientists to look for algorithms such as Backpropagation?**
- Deriving gradients for network optimization results in a lot of repeating terms, which can be reused.
- Deep networks usually has a lot of layers which makes them computationally expensive
- Deep networks usually takes a long time to train which makes them unusable
- Backpropagation cannot be associated with deep neural networks.
* `[ A ]`


---

**Q: Which of the following is WRONG?**
- Maximum likelihood measures the similarity between the data and the model.
- To maximize the Maximum likelihood, we can minimize the KL divergence.
- We use the Bernoulli cross entropy loss in logistic regression because we want to penalize really wrong predictions which is a weakness of using a square loss.
- In multi-class classification with K classes, we use "One-of-K" method to output the predicted class with label 1 and the rest classes with label 0.
* `[ D ]`


---

**Q: Which of the following is CORRECT?**
- Backpropagation is backward pass which is alternative way of doing forward pass.
- In backward pass, we compute and update the weights backwards. While in forward pass, we compute and update the weights forwards.
- Backpropagation requires all functions to be differentiable because it computes the derivatives of them.
- The computational graph indicates the topological ordering and not all nodes on it are differentiable. 
* `[ D ]`


---

**Q: Backpropagation is considered to be an engine that powers ``End-to-End'' optimization. What does the term ``End-to-End'' mean in this context?**
- The network accepts input from one end and produces output at the other end.
- Backpropagation works from one end to the network to the other.
- The output of such networks can directly be used by end-users.
- The network tries to optimize from the first layer to the last layer.
* `[ A ]`


---

**Q: Which of the following statements is false about backpropagation?**
- During the backward pass, previously computed values can be reused.
- There are two pass-types.
- It allows for modularity.
- During the backward pass, the loss is computed.
* `[ D ]`


---

**Q: Why is it possible to transform the product of maximum likelihood, to the sum of log maximum likelihoods?**
- Because the product goes to 0 and the logs also go to zero if you add enough items.
- Because the end result of the sum is the same as the end result of the product.
- Because log(A) > log(B) iff A > B
- Because log(A) < log(B) iff A > B
* `[ C ]`


---

**Q: Which statement about outliers and the results of a max log likelihood classifier are correct.
1. If we take it out the classifier will not be representative of the data any more so it will be less effective.
2. Because we take the log it has an extremely high influence on the result**
- Both are NOT correct
- Only 1 is correct.
- Only 2 is correct
- Both are correct
* `[ A ]`


---

**Q: Which of the following processes is not equivalent to the others?**
- Maximizing maximum likelihood
- Minimizing negative log likelihood
- Minimizing cross-entropy
- Maximizing cross-entropy
* `[ D ]`


---

**Q: Which of the following statements for backpropagation is false:**
- Gradient descent with backpropagation is guaranteed to find the global minimum of the error function
- Backpropagation is an efficient algorithm to compute gradients
- Backpropagation is "just" a clever application of the chain rule of calculus
- Backpropagation is used to train a huge majority of deep nets
* `[ A ]`


---

**Q: Is it possible to implement the back-propagation as message passing?**
- No, because of the dependencies.
- No, just for the first two layers.
- Yes
- No, because of numerical instability.
* `[ C ]`


---

**Q: What is the derivative of the ReLu function?**
- f'(x) = {0 if x=<0, else 1}
- f'(x) = {0 if x=<0, else x}
- f'(x) = {-x if x=<0, else 1}
- f'(x) = {0 if x=<0, else -x}
* `[ A ]`


---

**Q: What is typically used in a multi-class classification problem?**
- Targets are encoded using “One-hot” and the function used at output is softmax
- Targets are encoded using “One-hot” and the function used at output is sigmoid
- The loss function used is binary cross-entropy and the outputs are positive and sums to 1
- The loss function used is categorical cross-entropy and the outputs are positive and sums to the number of classes
* `[ A ]`


---

**Q: What is the FALSE statement regarding backpropagation?**
- It uses the topological ordering of the computational graph
- It efficiently re-uses already computed values
- It uses the chain rule of calculus
- It uses gradient ascent to minimize the loss function 
* `[ D ]`


---

**Q: What should a good loss function have**
- Single large errors have large impact on loss value.
Gradient is required over the complete interval
- Single large errors have increasing but limiting effect on the loss value
Gradient is optional over the complete interval
- Single large errors have increasing but limiting effect on the loss value
Gradient is required over the complete interval
- Single large errors have large impact on loss value.
Gradient is optional over the complete interval
* `[ C ]`


---

**Q: What is backpropagation?**
- Algorithm to compute gradients
- Algorithm used in deep learning
- Algorithm used within AI
- Algorithm used in deep learning developed by Google to improve the performance of the deepmind agent
* `[ A ]`


---

**Q: When calculating back propogation, why do we use chain rule and save the intermediate result?**
- Reduce the amount of calculation
- Make the result more accurate
- Make the convergence easier to reach
- Make the model of ML better
* `[ A ]`


---

**Q: What is most common encoding in Multiclass classication？**
- One-hot coding
- Gray coding
- Binary coding
- Decimal
* `[ A ]`


---

**Q: Which one of the following does not optimize cross-entropy?**
- Minimization of cross-entropy
- Minimization of negative log likelihood
- Maximization of KL divergence
- Maximization maximum likelihood
* `[ C ]`


---

**Q: What is the goal of backpropagation?**
- Create topological ordering of graph
- Compute the derivatives $\bar{w}$ and $\bar{b}$
- Calculate the parameters of loss function
- Compute all of the values in a forward pass
* `[ B ]`


---

**Q: Which of the following functions does not squash the output into [0, 1]?**
- Sigmoidal
- max(0, min(1, f(x)))
- hard limit transfer function (limited to [0, 1])
- sign function
* `[ D ]`


---

**Q: Can you come up with a quick solution to compute L loss derivatives of a deep neural network?**
- Efficient value re-use by computing the derivatives backwards using chain rule
- Calculate the derivatives from L completely
- Computing the derivatives in random order with some value-reuse
- None of the above
* `[ A ]`


---

**Q: Which of the following statements does NOT hold true for the back-propagation algorithm?**
- Back propagation requires the activation and error functions to be differentiable
- the back propagation algorithm may not return a weight vector that minimizes the training error for a given neural network
- the initial error weights do not affect the performance of the back-propagation algorithm
- back propagation can perform better when input vectors are normalized
* `[ C ]`


---

**Q: Which of the following statements is NOT a limitation of the back-propagation algorithm?**
- It can get stuck to local minima
- It is sensible to catastrophic forgetting
- The convergence is slow
- It can only converge when the learning rate is constant throughout the training process
* `[ D ]`


---

**Q: What is the problem in using binary classification for optimization?**
- Goes to zero at infinity
- Does not converge
- Gradient cannot be found
- No optimum available
* `[ C ]`


---

**Q: Most commonly used activation function**
- Log-sigmoid transfer function
- Symmetrical
- Positive linear function
- Parabolic
* `[ A ]`


---

**Q: Which following choice about Maximum likelihood estimation is not true:**
- The set of samples x = {x(1),x(2)...x(m)} doesn't need to drawn independently from the true but unknown data generating distribution Pdata(x).
- When calculating the expectation of the dissimilarity, we can leave out the data term logPdata(x) 
- When calculating the expectation of the dissimilarity, we can leave out the data term logPmodel(x)
- Minimize the KL divergence is equal to minimize cross entropy
* `[ C ]`


---

**Q: which following choice about backpropagation is not true:**
- Backpropagation is an algorithm used to compute gradients.
- Backpropagation is used to train a huge majority of deep nets.
- Backpropagation powers "End-to-End" optimization and representation learning.
- Backpropagation is an algorithm used to compute loss.
* `[ D ]`


---

**Q: Which statement is FALSE?**
- mean squared error is the cross-entropy between empirical distribution and a Gaussian
- In deep nets, we usually minimize cross-entropy between data distribution and model distribution
- Logistic regression aims to penalize very wrong predictions
- Term cross-entropy is only reserved for classification and cannot be used with other distributions like softmax or Bernoull
* `[ D ]`


---

**Q: Regarding backpropagation, which statement is TRUE?**
- It is clever application of the chain rule of calculus
- It can be used only with CNNs
- Each node passes message to its children
- Each node aggregates the error signal from its parents
* `[ D ]`


---

**Q: Why is a single number enough to represent the probabilities of each class in a 2 class problem?**
- In a 2 class problem the probabilities are always nonzero and identical
- In a 2 class problem the probabilities are always zero
- In a 2 class problem the probabilities both go to infinity
- The probability of class 2 is equal to 1 minus the probability of class 1, p2 = 1 - p1
* `[ D ]`


---

**Q: What is often the goal of using a logistic regression model over a regular sigmoid?**
- To provide a more exact solution
- To converge to a solution more quickly when close to the correct answer
- To converge to a solution more quickly when far away from the correct answer
- To avoid getting stuck in a local minimum
* `[ C ]`


---

**Q: Which of the following does not lead to a similar result? **
- Maximize KL divergence
- Minimize negative log likelihood
- Minimize cross-entropy
- Maximize maximum likelihood
* `[ A ]`


---

**Q: What would be an example of a so called one-hot encoded vector?**
- [1 1 0 0 1 0]
- [0 0 0 0 0 1]
- [9 3 2 7 8 2]
- [1 1 1 1 1 1]
* `[ B ]`


---

**Q: What is the most important benefit of using the logistic function as activation function?**
- It is faster to compute than the min/max function
- It does not need a threshold value 
- Samples with high confidence have a large loss
- It penalizes very wrong predictions
* `[ D ]`


---

**Q: Given: a=a(b), b=b(c,d), c=c(e,f), d=d(e,f), e=e(g,h) and f =f(g,h), where the notation d=d(e,f) means that ’d’ is dependent on ’e’ and ’f’.  How many unique terms do you need to compute all derivatives?**
- 8
- 10
- 12
- 16
* `[ A ]`


---

**Q: What of the following is a reason why backpropagation is efficient?**
- It reuses values already calculated.
- It is not very efficient.
- It is good at calculating derivatives.
- It can find most values before doing any calculations.
* `[ A ]`


---

**Q: If dL/dx = -x and , dx/dy = 0.5*y^2 then what is dL/dy?**
- y
- -x*0.5*y^2
- -xy
- -y
* `[ B ]`


---

**Q: What is the purpose of creating the topological ordering in the back propagation algorithm?**
- To find the dimensionality of the network.
- To reduce the dimensionality of the network.
- To make sure that the gradients of the children (in the forward pass) are calculated before that of the parents.
- To make sure that the gradients of the parents (in the forward pass) are calculated before that of the children.
* `[ C ]`


---

**Q: What is the benefit of using cross-entropy loss?**
- You get a sizable gradient signal even when the predictions are very wrong.
- It is a model inspired from thermodynamic principles.
- Neither A nor B.
- Both A and B.
* `[ A ]`


---

**Q: What is backpropagation?**
- An efficient algorithm for computing gradients
- A neural net architecture
- A 'smart' activation function
- A way to introduce feedback into a feed forward network
* `[ A ]`


---

**Q: What does the notation $\bar{y} $ mean, in the context of backpropagation**
- The mean of y
- An estimate of y
- The derivative of the loss function to y
- The derivative of y to the loss function
* `[ C ]`


---

**Q: For which of the sentence endings is the following statement wrong: Minimizing the Kullback-Leibler divergence is the same as**
- minimizing the negative log likelihood
- minimizing the cross-entropy
- minimizing the maximum likelihood
- maximizing the log maximum likelihood
* `[ C ]`


---

**Q: Take the model L = \frac1 2(\sigma(wx+b))-t)^2. When decomposed and rewritten with the chain rule, what is the amount of unique terms to compute both derivatives?**
- 2
- 4
- 6
- 8
* `[ B ]`


---

**Q: Why is back propagation an efficient way of learning?**
- It is easy to compute the partial derivatives every time for any size of neural network
- It stores the intermediate values of the derivatives so it does not have to compute those again when needed
- All of the above
- None of the above
* `[ B ]`


---

**Q: How are the maximization of the KL divergence , and the maximization of Maximum Likelihood related?**
- Maximizing KL divergence and Maximum Likelihood is the same
- Maximizing KL divergence gives better results
- Maximizing Maximum Likelihood gives better results
- None of the above
* `[ A ]`


---

**Q: Backpropagation has a

I. Forward pass

II. Backward pass**
- I
- II
- Both
- Neither
* `[ C ]`


---

**Q: Which are true:

I. Minimizing negative log likelihood <=> minimizing cross-entropy

II. Minimizing KL divergence <=> maximizing maximum likelihood**
- I
- II
- Both
- Neither
* `[ C ]`


---

**Q: Following are few statements about backpropagation:
(a) Initially, the weights are set to 0 so that every neuron starts from the same baseline for the weight update
(b) The change in weights in a layer are proportional to it's input
(c) The update made to the weights is proportional to the difference between desired and actual outputs**
- (a), (b) are correct and (c) is wrong
- (a) is wrong and (b),(c) are correct
- All statements are wrong
- All statements are correct
* `[ B ]`


---

**Q: Traditional way to calculate gradients is to write the loss function in terms of weights and perform partial differentiation. Which of the following is correct regarding this method**
- Calculations are very cumbersome
- Calculation procedure involves redundant work
- Final expression contains repeated terms
- All of the above
* `[ D ]`


---

**Q: The backpropagation algorithm can be divided into five steps, choose the option that completes the algorithm in the right order. 
1 – Compare the output and target value and compute error derivative with respect to output activations.
2 – We take the inputs and forward them through the network, layer by layer, to generate output activations and all the activations in the hidden layer.
3 – Update the weights.
4 – Calculate the error derivative with respect to weights, using the previously calculated derivatives for output and all hidden layers.
5 – Backpropagate to compute the derivative of the error with respect to output activations in the previous layer and do the same for the rest of the hidden layers.**
- 2 – 5 – 1 – 4 – 3 
- 2 – 5 – 1 – 3 – 4 
- 2 – 1 – 5 – 4 – 3 
- 2 – 1 – 5 – 3 – 4 
* `[ C ]`


---

**Q: Mark the false sentence. When estimating parameters, the objective is to:**
- Maximize maximum likelihood.
- Minimize the dissimilarity of distributions.
- Maximize cross-entropy.
- Minimize negative log likelihood.
* `[ C ]`


---

**Q: What are general limitations of back propagation rule?**
- local minima problem
- slow convergence
- scaling
- all of the mentioned
* `[ D ]`


---

**Q: Does backpropagaion learning is based on gradient descent along error surface?**
- yes
- no 
- can't be said
- it depends on gradient descent but not error surface
* `[ A ]`


---

**Q: Which cost function is best to classify a binary linear classifier?**
- Classification error loss function
- Surrogate loss function
- Logistic function
- Cross entropy loss function
* `[ D ]`


---

**Q: If the cross-entropy loss function is defined as -log(y) for t=1 and as -log(1-y) for t =0, which of the following statements are equivalent?**
- -t*log(y)  -  (1-t)*log (1–y)
- (1+y)*t*log(1-y)  
- -t*log(y)  -  t*log (1+y)
- - t*log (1+y)
* `[ A ]`


---

**Q: What is the disadvantage of using a square, linear loss-function?**
- We compute high losses for predictions with high confidence
- We compute low losses for predictions with low confidence
- Neither
- Both
* `[ D ]`


---

**Q: Using backpropagation, in the forward pass:**
- We compute all values
- We compute all derivatives
- We execute the backward pass
- Neither
* `[ A ]`


---

**Q: Backpropagation basically involves**
- applying the product rule in differentiation to minimise computational load
- linearisation of functions to find local gradients faster
- using the chain rule in differentiation to minimise computational load
- adding partial derivatives of the loss function in an optimal manner
* `[ C ]`


---

**Q: What step is not essential to backpropagation?**
- present a training sample and obtain an output label
- penalise model complexity to prevent overfitting
- compare result with the actual label
- update the weights of the model
* `[ B ]`


---

**Q: Which is not equal to the others?**
- minimize KL divergence
- minimize negative log likelihood
- maximize cross-entropy
- maximize maximum likelihood
* `[ C ]`


---

**Q: How many unique terms to compute the derivatives to w and b in the model L=0.5*(\sigma(w*x+b)-t)^2?**
- 2
- 4
- 6
- 8
* `[ B ]`


---

**Q: Minimizing the negative log-likelihood is the same as **
- maximizing KL divergence
- maximizing cross-entropy
- minimizing cross-entropy
- none of the above
* `[ C ]`


---

**Q: What is not an issue with gradient descent on the sigmoid activation function?**
- Small gradient when the prediction is really wrong
- Large gradient when the prediction is close to the boundary
- Not differentiable on the entire domain
- The function is non-linear
* `[ C ]`


---

**Q: Disadvantage of maximum likelihood is**
- It is mathematically complex
- It takes a lot of time to perform the computation
- Both A and B are correct
- None of the above
* `[ B ]`


---

**Q: Gradient of 0 is problematic because**
- Learning algorithm has no longer a guide to improve the parameters
- It means that you have reached the optimal solution
- Both A and B are correct
- None of the above
* `[ A ]`


---

**Q: The assumption of i.i.d.:**
- Is not the reason why it is allowed split the data set in a training set and testing set.
- Allows e dissimilarity of distribution between \^p$_{\text{data}}$ and p$_{\text{model}}$
- Variables are mutually dependant.
- Each random variable is drawn from the same probability distribution.
* `[ D ]`


---

**Q: Training a Classifier: which of the following statements is TRUE?**
- Squared error loss function is easier to optimize than cross-entropy (assuming a logistic activation function).
- We can optimize 0-1 loss with gradient descent.
- With a surrogate loss function a really confident prediction will have no consequences on other training samples.
- With logistic nonlinearity the learning algorithm does not have a strong gradient signal for confident wrong predictions.
* `[ D ]`


---

**Q: Statement 1: Minimize KL divergence <--> minimize negative log likelihood <--> minimize cross-entropy <--> maximize maximum likelihood.
Statement 2: Bar notation is used because it clutters less and it emphasizes value re-use**
- Both statements are true
- Statement 1 is true
- Statement 2 is true
- Both statements are false
* `[ A ]`


---

**Q: How many unique terms need to be computed for both derivatives in the backpropagation algorithm minimally? **
- 3
- 4
- 5
- 6
* `[ B ]`


---

**Q: How many distinct terms (for backpropagation) does the following formulas have?
I  dL/dr = dL/de * de/dr 
II dL/dw = dL/de * de/dw**
- 3
- 2
- 4
- 5
* `[ A ]`


---

**Q: What should one do to use 1D gradient descent on a multi-layer network?**
- Nothing
-  Calculate the average derivative.
- Use one coordinate for each weight/bias in all the layers and compute partial derivatives
- Use all of the above methods.
* `[ C ]`


---

**Q: When do you use mean squared loss and when do you use KL divergence loss?**
- They are interchangeable
- Use mean squared loss for regression problems and KL divergence for classification problems
- Use mean squared loss for classification problems and KL divergence for regression problems
- You don't use mean squared loss at all
* `[ B ]`


---

**Q: What is the benefit of backpropagation?**
- More efficient calculation of gradient
- Simplifies code to calculate gradient
- Each calculation is modular and is independent of later calculations
- All of the above
* `[ D ]`


---

**Q: Which one of the following is false regarding the maximum likelihood estimation method?**
- The data is assumed to be identically and independent distributed.
- Log function can be applied to make the computation easier because it is a non monotonous function.
- MLE is used to find the parameter values of an empirical distribution and minimize the dissimilarity between the data and the model.
- MLE is a common method in machine learning.
* `[ B ]`


---

**Q: What is the problem with using squared error loss and gradient descent?**
- Small gradient penalizes small
- Small gradient penalizes large
- It uses Bernoulli cross entropy
- It penalizes small differences of really wrong predictions
* `[ A ]`


---

**Q: What is true about the backwords propegation?**
- Caculate efficient the gradients.
- End-to-end take gradients in steps  (clever-chain rule)
- None of the above 
- All of the above
* `[ D ]`


---

**Q: What is true about the backwords propegation is used?**
- update weights in backwards-pass
- Calculate the Loss function for calculation the weighs
- None of the above 
- All of the above 
* `[ D ]`


---

**Q: what's the disvantage of using the Gradient Descent in a Binary Classification problem with a Sigmoid Function as activation?**
- it can takes small step when the prediction is very wrong
- it can takes large step when the prediction is very wrong
- it often does not find a minimum
- it finds too many solutions
* `[ A ]`


---

**Q: Which of the following statements is true about Gradient Descent Algorithm?**
-  It takes a step proportional to the gradient of the function at the current point
- If the point is nearer to the solution, the resolution will be faster
- It always converge to the global minimum
- The step is proportional to the magnitude of the gradient of the function in the specific point
* `[ D ]`


---

**Q: Which statement is false?**
- Mean squared error is the cross-entropy between empirical distribution and a Gaussian.
- Minimizing cross-entropy has very similar meaning with minimizing maximum likelihood.
- Usually, cross-entropy between data distribution and model distribution will be optimized in the deep net
- A single number is enough for two classes for Bernoulli distribution
* `[ B ]`


---

**Q: Which of the following method has the most unsatisfying performance when minimizing the error rate?**
- 0-1 loss
- Linear regression
- Logistic nonlinearity
- Cross-entropy loss
* `[ A ]`


---

**Q: Which of these statements is not equal to the others?**
- Minimizing KL divergence
- Minimizing negative log likelihood
- Minimizing cross entropy
- Minimizing maximum likelihood
* `[ D ]`


---

**Q: Which assumption on the dataset is assumed for maximum likelihood estimation?**
- The parametric family of probability distributions and the data have the same probability distribution, but are not mutually independent
- The parametric family of probability distributions and the data do not have the same probability distribution,  but they are mutually independent
- The parametric family of probability distributions and the data have the same probability distribution and are mutually independent
- The parametric family of probability distributions and the data neither have the same probability distribution nor are they mutually independent
* `[ C ]`


---

**Q: Maximizing the maximum likelihood is equivalent to:**
- Minimize KL divergence
- Maximize KL divergence
- Maximize the cross-entropy
- Maximize the negative log likelihood
* `[ A ]`


---

**Q: Which of the following statement is true when it comes to back-propagation message passing?**
- Each node passes the error signal to its children
- Each node passes messages to its parents
- Each node aggregates messages from its children
- Each node aggregates error signal from its parents
* `[ B ]`


---

**Q: Which of the following is true about logistic regression**
- Logistic regression can be used only for 2 class classification problem
- Logistic regression can be used only when the classes are linearly separable
- Logistic regression can be used when when the boundary is non linear
- None of the above
* `[ C ]`


---

**Q: Why is the activation function chosen to be continuous (differentiable)?**
- For the back-propagation algorithm to carry out the chain rule effectively
- For the ease of computations
- The activation function need not be differentiable
- To remove non linearity
* `[ A ]`


---

**Q: How can we estimate the parameters of the model?**
- By minimizing maximum likelyhood
- By minimizing KL divergence 
- By maximizing cross-entropy
- By maximizing negative log likelihood
* `[ B ]`


---

**Q: Why do we need to calculate partial derivatives?**
- To perform backpropagation
- Derivatives act as features
- To find the direction to the optima
- All mentioned above
* `[ C ]`


---

**Q: A method used for estimating the values of the parameters of a model is maximum likelihood estimation. These values are found such that the likelihood that the process described by the model produces the observed data is maximized. Which of the following are equivalent to maximizing the maximum likelihood?**
- Minimizing the KL divergence, minimizing the negative likelihood, minimizing the cross entropy
- Maximizing the KL divergence, maximizing the negative likelihood, maximizing the cross entropy
- Maximizing the KL divergence, minimizing the negative likelihood, minimizing the cross entropy
- Minimizing the KL divergence, minimizing the negative likelihood, maximizing the cross entropy
* `[ A ]`


---

**Q: Consider the model $L = 1/2(\sigma(wx+b)-t)^2$, which is decomposed in: $z=wx+b$, $y=\sigma(z)$, $L=1/2(y-t)^2$. How can we write the derivative of the LOSS with respect to the WEIGHT using the chain rule and according to the backpropagation procedure?**
- $\frac{dL}{dw} = \frac{dL}{dy} \frac{dy}{dz} \frac{dz}{db}$
- $\frac{dL}{dw} = \frac{dL}{dy} \frac{dy}{dz} \frac{dz}{dw}$
- $\frac{dL}{dw} = \frac{dz}{dw} \frac{dy}{dz} \frac{dL}{dy} $
- We cannot apply the chain rule in this case.
* `[ B ]`


---

**Q: How does forward and backward propagation work for Neural Networks?**
- Forward propagation is used to compute the loss and backward propagation is used to minimize the loss
- Forward propagation is used to compute the gradients and backward propagation is used to compute the loss
- Backward propagation is used to compute the loss and forward propagation is used to minimize the loss
- Backward propagation is used to compute the gradients and forward propagation is used to compute the loss
* `[ A ]`


---

**Q: What is the difference between the maximum likelihood and minimizing the cross-entropy?**
- There is only a difference in cases with more than 2 classes
- The output of the maximum likelihood depends more on the data than that of the cross-entropy
- The maximum likelihood maximizes the estimation, where the cross-entropy minimizes the error
- There is no difference between the output
* `[ D ]`


---

**Q: Which of the following is true about maximum likelihood estimation?**
- The i.i.d assumption allows us to rewrite the argmax(p(X,theta)) term as a product of individual data samples
- Introducing log allows the values from going to 0 with repeated multiplications of small values
- It aims to reduce the dissimilarity between p(data) and p(model)
- All of the above
* `[ D ]`


---

**Q: Which of the following is not a characteristic back propagation algorithm?**
- It is computationally inefficient 
- It goes not always guarantee a global minima and can get stuck in local minima.
- It uses chain rule to find gradients
- It is used to train majority of deep nets.
* `[ A ]`


---

**Q: What is the effect of taking the logarithm of the likelihood function**
- Moves the value of the maximum to x=0
- Maintains the same maximum value and simplifies calculations
- Maintains the same maximum location and simplifies calculations
- Reduces numerical error by making values used in calculations magnitudes smaller
* `[ C ]`


---

**Q: How does the multivariate chain rule help in back propagation?**
- It lets us use pre computed values in calculations
- It breaks the derivative into smaller equations that are easier to approximate numerically by computers
- It removes the need to calculate derivatives
- It linearizes all functions making them simple additions
* `[ A ]`


---

**Q: In creating a backpropagating algorithm, what is done first?**
- Calculate gradient from last node to a node earlier in the chain.
- A forward pass.
- A backward pass.
- Determine the least-square error at your final node.
* `[ B ]`


---

**Q: The model L = 2(S(wx+b)^2-t)^2, with t = true label, describes loss of your function. What do you need to do to start backpropagating?**
- Get the derivative of L
- Get the derivative of w
- Get the derivative of S
- Get the derivative of t
* `[ B ]`


---

**Q: What is Cross-entropy**
- Cross-entropy is commonly used to quantify the difference between two probability distributions.
- Cross-entropy is commonly used to quantify the overlap between two probability distributions.
- Cross-entropy is commonly used to quantify the difference between two probability networks.
- Cross-entropy is commonly used to quantify the difference between training sets.
* `[ A ]`


---

**Q: Back propagation relies on the following math trick**
- The chain rule
- the product rule 
- substitution
- linear regression
* `[ A ]`


---

**Q:  Following are the statements related to entropy/cross-entropy, which one is wrong?**
- The amount of uncertainty in an entire probability distribution can be  quantified by using the Shannon entropy.
- Shannon entropy give a lower bound on the number of bits needed on average to encode symbols drawn from a distribution.
- Distributions that are nearly deterministic have high entropy.
- A quantity that is closely related to the Kullback-Leibler  divergence is the
cross-entropy. 
* `[ C ]`


---

**Q: Which statement is wrong?**
- Mean squared error is the cross-entropy between the empirical distribution and a Gaussian model
- minimize cross-entropy<=>maximize maximum likelihood
- Backpropagation can be used to train a huge majority of deep nets.
- Hinge loss has the property that it's an lower bound on 0-1 loss.
* `[ D ]`


---

**Q: Which of the following statements is true:**
- Maximizing Log Likelihood is the same as maximising Cross-Entropy
- Maximizing Log Likelihood is the same as minimising Cross-Entropy
- Minimising KL divergence is the same as maximising Cross-Entropy
- Minimising Log Likelihood is the same as minimising KL divergence
* `[ B ]`


---

**Q: If we have a model $L = 1/2*(\sigma(wx+b)-t)^2$ and we decompose that in: $ z=wx+b, y=\sigma(z), L = 1/2(y-t)^2$, what is the derivative of the model with respect to b?**
- $\fraq{dL}{db} = \fraq{dL}{dy} \fraq{dy}{dz} \fraq{dz}{db} 
- $\fraq{dL}{db} = \fraq{dL}{dy} \fraq{dy}{dz} \fraq{db}{dz} 
- $\fraq{dL}{db} = \fraq{dL}{dz} \fraq{dz}{dy} \fraq{dy}{db} 
- $\fraq{dL}{db} = \fraq{dy}{dz} \fraq{dz}{db} \fraq{dL}{db} 
* `[ A ]`


---

**Q: Which of the following statements about information theorem is false?**
- The Shannon entropy of a distribution is the expected amount of information in an event drawn from that distribution. It gives a lower bound on the number of bits (if the logarithm is base 2, otherwise the units are diﬀerent) needed on average to encode symbols drawn from a distribution P.
- If we have two separate probability distributions P(x) and Q(x) over the same random variable x, we can measure how diﬀerent these two distributions are using the Kullback-Leibler (KL) divergence
- A quantity that is closely related to the KL divergence is the cross-entropy H(P, Q) =H(P) +DKL(P ||Q), which is similar to the KL divergence but lacking the term on the left.
- Minimizing the cross-entropy with respect to Q is equivalent to minimizing the Shannon entropy divergence of P.
* `[ D ]`


---

**Q: Which of the following statements about choosing a cost function is false?**
- The classification error loss, or the 0-1 loss is given by L_{0-1}(y,t) = {0 if y=t; 1 otherwise. It is possible to optimize 0-1 loss with gradient descent, but is not an efficient approach. 
- When we replace a loss function we trust with another one we trust less but which is easier to optimize, the replacement one is called a surrogate loss function. 
- The problem with squared error loss in the classification setting is that it doesn’t distinguish bad predictions from extremely bad predictions. 
- The logistic function squashes y to be between 0 and 1, but cross-empty draws big distinctions between probabilities close to 0 or 1. 
* `[ A ]`


---

**Q: In backpropagation**
- The derivatives with respect to the loss function are found to compute the loss
- Each node aggregates the error signal from its children
- Each node passes messages to its children
- The order of computation is independent
* `[ B ]`


---

**Q: The dimension of the vector of derivatives between two hidden layers depends on**
- The dimension of the children layer
- The dimension of the parents layer
- The number of batches
- The dimension of the output
* `[ B ]`


---

**Q: Which of the following statements is true:
A) Backpropagation utilizes the compute values from the forward feed to update the weights using the chain rule.
B) Logistic regresseion penalizes differences in the rating of the predictions.**
- A is correct
- B is correct
- A & B are correct
- Neither are correct
* `[ A ]`


---

**Q: What is meant with overfitting**
- Overfitting is making your Network as good as possible
- Overfitting is when your model can't predict results good enough, because it has had to few examples.
- Overfitting is training your model to precisely fit the training data
- Overfitting is the result of a network that can't predict the results good enough
* `[ C ]`


---

**Q: what is a problem with unbounded loss function**
- It prevents the loss function from being applied
- It allows for bounded loss
- It allows for infinite loss
- There is no problem
* `[ C ]`


---

**Q: What is a solution to the unbounded loss problem**
-  No such problem exists
- squash it to fit between 0 and 1
- scale it over the new bound
- apply a logarithm to limit the loss scaling
* `[ B ]`


---

**Q: Which of following way cannot minimize the dissimilarity of distributions?**
- Minimizing KL divergence
- Minimizing negative log likelihood
- Maximizing cross-entropy
- Maximizing maximum likelihood
* `[ C ]`


---

**Q: What is the problem using square loss in binary classification?**
- Huge penalization for high variance data
- No gradients 
- Cannot be optimized by gradient descent
- Little penalization when prediction is really wrong
* `[ A ]`


---

**Q: The backpropagation algorithm differs from (forward) differentiation by**
- Being more efficient when computing the derivatives for a single output (due to smart reuse of computed values) 
- Being more efficient when computing the derivatives of multiple outputs
- Being possible; Forward differentiation is incomputable when using multi-layer networks
- Being computable for non smooth functions
* `[ A ]`


---

**Q: The derivative of the reLU function**
- Is defined everywhere
- Is undefined at x=0
- is undefined at x > 1
- is undefined at x < 0
* `[ B ]`


---

**Q: Mean squared error is the cross-entropy between which of the following:**
- Gaussian model
- Empirical distribution 
- Both 1 and 2
- None of the above
* `[ C ]`


---

**Q: Which of the following conditions helps in ensuring that the maximum likelihood estimator has the property of consistency?**
- The true distribution p_data must lie within the model family p_model (.;theta)
- The true distribution p_data must correspond to exactly one value of theta. 
- Both 1 and 2
- None of the above
* `[ C ]`


---

**Q: To which terminology corresponds the global step of training a network? Steps: 1) Present a training sample. 2) Compare results. 3) Update Weights. Terminology: a) Forward pass b) Backward pass. c) Loss**
- 1 - b, 2 - c, 3 - a
- 1 - c, 2 - a, 3 - b
- 1 - a, 2 - b, 3 - c
- 1 - a, 2 - c, 3 - b
* `[ D ]`


---

**Q: What is the topological ordering of a computational graph?**
- Linear ordering of the nodes so that for every directed edge uv, node u comes before node v in the ordering
- Linear ordering of the edges so that for every directed edge uv, edge u comes before edge v in the ordering
- Linear ordering of the nodes so that for every edge uv, node u comes before node v in the ordering
- Linear ordering of the edges so that for every edge uv, edge u comes before edge v in the ordering
* `[ A ]`


---

**Q: Which of the following is false:**
- Cross-entropy between data distribution and model distribution is optimized in a deep network
- For y = max{0, min{1, w’h + b}} the gradients outside the [0, 1] interval determine very small adjustments for the weights, where w’ stands for transpose.
- Bernoulli cross-entropy loss penalizes more heavily very wrong predictions
- The assumption of maximum likelihood estimation is that the samples are i.i.d.
* `[ B ]`


---

**Q: Backpropagation:**
- Is the only algorithm for computing gradients of deep networks
- Has the disadvantage of not reusing previous computed terms
- Requires from the node of the network to know the state of the whole network
- Is based on applying the chain rule of calculus
* `[ D ]`


---

**Q: Backpropagation could be formulated as a dynamic programming problem**
- TRUE
- FALSE
- N.A.
- N.A.
* `[ A ]`


---

**Q: Which of these four functions is most suitable for gradient descent optimization?**
- Concave function
- Unit step function
- Logarithmic function
- Square root function
* `[ A ]`


---

**Q: When optimizing cross-entropy, which of the following arguments does NOT belong?**
- Minimize negative log likelihood
- Maximize cross-entropy
- Minimize KL divergence
- Maximize maximum likelihood
* `[ B ]`


---

**Q: What technique can be used to make the calculations done for backpropagation more efficient?**
- Linearize the loss function
- Re-use pre-computed values in the partial derivatives
- Change the topological ordering of the vertices
- Copy the value from the previous node
* `[ B ]`


---

**Q: Logistic function is?**
- Linear
- Sigmoidal (S-shaped)
- Exponential
- Constant
* `[ B ]`


---

**Q: What is backprapagation?**
- Activation function
- Efficient algorithm to calculate gradients
- Clever application of the chain rule of calculus
- B and C
* `[ D ]`


---

**Q: Which of the following is not  the kind of loss function for calculation of error. **
- Logistic non-linearity
- Cross-entropy
- Linear regression
- all 
* `[ D ]`


---

**Q:  which of the following is true abut backpropagation:**
- Effcient algorithm to compute gradients
- Used to train a huge majority of deep nets
- C.	Engine that powers “’End-to-End" optimization and representation learning
- all
* `[ D ]`


---

**Q: It was found that on applying gradient descent, the cost function increased with iteration.So what steps needs to be taken:**
- Change the activation function
- Decrease the learning rate
- Increasing the training sample size
- Increase the number of hidden layers
* `[ B ]`


---

**Q: Gradient descent is the preferred method for backpropagation in deep neural net. So which of the following points can we get struck in when applying Gradient descent
1)Global minima
2)Local Minima
3)Saddle point
4)Local Maxima**
- 1) and 2)
- 4)
- 3) and 4)
- 1), 2) and 3)
* `[ D ]`


---

**Q: What holds true for back propagation nodes?**
- Each node aggregates the error signal from its children
- Each node passes messages to its parents
- Each node passes messages to its children
- Both A and B
* `[ D ]`


---

**Q: What happens during a backward pass?**
- A training sample is presented
- The loss is evaluated
- The results are compared
- The weights are updated
* `[ D ]`


---

**Q: Which of the following statements are true?**
- Minimizing the Kullback–Leibler divergence gives the same result as maximizing the log likelihood.
- Maximizing cross-entropy gives the same result as minimizing the negative log likelihood.
- Maximizing the negative log likelihood gives the same result as minimizing the Kullback-Leibler divergence.
- Minimizing cross-entropy gives the same result as minimizing the log likelihood.
* `[ A ]`


---

**Q: The backpropagation algorithm computes the gradient by

1) Re-using the values of partial derivatives
2) Using the chain rule to increase the accuracy of the derivatives

Which statements are correct?**
- Statement 1
- Statement 2
- Both statements are correct
- Both statements are wrong
* `[ A ]`


---

**Q: Which following statement is wrong about maximum likelihood?**
- ML estimation is to view it as minimizing the dissimilarity between the empirical distribution.
- ML is an attempt to make the model distribution match the empirical distribution.
- The true distribution must lie within the model family.
- The negative log-likelihood can not become negative.
* `[ D ]`


---

**Q: Which following statement is wrong about Output Units?**
- The neural network unit that may be used as an output can also be used as a hidden unit. 
- A sigmoid output unit can return a negative result.
- The softmax output ( softmax(z) = softmax(z + c)) is invariant to adding the same scalar to all its input.
- The softmax function is more closely related to the arg max function than the max function.
* `[ B ]`


---

**Q: Which of the following is NOT an appropriate method to measure the generalisation performance of a classifier from a dataset standpoint?**
- Match the sub-class distribution of a new test set to the original dataset (say CIFAR-10)
- Collect X set of images in the test set, but evaluate X-Y (Where Y<<X) using a classifier of choice.
- Experiment using investigation of multiple hypotheses
- Retain the same test set on multiple classifiers to predict so that the classifiers provide better accuracy
* `[ D ]`


---

**Q: Which of the following are NOT example methods for justifying the significant empirical gains/advances in research papers around ML?**
- Clever problem formulations help in adding the empirical advancements. 
- Using high end optimisation heuristic tasks as part of experimental research
- Using hyper parameter tuning or perform ablation analysis on existing research work.
- Devising complex algorithms and have new models built as part of research and not worrying about their performance. 
* `[ D ]`


---

**Q: Write maximum likelihood model in terms data sample**
- Theta_ML = arg max Product{p_model}
- Theta_ML = arg min Product{p_model}
- Theta_ML = arg max Sum{p_model}
- Theta_ML = arg min Sum{p_model}
* `[ A ]`


---

**Q: Why do you often use log of maximum likelihood estimator?**
- Log shows exponential growth as linear.
- The value of the probabilities are often very large
- The value of the probabilities are often very small
- Maximum likelihood itself is logarithmicly shaped
* `[ C ]`


---

**Q: In which answer are all operations equivalent?**
- Minimize the KL divergence, minimize the cross-entropy, maximize maximum likelihood, maximize negative log-likelihood
- Minimize the KL divergence, maximize the cross-entropy, minimize maximum likelihood, minimize negative log-likelihood
- Minimize the KL divergence, minimize the cross-entropy, maximize negative likelihood, minimize negative log-likelihood
- Minimize the KL divergence, minimize the cross-entropy, maximize log-likelihood, maximize maximum likelihood
* `[ D ]`


---

**Q: What is the best solution to gradients disappearing outside the interval [0,1] using a linear max(min()) unit in Binary Classification?**
- Remove the min/max
- Do not bound the output within [0,1], but continue beyond and set a prediction $y$ threshold at $y = \frac{1}{2}$.
- Use a logistic function to squash the prediction within the interval [0,1].
- None of the above.
* `[ C ]`


---

**Q: What is true regarding cross-entropy?**
- It is the chaos contained in the system.
- Used to identify specifically the negative log-likelihood of a Bernoulli or softmax distribution.
- Any loss consisting of a negative log-likelihood is a crossentropy
between the empirical distribution defined by the training set and the
probability distribution defined by model.
- Mean squared error is NOT the
cross-entropy between the empirical distribution and a Gaussian model.
* `[ C ]`


---

**Q: Linear regression is better than binary?**
- Always
- Often 
- Sometimes
- Never
* `[ D ]`


---

**Q: What is KL-divergence?**
- A certain type of activation function
- A measure of how fast a classifier diverges from the correct label when overfitting.
- A measure of variance in estimates from a certain classifier.
- A measure of dissimilarity between the data distribution and model distribution
* `[ D ]`


---

**Q: look at graph in one slide 14 of backpropagation. Which is not a topological ordering of this graph?**
- (w,b,x,z,y,t,L,R,L_reg)
- (b,x,w,z,y,t,L,R,L_reg)
- (b,x,w,z,y,t,R,L,L_reg)
- (b,x,z,w,y,t,L,R,L_reg)
* `[ D ]`


---

**Q: What is true about back propagation**
- The goal is to update the weights of the network
- The goal is to determining the error
- No initial input is required
- None of the above
* `[ A ]`


---

**Q: What method is used by applying back propagation?**
- Any optimization can be used
- Stochastic gradient descent is often used
- A and B
- None of the above
* `[ B ]`


---

**Q: In maximum likelihood estimation, what is the problem with multiplying the likelihood of the model for all separate data samples, and how is it solved?**
- It is a multiplication of a lot of small (0 < x < 1) values, which goes to zero, and can be fixed by taking the logarithm of the values.
- It is a multiplication of a lot of small (0 < x < 1) values, which goes to zero, and can be fixed by replacing the multiplication by a summation over the logarithms of the values
- The likelihood can become too large since it is multiplied repeatedly and being maximized, causing numeric overflows. It can be fixed using techniques that allow for arbitary sized numbers.
- The likelihood can become too large since it is multiplied repeatedly and being maximized, causing numeric overflows. It can be fixed by employing regularization.
* `[ B ]`


---

**Q: When optimizing using gradient descent, what is a possible problem with using a logistic activation function.**
- The logistic functions gradient gets small pretty fast on large errors, while those are the ones that should be causing big changes, not small. 
- The logistic function is not differentiable.
- The logistic function is computationally expensive, causing an exponential increase in training time.
- The logistic function causes exploding gradients.
* `[ A ]`


---

**Q: Which of the following is true on regard to back propagation?**
- In back propagation, no feedback signal is provided at any stage.
- It is the transmission of error back through the network to adjust the inputs.
- It is the transmission of error back through the network to allow weights to be adjusted so that the network can learn.
- None of the mentioned
* `[ C ]`


---

**Q: Which of the following is true of logistic regression?**
- It can be motivated by ”log odds”.
- The optimal weight vector can be found using Maximum Likelihood Estimation.
- It can be used with L1 regularisation.
- All of above
* `[ D ]`


---

**Q: The problem of choosing the linear regression as a cost function for a binary classification problem is:**
- The discontinuities have as a result that Gradient Descent will change only on the boundaries
- The predictions are real valued rather than binary
- The predictions are allowed to take arbitrary real values which can’t provide any extra information about the binary problem
- The changes between the values of the predictions can’t provide any valuable information about the classification
* `[ C ]`


---

**Q: Backpropagation is a method which : **
- Update weights & bias of each layers based on error at that layer in order to minimize error at layer
- can be characterised as a learning algorithm for neural network
- Is based on linear regression 
- Calculate the error of each layer and corrects the weights every time a sample pass through a layer
* `[ A ]`


---

**Q: The following is not true about back-propagation :**
- Efficient algorithm to compute gradients
- Efficient algorithm to present training samples 
- Uses gradient decent to calculate the error function
- Used to train deep learning networks
* `[ B ]`


---

**Q: A false statement about Bernoulli distribution  :**
- Uses a single number between 0,1 
- Can’t be optimised by gradient descent
- Uses a single number for 2 classes
- Not a binary classification method
* `[ D ]`


---

**Q: Why do we take the sum of logarithms when calculating the Maximum Likelihood?**
- To make its computation faster
- To avoid its tendency to 0 (numerical underflow)
- To reflect cross-entropy minimization
- So the arg max gets closer to the true value
* `[ B ]`


---

**Q: What is the relation between Softmax and the sigmoid / logistic functions?**
- Both represent a probability distribution
- Each variable will be between 0 and 1
- Softmax is a generalization of the logistic function
- All are correct
* `[ D ]`


---

**Q: Let's say random variables X and Y are i.i.d. Which of the following statements hold?**
- $F_X(x) = F_Y(y)$
- $F_{X,Y}(x,y) = F_{X}(x) * F_{Y}(y)$
- Both of the above
- None of the above
* `[ C ]`


---

**Q: Which of the following statements hold regarding backpropagation?**
- Backpropagation finds global optima, rather than local optima
- Backpropagation needs normalized input vectors
- Both of the above
- None of the above
* `[ D ]`


---

**Q: Which of the following statements about log-likelihood is incorrect:**
- A maximum likelihood estimate for a dataset tries to find a theta to maximize the product of p_model(x_i; theta) for each object (x_i) in the dataset.
- Log-likelihood can be used for multiclass problems
- To compute maximize likelihood we generally use gradient descent.
- All are incorrect
* `[ C ]`


---

**Q: Which of of the following statements about logistic regression is correct:**
- It can be used for multiclass classification
- In combination with cross-entropy small difference in the wrong direction are penalized.
- In the multiclass version, the softmax function can be used to give an approximation of normalized probabilities. 
- All of the above
* `[ D ]`


---

**Q: Which of the following statements is/are correct?

1. The use of a square loss function creates the problem that high confidence (correct) classifications creates large losses, which is unwanted.
2. To limit the output to the interval of [0,1], one could use a output unit with the sigmoid function as activation function.**
- Only 1 is correct
- Only 2 is correct
- Both are correct
- Both are incorrect
* `[ C ]`


---

**Q: Which of the following is/are correct?

1.  Backpropagation is an efficient algorithm to calculate gradients and makes use of the product rule of derivatives from calculus.
2.  Modularity refers to the fact that nodes only need to calculate their derivatives with respect to their children nodes.**
- Only 1 is correct
- Only 2 is correct
- Both are correct
- Both are incorrect
* `[ B ]`


---

**Q: When optimizing the cost function the learning rate referes to:**
- introduction of any step size to optimizer of the cost function
- introduction of interative step size to the intercept
- the change in optimizing using stochastic gradient decent instead of regular gradient decent
- the assumption of a given distribution prior to regularize the cost function
* `[ A ]`


---

**Q: Which of the following statements are NOT true for backpropagation**
- Redundant derivatives are only computed once
- Backpropagation is a cleaver use of the calculus - "chain rule"
- Backpropagation is specialized for stochastic gradient descent
- Backprop is an algorithm to compute values
* `[ C ]`


---

**Q: Which of the following description is correct?**
- Loss functions used to computer/compare the result of training network
- Gradient descent can always find a global optimisation, which is its advantage
- The different orientations of detection would lead to different detection results
- As there might be multiple input, operations sometimes can have multiple output
* `[ A ]`


---

**Q: Which of the following description is correct?**
- To ensure the accuracy, we have to calculate every derivatives inside a neural network
- It is not necessary to create topological ordering of graph for back-propagation algorithm
- Different layer’s backward pass must be different
- A node in a neural network could be a scalar, vector or even matrix
* `[ D ]`


---

**Q: In back propagation, z is an important variable defined as z = wx+b. What does 'w' stand for?**
- An array of weights of the previous layer
- An array of weights of the next layer
- An array for the gradient coordinates
- none the above
* `[ A ]`


---

**Q: Forward pass computes one thing (a) and the backward propagation computes something else (b). which is which?**
- (a): loss, (b): derivatives
- (a):derivatives, (b): non-linearisation
- (a): non-linearisation, (b): derivatives
- (a): derivatives, (b): loss
* `[ A ]`


---

**Q: Concerning cross-entropy which of the following terms do not belong?**
- Minimisation of negative log likelihood.
- Minimisation of KL divergence.
- Maximisation of maximum likelihood.
- Minimisation of covariate shift.
* `[ D ]`


---

**Q: Which of the following non-linear activation functions are not present in contemporary use? **
- Sigmoid.
- ReLu.
- Hyperbolic tangent (tanh).
- Himmelblau function.
* `[ D ]`


---

**Q: Probabilities of classes is given by:
\begin{center}
\begin{equation}
 P(Y=k|X=x_{i})= \frac{e^{s_{k}}}{\sum_j{e^{s_{j}}}}
\end{equation}

\end{center}
Scores for three classes are given as follows: \newline
cat: -2.85, dog:0.86, car:0.28. What is the percentage of Softmax loss for class car?**
- 4.52
- 0.452
- 0.353
- 45.2
* `[ D ]`


---

**Q: If f(p,q,r)=(p+q)r and p=-2, q=5 and r=-4 are the inputs in a feed forward neural network. Then what are the updated values of the inputs, after the first iteration in backpropagation?**
- -4, 4, 3
- -4, -4, -3
- -4, -4, 3
- 4, -4, 3
* `[ C ]`


---

**Q: Given a iid parametric family of  probability distributions Pmodel(x;A), which equation below is not a maximum likelihood estimation**
- argmax(A) pmodel(X ; A)
- argmax(A) sum(from i=1~m) log pmodel(x_i;A)
- argmin(A) D(pdata || pmodel(X;A))
- argmin(A) Ex_pdata[log pmodel(x;A)]
* `[ D ]`


---

**Q: Which answer optimize the cross-entropy H(pdata, pmodel)**
- All of below
- minimize H(pdata)
- minimize log likelihood
- minimize D(pdata || pmodel)
* `[ D ]`


---

**Q: Which is a better approach for Bernoulli Output Distribution?**
- Linear unit and threshold the value to be a valid probability
- Sigmoid unit with maximum likelihood
- Sigmoid unit
- None
* `[ B ]`


---

**Q: What is(are) the properties of maximum likelihood?**
- The true distribution pdata must lie within the model family pmodel (.;θ)
- The true distribution pdata must correspond to exactly one value of θ
- Both A & B
- None
* `[ C ]`


---

**Q: What happens to the gradients outside the interval [0,1]**
- Gradient goes to infinity, can be optimized.
- No more gradients, cannot be optimized.
- No more gradients, can thus be optimized.
- Gradient goed to infinity, cannot be optimized.
* `[ B ]`


---

**Q: Which method is used in back-propagation?**
- K-nearest neighbor optimizer
- Stochastic gradient descent optimizer
- Least squares optimizer
- Only B and C
* `[ D ]`


---

**Q:  What is penalized in logistic regression?**
- small differences of really wrong prediction
- big differences of really wrong prediction
- small differences of very right prediction
- big differences of very right prediction
* `[ A ]`


---

**Q: In a loss model L=(1/2)*(sigma(wx+b)-t)2, we look to find the derivatives of w and b by decomposing the equation into different terms. After doing so, how many unique terms can be found in total over both derivative equations?**
- 3
- 4
- 5
- 6
* `[ B ]`


---

