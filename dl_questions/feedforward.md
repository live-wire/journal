# Questions from `feedforward` :robot: 

**Q: Which of the following about stochastic gradient descent (SGD) is true?**
- The cost per SGD update depends on the training set size m.
- The number of updates required to reach convergence does not increase with the training set size.
- SGD follows the estimated gradient uphill.
- On each step of the algorithm, we can sample a set of examples drawn uniformly from the training set.
* `[ D ]`


---

**Q: Which of the following is NOT a way to make the input to a neural network nonlinear?**
- Learning the representation.
- Designing feature extractors.
- Using kernel functions.
- Employing feature selection.
* `[ D ]`


---

**Q: The cost function of a Neural Network:**
- Gives an indication of how wrong the estimation is
- Estimates the computational complexity of each iteration
- Estimates the cost of using the network
- Needs to be maximized to increase the network accuracy
* `[ A ]`


---

**Q: How does the Stochastic Gradient Descent work?**
- It updates the gradient by selecting a random sample at each iteration
- It approximates the gradient by using all of the samples
- It selects the sample with the highest probablity for the gradient update
- None of the above
* `[ A ]`


---

**Q:  Which way that makes the input non-linear in a feed forward network is more flexible?**
- Designing feature extractors 
- Generic kernel function
- Learning the input
- All of the above are the same
* `[ B ]`


---

**Q: A deep forward network with the following function chain $f(x) = f^{(3)}(f^{(2)}(f^{(1)}(x)))$**
- Has a depth of 2 
- The first layer is $f^{(3)}$
- The hidden layer is $f^{(2)}$
- The Output layer is $f^{(1)}$
* `[ C ]`


---

**Q: Which of the following expressions is true?**
- A simple linear model cannot solve the XOR problem
- To solve the XOR problem we need to introduce non-linearity
- Compositions of several linear functions cannot solve the XOR problem
- All of the above
* `[ D ]`


---

**Q: Which of the following expressions about gradient descent is true?**
- Stochastic gradient descent will reach the global minimum, if it exists
- The learning rate should be high when the norm of the gradient of the loss function is close to zero
- With a fixed size for the mini batch, the cost of a single update of stochastic gradient descent doesn’t depend on the training set size
- None of the above
* `[ C ]`


---

**Q: What is NOT a valid moment to halt gradient descent?**
- When the number of iterations is high enough
- When the function gets close enough to zero
- When the steepness of the gradient gets above a threshold
- When improvement drops below a threshold
* `[ C ]`


---

**Q: How many hidden layers are needed to be able to properly represent the AND-operator using a neural network?**
- 0
- 1
- 2
- 3
* `[ A ]`


---

**Q: How can deep feed forward networks make their input non-linear?**
- Using a kernel function
- Designing feature extractors
- Learning it using a non-linear activation function
- All of the above
* `[ D ]`


---

**Q: Why is Stochastic Gradient Descent generally used instead of Gradient Descent to update a neural networks' weights?**
- Due to the computational limits for large datasets
- Stochastic Gradient Descent is more accurate
- Real-life applications can only be approximated using stochastic processes
- Some units in feed forward neural networks perform stochastic calculations
* `[ A ]`


---

**Q: In the Stochastic Gradient Descent method**
- the gradient of the function is not always taken into account, but neglected in a certain percentage of randomly selected cases.
- a random subset of samples is selected from the dataset.
- the gradient descent process is restarted at randomly selected seeding points, to reduce the risk of getting trapped in a local optimum.
- the learning rate is stochastically adjusted.
* `[ B ]`


---

**Q: What is a valid restriction on the choice of activation function when using a gradient descent method?**
- It needs to be monotonic.
- It needs to be linear.
- It needs to be differentiable.
- It needs to be non-linear.
* `[ C ]`


---

**Q: If $f: \mathbb{R}^n \to \mathbb{R}$ is a differentiable function, then $\nabla f$ is a vector with the following geometrical interpretation:**
- $\nabla f$ points to the direction of fastest growth of $f$
- $\nabla f$ points to the direction of fastest decay of $f$
- $\nabla f$ points to the nearest zero of $f$
- $\nabla f$ points to the global maximum of $f$
* `[ A ]`


---

**Q: A neural network with $N$ layers, all of which only have linear activation functions, can be reduced to a neural network with**
- $N-1$ layers
- $\lceil N/2 \rceil$ layers
- $1$ layer
- all of the above
* `[ D ]`


---

**Q: Which of the following is false. Using a batch size smaller than the total amount of training data (also known as stochastic gradient descent),**
- will give a better approximation for the gradient.
- will allow for faster weight updating.
- will add more randomness to the gradient descent.
- could lead to better exploration of the solution space.
* `[ A ]`


---

**Q: What is not a way to make the input non-linear:**
- Use generic kernel functions.
- Designing feature extractors.
- Learning it through mappings $\phi (x,\theta)$ of the input.
- Stack multiple linear functions after one another. 
* `[ D ]`


---

**Q: Given a fully connected feed forward network with 3 input nodes, 2 hidden nodes (1 hidden layer) and 2 output nodes. How many parameters does this network have?**
- 8
- 12
- 6
- 10
* `[ B ]`


---

**Q: In order to train a deep feed forward neural network, the network has to be presented with every element in the training data set at least:**
- One time for the amount of parameters of the network.
- One time.
- One time for every layer in the network
- Until the network has converged according to some objective function
* `[ D ]`


---

**Q: A feed forward network with no hidden layers can approximate a non-linear function**
- Never
- Yes, if linear activation functions are used
- Yes, if non-linear activation functions are used
- Yes, if biases are added to the weights
* `[ C ]`


---

**Q: Stochastic Gradient Descent updates the:**
- Parameters only
- Hyperparameters only
- Both parameters and hyperparameters
- Neither parameters or hyperparameters
* `[ A ]`


---

**Q: Which description most accurately describes the hidden layer**
- The layer which is not connected to either the input or the output
- The layer which is connected to the input but not the output
- The layer which is connected to the output but not the input
- The layer which hides the input and output
* `[ A ]`


---

**Q: The XOR problem is not linearly separable, what model do you need to be able to learn the XOR problem**
- A model with at least three layers: input-, hidden-, and output layer
- A non-linear activation function
- A model with at least two layers: input-, and output layer. The hidden layer is redundant
- None of the above, the XOR problem is linearly separable
* `[ A ]`


---

**Q: In the topic of Stochastic Gradient Descent, what does the equation: \theta' = \theta - \epsilon ∇f(x, \theta) represent?**
- Computation of the error function
- Estimation of the steepest gradient
- The computational cost
- Update of the fitting parameters
* `[ D ]`


---

**Q: Given the function f(x,W) = (min(0, W*x) + [1, 1]) * [2, -1]^T and the that the inputs x = [0, 0]^T, [0, 1]^T, [1,0]^T, [1,1]^T have the respective outputs: 1, 1, -3, -3, select the correct Weight function W (where ";" represents a new row):**
- W = [0, 0; 0, 0]
- W = [1, 0; 0, -2]
- W = [-2, 0; 0, 1]
- W = [1, 1; 1, 1]
* `[ C ]`


---

**Q: How does Stochastic Gradient Descent (SGD) differ from normal gradient descent (GD)**
- SGD uses a subset of the derivatives to compute which direction to move
- SGD uses a random sample of training data to compute the gradient, whereas GD has to use all training data
- SGD is the same as GD
- SGD uses a different loss function
* `[ B ]`


---

**Q: What is the requirement on a nonlinear function f for gradient descent to work**
- function f must be sufficiently smooth (derivative must be defined)
- function f must have an infinite domain
- function f must range between 0 and 1
- function f can not be polynomial
* `[ A ]`


---

**Q: Optimization refers to a task where one...**
- Finds the minimum or maximum value of x by taking the derivative of f(x), setting the value to zero and directly solving for x
- Finds the minimum or maximum value of f(x) by taking the derivative of f(x), setting the value to zero and directly solving for x
- Finds the minimum or maximum value of f(x) by taking the derivative of f(x), and then altering the x value
- Finds the minimum or maximum value of x by taking the derivative of f(x), and then altering the x value
* `[ C ]`


---

**Q: Gradient Descent of the function f(x) stops updating ONLY when**
- A point x which is the global minimum point is found
- A point x which is a local minimum point is found
- A point x which is a saddle point is found
- A point x where f'(x) = 0 is found
* `[ D ]`


---

**Q: The number of parameters in a deep feed forward net is given by:**
- Number of weights
- Number of bias terms
- Number of weights + number of bias terms
- Number of neurons
* `[ C ]`


---

**Q: How is the problem of huge datasets solved in deep nets?**
- Use SGD on stochastically chosen inputs
- Lower the number of inputs
- Use faster networks
- None of the above
* `[ A ]`


---

**Q: In the XOR problem, with of the following statements is false:**
- The input layer consists of 2 neurons.
- The space of the problem becomes linearly separable after the activations of the hidden layer have been computer.
- Although the least squares could be used as loss function, it is not the most appropriate for this type of problems.
- In this network, 8 parameters have to be trained.
* `[ B ]`


---

**Q: Which of the following is not a way of making the input non-linear:**
- Designing specific feature extractors.
- Composing linear functions multiple times (at each layer).
- Using a generic kernel function.
- Learning the actual mapping that is needed.
* `[ B ]`


---

**Q: How is the value of a neuron calculated?**
- Weighted sum of the values of the neurons in the previous layer
- Product of the values of the neurons in the previous layer
- Sum of the values of the neurons in all previous layers
- Dot product between the weight of the neuron and the values of all previous neurons
* `[ A ]`


---

**Q: How is SGD (Stochastic Gradient Descent) best described?**
- Updating the weights by an approximation of the gradient over a small set of stochastically chosen samples
- Updating the weights of a neural network by adding randomization, then checking whether the neural network has improved
- Doing a gradient descent on a subset of the weights that are stochastically chosen with a probability p
- Doing gradient descent using the whole dataset and then check the performance on a stochastically subset of samples
* `[ A ]`


---

**Q: The stochastic gradient descent of a dataset m is:**
- The avarage gradient over all samples m
- An approximation of the gradient from a large number of samples m'
- An approximation of the gradient from a small number of samples m'
- None of the above
* `[ C ]`


---

**Q: The depth of a neural network is described as:**
- The number of hidden layers
- The sum of the nodes in the hidden layers
- The sum of the nodes in the network
- The sum of the input, hidden and output layers.
* `[ D ]`


---

**Q: Why would we apply a non-linear function to a hidden layer?**
- Linear functions cannot separate certain data samples
- Linear functions have a high time complexity while non-linear functions do not have this problem
- You cannot add a bias to a linear function
- A non-linear function can be applied element-wise, while a linear function cannot be applied element wise
* `[ A ]`


---

**Q: What does a rectified linear unit (RELU) do ?**
- It serves as a kernel function
- It minimises f the activation function
- It makes a feed-forward network more rigid
- It computes the maximum between 0 and its input
* `[ D ]`


---

**Q: What is true a bout gradient descent? The gradient descent method...**
- always finds a global minimum
- updates only a small group of parameters per iteration
- can get stuck in a local minimum
- only works for a two layered network
* `[ C ]`


---

**Q: Which logic gate requires a non-linear classifier to be seperated?**
- AND
- OR
- NOR
- Neither of the above
* `[ D ]`


---

**Q: To update the weights of a feedforward network a loss function is derived. In what direction do you have to move to minimize this loss function?**
- in the opposite direction of the gradient
- in the direction of the gradient
- perpendicular to the gradient
- none of the above
* `[ A ]`


---

**Q: Suppose a feedforward network can be modelled with the following model 

f(w, x, b) = wT g(W,X,c) + b with hidden layer g(W, X, c) = max{0, WTX + c}

Determine the solution to the following XOR problem with the following matrices:

W = [1 1; 1 1]; c = [0; -1]; w = [1; -2]; b = 0; X = [1 0 1 0;0 1 1 0];**
- [1 1 0 1] (wrong, forgetting to take the max in last step)
- [0 1 1 0] (wrong, thinking it is the slide answer, which is wrong)
- [1 1 0 0]
- none of the above
* `[ C ]`


---

**Q: How many layers does this network have: $f^{(4)}(f^{(3)}(f^{(2)}(f^{(1)}(1))))$**
- 3
- 2
- 4
- 5
* `[ C ]`


---

**Q: Which functions are able to be learned by a feed forward net**
- And, Or, Xor, Nand, Nor
- Xor, Nand, Nor
- And, Or, Xor
- And, Or, Nand, Nor
* `[ D ]`


---

**Q: The rectiﬁed linear activation function is the default activation function recommended for use with most feedforward neural networks. What type is the output of this function?**
- linear function
- nonlinear function with two linear pieces
- nonlinear function with two nonlinear pieces
- curve function
* `[ B ]`


---

**Q: According to stochastic gradient descent(SGD), which one of the following is NOT contain in this procedure?**
- keep the weights as first calculated
- showing the input vector for a few examples
- computing the outputs and the errors
- computing the average gradient for those examples
* `[ A ]`


---

**Q: What is not true about Stochastic Gradient Descent?**
- It is a vector of all partial derivatives of a function
- It is a gradient of the loss on all samples
- It is a way to minimize cost/loss
- It is used to update the parameters of the function
* `[ B ]`


---

**Q: Deep learning is preferred over Machine learning when:**
- You have a lot of computational power at your disposal;
- You have a lot of data at your disposal;
- It is unclear what features to extract from your data;
- You have to classify images.
* `[ C ]`


---

**Q: Why are non-linear activation functions used in a multilayer perceptron?**
- To describe the model in a diﬀerent feature space in which a linear model is able to represent the solution
- To decrease the number of hidden layers of the model
- To transform the model in a linear model by applying a non-linear transformation on the input
- All the previous answers
* `[ A ]`


---

**Q: Why is Stochastic Gradient Descent (SGD) used in the updating of weights?**
- To increase the precision of the model
- To decrease the computational complexity
- To select the most important corrections
- To increase the learning rate
* `[ B ]`


---

**Q: How many nodes does a linear Feed Forward net require to learn a XOR problem?**
- 1
- 2
- 4
- A linear Feed Forward net can't learn a XOR problem
* `[ D ]`


---

**Q: What is the main advantage of Stochastic Gradient Descent(SGD) over regular Gradient Descent**
- SGD makes a probabilistic approximation of the gradient, making it more robust to noise.
- SGD does not require the entire dataset, making it faster
- SGD is less likely to get stuck in local optima due to a variable learning rate
- SGD has multiple runs per epoch, making it less likely to overfit on the data
* `[ B ]`


---

**Q: What is not one of the steps when training a network wile it is not converged?**
- Update the weights
- Present a training sample
- Compare the results
- All answers above are steps used when a network is trained
* `[ D ]`


---

**Q: What is a good way to make the input of a deep feed forward network (Multi-layered perceptron) non-linear?**
- Use a generic kernel function
- Designing feature extractors
- Learn it (with for example a model: x^Tw+b)
- All of the above 
* `[ D ]`


---

**Q: What is the fundamental goal of machine learning?**
- Generalize beyond the examples in a training set
- Find an exact representation of the training set
- Generalize from one topic to totally different topics
- Try to find the simplest representation of a training set 
* `[ A ]`


---

**Q: What is the purpose of a pooling layer in a convolutional neural network?**
- Pre-select striking features to improve the performance of the network
- Make detection invariant to small shifts and distortions of features
- Accumulate all outputs from the previous layer
- Make detection invariant to large shifts and distortions of features 
* `[ B ]`


---

**Q: The Stochastic Gradient Descent is called "STOCHASTIC" because, **
- It always converges to the minimum of the function.
- It is a time-dependent gradient descent method.
- It approximates the gradient from a smaller number of samples.
- It can be applied only on a convex objective function.
* `[ C ]`


---

**Q: What is the role of an activation function in a feed-forward neural network?**
- To introduce non-linearity into the neural network, so that it can learn more complex functions.
- To bound the neural network layers outputs to desired valued expected, so that valued beyond the bound are neglected.
- To represent the data in terms of sequential linear functions (in each node), so that non-linearity of the whole data representation is achieved.  
- All of the above.
* `[ A ]`


---

**Q: What are the three steps of training a multi-layered network?**
- 1. Present a training sample
2. Update the results
3. Update the weights
- 1. Train on training sample
2. Update the weights
3. Validate on test sample
- 1. Predict the weights
2. Train on training sample
3. Validate on test sample
- 1. Present a training sample
2. Compare the results
3. Update the weights
* `[ D ]`


---

**Q: How does the Stochastic Gradient Descent (SGD) update?**
- The SGD updates by updating the parameters using the gradient descent for all samples in the dataset given a certain learning rate
- The SGD updates by updating the parameters using the gradient ascent for all samples in the dataset given a certain learning rate
- The SGD updates by updating the parameters using the gradient descent for a small number of samples in the dataset given a certain learning rate
- The SGD updates by updating the parameters using the gradient ascent for a small number of samples in the dataset given a certain learning rate
* `[ C ]`


---

**Q: A function \psi(x, \theta) that defines a mapping y = f(x, \theta,w) is contained in which layer of the feedforward network?**
- Input layer
- Output layer
- Hidden layer
- None of the above
* `[ C ]`


---

**Q: What is the major advantage of using SGD in Deep Learning?**
- Decreased computational time over a small random data sample
- Improved generalization compared to the Kernel methods
- Result can always reach a global optimum
- All of the above.
* `[ A ]`


---

**Q: When is a layer called a hidden layer?**
- When it's an input-layer
- When it's an output-layer
- When it's between an input- and output-layer
- When it does not receive any input
* `[ C ]`


---

**Q: How can loss be reduced when updating weights?**
- By moving in the sign of the gradient
- By moving in the opposite sign of the gradient
- By moving along the second derivative in the sign of the gradient
- By using a loss-reduction-function
* `[ B ]`


---

**Q: You have a deep feed forward network, which is networked in the following function chain:  $$ f^{4}(f^{3}(f^{2}(f^{1}(x)))) $$. What are the hidden layers of this network?**
-  $$ f^{1} , f^{2},  f^{3} $$
-  $$ f^{2}, f^{3} $$
-  $$ f^{1}, f^{4} $$
- None of the above answers is correct.
* `[ B ]`


---

**Q: In what order should you train a network which is not converged?**
- 1: Check expected results, 2: Present a new training example, 3: Update the weights of the network
- 1: Present a new training example, 2: Compare expected result with obtained result 3: Update the weights of the network
-  1: Set the weights of the network, 2: Check expected results, 3: Present a new training example
-  None of the above answers is correct.
* `[ B ]`


---

**Q: What is a Stochastic Gradient Descent?**
- An SGD is the derivative of a function.
- An SGD is a formula to calculate the Gradient Descent in an optimal way.
- An SGD is an optimization method to find minima.
- An SGD is an approximation of the gradient based on a small number of samples.
* `[ D ]`


---

**Q: What is meant by: "a hidden layer"?**
- Variables of the linear model that are unknown
- The layer between the input and output layers.
- The layer before the output layer.
- All layers except the output layer.
* `[ B ]`


---

**Q: How many parameters does a fully connected Feed-forward Network have? 
Given $2^3$ input parameters, having two hidden layers with two units each with biases and a two percentage output unites, telling wherever y(x) is 1 or -1.

NOTE: (can either be given with raw solution e.g. "28" or formulaic e.g "2^3*2 + 2*2 + 2*2 + (2+2)")**
- 28 = 2^3*2 + 2*2 + 2*2 + (2+2)
- 24 = 2^3*2 + 2*2 + 2*2
- 16 = 2^3*2
- 8 = 2^3
* `[ A ]`


---

**Q: In deep learning - Stochastic Gradient Descent (SGD) is applied to:**
- the objective/loss/error/cost function
- the input data
- the covariance matrix
- the network weights
* `[ A ]`


---

**Q: What is the correct order of steps for training a network?**
- present training samples, update the weights, check for convergence
- present training samples, update the weights,  compare the results.
- present training samples,  compare the results, update the weights.
- compare the results, update the weights, present training samples , check for convergence.
* `[ C ]`


---

**Q: What are the three ways to make the input non-linear?**
- Generic kernel function, Designing feature extractors, Learn it.
- Generic kernel function, adding more input neurons, selecting a different optimization scheme
- adding more input neurons, non-linear activation function , Designing feature extractors.
-  Designing feature extractors, selecting a different optimization scheme, nonlinearly update the weights.
* `[ A ]`


---

**Q: Which of the following statements are true regarding gradient based optimization?**
- The Hessian is the Jacobian Matrix of second-order partial derivatives of a function.
- The second derivative of the optimization function is used to determine if we have reached an optimal solution.
- Both A and B.
- None of the above.
* `[ B ]`


---

**Q: Which of the following statements are true about an activation function?**
- We can perform non-linear transformations without an activation function.
- To predict the probabilities of n classes with the sum of p over all n equal to 1, we can use Softmax function as an activation function in the output layer.
- Both A and B.
- None of the above.
* `[ B ]`


---

**Q: Which of the following statements about Gradient Descent (GD) is FALSE?**
- With Batch GD, the gradients for every training set elements are summed and then the weights are updated
- With Mini-batch GD, multiple training set elements (say 50 of 1000) from the data set are summed and then the weights are updated
- With Stochastic GD, a single training set element is used to update the parameters after every epoch
- SGD is typically faster than Batch GD with large data sets.
* `[ C ]`


---

**Q: Which of the following is NOT a hyperparameter?**
- Number of units in a hidden layer
- Mini-batch size
- Learning rate for parameter updating
- The coefficients in a logistic regression 
* `[ D ]`


---

**Q: what is the purpose of the activation function**
- to define at which values the network returns true
- to define at which values the network returns false
- to define at which values the network is going to return a result
- none of the above
* `[ A ]`


---

**Q: what is the main purpose of a neural network as opposed to a single perceptron**
- it allows multiple inputs
- it allows multiple output
- it allows a more complex classification
- none of the above
* `[ C ]`


---

**Q: Which is NOT a way to make input non-linear?**
- Designing feature extractors.
- Generic kernel function.
- Use of stochastic gradient descent (SGD).
- Learn it.
* `[ C ]`


---

**Q: Which of the following statements is true?**
- Loss can be reduced by moving in the sign of the gradient.
- Stochastic Gradient Descent (SGD) is an approximation of the gradient from a small number of samples $m{'}$.
- A hidden layer is a layer you cannot see. 
- The depth of the network $f(x) = f^{(3)}(f^{(2)}(f^{(1)}(x)))$ is 4.
* `[ B ]`


---

**Q: The XOR problem can be learned by single layered feed-forward networks (no hidden layer) if...**
- ...the network is trained long enough.
- ...a linear function is used.
- ...it can't be learned at all.
- ...the input is right.
* `[ C ]`


---

**Q: Assume function g(x) is non-linear and function f(x) is linear. Then**
- g(f(x)) is linear
- f(f(x)) is linear
- g(f(x)) is not non-linear
- f(g(x)) is linear
* `[ B ]`


---

**Q: Which of these is not true regarding Stochastic Gradient Descent (SGD)?**
- SGD makes many small adjustments to the networks parameters, but uses less time to get feedback compared to regular Gradient Descent
- SGD works better than GD when the size of the training set is large.
- SGD can be easily parallelised because it gets feedback all at the same time. 
-  SGD gives a good approximation of the true gradient of the loss function.
* `[ C ]`


---

**Q: Which of the following is false regarding activation functions in a neural network?**
- The XOR problem cannot be solved using a linear activation function.
- Any non linear function can be used as an activation function
- The rectified linear unit function is nowadays considered the standard activation function used by neural networks.
- If the NN uses a linear activation function, the resulting behaviour of the NN will be linear no matter how may layers it has.
* `[ B ]`


---

**Q: Given is a Deep Feed-Forward Network with 4 nodes in each layer , including the input layer and excluding the output layer (there is only a single output node). The network has 84 parameters in total. What's the amount of hidden layers in this network?**
- 3
- 4
- 5
- 6
* `[ B ]`


---

**Q: In learning the parameters, how does Gradient Descent guard against taking too large steps when updating the weights?**
- Gradient Descent always takes a step of constant size, as specified by the user, in the right direction.
- Gradient Descent only looks at the partial derivative of the objective function with respect to a single weight.
- Gradient Descent uses a learning rate parameter which can be made sufficiently small.
- It does not. For Gradient Descent to become computationally less expensive, it is desirable that large updates to the weights are made in order to get to the (local) optimum faster.
* `[ C ]`


---

**Q: One of the following is not the way for an input for non-linear**
- generic kernel function 
- Designing feature extractor 
- stochastic gradient descent 
- learn it through training 
* `[ C ]`


---

**Q: In the training of feed forward neural network, we need one of the following to converge**
- update the weights
- present a training samples and analysis 
- compare results 
- all 
* `[ D ]`


---

**Q: What is the correct order to train a network while not yet being converged?

\begin{enumerate}
	\item Compare the results.
	\item Update the weights.
	\item Present a training sample.
\end{enumerate}**
- 2 3 1
- 3 1 2
- 1 2 3
- 3 2 1
* `[ B ]`


---

**Q: In order to minimize a certain criterion, a cost function can be used. With the results of the cost function, what is the right step to do if we want to minimize the criterion? I.e. how can we reduce the loss?**
- Move in the opposite sign of the gradient.
- Move in the sign of the gradient.
- Move in the opposite sign of the curvature.
- Move in the sign of the curvature.
* `[ A ]`


---

**Q: How the error of feed forward neural network is being minimized?**
- Compute a gradient at a sample and take a step in opposite direction of a gradient
- Compute a gradient at a sample 
- Compute a gradient at a sample and take a step in the direction of a gradient
- Compute a gradient for all sample points and take a step in direction of a smallest gradient for each point
* `[ A ]`


---

**Q: XOR problem is a problem of using a neural network to predict the outputs of XOR logic gates given two binary inputs. How XOR problem could be solved using feed forward network?**
- By introducing hidden layer with nonlinear activation function.
- By introducing nonlinear activation function in input layer
- By introducing additional hidden layer to feed forward network
- By extracting features from existing feature set
* `[ A ]`


---

**Q: What does gradient descent show?**
- The direction of the greatest decrease
- Something connected with colors in computer graphics
- The direction of the greatest increase
- The values of the second derivative
* `[ C ]`


---

**Q: What are the parameters of a feedforward neural network?**
- Input values and activation functions of the neurons
- Weights and activation functions of the neurons
- Input values and weights
- Number of neurons
* `[ B ]`


---

**Q: What is not part of training a network while it is not converged? **
-  Update the weights
- Maximize input value
- Compare the results
-  Present a training sample
* `[ B ]`


---

**Q: Which step is not a way to make a function non linear? **
- Learn it
- Adding a value
- Generic Kernel Function
- Designing feature extraction
* `[ B ]`


---

**Q: What is the deal with xor**
- The building blok of a the neurons of the nodes.
- xor is a gate that is slower than a nand port that gives performance problems in machine learning.
- It describes a situation issue that can happens, with normal data-sets. 
- Its used in machine learning to make activation functions.
* `[ C ]`


---

**Q: What is the best approach?**
- divide the data in a training data and test data equal
- use the test data and train data to train the machine learning and the last step is to test the data.
- First calculate the weights and then start training.
- non of the above
* `[ A ]`


---

**Q: How many trainable parameters are there in a 3-layer network? The input and hidden layers have 2 nodes each. The output layer has one node. Take into account biases.**
- 2
- 4
- 6
- 8
* `[ D ]`


---

**Q: Can a two-layer network learn an XOR relationship?**
- Yes, if there are many training data
- No
- Yes, as long as it trains for sufficiently many epochs
- Unknown
* `[ B ]`


---

**Q: In a feedforward network of deep learning, which is general correct statement?**
- Increasing number of hidden layers always increases the accuracy
- Increasing number of hidden layers always decreases the accuracy
- Increasing number of hidden layers may or may not increases the accuracy
- none of the above are correct 
* `[ C ]`


---

**Q: In a feedforward network of deep learning, what does a generic kernal function do?**
- It updates the weight
- It controls the number of hidden layers
- It makes the input non linear 
- It controls the design of the network
* `[ C ]`


---

**Q: Considering the architecture and mechanisms of a deep feedforward network, which of the following statements are NOT true?**
- The weight vector is updated by computing the gradient vector and moving in the opposite direction of the gradient vector.
- Stochastic Gradient Descent (SGD) represents a solution to the fact that computing the gradient of all the samples is impractical, as it approximates the gradient from a smaller number of samples.
- The number of parameters in a network is represented by the number of units in the hidden layers of the network.
- The activation function in a hidden layer transforms the input features in a non-linear way, thus making them linearly separable.
* `[ C ]`


---

**Q: In a deep feedforward network, in order to update and properly adjust the weight vector, some criterion needs to be minimized (e.g. a loss function). Based on the gradient vector, how is the weight vector updated in order to reduce the loss? **
- The update of the weight vector does not depend on the gradient vector.
- The weight vector is updated by moving in the same direction as the gradient vector.
- The weight vector is update by moving in the opposite direction of the gradient vector.
- The weight vector is updated by moving orthogonally to the direction of the gradient vector.
* `[ C ]`


---

**Q: Why is stochastic gradient descent a better choice than Gradient Descent ?**
- Better at finding global minima
- Computationally cheaper
- Good results even if the training data is small
- None of the above
* `[ A ]`


---

**Q: Which is a problem which cant be solved by single layer perceptron ?**
- AND 
- OR
- NOT
- XOR
* `[ D ]`


---

**Q: Given a feedforward network with 4 layers, how many hidden layers are there?**
- 1
- 2
- 3
- 4
* `[ B ]`


---

**Q: What is not a method for making the input non-linear?**
- Learning it
- Adding Gaussian noise
- Generic kernel function
- Designing feature extractors
* `[ B ]`


---

**Q: What is not a step in training a feed forward network?**
- Make input non-linear
-  Validate the results
- Get a training sample
- Assign weights
* `[ A ]`


---

**Q: A function chain f(x) = f3(f2(f(1(x))) can be...**
- x^3 = y
- k = l^2+m+2
- (a^2+ab+b^2)*3 = c
- All of the above
* `[ D ]`


---

**Q: What is usually not interesting in deep learning?**
- Theoretical bounds
- Architecture design
- Optimization
- Programming
* `[ A ]`


---

**Q: When training a neural network and updating the weights, how to reduce the loss/cost?**
- By moving in the sign of the first partial derivative
- By moving in the opposite sign of the first partial derivative
- By moving in the sign of the gradient
- By moving in the opposite sign of the gradient
* `[ D ]`


---

**Q: How many learnable parameters are there in a multi-layered perceptron with 2 inputs and 2 hidden layers?**
- 2
- 4
- 6
- 8
* `[ D ]`


---

**Q: Which of the following is a solution to the XOR problem?**
- W^\top \max\{0,w^\top Xb\}+c
- w^\top \max\{0,W^\top X+c\}+b
- w^\top \max\{0,W^\top Xc\}+b
- W^\top X + c
* `[ B ]`


---

**Q: In neural network, each layer consists of many units that act in parallel. What function does each of these units represent?**
- Vector to vector 
- Scalar to vector
- Vector to scalar
- Scalar to scalar
* `[ C ]`


---

**Q: Which of the following is the recommended activation function?**
- Sigmoid
- Tanh
- Logistic
- Relu
* `[ D ]`


---

**Q: To train a deep feed forward networks(multi-layered perceptron), the following three steps are needed while not converged:1.Present a training sample 2.Compare the results 3.Update the weights. Which is the correct squence?**
- 1, 2, 3
- 2, 1, 3
- 3, 1, 2
- 2, 3, 1
* `[ A ]`


---

**Q: For a mean suqared error loss function f(.), what of the follow method is a correct approach to reduce loss?**
- x<0 decrease f by moving leftward,x>0 decrease f by moving rightward
- x<0 decrease f by moving rightward,x>0 decrease f by moving leftward
- By moving in the same sign of the gradient
- By taking a small number of samples
* `[ B ]`


---

**Q: Which statement about the Stochastic Gradient Descent is true?**
- The Stochastic Gradient Descent (SGD) returns the exact gradient from a large number of samples m'
- The Stochastic Gradient Descent (SGD) is an approximation of the gradient from a small number of samples m'
- The Stochastic Gradient Descent (SGD) is an approximation of the gradient from a large number of samples m'
- The Stochastic Gradient Descent (SGD) returns the exact gradient from a small number of samples m'
* `[ B ]`


---

**Q: What is the correct order of training a network while it is not yet converged? Given the following steps have to be implemented in some order:

1. Update weights
2. Compare results
3. Present a training sample**
- 1 2 3
- 2 3 1 
- 3 2 1
- 2 1 3
* `[ C ]`


---

**Q: If we want to extend linear models to represent nonlinear functions of input vector x through a nonlinear transformation $\phi$, what is the strategy that is applied in deep learning to obtain $\phi$?**
- A very generic $\phi$ is chosen.
- $\phi$ is manually engineered.
- $\phi$ is learned.
- Deep learning makes the need for $\phi$ redundant.
* `[ C ]`


---

**Q: In order to get to a nonlinear function, after an affine transformation, an activation function is often used. Within modern neural networks, what function is considered to be the default recommended activation function?**
- Sigmoid
- ReLU (rectified linear unit)
- Tanh
- Leaky ReLU
* `[ B ]`


---

**Q: How does Stochastic Gradient Descent differ from Gradient Descent?**
- Gradient Descent approximates the gradient of the objective function across all training samples; Stochastic Gradient Descent approximates the gradient of the objective function across a subset of training samples
- Gradient Descent approximates the gradient of the objective function with an approximation formed by unweighted gradient estimates; Stochastic Gradient Descent approximates the gradient of the objective function with an approximation formed by gradient estimates, weighted by previous changes in the objective function
- Gradient Descent approximates the gradient of the objective function without any added noise; Stochastic Gradient Descent approximates the gradient of the objective function, adding noise to avoid nonglobal minimum convergence
- Gradient Descent approximates the gradient of the objective function and updates all parameters in the model; Stochastic Gradient Descent approximates the gradient of the objective function and updates only a random subset of the model's parameters
* `[ A ]`


---

**Q: What is the best definition for an objective function?**
- A function which evaluates the accuracy of our model in its current state; We aim to minimize the value of this function with the state of our model as the input to the objective function
- An unknown function that, on a given input, always produces the correct output, which we hope to replicate with a learned mathematical model; When we train our model, we aim to replicate the behavior of the objective function
- A function which turns the input of our machine from a subjective representation of our data point into an objective representation; With this objective representation, our model now has no bias when using this data point for learning
- A function which introduces non-linearity in our model, allowing us to create otherwise impossible functions that can correctly learn and classify our training data
* `[ A ]`


---

**Q: How many perceptron nodes are required to correctly classify a 2 dimensional XOR using a multilayer perceptron(not counting the input or output layer).**
- 1
- 2
- 3
- 4
* `[ B ]`


---

**Q: What is not a risk of using SGD in a multilayer perceptron**
- Coming in a local optimum
- Never Converging
- Adding unwanted layers
- Overfitting
* `[ C ]`


---

**Q: Can you combine multiple linear operations to get a non-linear transformation?**
- Yes, but only if you use more than 1 linear operator.
- Yes, even with a single linear operator.
- No, combining linear operators will always result in a linear operation.
- No, linear operators are typically not used in neural networks for that reason.
* `[ C ]`


---

**Q: What is 'stochastic' referring to in stochastic gradient descent?**
- Each time you run the procedure the result is different.
- You take random values for your initial weights.
- A random subset of data is used per iteration.
- A small subset of data gives a better gradient approximation than the full set.
* `[ C ]`


---

**Q: A multilayer perceptron with 4 nodes in the input layer, 3 nodes in a singular hidden layer and 2 nodes in the output layer, has how many tunable parameters? (Given no Bias on the input&output nodes)**
- 24
- 21
- 27
- 9
* `[ B ]`


---

**Q: The ReLU activation function is strictly better than the sigmoid activation function, because...**
- It is unbounded for positive values and hence able to represent a larger region of the dataset.
- Is able to represent sharp corners in separating classifications, due to the discontinuous gradient at x=0.
- It only has 2 values for it's gradient (0&1), it does not require the usage of the chain rule to find the partial derivative of multiple cascaded activation functions.
- It is not strictly better, both are perfectly valid options for designing a network, however the ReLU has some advantages as e.g. being more computationally efficient.
* `[ D ]`


---

**Q: What is main benefit of stochastic gradient descent over gradient descent?**
- Better accuracy
- Better parallelization
- Less computationally demanding
- None
* `[ C ]`


---

**Q: What must fulfill activation function if we want to use gradient descent?**
- It must be even function
- It must be odd function
- It must be continuous function
- It must go differentiate
* `[ D ]`


---

**Q: Which of the following is NOT a common step in training a feed forward network**
- Present the training samples
- Update the weights after comparing output
- Obtain the 5th order derivative for more accurate results
- Apply a non-linear function on the input for better discrimination
* `[ C ]`


---

**Q: Which of the following is a reason for using stochastic gradient descent for large training samples**
- Computationally expensive to compute the batch gradient descent for the entire training set
- It produces a more accurate estimation of the cost gradient
- It requires that the cost function be convex or pseudo-convex
- None of the above
* `[ A ]`


---

**Q: What is cross validation?**
- A way to obtain more training data
- Checking outcome with other results
- Checking with your colleagues if the answer is correct
- Splitting the training set in parts
* `[ D ]`


---

**Q: What is the difference between the size of an input layer and an hidden layer?**
- A hidden layer is larger.
- The input layer is larger
- They are the same size
- Input layer needs to have a size equal to the number of features of, while the hidden layer can have any size.
* `[ D ]`


---

**Q: What is not a way to make the input non-linear?**
- By designing feature extractors
- By learning the features
- By the use of stochastic gradient descent
- By the use of generic kernel functions
* `[ C ]`


---

**Q: Why do we only use a subset of the samples in stochastic gradient descent and not all of them?**
- SGD is not able to handle a lot of samples
- Using all the samples would be impractical as it leads to a higher computational cost
- We need the other subset of the samples to varify the correctness of the SGD
- Using all the data would make it more difficult to use multiple layers in a network
* `[ B ]`


---

**Q: Why we need activation function?**
- To make the network unlinear
- To slow down the convergence
- To to speed up the convergence
- To make the calculation interesting
* `[ A ]`


---

**Q: How many input vertices should be in the graph?**
- 2
- 3
- The same as the number of input features
- Decided by the programmers
* `[ C ]`


---

**Q: Which of the following is in general true about stochastic gradient descent using mini-batches in the context of neural networks?**
- Stochastic gradient descent updates the networks parameters (weights) exactly once per sample over the whole learning process.
- When looping through all the samples once, stochastic gradient descent has to evaluate the gradient of the loss function of a single sample less often than regular gradient descent would.
- Stochastic gradient descent performs worse on large datasets when compared to regular gradient descent.
- None of the above
* `[ D ]`


---

**Q: Given that the gradient of the loss function of all samples $\nabla_{\theta} J(\theta))$ for a regular gradient descent equals $[-2,1]^T$. How many samples is this network being trained on? Towards what direction will the weights of the network be updated?**
- Amount of samples: 2, direction $[-2,1]^T$
- Amount of samples: 2, direction $[-1,2]^T$
- Amount of samples: not enough information, direction: $[2,-1]^T$
- Amount of samples: not enough information, direction: $[-2,1]^T$
* `[ C ]`


---

**Q: One of the greatest advantages of Deep Learning is...**
- ...that it involves non-representational learning.
- ...how it allows for faster training times.
- ...that it is a form of unsupervised learning.
- ...that it avoids feature engineering.
* `[ D ]`


---

**Q: Stochastic Gradient Descent is...**
- ...a loss function.
- ...called stochastic because it randomly restarts, in order to avoid local minima.
- ...an optimization technique.
- ...none of the above.
* `[ C ]`


---

**Q: What are the differences between Deep Learning and the Machine Learning?

statement 1: There is a difference because tha machine learning is a part of bigger domain of deep learning 

statement 2: There is a difference because tha deep learning is a part of bigger domain of machine learning

statement 3: Machine learning heavily relies on representation learning which is not the case for DL.

statement 4: opposite of statement 3

statement 5: Typically DL relies on the millions of data point while the ML learning needs only thousands of data points

statement 6: Opposite of statement 5**
- statements 2 3 5 
- statements 1 4 6 
- statements 2 4 5 
- statement 2, 4
* `[ C ]`


---

**Q: In XOR example, the inital problem of using the linear acivation function is that it cannot separate two classes, because pairs of the same label data are sitting at the corners diagonally. What are the possible solutions to this problems.**
- This problem has no solution.
- Make change to the activation function so it is not linear anymore any function will fit.
- Use the RELU by adding the hidden layer
- Use RELU and don't add the hidden layer
* `[ C ]`


---

**Q: The function XOR can be learned with...?**
- Logistic Regression
- Perceptron
- Linear Regression
- Multi-layered perceptron
* `[ D ]`


---

**Q: Which of these is true about Gradient Descent Algorithm?**
- It always converges to a global optimum
- It takes a step proportional to the gradient of the function at the current point
- It takes a step proportional to the negative of the gradient of the function at the current point
- When it starts from a point near the solution, it will converge very quickly
* `[ C ]`


---

**Q: For which condition on the gradient below does the steepest descent algorithm reach a global minimum?**
- gradient > 0
- gradient < 0
- gradient = 0
- none of the above
* `[ D ]`


---

**Q: Why can a linear model applied directly to the original input cannot implement the XOR function?**
- The XOR x space cannot be represented by a linear function.
- In the original x space, the linear model cannot use the value of x1 to change the coefficient on x2. In a linear model a fixed w2 must be applied to x2
- A linear model is to rigid
- all of the above
* `[ D ]`


---

**Q: Which of the following is NOT a way to make the input of a feed forward network non-linear?**
- Designing and using a feature extractor
- Using a generic kernel function
- Using Stochastic Gradient Descent on the training samples
- Making the network learn a representation
* `[ C ]`


---

**Q: Which of the following statements is true?**
- A single perceptron without a non-linear activation function is enough to solve the XOR problem
- The XOR problem can be solved by adding a hidden layer with a non-linear activation function
- Adding a bias term to a single perceptron enables the perceptron to solve the XOR problem
- The XOR problem is unsolvable for any multi-layered perceptron
* `[ B ]`


---

**Q: Given a Multi-layered perceptron with 3 layers (l1, l2, l3 for input, hidden and output respectively), what is the correct function chain of this network:**
- f(x) = $f_3(f_2(f_1(x)))$
- f(x) = $f_1(f_2(f_3(x)))$
- f(x) = $f_1(x)f_2(x)f_3(x)$
- None of the above
* `[ A ]`


---

**Q: In what direction should the weights of a network be updated during training when SGD is used**
- The weights should move in the opposite direction of the gradient
- The weights should move in the same direction of the gradient
- The weights should move in the direction of $y_{pred}$ - y
- None of the above
* `[ A ]`


---

**Q: What does a MLP need to function as an XOR gate?**
- A fully differentiable loss function
- At least one hidden layer
- At least 4 inputs
- A positive loss function
* `[ B ]`


---

**Q: Can a MLP always rely on ReLu activation functions for SGD**
- Yes, they are fully differentiable
- No, It can happen that all derivatives become 0 and so no more learning takes place
- Yes, they are essentially linear
- No, the gradient is undefined at x = 0
* `[ B ]`


---

**Q: In the first lecture, we introduce Stochastic Gradient Descent (SGD) as an approximation of the gradient froma small number of samples m. Here, how to define such small numble of samples (I mean how to select these samples and decide the specific sample number)?**
- Set the size of samples as 50% of all data and randomly select them.
- Set the size of samples as 60% of all data and randomly select them.
- Set the size of samples as 70% of all data and randomly select them.
- Set the size of samples as 80% of all data and randomly select them.
* `[ C ]`


---

**Q: What common properties should activation functions have?**
- nonlinear and element-wise
- linear and element-wise
- nonlinear and non element-wise
- linear and non element-wise
* `[ A ]`


---

**Q: What is NOT a main component of the Machine Learning process?**
- Representation
- Evaluation
- Optimization
- Overfitting
* `[ D ]`


---

**Q: What is NOT an advantage of the distributed representations in Neural Networks?**
- Better generalization for combinations of learned features that were not in the training data
- More memory efficient compared to the logical paradigm network
- Captures local structures that can not be seen ahead of time by domain experts
- Reduces the computational time during training the network
* `[ D ]`


---

**Q: In a feed forward network the connection between two neurons is associated with a W_ij plus a bias term for each neuron which needs to be trained. In a simple feed forward network which performs a simple linear regression task with n neurons in the input layer and k neurons in the hidden layer, what is the total number of trainable parameters for the network?**
- (n+1)K
- n^k + k
- k(2+n)+1
- n(k+1)+2k+1
* `[ C ]`


---

**Q: Which of the functions below can not be used as an activation function a hidden layer of a neural network?**
- Sigmoid function
- Sinuous function
- Tanh function
- Step function
* `[ D ]`


---

**Q: Why is it called stochastic gradient decent? The term ‘stochastic’ refers to the fact, that iteratively a random**
- loss function is used for gradient descent
- subset of parameters is optimized with gradient descent
- subset of examples is used for gradient descent
- learning rate epsilon is picked for updating weights
* `[ C ]`


---

**Q: Which statement about updating weights is true?**
- To reach the minimum of a loss function f, one must move in the opposite sign of the gradient
- Once the first derivative of the loss function is f(x)’ is zero, we reached the minimum. Therefore, there cannot be any better parameter settings
- Using all available samples at the gradient descent approach is most effective
- The second derivative of the loss function determines the learning rate
* `[ A ]`


---

**Q: What is the gradient that is used to minimise the loss function in the SGD algorithm?**
- Another word for the partial derivative of a single variable
- A combined vector of all partial derivatives in the batch size
- The derivative of the average of all the input samples
- A combined vector of all partial derivatives in the entire data set
* `[ B ]`


---

**Q: How is a neural network able to deal with nonlinear data?**
- It is always able to deal with nonlinear data
- By normalising the data
- By creating a nonlinear mapping with an activation function
- It can never work with nonlinear data
* `[ C ]`


---

**Q: If we have a linear input, what procedure will never result in a non-linear output?**
- Multiplying with another linear input
- Subtracting another linear input
- Setting all the input's negative values to zero
- Setting all the input's positive values to zero
* `[ B ]`


---

**Q: If we have a system with 3 input nodes, a hidden layer with 2 nodes, and 1 output node, how many parameters are there?**
- 6
- 8
- 10
- 12
* `[ C ]`


---

**Q: What is not a way of making the input of a deep feedforward network non-linear?**
- Use a generic feature mapping.
- Design a feature extractor to manually engineer a feature mapping.
- Design a feature extractor to manually engineer a feature mapping.
- Manually engineer a feature mapping by grouping similar samples.
* `[ D ]`


---

**Q: What is the main reason that the XOR problem is a fundamental example in the field of deep learning research?**
- It can be solved using a single-layer perceptron.
- It is the most simple example of a linearly inseparable problem.
- It cannot yet be solved using deep learning.
- It is the most simple example of a linearly separable problem.
* `[ B ]`


---

**Q: A neural network with linear activation functions can: **
- Learn any representation possible
- Learn linear representations only
- Learn quadratic and linear representations only
- None of the above
* `[ B ]`


---

**Q: What is TRUE about Stochastic Gradient Descent**
- It converges always to the minimum
- It is faster than Gradient Descent but normally does not give an "optimal" result
- It does not have any advantage
- None of the above
* `[ B ]`


---

**Q: Which of the following statements are true regarding Sigmoid and RELU (Recified Linear Unit) activation functions?**
- Relu has slope values - {0,1} while sigmoid has slope values - [0,1]
- Over multiple training iterations, relu functions can saturate (output the same value), but sigmoid functions do not
- Over multiple training iterations, relu functions dont saturate (output the same value), and sigmoid functions dont saturate as well
- Relu functions are difficult to compute compared to sigmoid functions
* `[ A ]`


---

**Q: What is the range of slope values for a Rectified Linear Unit [RELU]**
- 0
- 1
- [0,1]
- {0,1}
* `[ D ]`


---

**Q: What is the goal of a feedforward network?**
- To give an overview of the output of each layer in a neural network.
- To allow linear models to be used to model non-linear mappings.
- To approximate a certain function by updating parameters in the network.
- To provide a set of features/representations describing the input.
* `[ C ]`


---

**Q: What is meant by an activation function**
- A function defined by the programmer that needs to be minimized in order for a neural network to be optimal.
- The function that needs to be approximated by changing parameters in the neural network.
- The function that defines the output of a node given certain inputs.
- The function that represents the feedforward network.
* `[ C ]`


---

**Q: What is an important benefit of using Stochastic Gradient Descent?**
- The gradient is an expectation, thus an approximation by a small set of samples is possible.
- This method always converges.
- The gradient is an partial derivative, thus can be easily computed.
- The cost per weight update depends on the training set size.
* `[ A ]`


---

**Q: Why would we sometimes want our network function 'f' to be non-linear?**
- The network has a higher accuracy when non-linear.
- The network might be to rigid to classify some sample sets.
- Non-linear networks always converge faster.
- Activation functions require less computation time.
* `[ B ]`


---

**Q: Which of the following statements is true when applying the gradient descent algorithm on a convex function f(x)?**
-  The found minimum is a local minimum
- The found minimum is a global minimum
- The found minimum is both a local and a global minimum
- Not enough information is available
* `[ C ]`


---

**Q: What are ways to make an input non-linear (multiple answers possible)?**
- Generic kernel function
- Training the network
- Apply first order Taylor expansion to input
- Using a backpropagation algorithm
* `[ A ]`


---

**Q: What is the loss function used for?**
- To calculate the error rate.
- To make sure none of the training data is lost during the training process (i.e. every training sample contributes to the final weights of a model).
- To optimize the model.
- To calculate the learning rate.
* `[ C ]`


---

**Q: What is the minimum depth a multi-layered perceptron must have for there to be at least one hidden layer.**
- 1
- 2
- 3
- 4
* `[ C ]`


---

**Q: For a function chain, f(3)(f(2)(f(1)(x))), what is f(3)?**
- Output
- first layer
- second layer
- input
* `[ A ]`


---

**Q: When approximating the function f(x) = x^{1/2}, and f'(x) > 0, what should we do with X in the next step?**
- Decrease
- Keep the same
- Increase
- Impossible to answer given this information
* `[ A ]`


---

**Q: What is the goal of deep forward networks?**
- Learn the nonlinear transfer function \theta(x)
- Design feature extractors
- Compute a general kernel function
- Drive f(x) to match f*(x)
* `[ D ]`


---

**Q: Why do we use the Stochastic Gradient Descent (SGD) algorithm to approximate gradients?**
- Nonlinearity prevents precise calculation of gradient
- SGD allows computation of hidden layer values
- Large training sets are computationally expensive
- SGD reduces the loss on all samples
* `[ C ]`


---

**Q: Why can the XOR-problem not by solved without a non-linear activation function? **
- Without a hidden layer the network does not have enough free parameters
- The XOR-problem has only 4 objects, this is not enough to train a linear neural network
- The 2 classes of objects in the XOR-problem can not be separated by a straight line 
- Neural networks perform better on high-dimensional data
* `[ C ]`


---

**Q: What is not a disadvantage of using gradient descend to upgrade the weights?**
- It is heavily dependent on the learning rate
- It can end up in a local minimum 
- Derivatives are computationally expensive 
- It needs all of the data to get the correct gradient
* `[ D ]`


---

**Q: Given the output of a network can be defined by the function:

f(x) =f(3)(f(2)(f(1)(x)))

What is the hidden layer?**
- f(1)
- f(2)
- f(3)
- There is no hidden layer
* `[ B ]`


---

**Q: What is not a method to make the input of a neural network non-linear?**
- Using a generic Kernel function
- Finding the anti-derivative
- Designing feature extractors
- Learning it
* `[ B ]`


---

**Q: In general, how many input layers needed to train a feedforward neural network considering a 2-dimensional case.**
- 1
- 2
- 4
- Could be any number
* `[ B ]`


---

**Q: Consider a 2-dimensional deep feed forward network, how many parameters is there?**
- 6
- 7
- 8
- It depends.
* `[ D ]`


---

**Q: Which of the following statements does \textbf{not} hold?**
- A deep feedforward of depth 2 network may contain arbitrary many hidden layers
- A feedforward network can learn the XOR function
- Given a fixed model size, the cost per SGD (stochastic gradient descent) update is independent of the training set size
- There are no loops in a feedforward network
* `[ A ]`


---

**Q: Gradient descent techniques…**
- … use the second derivative of the function the decide which step to take
- … are used to minimize or maximize objective functions
- … always find the global optimum of a function
- None of A, B, C holds.
* `[ A ]`


---

**Q: The Stochastic Gradient Descent (SGD) often converges faster than Gradient Descent(GD) because:**
- The error function is minimized more accurate and faster than in GD
- SGD uses one or a subset (randomly collected) of training sample points to update the parameters in every iteration
- The starting point of every iteration is collected randomly
- Because the size step of every iteration changes stochastically (randomly)
* `[ B ]`


---

**Q: The XOr (exclusive or ) problem can be classified better with a feed forward network because:**
- The feed forward network performs better than machine learning classifiers in such simple problems
- The feed forward network can deal with the non-linearity of this problem
- The feed forward network can be robust with large amount of data that can be generated for this problem
- The feature space is linearly separable
* `[ B ]`


---

**Q: How many parameters for a feedforward networks with 2 input units, 2 hidden units and one output units?**
- 7
- 8
- 9
- 10
* `[ B ]`


---

**Q: Which of the following situation could we get minimum cost function?**
- when gradient of loss equals to zero
- when gradient of loss is greater than zero
- when gradient of loss is smaller than zero
- Non of above
* `[ B ]`


---

**Q: Mark the false sentence.**
- The depth of a deep feed forward neural network is equal to the number of hidden layers.
- Deep learning usually requires extensive amount of training data.
- Gradient Descent is an optimization algorithm used in machine learning to learn values of parameters that minimize the loss function.
- Full batch gradient descent calculates the gradients for the whole training examples and update the parameters at each iteration only one time.
* `[ A ]`


---

**Q: On the process of training a neural network... (choose the correct option):**
- The training set of examples used is different from the one used to measure the performance of the system.
- The goal is to find parameters that minimize the cost function.
- The weight vector is adjusted in the opposite direction to the gradient vector.
- All the above.
* `[ D ]`


---

**Q: Stochastic Gradient Decent differs for Gradient Descent because:**
- It uses smaller learning rate
- Uses a small subset of the training samples in each epoch
- Does not update all the parameters
- It has completely different use cases
* `[ B ]`


---

**Q: What is an epoch?**
- An epoch is when all of the data in the training set is presented to the neural network once
- An epoch is when the training phase of the neural network has been completed 
- An epoch is when one sample of the training set has been presented to the neural network
- An epoch is when all the samples of the training set have passed through one layer of the neural network
* `[ A ]`


---

**Q: Which of the following is not true:**
- Linear models, such as logistic regression and linear regression, can be fit efficiently and reliably either in closed from or with convex optimization. 
- Linear models have the obvious defect that the model capacity is limited to linear functions.
- There are two approach to choose the mapping of extending linear models to nonlinear. 
- Cross entropy between the training data and the model's predictions can be used as the cost function. 
* `[ C ]`


---

**Q: Which of the following is not ture?**
- Nearly all of deep learing is powered by stochastic gradient descent.
- The minibatch size should grow with the increasing of the training set size.
- Stochastic gradient descent is the main way to train large linear models on very large datasets.
- The main way to learn nonlinear models was to use the kernel trick in combination with a linear model. 
* `[ B ]`


---

**Q: In a deep feedforward neural networks consisting of three layers (of following form 
f(x)=f^((3) ) (f^((2) ) (f^((1) ) (x))). Which of the following statements is FALSE? **
- First layer is input layer and third is output layer.
- First layer is input layer and second layer is hidden layer.
- First layer is input layer and third layer is hidden layer.
- Third layer is NOT hidden layer.
* `[ C ]`


---

**Q: In a deep feedforward neural function to update weights it is necessary to minimize cost/error/loss function. Which answer concerning the update of weight vector is TRUE?**
- We try to avoid calculating the gradient for all samples as this task is highly impractical.
- Instead of using approximated method we can get faster convergence by calculating gradient using simply partial derivatives.
- The weight vector is updated by moving in the direction of the gradient vector.
- Stochastic Gradient Decent approximation of the gradient from all samples in dataset.
* `[ A ]`


---

**Q: Can we solve the XOR problem with a linear model?**
- Yes, because XOR is a linear problem.
- Yes, because XOR has seperable classes.
- No, because there is no linear model that divides the two classes perfectly.
- No, because XOR is not a solvable problem.
* `[ C ]`


---

**Q: What statement is INCORRECT about gradient descent?**
- Gradient descent uses the derivative of f(x) to get closer to the minimum error.
- Gradient descent always finds the same results independent of initialization.
- Gradient descent always improves its result per computational step until finished.
- Gradient descent can get stuck in a local minimum.
* `[ B ]`


---

**Q: Given the following network chain; 

f(x) = f(3)(f(2)(f(1)(x)))

What type of layer is f(2)?**
- Hidden
- First layer
- Third layer
- Output layer 
* `[ A ]`


---

**Q: What is the problem caused by XOR inputs applied to feed-forward networks?**
- The inputs are not linear separable which makes it impossible to separate the 1 and 0 predictions with a single classification line
- The gradient is ineffective for larger losses which has a strong negative effect on training the network
- All of the above
- None of the above
* `[ A ]`


---

**Q: Assume we have a 3 dimensions data inout, we are going to use neural network to obtain a target out put, how many neurons can we have in the next layer of the input layer?**
- It only depends on the number of neurons of input layer
- It only depends on the number of neurons of output layer
- It depends on the both input and output layer (some relation between them)
- It does not depend on either input layer or output layer
* `[ D ]`


---

**Q: Assume we have a neural network, there are 3 neurons in input layer, 3 neurons in the next layer, how many parameters are there (between and inside these two layers)? **
- 9
- 12
- 3
- 18
* `[ B ]`


---

**Q: How can finding the second derivative of an objective function help you when doing stochastic gradient descent?**
- It does not help
- It tells you which direction steps should be taken
- It can be result in a good step size to use
- It helps to find the third derivative
* `[ C ]`


---

**Q: Why is the XOR problem a good example of how neural networks work well?**
-  Because it is not linearly separable
- Because it is a very famous problem
- Because it is important to solve for electrical engineering
- Because solving it will result in faster computers
* `[ A ]`


---

**Q: In which of the following ways can any linear input be made non-linear?**
- Generic kernel function
- Designing feature extractors
- Learning it
- All of them
* `[ D ]`


---

**Q: Which neural network solves XOR?**
- W = [1 1; 1 1], c = [0; -1], w = [1, -2]
- W = [1 1; 1 1], c = [1; 0], w = [1, -2]
- W = [1 1; 1 1], c = [0; -1], w = [-1, 2]
- W = [1 1; 1 1], c = [1; 0], w = [-1, 2]
* `[ A ]`


---

**Q: Which if the following statements is correct (if any)**
- SGD generally uses the whole dataset to train 
- The learning rate is equal to the number of samples in SGD
- During gradient descent, one can upgrade the weight by taking the derivative of the activation function.
- All are incorrect
* `[ D ]`


---

**Q: Which of the statement is correct**
- A perceptron is an example of a basic deep neural network.
- Perceptrons are linear units hence neural networks containing them cannot learning the XOR function.
- In an MLP every neuron is to every other neuron
- All are incorrect
* `[ D ]`


---

**Q: Which of the following statements is false?**
- Representable does not imply learnable.
- There is no connection between the number of parameters of a model and its tendency to overfit.
- Simplicity does not imply accuracy.
- Bias is the tendency to learn random things irrespective of the real signal.
* `[ D ]`


---

**Q: Which of the following statements is false?**
- Learning distributed representations enable generalization to new combinations of the values of learned features beyond those seen during training.
- We expect unsupervised learning to become far more important in the longer term.
- Stochastic gradient descent (SGD) consists of showing the input vector for a few examples, computing the outputs and the errors, computing the average gradient for those examples, and adjusting the weights accordingly.
- RNNs are very powerful dynamic systems, training them has proved to be successful because the backpropagated gradients either grow or shrink at each time step.
* `[ D ]`


---

**Q: Why is it not possible to use linear models in deep learning?**
- Linear models are hard to compute and usually require more trainable parameters.
- The composition of linear and affine transformations is linear and affine respectively, thus connecting multiple layers results in a design that could be replaced with one layer that is still unable to approximate non-linear functions.
- The range of output values in a linear model is varying, thus making decisions based on them is hard.
- Linear models are not differentiable, thus making the weight-update impossible.
* `[ B ]`


---

**Q: What is the output of a ReLu layer for the given input vector: tr([15 0 1 -1 5])?**
- tr([15 0 1 0 5])
- tr([15 0 1 -1 -5])
- tr([15 0 1 -1 5])
- tr([0 0 0 -1 0])
* `[ A ]`


---

**Q: In the process of training the feed-forward network, if the result is not the one expected after presenting a training sample we should:**
- Update the weights
- Take the first derivative of the loss function and update the weights
- Take the second derivative of the loss function and update the weights
- Design a feature extractor
* `[ A ]`


---

**Q: In order to solve the XOR problem, we can:**
- Change the input
- The XOR problem cannot be solved
- Make output linear by adding a hidden layer
- Make input non-linear by adding a hidden layer
* `[ D ]`


---

**Q: The Stochastic Gradient Descent:**
- Includes classes' prior probabilities
- Uses a subset of the training set
- Is impractical for very large datasets
- Can always be replaced by an analytical solution
* `[ B ]`


---

**Q: A neural network:**
- Has always the same activation function for each node
- Can only have the mean squared error as a cost function
- Updates its weights to maximise the objective function
- Its first layer only brings the input into the network 
* `[ D ]`


---

**Q: When using gradient descent a parameter, the learning rate, is used. What can be said about the learning rate?**
- It controls how many training samples are used for each iteration of training.
- It is used to indicate the amount of iterations of gradient descent should be performed.
- The learning rate controls the size of the steps gradient descent takes.
- The higher the learning rate, the better the model will fit to the training data.
* `[ C ]`


---

**Q: Why can choosing a learning rate that is too large be potentially bad?**
- There is no issue with a larger learning rate, as it simply trains the model faster.
- A learning rate that is too high could lead to divergence of the loss function.
- It doesn't matter what the learning rate is, as it only controls the speed at which variables are updated.
- A learning rate that is too high will cause the neural network to overfit on the training data.
* `[ B ]`


---

**Q: Stochastic Gradient Descent, uses a very important characteristic of the gradient of a multi-dimensional vector. What is this characteristic?**
- The gradient indicates the highest rate of descent
- The gradient estimates how far the input is from other inputs in the database
- The gradient makes the system non-linear
- The gradient indicates the highest rate of ascent
* `[ D ]`


---

**Q: Comparing the output of a neural network is compared with the label (desired output) using the least-square cost function. Which of the following functions is that least-square cost-function?**
- \frac{1}{n}\sum_{i=1}^{n}(h(x_i)-y_i)^2
- \frac{1}{n}\sum_{i=1}^{n}(h(x_i)^2-y_i^2)
- \sum_{i=1}^{n}(h(x_i)-y_i)^2
- \sum_{i=1}^{n}(h(x_i)^2-y_i^2)
* `[ A ]`


---

**Q: Which of the following steps is not a step when converging a network?**
- Compare the results
- Update the weights
- Present a training sample
- Show the result to the user.
* `[ D ]`


---

**Q: Which of the following statements are true?

I Designing feature extractors is a method to make input non linear

II Creating a generic kernel function is a method to make input non linear**
- Only I is true.
- Only II is true.
- Both I and II are true
- None of the statements is true.
* `[ C ]`


---

**Q: The name "Deep Learning" arose from the terminology "depth", which indicates the overall chain length of feedforward neural network. What's the output layer of f\left ( x \right ) = f^{\left ( 3 \right )}\left ( f^{\left ( 2 \right )}\left ( f^{\left ( 1 \right )}\left ( x \right ) \right ) \right )?**
- f^{\left ( 1 \right )}
- f^{\left ( 2 \right )}
- f^{\left ( 3 \right )}
- f^{\left ( x \right )}
* `[ C ]`


---

**Q: Which one of the statements is wrong?**
- A linear model could be extended to non-linear by applying a non-linear transformation on the input.
- Stochastic Gradient Descent (SGD) is an approximation of the gradient from a small number of samples.
- When exactly one of these binary values is equal to 1, the XOR function returns 0.
- A linear model is not able to represent the XOR function.
* `[ C ]`


---

**Q: Why does a neural network need multiple layers to be able to learn the XOR function?**
- XOR is not linearly separable, the input space needs to be transformed by an earlier layer before the classification layer can separate them.
- This is not true, XOR can be learned in one layer.
- A single layer can not capture the complexity of the function.
- Because the network needs to be non-linear.
* `[ A ]`


---

**Q: What is the difference between stochastic gradient descent and gradient descent?**
- Gradient descent calculates the gradient over all the samples, whereas stochastic gradient descent uses a subset, or just one sample at a time.
- Stochastic gradient descent incorporates Gaussian noise in the update of the weights.
- Stochastic gradient descent is more precise.
- Stochastic gradient descent minimizes the loss function, whereas gradient descent moves the weights in the opposite sign of the gradient.
* `[ A ]`


---

**Q: Why do we use *stochastic* gradient descent instead of gradient descent on the whole dataset?**
- SGD is more accurate
- Because using the whole dataset is not practical
- The derivative on the whole dataset is undefined
- Gradient descent on the whole dataset is impossible
* `[ B ]`


---

**Q: What of the following problems can not be learned by a linear model?**
- Predicting income based on education
- Classifying cars based on price
- Classifying people in income groups
- Based on earlier races between two horses, what horse is going to win the next race.
* `[ D ]`


---

**Q: Which of the following answers is NOT a step of training a network?**
- Compare results
- Update weigths
- Update weigths
- Update training sample
* `[ D ]`


---

**Q: What does f(x+e) approximate to?**
- f(x) + ef'(x)
- ef(x) + ef'(x)
- f'(x) + ef(x)
- ef(x) + eF(x)
* `[ A ]`


---

**Q: Which statements are true about linear deep feed forward network?
I. If the data set is when a data set is non-linearly separable, the result will be a hyperplane.
II. Input can be learned to be non-linear
III. A generic kernel function can make the input non-linear**
- I and II
- I and III
- III
- I, II and III are all true
* `[ D ]`


---

**Q: What is true about the stochastic gradient descent?**
- The learning rate is proportional to 1/iterations required till the weighting factors a converged 
- The samples are often divided in multiple data sets
- If is not diverging always converge to the global minimum
- Convergence is depending on what type of function is minimized
* `[ B ]`


---

**Q: A 2-input neuron has weights 2 and 3. The transfer function is linear with the constant of proportionality being equal to 5. The inputs are 3 and 20 respectively. The output will be:**
- 360
- 330
- 125
- 420
* `[ B ]`


---

**Q: For a neuron with inputs 0 and 1, the output is 1 and for inputs 1 and 1, it is 0. What can the possible function of the neuron be?**
- OR
- XOR
- NAND
- B or C
* `[ D ]`


---

**Q: What do we use to update the weights in a network?**
- The weights are randomly guessed.
- The derivative of the function(s)
- the second derivative of the function(s)
- Some other optimization algoritmn is used.
* `[ B ]`


---

**Q: What is the function of an activation function**
- To activate the network
- To keep all the results positive
- To make the system multi-dimensional
- To make the system non-linear
* `[ D ]`


---

**Q: What is the XOR problem in neural networks?**
- The solution space is negative and therefore not linearly separable
- A xor operation is linear and therefore not suited
- The solution space is not separable with a single line, in other words: XOR is not linearly separable.
- The solution space is separable with a single line. XOR is there fore not linearly separable 
* `[ C ]`


---

**Q: What are the basic three steps to train a network?**
- 1: Present a training sample, 2: Compare the results, 3: backpropagation
- 1: Present a training sample, 2: backpropagation, 3: Compare the results
- 1: Calculated the weights, 2: calculate the gradients, 3: update the weights
- 1: Present a training sample, 2: Compare the results, 3: Update the weights
* `[ D ]`


---

**Q: In order to reduce the loss of an objective cost function, you will need to:**
- Perform the second derivative of the loss function and move in the direction on the gradient.
- Perform the first derivative of the loss function and move in the opposite direction of the gradient
- Perform the second derivative of the loss function and move in the direction of the gradient.
- Perform the partial derivative of the loss function and move in the opposite direction of the gradient.
* `[ B ]`


---

**Q: The Rectifed linear unit function is:**
- An activation function applied to a non-linear input
- A linear unit function applied to a linear input
- A non-linear activation function.
- Unbound on values for x smaller than zero
* `[ C ]`


---

**Q: Which of the following about deep feed forward networks is false?**
- Mean squared error method is used to minimize the loss function
- Stochastic Gradient Descent method approximates the gradient using many samples
- In order to solve the XOR problem by learning, an additional hidden layer must be added.
- Activation function is used to compute hidden layer values
* `[ B ]`


---

**Q: Which of the following is not a step used in training networks?**
- Presenting the training sample
- Comparison of results
- Updating labels
- Updating weights
* `[ C ]`


---

**Q: Select the correct affirmation regarding the two loss optimization techniques: gradient descent (GD) and stochastic gradient descent (SGD):**
- for each iteration/epoch GD updates the coefficients after running through all samples from the training dataset
- for each iteration/epoch SGD updates the coefficients after running through all samples from the training dataset
- for each iteration/epoch GD updates the coefficients after each sample of the training dataset
- for each iteration/epoch SGD updates the coefficients after running through a large enough batch of samples from the training dataset
* `[ A ]`


---

**Q: What could be a possible reason for replacing Rectified Linear Unit – ReLU ($max(0, x)$) activation function with a another similar function such as Leaky ReLU ($0.01 x$ for $x < 0$ ; $x$ for $x >= 0$) in a Deep Feed-Forward Network?**
- Leaky ReLU is easier to compute
- If the input of the ReLU activation function is constantly negative, the gradient flowing to the unit will be 0 
- Leaky ReLU gradient is always zero for negative inputs
- If the input of the ReLU activation function is constantly positive, the gradient flowing to the unit will be 0
* `[ B ]`


---

**Q: Which feature of deep learning is chiefly responsible for enabling the learning of abstract features?**
- Stochastic Gradient Descent 
- Back Propagation
- Multiple layers / Hidden Layers
- Non-Linear Activation Functions
* `[ C ]`


---

**Q: What is necessary for implementing an XOR classifier using a neural network and why?**
- A non-linear activation function, since the input-space is not linearly separable.
- Hidden layers, since the inputs are Boolean values.
- Both hidden layers and a non-linear activation function, since the input-space is not linearly separable.
- Either hidden layers or a non-linear activation function, since the Boolean input-space are not linearly separable. 
* `[ C ]`


---

**Q: How many nodes are there in the hidden layer of g(x1w1+x2w2+c1)w5+g(x1w3+x2w4+c2)w6**
- 1
- 2
- 3
- 4
* `[ B ]`


---

**Q: What is the value of w in the XOR model of w^Tmax{0,W^Tx+c}+b?**
- [1 1]^T
- [1 2]^T
- [1 -2]^T
- [0 1]^T
* `[ C ]`


---

**Q: How many parameters have to be tuned if there are 2 input nodes, 2 nodes in a hidden layer and one output node ? (assume a bias and activation function for the hidden and output layer )**
- 7
- 8
- 9
- 10
* `[ C ]`


---

**Q: Stochastic Gradient Descent is mainly used **
- To improve the accuracy
- To reduce the power of computation needed
- When the classifier is non linear
- When the classifier is linear
* `[ B ]`


---

**Q: Which statement is false?**
- To reduce the loss, one should move in the opposite sign of the gradient.
- Stochastic Gradient Descent(SGD) is an approximation of the gradient from a small number of samples.
- The weights of a Deep Feed forward Network (Multi-layers perceptron) are updated at the very end of the training phase.
- Not all samples are used when computing the gradients because this is impractical for large datasets.
* `[ C ]`


---

**Q: What does this function chain represent: f(x) = f⁴(f³(f²(f¹(x))))**
- f¹ = first layer, f² = hidden, f³ = hidden, f⁴ = output, depth = 4
- f¹ = output, f² = hidden, f³ = hidden, f⁴ = first layer, depth = 4
- f¹ = first layer, f² = hidden, f³ = hidden, f⁴ = output, depth = 3
- f¹ = output, f² = hidden, f³ = hidden, f⁴ = first layer, depth = 3
* `[ A ]`


---

**Q: What does this do: \theta + \epsilon\nabla _{\theta} f(x;\theta) (f a cost function and \theta parameters)**
- The error of the model might decrease
- The error of the model might increase
- The error of the model increases
- The error of the model decreases
* `[ C ]`


---

**Q: Which distribution of datapoints can be properly classified by the following model (only two labels): x ^{T} w+b**
- two ball-shaped concentric point clouds
- two point clouds separable by a hyperplane
- two line-shaped cross intersecting point clouds
- 4 points at the corners of a rectangle, with alternating label
* `[ B ]`


---

**Q: Minimizing the loss function allows us to...**
- Get better data to feed the network with
- Update the weigths in the network
- Find the derivative of the gradient
- Move in the opposite side of the gradient
* `[ B ]`


---

**Q: How can we deal with non linear classification?**
- Using multy layered perceptrones
- Using the correct loss function
- Introducing activation functions in the network
- Avoiding features extractors
* `[ C ]`


---

**Q: We aim to construct a feedforward Neural Network which predicts the output of $f^{*}(x)=x^{2}$ where $x \in \left(-\infty . +\infty \right)$. How many hidden layers would such network at least require such that the output function $f$ satisfies $\lvert f^{*}-f\lVert\leq 10^{-6}$ **
- None
- 2
- 8
- infinite
* `[ D ]`


---

**Q: Consider a continuous function $f(\mathbf{x}): [A,B]^{\nu} \longrightarrow\Re^{\nu} $ where $A,B$  are both finite. Given that $H=J(\nabla f(\mathbf{x}) )^{T} $ and $\det(H)<0$ $\forall \mathbf{x} \in [A,B]^{\nu}$, we can conclude that:**
- there exists exactly one saddle surface
- there exists at least one direction where f exhibits maximum increase
- there exists at least one direction where f exhibits minimum increase
- all of the above 
* `[ D ]`


---

**Q: For a multi-layered perceptron, what is the best activation function when the input is a binairy image with clear shapes and enough perceptron for the output ?**
- Sigmoid function
- step function
- Sign function
- Linear function
* `[ B ]`


---

**Q: Can an XOR be realised by an perceptron?**
- Yes, perceptron is linear.
- Yes, perceptron can be made nonlinear because of an nonlinear weight.
- No, not all states in an XOR can be separated by an linear classifier/ perceptron.
Yes, when enough perceptrons are added it can be done.
- No, XOR can be realized by an anti-perceptron as an OR can be classified by and perceptron
* `[ C ]`


---

**Q: How many parameters we need to tune for 2-2-1 network (2 input units, 2 hidden units and 1output unit)?**
- 8
- 6
- 4
- 2
* `[ A ]`


---

**Q: The reason for using non-linear activation function is that:**
- Non-linear function is more complex than linear function in terms of representation.
- It could generate non-linear classifier to classify linearly non-separable classes.
- Linear function may lead to high bias.
- There's no difference in using non-linear or linear activation function.
* `[ B ]`


---

**Q: The sign of the derivative of f(x)**
- Indicates where the gradient of descent the highest
- Indicates in which direction to go in order to reach a local minimum of f(x)
- Does not give extra information about f(x)
- Indicates where the error rate is the lowest 
* `[ B ]`


---

**Q: What can be a downside of a linear deep feed forward network?**
- It can have unbounded conditions
- It can be too rigid
- The output can grow exponentially due to the lack of governing terms
- There are no downsides
* `[ B ]`


---

**Q: What are the methods that are suited for classification involving non-liner decision boundary in raw input data?**
- Transform the input into another space using Generic Kernel Function
- Feature extractors
- Multi layer neural network using  linear activation functions in different layers.
- Both a and b
* `[ D ]`


---

**Q: Stochastic gradient descent or mini-batch gradient descent is used to optimize the loss/error function in many of the deep learning applications. So among the following which are the possible solution that a gradient descent may end up in?**
- Local minima
- Saddle point
- Global minima
- All of the above
* `[ D ]`


---

**Q: Why do you want to only use part of the data to train an algorithm?**
- It usually is too much data to handle
- Using too much data will generate too noisy results
- A part of the data samples have to be saved to test the algorithm
- You do have to use all the data
* `[ C ]`


---

**Q: Which step is not part of the iterative process of training a basic network?**
- Compare the results of the algorithm w.r.t. the correct result
- Change the input such that the algorithm produces the correct result
- Update the weights of the network
- Present the network with a training sample
* `[ B ]`


---

**Q: In order to make a neural network that identifies 20x20 grayscale images into either cats or dogs, using one hidden layer, what would be the size of the input layer? The input layer here reassigns the matrix into vector form.**
- 2
- 20
- 400
- 19
* `[ C ]`


---

**Q: In the gradient descent method, parameters of the network are updated in the direction opposed to the gradient of the loss function, with steps proportional to the learning rate $\epsilon$. Ideally what should be the scale of the learning rate?**
- initially large, and decreases with the number of steps taken
- initially small, and increases with the number of steps taken
- set to a constant large value
- set to a constant small value
* `[ A ]`


---

**Q: Which function employs the rectified linear unit (ReLU)?**
- f(z) = z / (1 + |z|)
- g(z) = z 
- h(z) = (z + |z|) / 2
- i(z) = log(z)
* `[ C ]`


---

**Q: Which step is NOT a part of training a multi-layered perceptron?**
- Compare the results
- Feature selection
- Update the weights
- Present a training sample
* `[ B ]`


---

**Q: Which statement is wrong about Feedforward Network?**
- There are no feedback connections in which outputs of the model are fed back into itself. 
- The input of the network must have a proper representation. 
- The mapping function could be manually implemented.
- The inﬁnite-dimensional mapping function could be used.
* `[ B ]`


---

**Q: Which statement is wrong about Stochastic Gradient Descent?**
- For a ﬁxed model size, the cost per SGD update does not depend on the training set size.
- Gradient descent in general has often been regarded as slow or unreliable.
- To improve the performance, it is better to apply on all samples.
- We may ﬁt a training set with billions of examples using updates computed on only a hundred examples.
* `[ C ]`


---

**Q: Given a function f3(f2(f1(x))) = f(s); What is the name of f2**
- Output layer
- Under layer
- Input layer
- Hidden layer
* `[ D ]`


---

**Q: How should you train a network while it has not converged yet?

1.  Present a training sample
2. Compute Bayesian Model Averaging
3.  Update the weights
4.  Compare the result

What is the correct order?

4.**
- 1, 2, 3, 4
- 1, 2, 3
- 1, 4, 3
- 1, 4, 3, 2
* `[ C ]`


---

**Q: What is the meaning behind the use of the word "stochastic" in the concept of stochastic gradient descent (SGD)?**
- Non-convexity leads to parameter updates which are stochastic in nature due to overshoot which is affected by the learning rate.
- Each small set of examples gives a noisy estimate of the average gradient over all examples.
- The use of linear and non-linear transformations within the perceptrons (nodes) lead to erratic parameter updates.
- The update of the weights in a multi-layered hidden segment is indicative of probabilistic updates similar to Bayesian interference. 
* `[ B ]`


---

**Q: During the non converged phase of training a neural network, which of the following is not relevant in contemporary practice?**
-  Update the weights.
- Compare the results to labels.
- Calculate the second derivative.
- Present a training sample.
* `[ C ]`


---

**Q: After how many training samples does the stochastic gradient descent update the weights?**
- 1
- 5
- The whole data-set
- It does not update the weights
* `[ A ]`


---

**Q: What do you call the layer(s) between the input and output of a network?**
- Depth layers
- Hidden layers
- Secret layers
- They don't have a special name
* `[ B ]`


---

**Q: Which of the following steps is not part of the training process loop of a Deep Neural Network?**
- Update the weights
- Use a test sample to test the classifying performance of the network
- Compare the results
- Present a training sample
* `[ B ]`


---

**Q: Which of the following statements is true? 

1: The use of mini-batches leads to a more efficient use of available memory.

2: The use of relatively small mini-batches leads to more noise in the training process. **
- Only statement 1 is true
- Only statement 2 is true
- Both statements are true
- Both statements are false
* `[ C ]`


---

**Q: When is a neural network considered a deep neural network?**
- If the neural network contains many inputs.
- If the neural network contains many outputs.
- If the neural network has many hidden layers.
- If the neural network is trained with a huge dataset.
* `[ C ]`


---

**Q: Which of the following statements is true about Stochastic Gradient Descent (SGD)?**
- SGD considers all input data in every step.
- SGD is used to preprocess the input of a neural network.
- SGD is useful for exploring weights values of the objective function rather than exploiting such values.
- The difference between stochastic and non-stochastic gradient descent is that SGD uses random samples for every time step.
* `[ D ]`


---

**Q: Which procedure is not used in the training process of Deep Feed forward Networks?**
- Compare the results
- Update the weights
- Calculate network convergence 
- Present a training sample
* `[ C ]`


---

**Q: How to reduce loss in weight update process?**
- By moving in the opposite sign of the gradient
- By moving in the same sign of the gradient
- By moving in the opposite sign of the descent
- By moving in the same sign of the descent
* `[ A ]`


---

**Q: In feed forward neural networks what function(s) is computed at each node?**
- Linear function on input followed by activation function
- Linear function on the input followed by maximum 
- Activation function on the input followed by a linear function
- Activation function followed by summation of all nodes in that layer
* `[ A ]`


---

**Q: Which of the following is NOT an advantage of using Stochastic gradient descent?**
- It is computationally less demanding and more suitable when there are memory limitations
- More frequent updates 
- Converges much faster on larger datasets
- Makes efficient use of vectorization
* `[ D ]`


---

**Q: We say a feed-forward neural network is fully connected under which of the following?**
- All the nodes of same layer are connected to one another.
- All nodes at one layer are connected to all nodes in their following higher layer.
- All nodes in the network are connected to each other.
- All of the hidden layered nodes are connected to all of the output layered nodes.
* `[ B ]`


---

**Q: Which of the following is true about Stochastic Gradient Method (SGD)?**
- Rather than approaching the optimum, SGD (with a constant step size) converges to a region of high variance around the optimum
- SGD method is useful only for applications involving exact-inference solutions.
- SGD converges using just one example to estimate the gradient.
- All of the above. 
* `[ C ]`


---

**Q: Which of the following statement is correct?**
- Stochastic Gradient Descent (SGD) is an approximation of the gradient from
a large number of samples.
- An epoch is when all of the data in the test set is presented
to the neural network at once.
-  A perceptron is a feedforward network with an input layer, a hidden layer and an output layer.
- None of the above
* `[ D ]`


---

**Q: An XOR function returns 1, only if both arguments are different, else, it will return 0. This can be represented by the following table:


\begin{center}
\begin{tabular}{ c | c | c | c | c }
 x1   & 0 & 0 & 1 & 1 \\  
 x2   & 0 & 1 & 0 & 1 \\
 \hline
 x1 XOR x2   & 0 & 1 & 1 & 0 \\
\end{tabular}
\end{center}

Read the following statements and choose the correct options
\newline
a. The above function can be implemented by a single-layer perceptron.
\newline
b. It is not possible to implement the above function neither by a single unit nor by a single-layer perceptron.
\newline
c. The above function can be implemented using a feed-forward network with a hidden layer.
\newline**
- a and b are correct.
- b and c are correct.
- a and c are correct.
- None of them is correct.
* `[ B ]`


---

**Q: Stochastic gradient descent is an extension of gradient descent. Which kind of datasets makes stochastic gradient descent better to use than gradient descent?**
- Datasets with low dimensional training samples
- Datasets with high dimensional training samples
- Datasets with a low amount of training samples
- Datasets with a large amount of training samples
* `[ D ]`


---

**Q: Gradient descent is often used to optimize an objective function. Which of the following statements about gradient descent is/are true?

1. Gradient descent is not guaranteed to find the global optimal point.
2. Gradient descent's learning rate $\epsilon$ is always a fixed, very small value.**
- Only 1 is correct
- Only 2 is correct
- Both are correct
- Both are incorrect
* `[ A ]`


---

**Q: How is  SGD different from Gradient descent ?**
- Gradient descent is iterative, but SGD is not
- SGD uses only a subset of training data every time to evaluate the cost function
- Gradient descent doesn’t converge to global minimum, but SGD does
- Gradient descent uses adaptive learning while training and SGD doesn’t
* `[ B ]`


---

**Q: The Hinge loss is **
- A multi class classification
- function with no upper bound on 0-1 loss
- loss function, better than logistic cross entropy
- good at giving least convexity
* `[ C ]`


---

**Q: Which of the following statement is wrong?**
- The negative direction of gradient is the fastest direction the function can reduce.
- When objective function is convex function, solution gradient descent is global optimal
- gradient descent is faster than newton method
- Newton method doesn't need to calculate Hessian Matrix
* `[ D ]`


---

**Q: Generalization feature of a multilayer feedforward network depends on which factor?**
- architectural details
- learning rate parameter
- training samples
- all of the mentioned
* `[ A ]`


---

**Q: Given full model f(x,W,c,w,b)=transpose{w} * max{0,transpose{W} * x,  +c} + b with:

W=[[1,1],[1,3]]
c=[-1,-3]
w=[3,2]
b=0
X=[[0,0],[0,1],[1,0],[1,1]]

The model represents:**
- an AND gate.
- an OR gate.
- an XOR gate.
- none of the above.
* `[ D ]`


---

**Q: Stoichastic Gradient Descent can be used for weight adaptation in neural networks. This is done by:**
- Computing the average of the gradient of the dataset.
- Approximating the gradient using only a few samples of the dataset.
- Using a separate training set to compute the gradient.
- Starting with a random change of weight and using the objective function iteratively to improve the change of weight.
* `[ B ]`


---

**Q: You are given a dataset which is distributed in a two dimensional space ($x_1, x_2$). Data belonging to class A are uniformly distributed inside a circle of radius 1 unit and those belonging to class B, are distributed within +/- 0.2 units from the circumference of a circle with radius 5 units. Both circles are centered at (0,0). Assuming you are using a Neural network with large enough training data and no restriction on size of the network, which of the following combination of input features and activation function does not result in a good classification performance**
- $x_1, x_2, x_1+x_2$ - Linear 
- $x_1, x_2$ - ReLU
- $x_1^2 x_2^2$ - Linear
- All of the above
* `[ A ]`


---

**Q: The general type of decision region obtained when using a feed forward network with no hidden layers is?**
- Regions divided by convex decision boundary
- Arbitrary decision regions
- Regions separated by a hyperplane
- None of the above
* `[ C ]`


---

**Q: Choose the incorrect option on regard to feed forward network.**
- Feed forward network is used for function approximation.
- Feed forward network is a finite directed cyclic graph. 
- Multiple layers of neurons in multilayer perceptron network allows for more complex decision boundaries than a single layer.
- The connections between units are not all equal: each connection may have a different weight.
* `[ B ]`


---

**Q: In reference to gradient of a continuous and differentiable function, which of the following is false.**
- is zero at a minimum 
- is non-zero at a maximum
- is zero at a saddle point
- decreases as you get closer to the minimum
* `[ B ]`


---

**Q: Using gradient descent:**
- Gradient should be exactly 0 at the end of training or the net won’t work.
- We are limited to single variable functions, gradient can’t be used with functions with multiple inputs
- We don't need to compute the result for all samples to update the weights, a batch could suffice
- If the gradient is 0 we know it is in a minimum
* `[ C ]`


---

**Q: Activation functions:**
- Are used to make the input non-linear.
- Should be linear transformations.
- Are used as feedback connections.
- Its parameters should be defined beforehand.
* `[ A ]`


---

**Q: How is the gradient (minimizing the function) for multiple input functions computed?**
- Derivative of the  function
- Partial derivative w.r.t to each input
- Integral 
- None of the above
* `[ B ]`


---

**Q: Method to make input non-linear**
- use generic set of features describing the input
- design feature extractors/ manually engineer feature set
- use a model and learn the set of features describing the input
- all of the above
* `[ D ]`


---

