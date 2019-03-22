# Questions from `optimisation` :robot: 

**Q: Which method is most suited for smoothing out noisy time series?**
- Convolution
- Weighted average
- Infinite impulse response
- Weighted average and convolution
* `[ C ]`


---

**Q: While tuning the learning rate, what is a result of a too low learning rate?**
- No instantiation is found
- It will take a long time for a good instantiation to be found
- A bad instantiation is found
- It does not matter how high the learning rate is for an instantiation to be found
* `[ B ]`


---

**Q: Which of the following about exponentially weighted moving averages is FALSE?**
- If p = 0, then none of the past values have an impact
- The smaller the hyperparameter p, the more weight is given to past values.
- A larger p increases the border effect since it is unknown what happens before time.
- EMWA is used within the Adam algorithm.
* `[ B ]`


---

**Q: Which of the following regarding learning rates is FALSE?**
- The learning rate is a hyperparameter that can be tuned.
- If the learning rate is too high, it will never find a good instantiation.
- Increasing the learning rate over time typically allows for better convergence.
- It is possible to use different learning rates for each parameter.
* `[ C ]`


---

**Q: Recap the 2D ellipse representing contour plot of the loss and red lines representing the trajectory of weight. Which statement below is wrong?**
- The learning rate is arguably the most important parameter to tune
- If LR is too high it will never end a good instantiation
- If LR is too low it will take very long to end a good instantiation
- Smaller learning rate should be used along x axis (major axis)
* `[ D ]`


---

**Q: Which one is wrong about the bias correction of exponentially weighted moving average?**
- Bias correction is needed because the initial values are very low compared with original data
- It is possible to exactly normalizes the average
- For large dataset, the influence of bias correction on the data at the end is very large
- After bias correction, the curve will be more accurate especially at the beginning
* `[ C ]`


---

**Q: Which of the following is true about learning rate values in stochastic gradient descent (SGD)?**
- A very high learning rate enables the network to learn more information.
- A low learning rate allows SGD to converge faster.
- Decaying learning rate over time introduces another hyperparameter to optimize.
- None of the above.
* `[ C ]`


---

**Q: Which of the following statements is true about the Momentum and RMSprop gradient update methods?**
- Momentum smooths the average of noisy gradients using the exponentially weighted moving average technique.
- RMSprop takes advantage of the fact that the variance might be high in the correct direction.
- Out of the two, Momentum is the only one makes use of past gradient update statistics.
- A combination of these two methods is infeasible.
* `[ A ]`


---

**Q: What is momentum in the context of training the parameters of a neural network?**
- The next update is partly determined by the new gradient and partly by the updates of the previous steps
- The time when the learning rate is the highest for training the parameters of the network
- When the gradients of each consequent step are approximately the same
- None of the above
* `[ A ]`


---

**Q: Consider the following statements about the parameter $\rho$ of exponentially weighted moving average:
1. When $\rho$=0, the next update of the parameter is completely dependent upon the most recent gradient
2. When $\rho$=1, the next update of the parameter is completely dependent on the previous updates**
- Only 1 is true
- Only 2 is true
- Both 1 and 2 are true
- None is true
* `[ C ]`


---

**Q: What will happen if the LR is low when finding a good instantiation?**
- it will take long a long time to find the instantiation
- it will find the instantiation very fast
- It will never find a good instantiation
- the effect of noise will not be as big as with a high LR
* `[ A ]`


---

**Q: Which algorithm makes use of both the average and the variance?**
- Adam
- Momentum
- RMSprop
- PCA
* `[ A ]`


---

**Q: What is the difference between machine learning and deep learning?**
- Deep learning does representation learning instead of pattern recognition followed by decision rules
- Deep learning does representation learning instead of pattern recognition followed by a classifier 
- Deep learning does representation learning instead of feature extraction followed by decision rules
- Deep learning does representation learning instead of feature extraction followed by a classifier
* `[ D ]`


---

**Q: What is a Toeplitz matrix?**
- A matrix that is a multiple of the unit matrix
- A matrix that is diagonal-constant
- A matrix that contains the eigenvalues in its diagonal
- A matrix that contains the result of a convolution operation
* `[ B ]`


---

**Q: Which of the following statements about the learning rate in a deep learning network is most correct?**
- Finding a higher optimal learning rate will always result in a better solution
- If the learning rate is too high it will never find a good instantiation
- If you have a high learning rate it will take longer to find a good instantiation
- Optimizing the learning rate will always result in gradient descent finding global optima
* `[ B ]`


---

**Q: Take a look at the following two statements about stochastic gradient descent:
\begin{enumerate}
    \item SGD can have noisy gradient stepts even when close to convergence.
    \item EWMA denoises the data and brings it closer to the original function.
\end{enumerate}
Which of the statements are true?**
- Statements 1 and 2 are true
- Statement 1 is true and statement 2 is false
- Statement 2 is false and statement 2 is true
- Statement 1 and 2 are false
* `[ A ]`


---

**Q: What happens if the learning rate is to high and what happens if the learning to low?**
- In case the learning rate is to high then it will take forever to reach a good initialization. 
In case the learning rate is to low then you will never find a good initialization.
- In case the learning rate is to high then you can never find a good initialization. 
In case the learning rate is to low it will take forever to reach a good initialization.
- In case the learning rate is to high then you can never find a good initialization. 
In case the learning rate is to low then it will not learn enough
- In case the learning rate is to high then you can never find a good initialization.
In case the learning rate is to low then the initialization will become to random
* `[ B ]`


---

**Q: In case you have the Exponentially weighted moving average (EWMA): $S_t = (\rho S_{t-1})+(1-\rho)y_t$. What happens in case you increase $\rho$ from 0.5 to 0.9 without having a bias?**
- In case you increase $\rho$ from 0.5 to 0.9 then the sooner it catches up with the curve of the EWMA, but it is less smooth. While at start it has less affected by the border effect, because past sample matter less
- In case you increase $\rho$ from 0.5 to 0.9 then the later it catches up with the curve of the EWMA, but it is smoother. While at start having a border effect, because it is unknown what happened in past samples.
- In case you increase $\rho$ from 0.5 to 0.9 then the later it catches up with the curve of the EWMA, but it is smoother. While at start it has less affected by the border effect, because past sample matter less.
- None of the above answers is correct
* `[ B ]`


---

**Q: What is the influence of the learning rate on the network’s algorithm?**
- a) The choice for the learning rate influences the accuracy of the network
- b) The choice for the learning rate influences the noise on the optimal solution
- c) The choice for the learning rate influences the robustness of the network
- d) The choice for the learning rate influences the convergence to an optimal solution
* `[ D ]`


---

**Q: The Exponentially Weighted Moving Average (EWMA) uses the weighting factor rho (0 < rho < 1) to weigh data points when calculating the average. What happens to the data points in the past when increasing rho?**
- a) Increasing rho takes older data points shorter into account
- b) Increasing rho takes older data points longer into account
- c) Increasing rho converges the average faster to the true average
- d) None of the above
* `[ B ]`


---

**Q: Which of the following is true about the learning rate (LR)?**
- If LR is high the parameter values might overshoot
- If LR is low it takes long time to converge
- LR depends on the number of parameters
- None of the above
* `[ B ]`


---

**Q: Which of the following is represented by RMSprop algorithm?**
- On average in the good direction
- Variance in the wrong direction is high
- Combination of momentum and adam algorithm
- Both a and c
* `[ B ]`


---

**Q: What is the best way to do a Stochastic Gradient Descent?**
- Stochastic Gradient Descent with Adam
- Stochastic Gradient Descent with momentum
- Stochastic Gradient Descent with RMSprop
- None of the above. Parameter/learning rate optimization is more art than science
* `[ D ]`


---

**Q: RMSprop ...**
- smooths the zero-centered variance of noisy gradients with EWMA
- smooths the average of noisy gradients with EWMA
- combines momentum with EWMA
- applies normalization for each hidden layer
* `[ A ]`


---

**Q: Match the learning rate adjustment techniques and the problems which they adjust for:
1) RMSProp
2) Adam
3) Momentum
a) Average gradient direction
b) Variance of gradient's components
c) Combination of both**
- 1 -> a
2 -> b
3 - > c
- 1 -> b
2 -> a
3 -> c
- 1 -> b
2 -> c
3 -> a
- 1 -> c
2 -> b
3 -> a
* `[ C ]`


---

**Q: What is the benefit of normalizing learned features in a neural network?**
- Normalization reveals otherwise undetectable features of the network's input
- The network doesn't use a much memory
- The neural network can interpret inputs from different encodings
- The network is trained more quickly
* `[ D ]`


---

**Q: Statement 1: The learning rate is arguably the most important parameter to tune. 
Statement 2: If the learning rate is too low, it will never find a good instantiation. **
- Both statements are true
- Statement 1 is true
- Statement 2 is true
- None of the statements are true 
* `[ B ]`


---

**Q: Statement 1: It is a good idea to decay the learning rate over time. 
Statement: It is nog a good idea to take different learning rates for different parameters. **
- Both statements are true
- Statement 1 is true
- Statement 2 is true
- None of the statements are true
* `[ B ]`


---

**Q: During Stochastic gradient descent it could happen that the training takes very long because the steps are very small, what is wrong?**
- The learning rate is too low
- The learning curve is too steep
- The learning curve has too much local optima
- The learning rate is too high
* `[ A ]`


---

**Q: Given is a sample X with 200 datapoints which looks like a ellipse when plotted. How can this ellipse shaped sample be transformed into a zero-centered circle?**
- Subtract the mean and divide all axis by the standard deviation
- Subtract the standard deviation and divide by the mean
- Take only the samples from X such that you get a zero-centered circle
- It is not possible to do this 
* `[ A ]`


---

**Q: What does feature normalization do?**
- It places the mean of inputs at 0
- It sets the variance of inputs across all dimensions equal
- Both of the above
- None of the above
* `[ C ]`


---

**Q: What does momentum do for training a neural network?**
- It uses past gradient statistics to improve learning speed.
- It smooths the average of noisy gradients
- Both of the above
- None of the above
* `[ C ]`


---

**Q: What is true about the learning rate?**
- A too high learning rate has the possibility that it never converges to a good instantation
- A too low learning rate has the possibility that it never converges to a good instantation
- The learning rate is not an important parameter to tune
- The learning rate parameter always tunes itself
* `[ A ]`


---

**Q: Which statement is true for filtering noisy time series data as values are coming in?**
- Convolution can be used because it does not use future values.
- A weighted moving average can be calculated as the size of the to-be-stored data does not matter
- An infinite impulse response filter solves the problem, using a recursive online average, for example
- Linear regression can be used to get rid of the noise
* `[ C ]`


---

**Q: Which one of these is NOT a reason to do Batch Normalization?**
- To improve gradient flow throughout the network
- To allow the use of higher learning rates
- To guarantee convergence to the global minimum for a large data set
- To reduce the strong independence on initialization of weights in the network
* `[ C ]`


---

**Q: If the learning parameter rate is very high, SGD will?**
- Not converge to a good solution
- Converge very slowly
- Work just fine
- Keep the same distribution of network weights that they had when randomly initialized
* `[ A ]`


---

**Q: The following algorithm doesn’t use past gradient update statistics:**
- Momentum
- Stochastic Gradient descent 
-  RMSprop
-  Adam
* `[ B ]`


---

**Q: Batch Norm**
- Normalises learned features
- Applied to only some of the mini batches.
- Doesn’t use weighted average of mini batches
- Computes only average for every mini batch
* `[ A ]`


---

**Q: By which of the following techniques we can NOT improve SGD so that it fast reaches the optimum?**
- Smooths the average of noisy gradients with EWMA (exponentially weighted moving average).
- Smooths the zero-centered variance of noisy gradients with EWMA
- Combine the techniques in A and B.
- Always use a small learning rate to ensure that we reach the optimum.
* `[ D ]`


---

**Q: Which of the following statements is true regarding SGD and Gradient Descent?**
- We can use the techniques on SGD to improve GD.
- Gradient descent is always better than SGD because it won't move towards the  wrong direction.
- SGD is faster than gradient descent because it requires less data to update weights once.
- Gradient descent doesn't need batch normalization because it doesn't move to wrong direction.
* `[ C ]`


---

**Q: For exponentially weighted moving average filters**
- larger $\rho$ parameter value increases the reaction time of the system.
- smaller $\rho$ parameter value results in less noisy output.
- for $\rho=1$ the output value equals to the input value.
- there is no border effect, because we can always compute the missing information accurately from the obtained data.
* `[ A ]`


---

**Q: Choose the correct statement about the Momentum, RMSprop and Adam algorithms.**
- The Adam algorithm combines momentum and RMSprop
- Only the Momentum algorithm makes use of past gradient update statistics
- The variance of the momentum algorithm in the wrong direction is high
- RMSprop smooths the average of noisy gradients with exponentially weighted moving average
* `[ A ]`


---

**Q: Which of the following two statements are true in general?
I If the learning rate is too high it will take very long to find a good instantiation
II The Adam algorithm combines the Momentum and RMSProp alogrithms**
- I is true, II is true
- I is true, II is false
- I is false II is true
- I is false II is false
* `[ C ]`


---

**Q: How can smooth out a noisy time series?**
- We can use a convolution since we have access to future values
- We can use recent values and compute a weighted average
- We recursively compute the exponentially weighted average
- We set the learning parameter to a very small value, so the algorithm takeslonger but smooths out the series
* `[ C ]`


---

**Q: How can you best smooth out a noisy time series as values are coming in?**
- By using a convolution.
- By keeping recent values, and computing a weighted average.
- By using an infinite impulse response filter to recursively compute online average.
- By using a Gaussian filter and computing the least squares.
* `[ C ]`


---

**Q: What is NOT true about exponentially weighted moving average (EWMA)?**
- The lower parameter $\rho$, the higher the bias.
- Because it is unknown what happened before time there is a border effect.
- The formula for bias correction is: $\hat{S}_t=\frac{S_t}{1-\rho^t}$.
- The formula for EWMA is $S_t = (\rho S_{t−1}) + (1 − \rho)y_t.
* `[ A ]`


---

**Q: Training with Stochastic gradient descent can be quite noisy. How can this be improved?**
- Choose the learning rate to be as small as possible
- Use Exponential Weighted Moving Average to smoothen noisy gradients
- Use a larger batch size to get a better estimate of the gradient
- Increase the learning rate over time
* `[ B ]`


---

**Q: What is meant by batch normalization?**
- Choosing always the same batch size instead of different ones
- The output of a previous layer is normalized by subtracting the batch mean and dividing by the batch standard deviation of the previous batches using a moving average
- The output of a previous layer is normalized by subtracting the mean and dividing by the standard deviation of that output
- Normalize each batch by subtracting the batch mean and dividing the batch standard deviation before feeding the data to the network
* `[ B ]`


---

**Q: Which of the following statements is FALSE**
- In EWMA, the larger $\rho$, the longer it takes to the curve to stabilize
- Momentum, RMSprop and Adam are algorithms used in SGD
- Adam is a combination of Momentum and RMSprop
- The closer we are to the minimum of the loss function, the larger should be the steps of the SGD
* `[ D ]`


---

**Q: Which of the following statements about the bias correction in EWMA is FALSE**
- The further in time we go, the closer we get to the real value of $S_t$
- The correction exactly normalizes the average of the weights
- The result of the bias correction is a line following the trend since the beginning and not after a while
- The resulting line is noise-free
* `[ D ]`


---

**Q: What is the learning rate?**
- A parameter that controls how much the weights of the network are adjusted with respect the loss gradient.
- A hyperparameter that controls how much the weights of the network are adjusted with respect the loss gradient.
- A hyperparameter that defines the duration of the learning process.
- A hyperparameter that defines the minimal amount of new information that is gained per layer of the network.
* `[ B ]`


---

**Q: How can we normalize all the feature representations in a deep neural net, including the ones in hidden layers?**
- Apply normalization to the output layer.
- Apply normalization to the input layer.
- Apply normalization to all layers except the hidden layers.
- Apply normalization to all layers, including the hidden layers.
* `[ D ]`


---

**Q: Stochastic Gradient Descent is typically noisy. Which of the following statements is true?**
- It is noisy because we randomly choose one direction to update the weights of the network
- The statement is false. Stochastic gradient descent is as noisy as gradient descent.
- It is noisy because we compute the gradient using only a subset of samples.
- None of the above
* `[ C ]`


---

**Q: The ADAM optimization algorithm:**
- Assumes that on average the updates go in the right direction
- Assumes that the directions with high variance are wrong
- It is a combination of Momentum and RMSprop
- All of the above
* `[ D ]`


---

**Q: Why do you have a bias in the exponentially weighted moving average and how do you compensate for it?**
- Because of the border effect: there is unknown what happened before time. You can compensate for this by introducing a bias term.
- Because of the border effect: there is unknown what happened before time. You can compensate for this by subtracting the mean from you data and scale it with the variance.
- Because of zero padding, there are some zero’s added which decrease the average. You can compensate for this by using a bias term.
- Because of zero padding, there are some zero’s added which decrease the average. You can choose a small /rho, so the curve ‘catches up’ way faster.
* `[ A ]`


---

**Q: Which statement is correct?**
- The learning rate is arguably the most important parameter to tune.
- If the learning rate is too low, it will take very long to find a good instantiation.
- It could be useful to let the learning rate decay over time but then you have to tune another hyperparameter.
- All of the above statements are correct.
* `[ D ]`


---

**Q: what is a property of kernel weights**
- they are not an abstraction of a moving neighbourhood average
- they have an associated color
- they are feature detectors
- none of the above
* `[ D ]`


---

**Q:  what is the purpose of spatial spooling in a convolutional network?**
- to pass a learned kernel over the original input
- to remove negative values from the kernelized processed original input
- to allow combining multiple channels into one output
- None of the above
* `[ C ]`


---

**Q: Regarding the Exponentially weighted moving average (EWMA), what is/are the main difference(s) between having a weight of 0.5 or 0.9?**
- A weight of 0.9 results in a smoother gradient than a weight of 0.5
- A weight of 0.9 results in a higher bias than a weight of 0.5
- Both A and B
- Neither A or B
* `[ C ]`


---

**Q: What is the main advantage of normalization of the learned features?**
- The cost gradient becomes less noisy
- The speed of the training is increased
- The learned features are easier to compare with each other
- All of the above
* `[ B ]`


---

**Q: Which oh these is NOT a Gradient Descent optimization algorithm?**
- Momentum
- Adam
- Stochastic gradient decent
- RmsProp
* `[ C ]`


---

**Q: When is usually applied normalization ?**
- only in the first layer
- only in the last layer
- in all the layers
- never in the layers
* `[ C ]`


---

**Q: Which one is true?
When adapting the learning rate with the exponentially weighted moving average (EWMA):**
- A lower discount factor means that the previously observed data influences the learning rate less.
- A higher discount factor means that the previously observed data influences the learning rate more.
- Introducing a bias makes previously observed data influence the learning rate less.
- Introducing a bias makes previously observed data influence the learning rate more.
* `[ B ]`


---

**Q: Regarding learning rates, which of the following statements is true?**
- A low learning rate will outperform a high learning rate, given enough time.
- A high learning rate always converges quickly.
- When using a changing learning rate, a learning rate which starts high and ends lower is wanted
- Exponentially weighted moving average methods for updating the learning rate are not feasible in high-dimensional time series due to memory constraints.
* `[ C ]`


---

**Q: Which of the following statements about smoothing out an optimization time series does not unequivocally hold?**
- The use of convolution is not possible due to access to future values being required.
- Keeping recent values is typically not feasible, due to the space complexity needed to store past configurations.
- Recursively computing an online average consists of weighing past values and incorporating them in the current average.
- An Exponentially Weighted Moving Average (EWMA) weighs past values more heavily than recent values.
* `[ D ]`


---

**Q: Which of the following statements about BatchNorm does not unequivocally hold?**
- A network using BatchNorm can learn to undo normalization procedures.
- BatchNorm is not applied to hidden layers.
- BatchNorm is typically applied to all learned features.
- BatchNorm normalizes the shape of an optimization landscape to have similar variance across dimensions.
* `[ B ]`


---

**Q: What are the components of a typical convolutional neural network layer?**
- detector stage -> convolution -> pooling stage
- Convolution -> detector stage -> pooling stage
- feature mapping -> pooling stage -> convolution
- feature exctraction -> classifier
* `[ B ]`


---

**Q: What is/are the main advantages of convolution?**
- sparse interactions
- parameter sharing
- equivariant representations
- all of the above
* `[ D ]`


---

**Q: Which statement about convolutional net kernels is true?**
- Using the identity matrix as kernel will give the same result as the moving neighborhood average
- The weights of a kernel are feature detectors
- Using a normalized kernel A and kernel B = 2*A will give the same output, even before normalization of B
- Noise in input images cannot be removed with kernels
* `[ B ]`


---

**Q: Which statement is false?**
- Padding prevents the input data getting smaller in each layer
- Pooling can be used to reduce the memory usage
- Striding can be used to increase the receptive field of a node in a certain layer
- All of the above are true
* `[ D ]`


---

**Q: How can we reduce use of memory to a minimum when computing EWMA?**
- Use a rho-value close to 0.1
- Use a rho-value close to 0.9
- Use a rho-value greater than 1
- Use a rho-value close to 0.5
* `[ A ]`


---

**Q: Which is no efficient algorithm to improve SGD?**
- Center inputs at zero and equal dimension variance
- Smooth the zero-centered variance of noisy gradients with EWMA
- Update the new weights by the average of all past gradients multiplied by a overtime decreasing Learning Rate
- No correct answer provided
* `[ C ]`


---

**Q: Which of the following statements are true?

1. RMSprop: On average in the good direction.
2. Momentum: Variance in the wrong direction.**
- Statements 1 is correct, statements 2 is incorrect
- Statement 1 is incorrect, statement 2 is correct
- Both statements are correct 
- Both statements are incorrect
* `[ D ]`


---

**Q: What is the reason that for a larger ρ, the later the Exponentially weighted moving average (EWMA) ‘catches’ up.**
- Border effect, unknown what happened before time.
- Out of memory, this method requires a lot of memory.
- The system sometimes is too rigid, need to add more parameters.
- None of the above.
* `[ A ]`


---

**Q: What is the use of the bias correction( dividing by (1 - \rho^t)) in the calculation of the exponentially weighted moving average?**
- To normalize the error across various dimensions.
- To remove the border effect caused by the lack of information before the start.
- Both A and B.
- None of the above.
* `[ B ]`


---

**Q: P smooths the average of the noisy gradients with EWMA. Q smooths the zero-centred variance of noisy gradients with EWMA. R combines P and Q. Name P, Q and R.**
- P: Stochastic Gradient Descent with RMSprop, Q: Stochastic Gradient Descent with momentum, R: Stochastic Gradient Descent with Adam.
- P: Stochastic Gradient Descent with momentum, Q: Stochastic Gradient Descent with RMSprop, R: Stochastic Gradient Descent with Adam.
- P: Stochastic Gradient Descent with momentum, Q: Stochastic Gradient Descent with Adam, R: Stochastic Gradient Descent with RMSprop.
- None of the above are correct.
* `[ B ]`


---

**Q: In practice, which of the following is not a threat to the convergence of the gradient descent?**
- Too large learning rate.
- Noisy gradient descent.
- Too low learning rate. 
- A lot of non-acceptable local minima in the target function.
* `[ D ]`


---

**Q: The main ideas behind Momentum, RMS and Adam optimizations are (respectively):**
- Use the average, use the variance, and combine average with variance.
- Use the variance, use the average, and combine the average with the variance.
- Use the average, combine the variance with the average, and add to the combination a temporal weighting term.
- Use the average with a temporal weighting term, use the variance with the average, and use only the variance.
* `[ A ]`


---

**Q: Which statement is true about the Learning Rate in Gradient Descent?**
- If the learning rate is too high, you usually don't converge.
- If the learning rate is too low, you usually don't converge.
- If the learning rate is too high, converging becomes very computationally expensive.
- None of the above.
* `[ A ]`


---

**Q: When normalizing features by subtracting the mean and dividing by the variance, what is a problematic assumption we make?**
- The variance of all features is approximately equal.
- The mean of all features is approximately equal.
- The ellipsoid containing the data is aligned with the axes.
- All of the above.
* `[ C ]`


---

**Q: What is not the result of a learning rate that is set too high?**
- The model might not converge to the optimal value.
- Training procedure typically takes a very long time.
- The model achieves a very low accuracy on the test set.
- The model achieves a very low accuracy on the train set.
* `[ B ]`


---

**Q: What is true about using Exponentially weighted moving averages?**
- It is a more accurate representation of the true mean of the data.
- It can increase the computational burden compared the standard mean.
- It requires less memory than computing the standard mean.
- It leads to better model generalization.
* `[ C ]`


---

**Q: Which algorithm uses exponentially weighted moving average (EWMA)?**
- Momentum
- RMSprop
- Adam
- all of the above
* `[ D ]`


---

**Q: Adam smoothes:**
- the average of noisy gradients
- the zero-centered variance of noisy gradients
- both
- nothing
* `[ C ]`


---

**Q: Which of the following statement is not correct?**
- Small learning rate can cause the cost function to converge to a local minimum
- Noisy time series data can not be fed into CNNs because of access to the feature values
- Normalization is not required in the CNNs
- An adaptive learning rate can help the network to find the best hyperparameter faster
* `[ C ]`


---

**Q: Which of the following statement is correct?**
- Shallow nets can sometimes learn the functions learned by the deeper nets, even when restricted to the same number of parameters.
- A shallow net  have less units in each layer to be able to mimic a deeper network with the same number of parameters.
- Data augmentation to create new dataset from the existing datasets always results in overfitting .
- Training shallower models without convolution layers can result in faster training time and better accuracy
* `[ A ]`


---

**Q: Why does SGD not converge directly towards the optimal parameters?**
- SGD contains a lot of noise
- SDG does not converge to optimal parameters
- The step size for the algorithm is too large
- SDG can get stuck at so called "saddle points"
* `[ A ]`


---

**Q: What is the main advantage of computing the average of a noisy time series recursively?**
- To avoid the need of having to store the history for every parameter
- To apply noise reduction functions more efficiently
- Recursive methods run much faster than traditional averaging
- To compensate for the missing future values needed for traditional convolution
* `[ A ]`


---

**Q: SGD is ...**
- Noisy only near the optimum
- Noisy far away from the optimum
- Noisy everywhere
- Noisy nowhere
* `[ C ]`


---

**Q: The advantage of exponentially weighted moving average is**
- that you only need to store one value from previous iterations
- a global optimum is always obtained after iterating
- you can stop after 100 steps, for machine precision gives a lower bound on the distance from the optimum
- you do not have to set additional hyper parameters
* `[ A ]`


---

**Q: How can learning rate influence your deep learning system?**
- If learning rate is too high, deep learning system might end up with high classification error.
- If learning rate is too high, deep learning system can still recover and end up with good classification error
- If learning rate is too low, deep learning system will have a very large classification error
- If learning rate is too low, deep learning system can still recover and end up with good classification error
* `[ A ]`


---

**Q: Suppose you are trying to normalize your deep neural network in order to avoid excessively large or small weights. How would you apply such a normalization?**
- Normalize data before passing it through the network, and also normalize data coming out of activation functions at each layer
- Normalize data before passing it through the network, thus there is no need to normalize at each layer
- normalize data coming out of activation functions at each layer, thus there is no need to normalize 
- Normalize data before passing it through the network and normalize only on the output layer
* `[ A ]`


---

**Q: Which statement is correct?**
- Tuning learning rate in deep network  always increases the error rate
- Tuning learning rate in deep network  always decreases the training  error rate
- Tuning learning rate in deep network  always increases the true error rate
- none of these are correct in general
* `[ D ]`


---

**Q: In Exponentially weighted moving average, **
- Bias is always high
- time series is random
- Bias is always low
- Nothing can be generalised like (a), (b)  and (c) 
* `[ D ]`


---

**Q: What do the kernel weights represent?**
- Features
- Pixels
- Both A and B
- None of the Above
* `[ A ]`


---

**Q: How do convolutional networks output images of equal resolution as the input image?**
- Padding
- Interpolating
- Extrapolating
- None of the above
* `[ A ]`


---

**Q: Choose the best answer:**
- The same learning rate e for x-axis and y-axis parameter should be used for a contour plot of the loss with an ellipse shape
- To smooth out a noisy time series with values coming convolution can be used but is not recommended
- To smooth a noise time series using with exponentially weighted moving average, increasing the coefficient parameter, a noise time series will become smoother
- Weight decay should be used for bias correction with exponentially weighted moving average
* `[ C ]`


---

**Q: Which of the following notions is TRUE?**
- Momentum:  Variance in the wrong direction is high
- RMSprop:  On average in the good direction
- Adam:  Combines momentum and RMSprop
- RMSprop: makes use of future gradient update statistics
* `[ C ]`


---

**Q: Which of the following is not true about Stochastic Gradient Descent?**
- A crucial parameter for the SGD algorithm is the learning rate.
- In practice, it is necessary to gradually decrease the learning rate over time.
- The learning rate of Stochastic Gradient Descent is usually best to choose by monitoring learning curves that plot the objective function as a function of time.
- The SGD gradient estimator doesn't introduce the source of noise of random sampling of m training examples. 
* `[ D ]`


---

**Q: Which of the following is not true about Momentum?**
- The method of momentum is designed to accelerate learning.
- Momentum aims to solve two problems: poor conditioning of the Hessian matrix and variance in the stochastic gradient.
- The name momentum derives from a physical analogy, in which the positive gradient is a force moving a particle through parameter space.
- In the method of Stochastic gradient descent with momentum, the size of the step depends on how large and how aligned a sequence of gradients are. 
* `[ C ]`


---

**Q: When tuning the learning rate in a 2D space, the line that connects the update values tends to have a zigzag shape when trying to reach the global minimum of the gradient. Why is this?**
- Because the gradient is stochastic.
- Because tuning the gradient is hard.
- Because the gradient is computed for two dimensions.
- Because this is usual for gradient descent.
* `[ A ]`


---

**Q: Which of the following algorithms tries to capture the good direction on average for gradient updates?**
- RMSprop
- Adam
- Momentum
- RANSAC
* `[ C ]`


---

**Q: Which of the following are good ways to smooth out a  noisy time series:**
- Apply convolution.
- Keep recent values and compute a moving average.
- Recursively compute online average.
- All of the above.
* `[ C ]`


---

**Q: In the Exponentially Weighted Moving Average (EWMA), which of the following is false:**
- Increasing $\rho$, will increase the importance of recent values.
- Increasing $\rho$, will decrease the importance of recent values.
- Without bias correction, increasing $\rho$ will increase the time before the average 'catches up' to the time series. 
- With bias correction, increasing $\rho$ will increase the noise of the average at the start of the time series. 
* `[ A ]`


---

**Q: How to overcome noisy SGD?**
- Increase learning rate
- Decrease learning rate
- Decrease learning rate over time
- Increase learning rate over time
* `[ C ]`


---

**Q: How to smooth out a noisy time series?**
- Recursively compute online average
- Keep recent values, and compute a weighted average
- Convolution
- None of the above
* `[ A ]`


---

**Q: Why do wee need bias correction for EWMA?**
- The algorithm is unstable without proper bias.
- In the initial phase the output value doesn't follow the input data because of the zero initial value.
- To reduce the variance of the output.
- To bias the output towards the latest input values.
* `[ B ]`


---

**Q: What is the transfer function of an EWMA IIR stage?**
- H(z) = \frac{\alpha}{1-(\alpha)z^{-1}}
- H(z) = \frac{(1-\alpha)}{1-(1-\alpha)z^{-1}}
- H(z) = \frac{\alpha}{1-(1-\alpha)z^{-1}}
- H(z) = \frac{(1-\alpha)}{1-(\alpha)z^{-1}}
* `[ C ]`


---

**Q: SGD is noisy. How can we limit its noisy behaviour close to a lower loss? **
- By increasing the learning rate over time
- By decaying the learning rate over time
- By switching to normal GD close to the wanted loss
- That is impossible
* `[ B ]`


---

**Q: What statement is true:

1. With EWMA, a larger rho gives a smoother average
2. With EWMA, a smaller rho 'catches up' later at the beginning of a sequence.**
- 1
- 2
- both
- neither
* `[ A ]`


---

**Q: What learning rate would you choose for a deep neural network and why?**
- A constant high one so that the network is trained fast!
- A constant low one so that the loss converges smoothly.
- A decaying learning rate so that the loss could converge smoothly and the network would be trained quicker!
- Impossible to say without more information.
* `[ C ]`


---

**Q: What is the most effective gradient descend optimization algorithm to dampen oscillations? **
- Momentum
- Adam
- RMSprop
- Impossible to answer
* `[ A ]`


---

**Q: Which of the following statements about Convolution Networks is false?**
- Convolution networks are simply neural networks that use convolution in place of general matrix multiplication in at least one of their layers.
- A pooling function replaces the input of the net at a certain location with a summary statistic of the nearby input. 
- Discrete convolution can be viewed as multiplication by a matrix, but the matrix has several entries constrained to be equal to other entries. This is known as a Toeplitz matrix.
- Convolutional networks typically have sparse interactions. This is accomplished by making the kernel smaller than the input. For example, when processing an image, the input image might have thousands or millions of pixels, but we can detect small, meaningful features such as edges with kernels that occupy only tens or hundreds of pixels.
* `[ B ]`


---

**Q: Evaluate the following two statements:
1.	Kernel weights are feature detectors
2.	First translating, and then convolving, is the same as first convolving and then translating**
- Both are true
- None are true
- 1 is true and 2 is false
- 1 is false and 2 is true
* `[ A ]`


---

**Q: Which of the following is NOT a useful algorithm related to Gradient Descent?**
- Momentum
- Distribution
- RMSprop
- Adam
* `[ B ]`


---

**Q: What happens if the learning rate is too large when using Gradient Descent?**
- Minima will not be found because the steps are too large to converge.
- Learning will halt as it gets stuck in a minimum.
- Connections in the neural networks will become unusable.
- None of the above.
* `[ A ]`


---

**Q: What is the reason behind the need of bias correction in exponentially weighted moving average of a noisy time series?**
- What happened before the start of the time series is unknown
- Values are very low at the beginning
- Border effect
- All of the above
* `[ D ]`


---

**Q: Which of the following statements is not true?**
- Momentum smooths the average of noisy gradients with Exponentially Weighted Moving Average (EWMA)
- RMSprop smooths the zero-centered variance of noisy gradients with EWMA
- Adam makes use of past gradient update statistics
- None of the above
* `[ D ]`


---

**Q: What is not important about stochastic gradient descent's convergence?**
- Initial point
- Learning rate
- Regularization factor
- All of the above
* `[ C ]`


---

**Q: When to normalize learned features?**
- Before activation
- After activation 
- Does not matter
- It is debated
* `[ D ]`


---

**Q: A possible case of the gradient descent moving in the opposite direction of the minima of the loss function can happen if the learning rate is**
- too low
- zero
- too high
- None of the above
* `[ D ]`


---

**Q: Which can be a good strategy to tune the learning rate hyper-parameter ?**
- Initialize with small value and increase it exponentially
- Initialize with large value and decrease it exponentially
- Initialize with a small value and keep the learning rate constant
- Initialize with a large value and make it small and then large in every alternate step 
* `[ B ]`


---

**Q: What strategy is not used for smoothing nosiy time series as values are coming in?**
- Convolution
- Recursively compute online average
- Exponentially weighted moving average
- Bias correction
* `[ A ]`


---

**Q: What of the following statement of algorithm is not true?**
- Momentum: On average in the good direction
- RMSprop: Variance in the wrong direction is high
- Adam: Combines momentum and RMSprop
- SGD: If Learning rate is too low it will not take very long to find a good instantiation
* `[ D ]`


---

**Q: What approach is used in noisy SGD?**
- Reduce the LR over time
-  Keep LR in a low value
-  Keep LR in a high value
-  Increase the LR over time
* `[ A ]`


---

**Q: Which of the following is TRUE for batch normalization?**
- In general speaking, it is an optimal algorithm.
- Batch norm presents an elegant approach that can almost re-parameterize all deep networks
- It use forward propagation to calculate the mean or standard deviation.
- δ should be set to zero.
* `[ B ]`


---

**Q: When training a network using Stochastic gradient descent, it is observed that the parameters being learned move in a "zig-zag" manner towards the optimal value, what might be a reason for this?**
- The zig-zag movement is a property of the optimization techniques being used on the learning parameter
- The zig-zag movement is due to the use of random sampling of the gradient as opposed to the true gradient of the data which would allow for a more direct approach to the optimal position
- The zig-zag movement is due to an improperly chose learning parameter and can be eliminated by using proper optimization techniques.
- None of the above
* `[ B ]`


---

**Q: Which of the following describes the effect of having a large $\rho$ ( $>0.9$) value in the Exponential weighted moving average**
- The more recent values have the largest effect on the calculated average
- A larger number of former values contribute to the moving average
- The oldest values under consideration contribute the most to the moving average
- None of the above
* `[ B ]`


---

**Q: Which of the following statements about the learning rate is true?**
- A learning rate is a hyper-parameter that controls how much the weights of the network are adjusted with respect to the loss gradient.
- If the learning rate is too low it will never find a good instantiation.
- If the learning rate is too high it will take very long to find an instatiation.
- It is best to use the same learning rates for x-axis and y-axis parameters. 
* `[ A ]`


---

**Q: Which of the following statements about stochastic gradient descent (SGD) algorithms is false?**
- SGD with momentum works better and is faster than SGD.
- Adam is a combination of momentum and RMSprop.
- RMSprop smooths the average of noisy gradients with EWMA.
- Momentum, RMSprop and Adam make use of past gradient update statistics.
* `[ C ]`


---

**Q: What is the main difference between stochastic gradient descent and 'regular' gradient descent?**
- SGD approximates gradient from small sample set
- SGD stochastically determines the learning rate
- SGD adds white noise to the data, to be more robust.
- SGD starts at multiple random initial positions
* `[ A ]`


---

**Q: What does RMSprop do?**
-  It smooths the average of noisy gradients with EWMA
-  It adds noise to the average of the gradients, for robustness
- It smooths the zero-centered variance of noisy gradients with EWMA
-  It adds noise to the zero-centered variance of the gradients, for robustness
* `[ C ]`


---

**Q: Which of the following statements is true?**
- The momentum smooths the zero-centered variance of noisy gradients with EWMA
- RMSprop combines adam with momentum
- When using adam, one needs to set two rhos as hyperparameter
- RMSprop makes use of stogastic gradient descent while momentum does not.
* `[ C ]`


---

**Q: What is not an advantage of the exponentially weighted moving average?**
- It has access to future values
- It has no need for hyperparameter optimization
- It can deal with huge high-dimensional time series
- It is able to deal with noisy time series
* `[ B ]`


---

**Q: Which statements about the Adaptive Optimization Methods are true?

Statement 1: In momentum method "Coefficient of Momentum" which is the percentage of the gradient retained every iteration. For an update, this adds to the component along (one type of the weights w2), while zeroing out the component in w1 direction (second type of weights). This helps us move more quickly towards the minima. For this reason, momentum is also referred to as a technique which dampens oscillations in our search.

Statement 2: Suppose if we are heading towards the minima, and we want to slow down so as to not to overshoot the minima. RMSProp automatically will decrease the size of the gradient steps towards minima when the steps are too large (Large steps make us prone to overshooting)

Statement 3: While momentum accelerates our search in direction of minima, RMSProp impedes our search in direction of oscillations.

Statement 4: Adam (adaptive moment optimization) essentially combines the heuristics of both Momentum and RMSProp. Here are the update equations.**
- 1 and 4
- 2 and 3 
- 1 2 and 3 
- All true 
* `[ D ]`


---

**Q: Example of Batch normalization can be: take the dataset that is distributed in terms of values from 0 to 1000 and normalize it such that it is ranging from zero to one. Which statements about Batch Normalization are true?

Statement 1: Batch Normalization is a method to reduce internal covariate shift in neural networks,  leading to the possible usage of higher learning rates

Statement 2: Consider the deep neural network consisting of the 15 layers. What we do by providing the batch normalization is: “It makes weights later (or deeper) in the neural network, say the weights in the layer 10, to be more robust to changes to the weights in the earlier layer (say in the layer 1)”. What we do is we not tweak the exact values of weights and bias but sustain the average and mean. (Say mean 0 and variance 1 ) Mean and variance can be controlled by two explicit parameters such as beta and gamma that can be trained as well.

Statement 3: The Batch norm essentially reduces the problem of input values changing. It causes these input values to become more stable; so the later layer in the neural network have more firm ground to stay on.

Statement 4: Having the sigmoid activation functions in your network you would preferer to have the zero mean and variance 1 normalized weights**
- 1 2 and 3
- 2 3 and 4
- 1 and 2 
- All answers are true
* `[ A ]`


---

**Q: The effect of batch normalization...**
- Makes SGD converge at the faster rate.
- Normalizes the variance of the sample.
- Centralizes the mean of the sample around point 0,0.
- All of the above.
* `[ A ]`


---

**Q: Adam algorithm combines**
- RMSprop and momentum
- RMSprop and training sample transformation
- RMSprop and variable step size
- Momentum and directional search
* `[ A ]`


---

**Q: What happens when you convolve a picture with the negative of a vertical corner detection kernel?**
- Now horizontal corners are highlighted
- Now corners are highlighted
- You get the negative of the image you received previously
- It will no longer be able to highlight anything
* `[ C ]`


---

**Q: Can a single convolution with a kernel be used to translate an image?**
- No, because the image will always reduce in size
- No, this requires multiple convolutions
- Yes, with the right kernel this is possible
- Yes, but only with a certain type of image
* `[ C ]`


---

**Q: Which of these is NOT optimalization algorithm?**
- Momentum
- Adam
- Eve
- RMSprop
* `[ C ]`


---

**Q: Why we cannot keep recent values, and compute a weighted average during smoothing out a noisy time series as values are coming?**
- We ran out of memory
- Is too computacily demanding
- It would not help
- We can
* `[ A ]`


---

**Q: Which statement about the algorithms for optimization learning rate is TRUE?**
- Momentum is on average in to the good direction
- In RMSprop variance is high in the right direction
- RMSprop combines Adam and Momentum
- Adam combines Stochastic Gradient Descendant and Momentum
* `[ A ]`


---

**Q: Which statement about batch normalization is TRUE?**
- Batch normalization is applied only to input and output nodes
- We only use batch normalization before activation function
- Batch normalization is applied for all training set
- For normalization on test set, weighted average of last mini batches is used
* `[ D ]`


---

**Q: About batch normalization, it can be said that:**
- Is applied on every input batch by centering the values at zero
- Is applied in the same way on both training and testing time
- Is applied for every mini-batch to normalize the learned features at the first training epoch
- When applied at test time it uses a weighted average of the last mini-batches 
* `[ D ]`


---

**Q: Which of the following is a consequence of the border effect in EWMA?**
- A large $\rho$ generates a smooth curve
- A large $\rho$ helps to “catch up” fast
- The usage of bias correction to help “catching up” faster
- The usage of past gradient update statistics
* `[ C ]`


---

**Q: What is not a danger of a wrongly chosen learning rate?**
- Taking long to find a good instantiation
- Not finding a good instantiation
- Getting stuck in a local optimum
- Overfitting on the data set
* `[ D ]`


---

**Q: What is the main goal of batch normalization?**
- Improving stability and performance
- Reducing the chance of getting stuck in a local optimum
- Improving efficiency
- Reducing overfitting
* `[ A ]`


---

**Q: Which of the following options makes the sentence false? The method of momentum…**
- aims to solve the variance in the stochastic gradient.
- doesn’t introduce another hyperparameter.
- accumulates an exponentially decaying moving average of past gradients and continues to move in their direction.
- is designed to accelerate learning.
* `[ B ]`


---

**Q: Which of the following options makes the sentence true? The learning rate should…**
- correspond to a low value.
- remain a constant value, during the training.
- correspond to a high value.
- be reduced over time, during training.
* `[ D ]`


---

**Q: Why do we need bias correction when using optimizer?**
- To make the result better at the starting phase
- To make the result better at the ending phase
- To make the result better at all the process
- To make the result more stable at the ending phase
* `[ A ]`


---

**Q: Which is NOT the reason of using Exponentially weighted moving average (EWMA)?**
- Give new value a higher weight
- Reduce the usage of memory
- Speed up calculation
- Make the convergence slower
* `[ D ]`


---

**Q: Can we use convolution in a noisy time series?**
- No, we have too many values
- No, because we don’t have access at the future values
- Yes, but is computationally expensive
- Yes, we can memorize all the values
* `[ B ]`


---

**Q: Why do we need a bias correction in the Exponentially weighted moving average (EWMA)?**
- To correct a border effect, it’s unknown what happened before time
- We want to change the values at the beginning, keeping the same values for large t
- We want to avoid EWMA to start from 0
- All the previous answers
* `[ D ]`


---

**Q: Which is the great advantage of using an exponetially weighted moving average to keep track of previous values (in time series for instance)?**
- It is computed online so it reduces the use of memory
- It gives more weight to the more important samples, that are the starting ones
- It does not need any correction of biases during its proper evaluation
- It is basically equivalent to convolving the time series
* `[ A ]`


---

**Q: If we want a deep net to learn normalized features, we can apply normalization on the input dataset before feeding it to the net. Is it enough?**
- Yes, as normalization is kept as it is also in the following forward passes in the net.
- No, we need to normalize just the input dataset once in reaches the layer right befor the output one (to be sure that the output is really normalized)
- No, we need to apply normalization also for each hidden layer
- No, it is not possible for a deep net to learn just normalized features
* `[ C ]`


---

**Q: Which statement about exponentially weighted moving average is correct?**
- A moving average with an increased rho means the moving average is more noisy
- It uses convolution to determine an average based on past and future values
- A bias correction is needed especially for moving averages with decreased rho’s
- A moving average with a decreased rho means only the near-previous values are relevant
* `[ D ]`


---

**Q: For batch normalisation, which parameters are normalised?**
- Input (x)
- Weights (w)
- Input to activation function (z)
- Output (y)
* `[ C ]`


---

**Q: How would you optimally alter the learning rate while training a neural network as it goes through multiple epochs?**
- Start big and decrease over time
- Start small and increase over time
- Keep it constant and only change the optimizer type from gradient descent to Adam.
- None
* `[ A ]`


---

**Q: What could be one reason why our gradient descent might not converge if we only consider the property of our learning rate hyperparameter**
- Too high
- Too low
- Avg of gradients in previous step
- None
* `[ A ]`


---

**Q: What is a problem of having a learning rate too high?**
- The best solution will be found too quickly such that not all solutions are explored.
-  The best solution may never be found.
- Learning will take too long
- The solution is more likely to end up in a local minima
* `[ B ]`


---

**Q: What of the following is NOT a good method to optimize a learning rate?**
- Decay
- Different learning rates for different parameters
- Using the average/variance of each dimension
- Increasing it for each time step
* `[ D ]`


---

**Q: Which of the following options are correct?**
- In batch normalising, gradient updates are coordinated across multiple layers at the same instant
- Full batch gradient descent is better that stochastic gradient descent because calculated gradient is closer to the true gradient of the data
- Both of the above options are correct
- None of the options are correct
* `[ A ]`


---

**Q: Which of the following options are correct?**
- Learning rate is best set by analysing learning curve that plot the objective function as a function of time
- RMSProp uses an exponential decaying average to discard history from being extreme past while updating weights
- Both of the above options are correct
- None of the above options are correct
* `[ C ]`


---

**Q: Which of the following statements is true:**
- Suppose that we know our loss function is convex. Making the learning rate very small will cause to not convergence.
- Decay will make sure that the learning rate will increase over time.
- RMSprop adds momentums
- None of the above
* `[ D ]`


---

**Q: Suppose that we have gathered some gradient statistic and found that past gradient update moved in the right direction on average. Based on this which of the following algorithms might be useful to adapt the learning rate:**
- SGD
- EWMA
- RMSProp
- Momentum
* `[ D ]`


---

**Q: Which of the following statements about the learning rate of the stochastic gradient decent is false?**
- If learning rate is too high, the algorithm will never find a good instantiation
- If learning rate is tool low,  it will take very long to find a good instantiation
- A common technique used is the reduction of the learning rate over time
- Learning rate is the less important parameter to tune in SGD
* `[ D ]`


---

**Q: Which of the following statements about the following (related to SGD) algorithms is false?**
- RMSprop algorithm considers that the variance in the wrong direction is high
- Momentum algorithm considers that the average of the previous updates is on the right direction
- Both Momentum and RMSprop algorithms make use of past gradient update statistics
- Both Momentum and RMSprop use the same learning rate per parameter
* `[ D ]`


---

**Q: Recall that if $y_n$ is a sequence of values and $0 \leq \rho \leq 1$ is a parameter, one can define the exponential weighted moving average (EWMA) of the time series $\{y_n\}$ by $S_0 = 0$, $S_{t} = \rho S_{t-1} + (1 - \rho)y_t$. Having that defined, we can define $\hat{S}_t$, the bias correction of the exponential weighted moving average, by $\hat{S}_t = \frac{S_t}{1 - \rho^t}$, where we divided by $(1 - \rho^t)$. We divide by $1 - \rho^t$ because**
- dividing $S_t$ by $1 - \rho^t$ means that $S_t$ is a weighted average of the values $y_1, \cdots, y_t$ where the coefficients add up to one.
- $1 - \rho^t < 1$, which means $\hat{S}_t > S_t$.
- $1 - \rho^t \to 1$ when $t \to \infty$, meaning $\hat{S}_t \to S_t$ as $t \to \infty$.
- no particular reason at all; any number in $[0, 1]$ would serve the purpose of bias correction.
* `[ A ]`


---

**Q: Let $x_i$ be a vector representing a data point, out of $N$ total data points. Defining $\bar{x} = \frac1N \sum_{i=1}^N x_i$, the average of all the data points, and $\sigma^2 = \frac1N \sum_{i=1}^N (x_i - \bar{x})^2$, the variance of the data points (calculated component-wise), then we know that**
- defining $x_i'$ as $x_i' = x_i - (\bar{x} + v)$ gives a new data set $x'$ that has been centred around the vector $v$.
- defining $x_i'$ as $x_i' = x_i - \bar{x}$ defines a new data set $x'$ centred around the origin and with unit variance.
- defining $x_i'$ as $x_i' = \frac{x_i + \bar{x}}{\sqrt{\sigma^2}}$ gives a new data set $x'$ that has been centred around the origin and with unit variance.
- defining $x_i'$ as $x_i' = \frac{x_i}{\sqrt{sigma^2}} - \bar{x}$ recenters the data around the origin and sets its variance to $1$.
* `[ A ]`


---

**Q: Why is the learning rate an important parameter to tune in SGD?**
- Because if it is too small, the algorithm will take too much time to get a good solution
- Because if it is too big, the algorithm may not get a good solution
- All of the above
- It is not important
* `[ C ]`


---

**Q: Why is feature normalization useful?**
- It makes the solution change to a better one without depending on the initialization of the training
- It helps speeding up the training
- All of the above
- None of the above
* `[ B ]`


---

**Q: If you have multiple dimensions, would you use the same learning rate for each dimension?**
- Yes, because this is what is called "Batch Optimization"
- Yes, as long as we take a small enough learning rate
- No, because each dimension has different scales towards the loss
- No, because we only need to update one single dimension consistently
* `[ C ]`


---

**Q: What is the benefit of doing EWMA to keep track of past gradient update statistics?**
- No large memory required
- Smooth out noisy values as new values come in
- No need to access future values
- All of the above
* `[ D ]`


---

**Q: What’s the point difference between the classic Stochastic Gradient Descent and its extended algorithm(SGD with momentum, SGD with RMSrop, SGD with Adam)?**
- They converge much faster than classic DG because they randomly start in different positions every time the algorithm starts
- They converge much faster than classic SDG because they don’t check the whole set of values of the objective function but a batch of values to minimise that function
- They converge much faster than classic SDG because they adapt their learning step to the objective function they want to minimise
- The point difference of between these algorithm doesn't lie on the convergence of the minimisation problem
* `[ C ]`


---

**Q: The normalisation of the learned features at test time depends on:**
- Performing a batch norm for the last mini-batch
- The computation of the mean value without the need of the variance
- Using the weighted moving average of every mini-batch
- Performing a batch norm for every mini-batch
* `[ D ]`


---

**Q: Why is an IIR filter instead of an FIR filter used to smooth noisy time series**
- Because IIR filters are recursive and automatically keep track of previous values without storing them
- Because IIR filters are easier to implement
- Because IIR filters use convolution which is very powerful
- There is no advantage in choosing one over the other
* `[ A ]`


---

**Q: Why does the exponentially weighted moving average need bias correction?**
- Because the current output will be biased by past values
- Because there are no past values at t=0
- To smooth out the effect of outliers
- To make sure that the average goes through the center of all points
* `[ B ]`


---

**Q: 	Finding an optimum value is strongly dependent on the learning rate. For this reason, the learning rate should be chosen adequately. Which of the following statements is wrong:**
- The case of having a low learning rate is very computational expensive
- The case of having a very high learning rate is ideal to find the optimum
- The case of having a low learning rate will lead to finding a value close to the optimum
- Decreasing the learning rate step over time will lead to finding an optimum
* `[ B ]`


---

**Q: Which of the following statements is false:**
- The Momentum algorithm performs better than Stochastic Gradient Descend
- The RMSprop algorithm has a high variance in the wrong direction
- Adam algorithm is a combination between Momentum and RMSprop
- Stochastic Gradient Descend is the best algorithm to find the optimum
* `[ D ]`


---

**Q: What hyperparameter can be tuned with momentum?**
- Learning rate
- Batch size
- Activation function
- Number of epochs
* `[ A ]`


---

**Q: What is not the purpose of batch normalization?**
- Remove effect of covariate shift
- Improve ability of model to generalize
- Speed up learning
- Allow model to learn from unseen data
* `[ D ]`


---

**Q: the momentum algorithm is based on which of the following assumption?**
-  On average the gradient updates are in the right direction
- the variance in the wrong direction is high
- the variance in the correct direction is high
- On average the gradient updates are in the wrong direction
* `[ A ]`


---

**Q: In the exponential weighted moving average algorithm what happens when the value of p is high (say 0.98) compared tot when it’s low (0.1) ?**
- The present values are taken more into consideration when smoothing.
- The algorithm converges very slowly .
- The past values are taken more into consideration when smoothing.
- The algorithm converges very fast.
* `[ C ]`


---

**Q: Which of the following algorithms is NOT useful for the optimization of the learning rate?**
- RMSprop
- Adam
- Momentum
- Batch normalisation
* `[ D ]`


---

**Q: When applying the exponentially weighted moving average (EWMA) on some noisy time series, what is the main purpose of ‘bias correction’?**
- To prevent the values at the start of the curve from being too low
- To prevent the values on the entire curve from being too low
- To smooth out the result from EWMA
- To prevent the values on the entire curve from being negative
* `[ B ]`


---

**Q: A convolutional filter given by: $\begin{bmatrix}
1 & 0 & -1\\ 
1 & 0 & -1\\ 
1 & 0 & -1
\end{bmatrix}$
, results in the extraction of which features?**
- Horizontal Edges
- Circles
- Vertical Edges
- Light patches
* `[ C ]`


---

**Q: Can any convolutional network be represented by a fully connected feed-forward network?**
- No, as they are inherently different operations.
- Yes, because any function can be represented by a fully connected network.
- No, as fully connected networks loose all spacial information due to their fully connected nature.
- Yes, however to achieve the same complexity a lot more parameters are required.
* `[ D ]`


---

**Q: You convolute an image with the following kernel: [0 0 0; 0 0 1;0 0 0]; what happens?**
- The image shifts to right
- The image shifts to left
- The image gets 1 value darker
- The image gets 1 value lighter
* `[ B ]`


---

**Q: You convolute an image with the following kernel: [0 0 0; -1 0 1;0 0 0]; what happens?**
- The image flips left vertically.
- The image flips horizontally.
- The image shows horizontal edges.
- The image shows vertical edges.
* `[ D ]`


---

**Q: What is the consequence of increasing the value of 'ρ' in the calculation of exponential weighted moving average? S[t] = ρ (S[t-1]) + (1- ρ)y[t]**
- More influence by the previous average value.
- More influence by the current data value.
- Achieves Bias Correction.
- Both A and C.
* `[ A ]`


---

**Q: What does batch normalization achieve?**
- Speeds up learning & optimal convergence by reducing the internal covariance shift of inputs of internal layers.
- Speeds up learning & optimal convergence by tuning the learning parameter stochastically during the back propagation.
- Speeds up learning & optimal convergence by tuning the learning parameter stochastically during the forward pass.
- All of the above.
* `[ A ]`


---

**Q: What is the downside of using a relatively high constant learning rate?**
- a constant learning rate introduces another hyperparameter to tune
- it takes a very long time to find a good instantiation 
- a good instantiation is never found
- the average instantiation will be too large for all the parameters
* `[ C ]`


---

**Q: which of these statements about algorithms that help in finding a good instantiation is true?

I: The momentum algorithm heads in the right direction on average.
II: The RMSprop algorithm makes sure variance in the wrong direction is low**
- Only statement I is true
- Only statement II is true
- Both statements are true
- Both statements are false
* `[ A ]`


---

**Q: What's the problem with a big learning rate in SGD**
- It takes a very long time to find a good value
- It may never find a good optimum
- It's hard to compute bigger steps
- It will quickly get too infinity
* `[ B ]`


---

**Q: What is the impact of on y_45 on S_50 when computing the EWMA, $S_t = (pS_t-1) + (1-p)y_t$, with p = .95**
- 0.774
- 0.05
- 0.041
- 0.039
* `[ D ]`


---

**Q: What is false about the exponential weighted moving average? S_{t} = (\rho S_{t-1}) + (1 - \rho) y_{t}**
- For a noisy data set, only increasing rho makes the curve more smooth
- For a noisy data set, only increasing rho makes the training set error smaller
- Only increasing rho results in poor estimations for values in the beginning
- Bias correction has no effective influence for values at the end
* `[ C ]`


---

**Q: What can be said about the learning rate?**
- Stochastic gradient descent is generally the best algorithm to use
- Adam algorithm uses only the mean of the past updates to determine the next LR
- An Additional layer can be added to determine the learning rates
- None of the above
* `[ C ]`


---

**Q: Given a neural network which makes use of the Stochastic Gradient Descent (SGD) optimization to train. What will happen if the learning rate is too high?**
- The trajectory of updated weights will deviate so much, it will never find a good instantiation.
- The trajectory of updated weights will barely deviate, it will never find a good instantiation.
- The trajectory of updated weights will deviate so much,  it will take a lot of time to find a good instantiation.
- The trajectory of updated weights will barely deviate,  it won't take long until it is stuck in a local minimum.
* `[ A ]`


---

**Q: Given is the exponentially weighted moving average (EWMA) : $S_t = (\rho S_{t-1}) + (1 - \rho)y_t$. What is the effect of $\rho$ on the EWMA?**
- It decides how fast the algorithm will optimize
- It decides how much the last value is taken into consideration for the new computation
- It decides how much the current value is taken into consideration for the new computation
- It decides how much the last and current value is taken into consideration for the new computation
* `[ D ]`


---

**Q: The learning rate:**
- Is chosen to avoid hitting local minima
- Leads to stable results regardless of the magnitude of its value
- Shortens the time until convergence if is chosen small
- Shortens the time until convergence if is set to decay
* `[ D ]`


---

**Q: The bias correction:**
- Reduces the effect of the border for calculating the moving average parameters
- Defines an offset for the smoothed moving average slope
- Is very small for parameters close to time 0
- Is only useful when dealing with a particular type of time series
* `[ A ]`


---

**Q: How to smooth out a noise time series as values are coming in? **
- Use convolution
- Keep recent values and compute weighted average
- Recursively compute online average
- None of the above
* `[ C ]`


---

**Q: Which of the following algorithms makes use of variance?**
- Momentum
- RMSprop
- Adam
- None of the above
* `[ B ]`


---

**Q: Which of the following is the best approach for tuning the learning rates?**
- SGD with High amplitude
- SGD with Low amplitude
- SGD with decaying amplitude
- None of above
* `[ C ]`


---

**Q: Which of the following algorithms use SGD with past gradient update statistics?**
- Momentum
- RMSprop
- Adam
- All of above
* `[ D ]`


---

**Q: Which of the following statements does NOT hold true?**
- The training of a Neural Network can halt even when the gradients are still large
- Stochastic Gradient Descent returns an unbiased estimate of the true gradient but with relatively large variance
- Estimators which use  moving average techniques are biased because they need to be initialized 
- In a fully connected layer which precedes a single activation function applied to all neurons , the relevant weights must be initialized with different values
* `[ B ]`


---

**Q: Which of the following statements does NOT hold true?**
- Batch normalization weakens the coupling between different layers of a neural net
- Batch normalization adds noise to the activation of the hidden layers
- SGD is not affected by Batch Normalization of the hidden layers
- Batch normalization permits the use of larger values in the learning rate of a neural network
* `[ C ]`


---

**Q: Which of the following statements about the learning rate is not true?**
- Noisy stochastic gradient descent can be remedied by decreasing the learning rate over time, also known as decay.
- A too small learning rate highgly gaurantees finding a good instantiation, but at the cost of taking lots of time to find it.
- A too high learning rate can cause the situation where a good instantiation can never be found.
- The learning rate is not fixed for all dimensions, it is often smaller for dimensions where the contour plot has lines closer together.
* `[ D ]`


---

**Q: Which of the following statements is/are correct?

1. Bias correction normalizes the result of the exponential weighted moving average, which is most noticeable late in the time series.
2. Convolution would not be practical for smoothing out noisy time series as it would only take a small fraction of the previous values into account.**
- Only 1 is correct
- Only 2 is correct
- Both are correct
- Both are incorrect
* `[ D ]`


---

**Q: In CNN what is the importance of using different filter kernels**
- To extract different features (shapes or edges)
- To reduce the size of data
- all
- none
* `[ A ]`


---

**Q: Why is padding important in CNN**
- Extract feature
- Sample the features
- Keep size of images
- all
* `[ C ]`


---

**Q: Consider following two statements: 

\begin{enumerate}
	\item If the learning rate is too high, it will take very long to find a good instantiation.
	\item If the learning rate is too low, it will never find a good instantiation.
\end{enumerate}

Which one(s) are true or false?**
- 1 true and 2 true.
- 1 true and 2 false.
- 1 false and 2 true.
- 1 false and 2 false.
* `[ D ]`


---

**Q: Is it possible to smooth out a noisy time series using convolution as values are coming in? Pick the most correct answer.**
- Yes; it is the most efficient way to do so.
- Yes; although it is not ideal and will take a lot of computational power.
- No; you do not have access to future values.
- No; the results would be too biased.
* `[ C ]`


---

**Q: Which algorithm does not make use of past gradient update statistics?**
- Momentum
- RMSprop
- Adam
- AdaBoost
* `[ D ]`


---

**Q: What should we do to solve the beginning bias low value?**
- Generalisation
- Normalisation
- Centralisation
- Regularisation
* `[ B ]`


---

**Q: Which of the statements below are true for learning rate?**
- For good instantiation, learning rate should be high.
- If Learning rate is too high, it will never find a good instantiation.
- Learning rate is arguably the most important parameter to tune.
- B and C
* `[ D ]`


---

**Q: Which of the below mentioned algorithms make use of past gradient update statistics?**
- Momentum
- RMSprop
- Adam
- All of the above
* `[ D ]`


---

**Q: What is decay used for?**
- To remove noise from SGD
- To determine the correct learning rate
- To reduce the learning rate over time
- To enlarge the learning rate over time
* `[ C ]`


---

**Q: What is a bias correction used for?**
- To normalize the average and make-up for the border effect
- To correct wrong estimates
- To predict what happened before time
- All of the above 
* `[ A ]`


---

**Q: Which one of the following is not an optimization method for deep learning?**
- Momentum
- EWMA
- RMSprop
- Adam
* `[ B ]`


---

**Q: Which of the following statements regarding EWMA is false?**
- The importance of the past data decreases exponentially
- EWMA is used for better forecasting and noise reduction
- A higher exponential factor gives more importance to new data
- The initialization factor and the exponential factor used in EWMA calculation must be the same
* `[ D ]`


---

**Q: Which statement about the learning rate of a neural network is incorrect?**
- The learning rate is arguably the most important parameter to tune
- Using different learning rates for different axes will not result in an improved learning rate
- Decay is the reduction of the learning rate over time
- If the learning rate is low it will take a long time to find a good instantiation
* `[ B ]`


---

**Q: Which of these can not be used to beat overfitting?**
- Using more data
- Reducing the number of features
- Reduce flexibility of the model
- Learning rate decay
* `[ D ]`


---

**Q: What is the downside of having a learning rate that is too high?**
- Since SGD is noisy, you may overshoot a good place in the weight space and not converge to a good minimum.
- It makes the gradient noisy.
- It leads to overfitting since the network picks up information more quickly.
- There is no real downside except that training takes a little longer.
* `[ A ]`


---

**Q: How does Adam relate to Momentum and RMSprop?**
- Adam is the combination of Momentum and RMSprop.
- Adam uses variance, while RMSprop and Momentum use, well.. momentum.
- They are essentially the same, and use past gradient update statistics.
- Adam adds weight decay to RMSprop.
* `[ A ]`


---

**Q: What happens when Learning rate for stochastic Gradient descent is very high?**
- Reaches global optimum very fast
- Get stuck in local optimum
- Will never arrive at a good estimate as solution oscillates
- None of the above
* `[ C ]`


---

**Q: In regards to learning rate what practices can lead to a good instantiation of the the gradient descent.**
- Use very high learning rate
- Use very low learning rate
- reduce the learning rate over time(decay)
- Both B) and C)
* `[ D ]`


---

**Q: Which algorithm is not designed for using past gradient update statistics?**
- Momentum
- RMSprop
- SGD
- Adam
* `[ C ]`


---

**Q: Why center inputs at zero and equal dimension variance?**
- Robustness
- Speed up training
- Reduce training error
- Smooth training
* `[ B ]`


---

**Q: Which of the following statements about the learning rate is false?**
- A high learning rate can overshoot the optimum solution
- A low learning rate increases the learning time of the network
- The learning rate can be different for each dimension
- Increasing the learning rate over time decreases the training time of the network
* `[ D ]`


---

**Q: Write out the exponential moving average with out a bias correction for $S_3$ as an combination of the inputs $y_i$. **
- $S_3 = (1-p) y_3 * (1-p) p y_2 + (1-p) p^2 y_1 $
- $S_3 = (1-p) p^2 y_3 * (1-p) p y_2 + (1-p) y_1 $
- $S_3 = p y_3 * (1-p)p y_2 + (1-p)^2 y_1 $
- $S_3 = p y_3 * p^2 y_2 + p^3 y_1 $
* `[ A ]`


---

**Q: When using a learning rate that is too high what will probably happen?**
- A good instantiation is never found
- It will take very long until an instantiation is found
- The instantiation is found directly
- The instantiation does not depend on the learning rate.
* `[ A ]`


---

**Q: When using noisy data one would like to use exponentially weighted moving average, however this is not always the best choice to use to smooth out a time series. In which of the following cases would give the best result?**
- When using unbiased EWMA at the 1st time step and a rho of 0.9.
- When using unbiased EWMA at the 1st time step and a rho of 0.5
- When using unbiased EWMA at the 1000th time step and a rho of 0.5
- When using unbiased EWMA at the 1000th time step with a rho of 0.99
* `[ D ]`


---

**Q: Which of the following technique in not used as optimization algorithm?**
- Momentum
- RMSprop
- Adam
- Smoothing
* `[ D ]`


---

**Q: To smooth out a noisy time series as values are coming in, which of following statement is wrong?**
- One can use convolution.
- One can compute a weighted average of recent values if time series are low-dimensional.
- One can apply dynamic programming.
- One can apply Exponentially weighted moving average (EWMA).
* `[ A ]`


---

**Q: The questions are about hyperparameters**
- Hyperparameters can change depending parameters
- Hyperparameters are parameters that needs to be tuned by hand 
- Hyperparameters are the values that cannot changes by and not depending on the kind of data
- Hyperparameters are parameters that are hypervision values
* `[ B ]`


---

**Q: Keep recent value, and compute a weighted average for then next computing why do the do this? What is most true.**
- At zero you cannot do anything
- Value are outdoted
- Millions of points takes allot of ram
- To lower the avaerage erro 
* `[ C ]`


---

**Q: Why is it important to correctly determine the learning rate**
- Setting the learning rate too high means that it can fail to converge to a low error
- Setting the learning rate too high means that it can't learn at all
- Setting the learning rate too high limits the number of layers you can use
- Setting the learning rate too low limits the number of layers you can use
* `[ A ]`


---

**Q: What purpose does exponentially weighted average serve in optimization**
- Efficiently remembering values for use in optimization algorithms
- Stabilizing gradient in gradient descent
- Optimizing nodes by adding decay for nodes that don't often contribute to the decision result
- Predicting gradients
* `[ A ]`


---

**Q: Which of the following is NOT true about learning rate:**
- If learning rate is too high, cost function may diverge
- If learning rate is too low, cost function may take too long to converge
- Cost function is guaranteed to converge to optimal solution with low learning rate
- If learning rate is high, the gradient may overshoot the minima
* `[ C ]`


---

**Q: Which of the following is False:**
- Momentum smooths the average of noisy gradients with EWMA
- RMSprop smooths the zero-centered variance of noisy gradients with EWMA
- In RMSprop, variance in wrong direction is high 
- In Momentum variance is zero in wrong direction
* `[ D ]`


---

**Q: Which one is wrong?**
- CNN have neurons arranged in 3 dimensions which are width, height, depth. Depth means that depth of the network.
- A layer transforms an input 3D volume to an output 3D volume with some differentiable function which may or may not have parameters
- Weight sharing: share and train the weights jointly for those inputs that contain the same information
- For CNN Convolutional layer, the trick is, back propagation over the entire layer is computed, but at the end all the backpropagated errors are summed together and the weights of the filter are updated just once
* `[ A ]`


---

**Q: Which one is wrong?**
- The sparse connectivity property means that each neuron is connected to only a limited number of other neurons. 
- Sparse connectivity incorporates a sparse data flow: The state of a neuron of the output layer depends on the state of only a small number of neurons from the input layer.
-  With sparse connectivity, the state of the output layer is inferred  less efficiently than with dense connectivity, as less information has to be processed.
- Pooling layers have several roles:  allow flexibility in the location of certain features in the image, therefore allowing non-rigid or rotated or scaled objects to be recognized ; allow unions of features to be computed; reduce the output
image size. 
* `[ C ]`


---

**Q: Which of the following is true about RMSProp?

A It modifies AdaGrad to perform better in the nonconvex setting by changing the gradient accumulation into an exponentially weighted moving average. 
B It uses an exponentially decaying average to discard history from the extreme past.**
- A
- B
- A and B
- None of the above
* `[ C ]`


---

**Q: Which of the following is true about Adam?

A In Adam, momentum is incorporated directly as an estimate of the first-order moment of the gradient. 
B Adam includes bias corrections to the estimates of both the first-order moments and the second-order moments to account for their initialization at the origin. **
- A
- B
- A and B
- None of the above
* `[ C ]`


---

**Q: What is not a problem with pure SGD?**
- A small learning rate has to be used otherwise SGD may overshoot an optimum towards the end.
- There is a 50% chance that SGD will go in the wrong direction as it is purely random.
- Choosing an appropriate learning rate is difficult as it can not be changed during runtime.
- The loss function has to be circular so there is no bias towards the x and y direction.
* `[ B ]`


---

**Q: Which of the following is false?**
- Momentum can be used to decrease the learning rate.
- Momentum can be used to increase the learning rate.
- Adam uses the inverse of the momentum for even faster learning.
- RMSprop has a high variance in the wrong direction
* `[ C ]`


---

**Q: What does adding a bias correction solve when computing the EWMA for a noisy time series? **
- Prevent running out of memory
- The effect of outlying datapoints throughout the series
- expensive parameter tuning
- Relatively low values at the beginning
* `[ D ]`


---

**Q: Centering inputs at zero and equal dimension variance..**
- minimizes memory needed
- reduces chances of overfitting
- increases flexibility
-  speeds up training
* `[ D ]`


---

**Q: Which statement is FALSE regarding optimization methods:**
- Momentum updates the weights on average in the right direction
- The variance of updates in the wrong direction is low when using RMSprop
- Adam is a combination of momentum and RMSprop
- Momentum first corrects the velocity after which a step is taken into the velocity direction
* `[ B ]`


---

**Q: Which statement is FALSE regarding batch normalization:**
- Batch normalization is applied for each mini-batch
- Batch normalization is typically applied before the activation function
- Batch normalization uses a combination of the mean and standard deviation of a layer
- Batch normalization cannot be used a test-time
* `[ D ]`


---

**Q: How can we reduce the noise in SGD close to the "good place".**
- Reduce the learning rate.
- Reduce the learning rate over time
- Increase the dataset.
- Increase the learning rate.
* `[ B ]`


---

**Q: How to smooth out a noisy time series as values are coming in?**
- Use convolution.
- Keep recent values, and compute a weighted average?
- Filter out extreme values with a gaussian kernel.
- Recursively compute online avarage.
* `[ D ]`


---

**Q: Convolutional networks are used for image pattern recongnition because:**
- Fast computing time compared to classifiers like SVM, Robustness for distorted inputs & the network interpretation function close to how humans view images.
- Robustness to shifts and distortions in the input, small amount of memory required & easy to apply already established neutral network training methods.
- Was a neutral extension of already known image processing tools, easy to apply to important fields e.g. medical imaging & The network evade the curse of dimentionallity
- Removing feature extraction - let the Machine learn the image features, reliable on a large amount of datasets & first method to show good results thus having early addaption
* `[ B ]`


---

**Q: In CNNs the distribution of the impact within the receptive field is:**
- asymptotical Gausian distributed
- approximately $chi^2$ distributed
- strict Laplacian distributed
- approximately Wishart distributed
* `[ A ]`


---

**Q: What is true if the variance of the gradient is low?**
- The gradient will converge slowly
- The gradient will converge fast
- The gradient takes large steps in general
- None of the above
* `[ A ]`


---

**Q: What is NOT true about batch normalization?**
- Batch normalization operates only on the first layer
- Batch normalization operates on all layers
- Batch normalization subtracts the batch mean and divides by batch standard deviation for each object in the batch
- Batch normalization adds parameters to a layer
* `[ A ]`


---

**Q: What are the advantages and disadvantage of setting a 'very' small learning rate**
- Your learner will converge to an optimum and will do so fast.
- your learner will converge to an optimum, but it may be caught in a local optimum
- Your learner will converge to a global optimum, but requires a lot of computational time
- Your learner is very fast, but may not converge to an optimum.
* `[ B ]`


---

**Q: When is untrue about time series.**
- When a process is static, its mean and variance don't change over time.
- Time series can be approximated by the exponentially weighted moving average 
- Time series can not be approximated by a convolution
- With the exponentially weighted moving average you can predict data from the past if you know what will happen in the future.
* `[ D ]`


---

**Q: Which of the following is a true statement about Learning rate (LR)?**
- If LR is very high then it will never find a good instantiation.
- If LR is too low then it will take too long to find a good instantiation.
- It's better to reduce the LR over time.
- All of the above
* `[ D ]`


---

**Q: In exponentially weighted moving average(EWMA), 
\begin{equation}
S_{t}=(\rho  S_{t-1})+(1- \rho )y_{t}, \texttt{ where t is time, and }  0\leq \rho \leq 1 \texttt{ and }  S_{t-1} = 0; \texttt{ if } t = 1
\end{equation}
Select the correct option.**
- If $\rho$=0.5, then graph will be more noisy and adapts slowly.
- If $\rho$=0.5, then graph will be less wavy and more smooth.
- If $\rho$=0.9, then graph will be more wavy but adapts quickly.
- If $\rho$=0.9, then graph will be more smooth but adapts slowly.
* `[ D ]`


---

**Q: Which of the strategy of tuning the learning rate is the most reasonable?**
- Learning rate should be large, because othervise it will take very long time to find an optimum
- Learning rate should be small in order not to jump over the optimum
- No matter if it is large or small, it should be just constant
- Learning rate should be reduced over time
* `[ D ]`


---

**Q: Which of these algorithms helps to reduce noise in time series?**
- Momentum
- RMSprop
- EWMA
- All mentioned
* `[ C ]`


---

**Q: The Exponentially Weighted Moving Average method is based on the following formula: $S_t = (\rho S_{t-1}) + (1- \rho)y_t$. How do the values of \rho affect the computation regarding the number of past values that are taken into account?**
- The higher the value of \rho, the more values from the past are considered.
- The smaller the value of \rho, the more values from the past are considered.
- All the values from the past are taken into account, as \rho term can never be 0. \rho however controls their contribution.
- \rho does not have any connection with the number of past values from the time series that are considered.
* `[ A ]`


---

**Q: One of the algorithms that improves over Stochastic Gradient Descent is RMSprop. What can you say about the value of the variance for each of the tuning space dimensions?**
- The variance in the wrong direction is small
- The variance in the wrong direction is high
- The RMSprop does not take the variance regarding the dimensions into account, but the average in the good direction
- The RMSprop does not take the average or the variance regarding the dimensions into account.
* `[ B ]`


---

**Q: Which statement is wrong?**
- The convolution function means an operation that consists of many applications of convolution in parallel.
-  Convolution with a single kernel can extract several kinds of features.
- The multichannel operations in convolutional networks are only commutative if each operation has the same number of output channels as input channels
- For convolutional layers, it is typical to have one bias per channel of the output and share it across all locations within each convolution map.
* `[ B ]`


---

**Q: Which statement is wrong?**
- Pooling helps to make the representation approximately in variant to small translations of the input
-  It is possible to use fewer pooling units than detector units.
- Pooling is essential for handling inputs of varying size. 
- Non of them.
* `[ D ]`


---

**Q: Given a Neural Net with N input nodes, no hidden layers, one output node, with Entropy Loss and
Sigmoid Activation Functions, which of the following algorithms (with the proper hyper-parameters and initialization) can be used to find the global optimum?**
- Gradient Descent with restarts
- Stochastic gradient descent
- Mini-batch Gradient descent
- All of the above
* `[ D ]`


---

**Q: What is the starting value of the EWMA?**
- zero
- target mean
- target standard deviation
- target variance
* `[ B ]`


---

**Q: Learning Rate is one of the most important hyper parameters, because it defines the optimisation steps. However there is a way to avoid getting stuck in local sub-optima. which of the following answers is the "way" meant here?**
- Keeping large steps makes the optimization converge faster towards the global minimum/maximum. This is wanted because it saves computing power and time.
- Keeping small steps, avoids the optimization to overshoot the global minimum/maximum.
- Making the steps smaller during the optimization process enables for more precision near the goal, whereas it keeps fast optimization when little accuracy is needed.
- None of the above
* `[ C ]`


---

**Q: Sometimes the use is made of "momentum" within algorithms. Why?**
- To check if the algorithm is moving in the right direction
- To keep the algorithm's speed that it gathered through previous steps.
- To avoid the algorithm to spin in circles, the momentum algorithm stretches arms to stop the spinning and go towards the goal
- none of the above
* `[ A ]`


---

**Q: What is one disadvantage of using gradient descent optimization algorithms (such as Momentum, RMSprop and Adam)?**
- There is no space efficient manner of keeping track of past gradient update statistics.
- Learning rate is the only effective hyper-parameter. These algorithms increase complexity, without results.
- Additional hyper-parameters need to be tuned.
- They dramatically increase training time.
* `[ C ]`


---

**Q: In batch normalization, why are the learnable parameters $\gamma$ (gamma) and $\beta$ (beta) added in?**
- In order to allow the network to return to the original representation of the data.
- In order to allow the network to return to the original representation of the data or any other it may prefer.
- In order to normalize the inputs of the previous layer.
- They are not added to every layer, only those that do not need normalization.
* `[ B ]`


---

**Q: What happens if the learning rate of a SGD is too high?**
- The SGD will converge too fast, usually resulting in a sub-optimal local optimum.
- The SGD will converge too slow, taking too long to find the optimum.
- The SGD will overshoot, a high change in weights causing it to fly past the optimum.
- The SGD will undershoot, a low change in weights resulting in a very long update time.
* `[ C ]`


---

**Q: For what reason is bias correction applied to EWMA?**
- To increase the variance.
- To eliminate noisy averages at the beginning.
- To reduce the necessary sample size.
- To prevent overfitting on the averages.
* `[ B ]`


---

**Q: Using EWMA, which rho value takes most the most historical values into account?**
- 0.1
- 0.5
- 0.9
- 0.95
* `[ D ]`


---

**Q: When could RMSprop be a useful algorithm to implement?**
- When the variance in the direction of the optimal path is high
- When the variance in the 'wrong' direction (in 2D: perpendicular to the optimal path) is high
- When the optimization algorithm is stuck in a saddle point
- When the step size is a fixed value
* `[ B ]`


---

**Q: The difference of convolutional neural networks and feed forward networks is ...**
- ...that convolutional neural networks are more complex than feed forward networks.
- ...that convolutional neural networks have more non-zero weights than feed forward networks.
- ...that cnn are a limited version of feed forward networks.
- ...that cnn are computationally less expensive than feed forward networks.
* `[ C ]`


---

**Q: The equivariance of cnn adds...**
- ... new neurons.
- ... more complexity.
- ... nothing.
- ... prior knowledge.
* `[ D ]`


---

**Q: Which of the following is not exclusive only to convolution neural networks?**
- Learned Filters
- Convolution
- Non-linearity
- Spatial Pooling
* `[ A ]`


---

**Q: Which of the following is true?**
- CNN is an extended parameter version of feed-forward.
- CNN is a limited parameter of feed forward.
- The curse of dimensionality is higher in CNN than feed forward. 
- None of the above. 
* `[ B ]`


---

**Q: Suppose you consider 2 convolutional network layers, each with 5 nodes. The nodes in the detector stage contain:
0.4 1 0.6 1.2 0.5. What will the 3rd node from the left of the pooling stage contain if one shifts 1 pixels to the right and consider a decetor response of 3?**
- 1
- 0.4
- 0.6
- 1.2
* `[ A ]`


---

**Q: When considering convolutional neural networs, the following properties hold:**
- Covnets learn feature representations where the kernel weights are feature detectors and the learning weights are learning features
- Covnets learn feature representations where the kernel weights are learning features and the kernal weights are feature detectors
- Covnets learn kernal weights where the kernel weights are feature detectors and the learning weights are learning features
- Covnets learn feature representations where the kernel weights are feature detectors and the learning weights are feature detectors
* `[ A ]`


---

**Q: While using SGD, which algorithm have both the advantage of smoothing the average of noisy gradients and smoothing the zero-centered variance?**
- Adam
- RMSprop
- Momentum
- None of the three
* `[ A ]`


---

**Q: The SGD take very long to find a good instantiation, what do you think would not be a reason of that?**
- LR too high
- LR too low
- SGD is too noisy
- none of the three
* `[ D ]`


---

**Q: Which of the following is true?**
- By increasing the learning rate,  we will eventually find a good instantiation
- In a time series input data, convolution reduces noise by using the surrounding average.
- One advantage of using an exponentially weighted moving average is that you can tune another hyperparameter, to improve your results.
- The bias correction for the exponentially weighted moving average approaches the exponentially moving average as time increases.
* `[ D ]`


---

**Q: Which of the following is true?**
- By finding the global optimum (as opposed to local) to improve the ability of your model generalizing.
- Momentum, RMSProp and Adam algorithms make use of past gradient update statistics
- When updating the weights, using RMSprop, the higher the variance in one dimension, the bigger the update for the parameter in that dimension
- The bias for the exponentially weighted moving average is \frac{S_{t}}{p^{t}-1}
* `[ B ]`


---

**Q: In SGD, the learning rate is gradually decreased, what should an optimal initial learning rate be?**
- As high as possible
- Very low
- Value that gives best performance after first few iterations
- A value higher than the best performing learning rate but not so high that it causes instability.
* `[ D ]`


---

**Q: In a time-series to keep track of past gradient update statistics, how to smooth out a noisy series as the values come in?**
- Keep recent values, and compute a weighted average
- use convolution
- Use inﬁnite impulse response ﬁlter to recursively compute online average
- None
* `[ C ]`


---

**Q: Consider a noisy time series, which are used to form the exponentially weighted moving average from an incoming value?**
- All past and future values of the series.
- only past values of the series.
- only future values of the series.
- the current and all past values of the series.
* `[ D ]`


---

**Q: Consider a noisy time series and the EWMA formed at each timestep. What are the coefficients, used in the weighted sum of incoming values, adding up to? **
- 1
- Can't be determined from the given information, depends on the time series.
- Can't be determined from the given information, depends on the noise.
- n, where n is the number of values up to the current timestep.
* `[ A ]`


---

**Q: Which following algorithm is not used to update SGD?**
- Momentum
- RMSprop
- Adam
- bagging
* `[ D ]`


---

**Q: Which following is not true for bias correction?**
- It significantly helps normalize value when in the beginning.
- When epoch is large, the value obtained with bias correction will approximately equal to value obtained without bias correction.
- Bias correction is effective to smooth the value.
- Bias correction is expensive to compute.
* `[ D ]`


---

