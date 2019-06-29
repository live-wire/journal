# Questions from `paper_9` :robot: 

**Q: What makes Batch Normalization so effective in accelerating the training of Deep Neural Networks?**
- Batch normalization increases the Internal Covariance Shift.
- Batch normalization reduces the Internal Covariance Shift.
- Batch normalization makes the landscape of the optimization problem less smooth.
- It makes the loss and the gradient of the loss change at a smaller rate and both of their gradients have a smaller magnitude.
* `[ D ]`


---

**Q: According to the paper ‘How Does Batch Normalisation Help Optimization?’, what is speculated to be the reason that batch normalisation (BN) helps in training networks?**
- BN reduces the internal covariate shift.
-  BN adds noise to the layers during training
- BN smooths the loss landscape, increasing gradient predictiveness
-  BN reduces the norm of the weights, thus improving training efficiency.
* `[ C ]`


---

**Q: Which of the following is the main reason that batchnorm can help during training?**
- Prevention of exploding or vanishing gradients.
- Increased robustness to different settings of hyperparameters.
- Makes landscape of the function to optimize smoother.
- Keeps most of the activations away from saturation regions of non-similarities.
* `[ C ]`


---

**Q: Which statement below is wrong?**
- This paper validates the claim that the performance gain due to Batch- Norm stems from increased stability of layer input distributions
- BatchNorm does significantly improve the training process
- BatchNorm might not even be reducing internal covariate shift
- BatchNorm makes the optimization landscape significantly smoother
* `[ A ]`


---

**Q: What is the key effect of BatchNorm on the training process?**
- It reparametrizes the underlying optimization problem to make it more stable which will eventually enables faster and more effective optimization.
- It shifts the covariance and mean to 0 which will eventually enables faster and more effective optimization.
- The BatchNorm causes the a change in the smoothness of the landscape (of gradients) which results in a slower gradient decent. 
- All of the above mentioned points are key effects.
* `[ A ]`


---

**Q: Review the following two statements about the results the authors of the paper "How Does Batch Normalization Help Optimization?" found on Batch Normalization:
\begin{enumerate}
    \item BatchNorm affects network training by making the landscape of the optimization problem smoother
    \item BatchNorm reduces the internal covariate shift of the network significantly
\end{enumerate}
Which of the statements are true?**
- Statement 1 and 2 are true
- Statement 1 is true and statement 2 is false
- Statement 1 is false and statement 2 is true
- Statement 1 and 2 are false
* `[ B ]`


---

**Q: Which of the following about Batch Normalization is FALSE?**
- Batch normalization fixes the first order moment of the activations.
- It aims at stabilizing the distribution over a minibatch of inputs to a network layer during testing. 
- Batch normalization makes the optimization landscape more smooth.
- Its goal was to reduce the change in the distribution of layer inputs caused by updates to the preceding layers.
* `[ B ]`


---

**Q: Which of the following is true about Batch normalization?**
- It is applied after the non-linearity of the previous layer.
- Removes layers from the network.
- Introduces layers that set the mean and variance to specific values.
- Makes the training of a network slower due to its computational complexity.
* `[ C ]`


---

**Q: The contribution of Batch Normalization is believed to be the reduction of the “internal covariate shift”. What is the real contribution of a batch normalization in a deep neural network instead?**
- It reduces the problem of the change on the distribution of layer inputs caused by updates to the preceding layers
- It reparametrizes the optimization problem to make it more stable and smoother during the training process inducing a more predictive and stable behaviour of the gradients
- It creates a simpler model enabling faster and more effective optimization, but the gradient is less predictive
- It is the only technique that provide a “smoothing effect” and the only technique with good performance gain
* `[ B ]`


---

**Q: What is NOT an advantage of batch normalization?**
- It makes the landscape of the corresponding optimization problem significantly more smooth
- It increases robustness to hyperparameter settings
- It avoids gradient explosion / vanishing
- It decreases internal covariate shift
* `[ D ]`


---

**Q: What is the aim of batch normalization?**
- to improve accuracy
- to reduce the size of the network
- to stabilize the distribution of inputs in a given layer (zero mean and unit variance)
- none of the above
* `[ C ]`


---

**Q: Batch Normalization is achieved by augmenting a neural network with BatchNorm layers, that take the output $y$ of the previous layer, rescale it by $\hat{y} = \frac{y - \mu}{\sqrt{\sigma^2}}$ so that $\hat{y}$ now has 0 mean and unit variance, and then scale and shift $\hat{y}$ by computing $z = \alpha \hat{y} + \beta$ with $\alpha, \beta$ trainable parameters. For what values of $\alpha$ and $\beta$ would the BatchNorm do nothing?**
- $\alpha = \sqrt{\sigma^2}$, $\beta = \mu$.
- $\alpha = \sigma^2$, $\beta = \mu$.
- $\alpha = \sqrt{\sigma^2}$, $\beta = -\mu$.
- $\alpha = \sigma^2$, $\beta = -\mu$.
* `[ A ]`


---

**Q: Which is true about batch normalization?**
- Batch normalization makes the gradients of the loss function less Lipschitz
- Batch normalization makes the optimization landscape rougher so local minima are more pronounced
- The smoothness of batch normalization optimization landscape induces a less predictive behavior of the gradients, allowing for a smaller range of learning rates.
- Internal covariance shift is the measure of difference between the distribution of an input after it has been changed by updates to the preceding layer.
* `[ D ]`


---

**Q: Which statement is correct about Batch Normalisation?**
- BatchNorm’s performance stem from controlling internal covariate shift
- BatchNorm leads the reduction of the training time. 
- BatchNorm could reduce the internal covariate shift.
- Deep networks with BatchNorm always perform better than the ones without BatchNorm. 
* `[ B ]`


---

**Q: According to this paper, what is the relationship between Batch norm and ICS?**
- strong relevant
- tenuous
- no relevant
- nonlinear relevant
* `[ B ]`


---

**Q: Over what dimension does BatchNorm normalize**
- Over the X dimension of the images
- Over the Y dimension of the images
- Over multiple samples
- Over multiple features
* `[ C ]`


---

**Q: What is Batch normalization?**
- Batch normalization is a mechanism that aims to stablilze the distribution (over a mini-batch) of inputs to given network layer during training. Which is achieved by normalizing the input mini batch
-  Batch normalization is a mechanism that aims to stablilze the distribution (over a mini-batch) of inputs to given network layer during training. Which is achieved by augmenting the network with additional layers that set the first two moments (mean and variance) of the distribution of each activation to be zero and one respectively. Then, the batch normalized inputs are also typically scaled and shifted based on trainable parameters to preserve model expressivity 
- Batch normalization is a mechanism that aims to stablilze the distribution (over a mini-batch) of inputs to given network layer during training. Which is achieved by setting the first two moments (mean and variance) of the distribution of each activation to be zero and one respectively. Then the normalized mini-batch will be used to achieve this.
-  All of the above answers are wrong
* `[ B ]`


---

**Q: Which of the following regarding models with BatchNorm layers is true?**
- The difference between models with noisy layers an non-noisy layers is significant
- The difference between the optimisation landscape of the optimisation parameters $W_i^{(t)}$ is problematic for training
- Batchnorm makes the training significantly more sensitive to hyperparameter choices
- Empirically, For all types of networks batch norm normalisation is the better performing  than l1 normalisation
* `[ B ]`


---

**Q: Which of following statements is NOT the effect of BatchNorm?**
- BatchNorm helps reduce the internal covariate shift.
- BatchNorm makes the training process more stable (in the sense of loss Lipschitzness).
- BatchNorm makes the training process more smooth (in the sense of “effective” β-smoothness of the loss).
- BatchNorm helps avoid gradient explosion/vanishing.
* `[ A ]`


---

**Q: What is the advantage of using batch normalisation?**
- It does improve the training process
- It improves the classification process
- It improves the normal distributions
- It improves the hidden layers
* `[ A ]`


---

**Q: It has been discovered that other normalization techniques exist achieving the same result as the use of standard Batch Normalization. Some of them are the L-1 normalization and the L-2 normalization. Which is their beneficial effect on the net?**
- They reduce internal covariate shift (ICS) in the net, as BatchNorm does as well
- They increase the gradient of the loss function in the net, allowing for faster training.
- They smoothen significantly the optimization landscape for the given problem
- They allow to use lower weights in the net
* `[ C ]`


---

**Q: Statement 1: BatchNorm is a mechanism that aims to stabilize the distribution of inputs to a given network layer during training, which is achieved by augmenting the network with additional layers that set the first two moments of the distribution of each activation to be zero and one respectively. 
Statement 2: BatchNorm reparametrizes the underlying optimization problem to make its landscape significantly more smooth. **
- Both statements are correct
- Statement 1 is correct
- Statement 2 is correct 
- None of the statements are correct 
* `[ A ]`


---

**Q: Why does BatchNorm work ?**
- BatchNorm’s reparametrization makes gradients of the loss exhibit a significantly better “effective” β-smoothness.
- BathNorm's normalisation, enables the loss during gradient descent, change at a faster rate and the magnitudes of the gradients are larger too. Hence a faster convergence.
- BathNorm's normalisation, changes the layer's input distributions during training to reduce the “internal covariate shift”.
- All of the above.
* `[ A ]`


---

**Q: What is NOT true about Batch Normalization?**
- BatchNorm is a technique that aims to improve the training of neural networks by stabilizing the distributions of layer inputs.
- BatchNorm introduces additional network layers that control the first two moments (mean and variance) of the distributions of the input layers.
- BatchNorm reduces the so-called 'internal covariate shift'.
- BatchNorm makes the landscape of the corresponding optimization problem significantly more smooth.
* `[ C ]`


---

**Q:  Consider the situation when you are using batch normalization in order to improve classification performance of your deep network. What holds true?**
- Batch normalization usually helps to achieve better performance by normalizing first and second moments at each layer, however improvement is not necessarily linked to internal covariate shift (ICS)
- Batch normalization usually helps to achieve better performance by normalizing first and second moments at each layer, the improvement is linked to internal covariate shift (ICS)
- There is no reliable evidence that batch normalization helps to improve performance of classification
- Batch normalization usually does not improve performance and is not linked to internal covariate shift
* `[ A ]`


---

**Q: Instead of ICS, what seems to be the true strength of batch normalization?**
- it smoothes out the gradients, making them more predictable
- it reduces the dimensionality of the input data
- it decreases data variance, making the node weights more effective
- it decreases the size of the network, making training faster
* `[ A ]`


---

**Q: Which of the following are benefits of smoothing the loss landscape?
I: Training is less sensitive to the choice of hyperparameters
II: Local minima are "larger" and "flatter", making them easier to find in gradient descent
III: Our gradient descent algorithms can more confidently take larger steps**
- I
- II
- I and III
- II and III
* `[ C ]`


---

**Q: Which of the following is NOT correct?**
- Batch normalization might not reduce internal covariate shift (ICS)
- Batch normalization significantly improves the training process of deep networks
- If the internal covariate shift (ICS) in a network is lowered, then the performance of this network will always increase
- Batch normalization makes the landscape of the underlying optimization problem more smooth
* `[ C ]`


---

**Q: How does batch normalization help optimization?**
- it reduces the “internal covariate shift”
- it makes the optimization landscape significantly smoother
- it reduces the input noise
- it reduces the output noise
* `[ B ]`


---

**Q: What is FALSE about Batch Normalization?**
- it's achieved by adding layers that control mean and variance
- it makes the landscape of the cost function more smooth
- it allows better performance because it increases stability of layer input distributions
- seems that there is not a correlation between BatchNorm and Internal Covariate Shift (ICS)
* `[ C ]`


---

**Q: According to the authors of the article, batch normalization…**
- Improves training behaviour due to less internal covariate shift
- Improves taining behaviour due to a smoother optimization landscape
- Improves testing performance due to less internal covariance shift
- None of the above
* `[ B ]`


---

**Q: What is the reason BatchNorm enables faster and more stable training of DNNs?**
-  It makes the optimization landscape smoother.
- It reduces the internal covariate shift
- It introduces noise which makes the DNN more robust
- None of the above answers are correct
* `[ A ]`


---

**Q: Why batch normalization works as good as it is?**
- Due to its smoothing effect
- Due to its optimization algorithm
- Due to its low complexity 
- Both a and b
* `[ D ]`


---

**Q: How does BatchNorm helps?**
- Reduce the internal Covariate shift (ICS)
- Reduces Lipschitz const and makes gradient more predictive 
- Smootshes the landscape 
- All of the above
* `[ B ]`


---

**Q: Which of the following is not typically a step in the process of Batch Normalization?**
- Augmenting the network with additional layers
- Removing superfluous layers of the network
- Setting the batch mean and variance to 0 and 1 respectively
- Scaling the inputs based on the trainable parameters
* `[ B ]`


---

**Q: What is the effect of Batch Normalization?**
- The loss gradient is smoothed
- The training of the network is faster
- The Lipschitz constant is improved
- All of the above
* `[ D ]`


---

**Q: Why does Lipschitz constant of t the loss play a crucial role in optimization?**
- Because it controls the amount by which the loss can change when taking a step
- Because it controls amount of epochs
- Because it limits the maximal value of the loss
- Because it limits the minimal value of the loss
* `[ A ]`


---

**Q: Which of the following statements is true?

1. The effectiveness of BatchNorm is related to internal covariate shift.
2. BatchNorm’s stabilisation of layer input distributions is effective in reducing ICS.**
- Statements 1 is correct, statements 2 is incorrect
- Statement 1 is incorrect, statement 2 is correct
- Both statements are correct 
- Both statements are incorrect
* `[ D ]`


---

**Q: What is FALSE about BatchNorm?**
- BatchNorm reparametrizes the underlying optimization problem to make it more stable
- A smoother optimization problem implies that the gradients used in training are more predictive
- More predictive and well-behaved gradients enable faster and more effective optimization
- Batchnorm will always reduce the internal convariate to some extend
* `[ D ]`


---

**Q: In "How does Batch Normalization help optimization" the author discuss the effects of the now widely used technique of Batch Normalization. Batch normalization is a technique that aims to control the first two moments (mean and variance). Which of the following statements about Batch normalization (according to their) conclusions regarding is true?**
- Batch normalization works by reducing the internal covariate shift.
- Batch normalization works by randomly inserting noise after a layer to make sure that the training process is smoother by increasing the Lipschitzness (beta-smoothness).
- Batch normalization reparametrizes the underlying training optimization problem to make it more smooth.
- Though other techniques are available for batch normalization is the best way to reduce smoothness especially for deep linear networks. 
* `[ C ]`


---

**Q: What is true about BatchNorm?**
- To see if controlling mean and variance of the input is directly linked to improved training performance, one could inject random noise before BatchNorm layer and compare its performance with regular normalization procedure. The noise is sampled from a non-zero mean and non-unit variance distribution.
- Compared to non-BatchNorm networks, training a BatchNorm system is less sensitive to the choice of the learning rate.
- So far BatchNorm is the only way to make training algorithms more stable in terms of exploding or vanishing gradients.
- BatchNorm can only be efficiently applied to the input layer
* `[ B ]`


---

**Q: According to Shibani et al. in their paper on the effectiveness of BatchNorm,**
- the existence of Internal Coveriate Shift (ICS) is a good predictor of training performance.
- the method (BatchNorm) effectively reduces ICS.
- the method (BatchNorm) smoothens the optimization landscape, leading to more predictive optimization behavior.
- the conventional wisdom on BatchNorm's tendency to improve generalization is incorrect.
* `[ C ]`


---

**Q: BatchNorms (a naturalisation technique) performance gain was earlier attributed to**
- Reduction in Internal covariance shift. 
- making optimisation problem more smooth
- Enabling large range of learning rates and faster network convergence.
- Improving Lipschitzness of loss and gradient.
* `[ A ]`


---

**Q: What is the common believe behind the success of batch normalisation before this paper was written?**
- BN makes the gradients of the loss more lipschitz
- effective beta-smoothness
- internal covariate shift
- gradient predictiveness
* `[ B ]`


---

**Q: Which of the following is NOT a reason why Batch Normalization improves the effectiveness of training of deep neural networks?**
- Batch Normalization reduces the Internal Covariate Shift, which is the key factor in its performance gain compared to standard neural networks.
- Batch Normalization makes the underlying optimization problem more stable.
- Batch Normalization makes the underlying optimization problem more smooth.
- None of the above.
* `[ A ]`


---

**Q: Batch Normalization has a key effect on the training process:**
- It reparametrizes the underlying optimization problem to make it more stable and smooth
- It minimizes the two moments (mean and variance) of the distribution of each activation
- It normalizes the mean and variance of the augmented training samples 
- It makes the input data translation invariant
* `[ A ]`


---

**Q: What does BatchNorm do according to the paper?**
- It normalizes the output of a network
- It stabalizes the inputs of a networks layer during training
- It increases the internal covariate shift of a network
- It improves networks by normalizing different batches of training data
* `[ B ]`


---

**Q: What is the correct description of Group Normalisation (GN) and Batch Normalisation (BN)?**
- GN is obviously better than BN when batch size is small
- GN is obviously worse than BN when batch size is large
- GN is similar with BN when batch size is small
- GN is always worse than BN regardless of the batch size
* `[ A ]`


---

**Q: Which of the following statements regarding BatchNorm is true: I: There is a strong link between the performance of BatchNorm and the reduction of internal covariate shift. II: BatchNorm makes the landscape of its corresponding optimization problem significantly more smooth.**
- I & II
- II
- I
- None.
* `[ B ]`


---

**Q: Which statement about Batch Normalization (BatchNorm) is false?**
- BatchNorm aims to stabilize the distribution of layer inputs by introducing additional network layers that control the mean and variance of these distributions.
- Benefits of BatchNorm are robustness to hyper parameter setting and avoiding gradient explosion/vanishing.
- The smoothing effect is unique to BatchNorm. No other natural normalization strategies have similar impact and result in a comparable performance gain.
- The key effect that BatchNorm has on the training process is that it reparameterizes the underlying optimization problem to make it more stable and smooth. 
* `[ C ]`


---

**Q: What is the given explanation for batchnorm to work?**
- The internal covariate shift is reduced
- The optimum weights are always between -1 and 1
- The optimization landscape is smoothened
- They do not give a clear indication what the reason possibly could be
* `[ C ]`


---

**Q: Which of the following is true about the gamma and beta parameters used in Batch Norm?**
- They should remain constant for all the hidden layers
- They vary for each layer
- They have to be equal to each other to normalize
- None of the above
* `[ B ]`


---

**Q: Ioffe and Szegedy identify a number of properties of Batch Norm's (BN) that improve the training process of a Neural Network.
Which of the following was demonstrated NOT to be the case?**
- BN's stabilization of layer input distributions is effective in reducing Internal Covariate Shift
- BN prevents exploding or vanishing gradients
- Makes the NN robust to different settings of hyperparameters such as learning rate and initialization scheme
- BN keeps most of the activations away from saturation regions of non-linearities
* `[ A ]`


---

**Q: Where is a BatchNorm layer usually introduced in a hidden layer - h?**
- After the non-linear activations and before the pooling layers
- After the pooling layers in a hidden layers
- Prior to the non-linear activations and pooling layers
- None
* `[ C ]`


---

**Q: Which of the following statements regarding BatchNorm is/are true?

A It is a mechanism that aims to stabilize the distribution of inputs to a given network layer during training. This is achieved by augmenting the network with additional layers that set the first two moments of the distribution of each activation to be zero and one respectively. 
B Batch normalized inputs are typically scaled and shifted based on trainable parameters to preserve model expressivity. **
- A
- B
- A and B
- None of the above
* `[ C ]`


---

**Q: How does batch normalization not help optimization?**
- It reduces internal covariate shift.
- It makes the optimization landscape smoother.
- It increase the execution speed of the training phase.
- It stabilizes the training phase.
* `[ A ]`


---

**Q: According to the paper "How Does Batch Normalization Help Optimization?", for which of the below options is NOT a reason that Batch Norm helps optimization?**
- It makes the underlying optimization more smooth
- It makes the underlying optimization more stable
- It causes a reduction of internal co-variate shift
- It avoids the vanishing gradient problem
* `[ C ]`


---

**Q: What is internal covariant shift?**
- Change in outputs due to updates in the preceding layer
- Change in distribution of inputs due to updates in the preceding layer
- Change in bias
- None of the above
* `[ B ]`


---

**Q: How Does Batch Normalization Help Optimization?**
- It is unknown
- It doesn't help at all
- The popular belief is that this effectiveness stems from controlling the change of the layers but it is not fully known
- Batch normalization allows the network to adapt better to the different features set which counters over-fitting and then allows for a faster training time.
* `[ C ]`


---

**Q: Which one of the following about BatchNorm is not true?**
- From an optimization viewpoint, BatchNorm might not be even reducing that shift.
-  The key effect of BatchNorm is that it reparametrizes the underlying problem to make it more stable.
- Smoothing effect  is not unique to BatchNorm.
-  BatchNorm can reduce the internal covariance shift effectively.  
* `[ D ]`


---

**Q: Some researchers claim that batch normalization techniques render "a smoother optimization landscape" which leads to faster convergence of a network. Which of the following statements explains this link?**
- The output of a batch-normalized network is more predictable
- Variance and mean scaling help to identify the principal components of information in the input
- The quick identification of a local minimum relies on the topology of the loss function
- All of the above
* `[ C ]`


---

**Q: Batch Norm is found prior to ?**
- Input Layer
- Output Layer
- After non-linear activation
- Before pooling layers
* `[ D ]`


---

**Q: Which of the following statements is correct about BatchNorm?**
- BatchNorm is robust to hyperparameter setting and prevents gradient explosion/vanishing.
- BatchNorm is not the only normalization strategy that has smoothing effect.
- BatchNorm has a weak effect in reducing the internal covariate shift.
- All of the above
* `[ D ]`


---

**Q: According to the paper: "How does batch normalization help optimization", which of the following do the authors put forward as reasons for the success of batch normalization**
- It reduces the internal covariate shift between different distributions of inputs to a layer in a neural network
- It "smoothens" the optimization landscape allowing the use of larger learning rates without affecting predictability or stability of gradient descent.
- It reduces the sensitivity of the algorithm to weight selection hence making training faster
- None of the above
* `[ B ]`


---

**Q: Why the  BatchNorm technique for training a deep neural network make the performance of the deep neural network in comparison with a classic training of a deep neural network?**
- Because of the unique smoothing ability of the BatchNorm technique on the input data
- BatchNorm stems from increased stability of layer input distribution
- BatchNorm reduce the internal covariance shift (ICS)
- The gradients used in training phase are more predictive and well behaved
* `[ C ]`


---

**Q: According to this paper, why does batch normalization work?**
- It reduces internal covariate shift
- It smoothes the optimization landscape
- It solves the vanishing gradient problem
- It makes the training data linearly separable
* `[ B ]`


---

**Q: Choose the correct statement about batch normalization:**
- its contribution to the training process comes from smoothing optimization landscape
- it is deeply connected with internal covariance shift
- its reparametrization makes gradients less predictive
- leads to a more accurate, but slower training process
* `[ A ]`


---

**Q: Which of the following is not a property of Batch Normalization?**
- Prevention of exploding or vanishing gradients
- Robustness to different settings of hyperparameters such as learning rate and initialization scheme
- Keeping most of the activations away from saturation regions of non-linearities
- Reducing internal covariate shift, therefore increasing the stability of the mean and variance of input distributions
* `[ D ]`


---

**Q: Is smoothening effect a unique feature of BatchNorm?**
- no
- Yes
- yes, but other can have similiar features
- We don't know
* `[ A ]`


---

**Q: What could Batch Normalization NOT benefit?**
- faster and more effective optimisation
- robustness to hyperparameter setting
- reducing internal covariate shift
- avoiding gradient explosion/vanishing
* `[ C ]`


---

**Q: What is the use of epsilon in batch normalization?**
- Avoid log(0)
- avoid division by 0
- avoid 1/log(0)
- None of them
* `[ B ]`


---

**Q: How does batch normalization contribute to the speed of the training process?**
- Reduces internal covariance shift, thus making training faster.
- Reparametrizes the underlying optimization problem to make it more stable.
- Prevents weight decay.
- Prevents underfitting.
* `[ B ]`


---

**Q: What values does Batch Normalization try to set with augmentation?**
- Mean = 1, Variance = 1.
- Mean = 0, Variance = 1.
- Median = 1, Standard Deviation = 1.
- Mean = 0, Variance = 0.
* `[ B ]`


---

**Q: What is Internal Covariate shift?**
- The change in the distribution of layer inputs caused by updates to the preceding layers.
- Updation of layer inputs by a random weight.
- Both A and B.
- None of the above.
* `[ A ]`


---

**Q: Where does the performance of BatchNorm comes from?**
- The stabilization of the covariate shift
- The smoothness produced in the optimization, making the gradients more reliable
- The centrality of the measures
- None of the above
* `[ B ]`


---

**Q: In some experiments, Batch Normalisation has shown exceptional results. However, there are still some problem(s) associated with it. What? **
- It isn’t always as fast
- It isn’t always stable
- It is stable and fast, but we don’t understand why
- None of the above mentioned problems are true
* `[ C ]`


---

**Q: What is the mechanism of BatchNorm?**
- Stabilize the distribution (over a minibatch) of inputs to a given network layer
- Increase the variance of data to a given network layer
- Stabilize the convergence(over a minibatch) of inputs to a given network layer
- Increase the mean value of data to a given network layer
* `[ A ]`


---

**Q: Which of the following challenges can be addressed by Batch Normalization?**
- Slow learning rate
- Overfitting
- Underfitting
- All of above
* `[ A ]`


---

**Q: which of the following statements is true?
I. neural networks with batch normalization often exhibit an increase in internal covariate shift
II. The reduction of internal covariate shift is the main reason for the effectiveness of batch normalization**
- Only I is true
- Only II is true
- I and II are true
- Neither I or II is true
* `[ A ]`


---

**Q: The authors test if BatchNorm's effectiveness was related to Internal Covariate Shift reduction by injecting noise after BatchNorm layers of a network. What did they observe?**
- The performance of "noisy" Batch Normalized networks was similar to that of non-noisy ones. 
- The performance of "noisy" Batch Normalized networks was vastly inferior to that of non-noisy ones. 
- The performance of "noisy" Batch Normalized networks was vastly superior to that of non-noisy ones. 
- The performance of "noisy" Batch Normalized networks when compared to non-noisy ones was highly dependent on hyperparameter choices such as learning rate.
* `[ A ]`


---

**Q: Which of the following about BatchNorm is NOT correct?**
- BatchNorm helps reduce the internal covariant shift and thus gains better performance.
- It induces a more predictable and stable behavior of gradients.
- It leads to a significant smoothness which allows for faster training.
- BatchNorm layers usually located before the non-linearity of the previous layer.
* `[ A ]`


---

**Q: Batch Normalization aims to improve the training of neural networks by stabilizing the distribution of layer inputs. How is this achieved?**
- By updating each of the layers according to these distributions.
- By introducing additional network layers that control the first two moments (mean and variance) of these distributions.
- By reducing the number of network layers.
- None of the above
* `[ B ]`


---

**Q: What is the aim of Batch Normalization?**
- Improve the training of neural networks by stabilizing the distribution of layer inputs
- Improve generalization of the model by adding and dropping hidden nodes
- Organizing data in batches to speed up the training process
- Introducing Batch type layer to perform better convolutional capabilities of network
* `[ A ]`


---

**Q: Which of the following statements is NOT true about batch normalization?**
- It reduces the internal covariance shift of the layer inputs by a large amount
- It makes the optimization landscape of the network loss function more smooth
- It prevents exploding or vanishing gradients
- It makes the network more robust to different hyperparameter settings
* `[ A ]`


---

**Q: What does batch normalisation do?**
- normalise the output of the first layer of a neural network.
- normalise the output of each layer of a neural network.
- normalise the inputs of the neural network.
- normalise the weights of the neural network .
* `[ B ]`


---

**Q: Which of the following statements is true?
I Batch normalization reparametrizes the optimization problem.
II Batch normalizations effectiveness comes fully from the internal covariate shift reduction.**
- Only I is true
- Only II is true
- Both statements are true
- None of the statements are true
* `[ A ]`


---

**Q: According to the author of the paper: "How Does Batch Normalization Help Optimization?", what seems to be the correlation between BatchNorm and internal covariate shifts?**
- By applying batch normalization, you stabilize the distributions of the layer inputs, which may significantly reduce the internal covariate shift.
- By applying batch normalization, you stabilize the distributions of the layer inputs, which may significantly increase the internal covariate shift.
- By applying batch normalization, you stabilize the distributions of the layer inputs, which barely has any impact on the reduction of the internal covariate shift.
- By applying batch normalization, you stabilize the distributions of the layer inputs, which barely has any impact on the increasement of the internal covariate shift.
* `[ C ]`


---

**Q: What are the roots of BatchNorm's success?**
- It makes the landscape of the corresponding optimization problem more smooth
- It reduces the internal covariate shift
- It makes the algorithm invariant to hyperparameters
- It reduces the amount of layers
* `[ A ]`


---

**Q: BatchNorm reparametrizes the underlying optimization problem to make it more stable and smooth. This implies that the gradients used in training are more predictive and well-behaved, which enables faster and more effective optimization. What is not a benefit of BatchNorm that this explains?**
- robustness to hyperparameter setting
- better generalization
- avoiding gradient explosion
- avoiding gradient vanishing
* `[ B ]`


---

**Q: Which statement is FALSE about BatchNorm?**
- BatchNorm significantly controls internal covariate shift
- BatchNorm significantly makes the gradient more reliable and predictive
- BatchNorm significantly reparameterizes the underlying optimization problem
- All statements are true
* `[ D ]`


---

**Q: Which statement about BatchNorm is correct? **
- BatchNorm makes sure that in the initial phases of training there is a wide range of values along the direction of the gradient of the loss
- The implication of BatchNorms reparameterization is that the gradients are more predictive even though they get less reliable
- The effectiveness of BatchNorm stems from the change of the layers input distribution to reduce the internal covariate shift
- BatchNorm allows the training algorithm to take larger steps without the danger of running into a sudden minimum
* `[ D ]`


---

**Q: Batch Normalization is a mechanism that aims to stabilize the distribution (over a mini-batch) of inputs to a given network layer during training. This is achieved by augmenting the network with additional layers that set the mean and variance of the distribution of each activation to be, what?**
- mean and variance to be zero and one, respectively
- mean and variance to be one and zero, respectively
- mean and variance to be one and one, respectively
- mean and variance to be zero and zero, respectively
* `[ A ]`


---

**Q: What does covariate shifts mean?**
- Normalization of the input
- Shifts of the distributions input
- Covariance shifts
- Translation of an image object
* `[ B ]`


---

**Q: Which of the following options, according to the empirical results in this paper, is false?**
- The smoothing effects show improvements in the training performance.
- The existence of internal covariate shift is a good predictor of training performance.
- The smoothing effect implies that the gradients used in training are more predictive and well-behaved.
- With Batch Normalization the training is less sensitive to hyperparameter choices.
* `[ B ]`


---

**Q: Which of these statements concerning batch normalization is incorrect?**
- The existence of internal covariate shift, at least when viewed from the distributional stability perspective is a good predictor of training performance
- Batch normalization reparametrizes the underlying optimization problem to make it more stable and smooth
- Batch normalization makes the gradients used in training more predictive and well-behaved which enables faster and more effective optimization
- Flat minima improve generalization
* `[ A ]`


---

**Q: How does BatchNorm influence the Lipschitzness of gradient of the loss of a network?**
- BatchNorm improves the Lipschitzness
- BatchNorm eliminate the Lipschitzness
- Both A and B
- None of the above
* `[ A ]`


---

**Q: Which of the following is not true?**
- BatchNorm makes the landscape of the optimization problem significantly smoother
- The smoothening effect introduced by BatchNorm is shared by other optimization techniques
- Internal covariate shift is not a good predictor of training performance
- Batchnorm is guaranteed to reduce the internal covariate shift
* `[ D ]`


---

**Q: What is the effect of BatchNorm on training?**
- Using batches allows for more frequent gradient updates and so it speeds up learning
- Reduces ICS and so makes training easier
- Makes the loss function more smooth and so easier to predict/learn
- Allows for more training data as different combinations can be taken in batches.
* `[ C ]`


---

**Q: What does internal covariate shift refer to?**
- The change in the distribution of layer inputs caused by updates to the preceding layers
- The change of the features used in order to reduce the error
- The change in the distribution of the error
- The change of the covariances of the layers’ weights
* `[ A ]`


---

**Q: How does batch normalization help optimization**
- It helps controlling the change of layers' input distribution
- It reduces internal covariate shift
- It makes the optimization landscape smoother
- It forces the input data to act as a Gaussian distribution 
* `[ C ]`


---

**Q: Which following statement is true for batch normalization?**
- It helps enhance the predictability of the loss.
- It helps improve the Lipschitzness of the loss.
- It can significantly reduce internal covariate shift.
- It help make the loss of gradients more smooth.
* `[ C ]`


---

**Q: What is meant by internal covariate shift?**
- It is the penomenon wherein the distribution of inputs to a layer in the network changes due to an update of parameters of the previous layers.
- The change in covariate matrix of the input distribution caused by training.
- The shift in covariance between the layer weights caused by training.
- The shift in layer weights caused by batch normalization.
* `[ A ]`


---

**Q: Why does batch normalization help in optimization?**
- Normalizing the output of the activations of the network accounts for internal covariate shift.
- Batch normalizing has a smoothing effect on the loss function.
- The gradient descent procedure has faster convergence when applied in batches.
- Normalizing the input features of the deep network helps to stabilize the gradients.
* `[ B ]`


---

**Q: Which staement(s) is/are true?
I. Batch normalization increases the smoothness of the loss landscape
II. Batch normalization increase the internal covariate shift**
- Only I is true
- Only II is true
- Both are true
- None of them is true
* `[ A ]`


---

**Q: Which is not improved by Batch Normalization?**
- Reduce the internal covariate shift
- More predictive gradients
- Faster convergence
- More smooth
* `[ A ]`


---

**Q: BatchNorm performs so well:**
- By obtaining a more stable distribution of the layer inputs
- By increasing the internal covariance shift
- By making the gradients more reliable and predictive 
- By making gradient descent optimization sensitive to the choosing of learning rate and initialization
* `[ C ]`


---

**Q: what does batchnorm reduce?**
- internal covariate shift
- the amount of batches to run
- external variate shift
- None of the above
* `[ A ]`


---

**Q: Which statement about the smoothing effect of batch normalisation is correct?**
- If the optimisation landscape is smoother, bigger steps can be taken in updating weights since gradients are more reliable and predictive 
- Smoothing increases the variation in the loss and changes in gradients, which reduces the chance of exploding or vanishing gradients
- The smoothing effect comes with worse Lipschitzness (predictability of gradients)
- Parameter initialisation becomes more critical with increased smoothness
* `[ A ]`


---

**Q: Which of the following is NOT a type of normalization technique used in DL?**
- Batch normalization
- Group normalization
- Instance normalization
- They are ALL existing normalization techniques
* `[ D ]`


---

**Q: Which one is wrong?**
- Using BatchNorm leads to faster convergence
- BatchNorm helps by alleviating something called internal covariant shift. BatchNorm reduce its effect by controlling the mean and variance of layer 
- In the paper they measure the change in gradient caused by concurrently updating previous layer and then normalized  network show the  reduction in this notion of covariant shift.
- BatchNorm smoothes the landscape.
* `[ C ]`


---

**Q: 1. An internal covariate shift is considered a good predictor for training performance 2. Batchnorm reparametrizes the underlying optimization to make it more stable and smooth .**
- Statement 1 is true, statement 2 is false
- Statement 1 is false, statement 2 is true
- Both statement are false
- Both statements are true
* `[ B ]`


---

**Q: Which of the following statements are true? (Based on "How Does Batch Normalization Help Optimization:"

stat1: A fundamental impact of BatchNorm on the training process is to turn a optimization landscape in convex shape landscape(signficantly better "effective" Beta-smoothness.

stat2: A fundamental impact of BatchNorm on the training process is to make a optimization landscape significantly smoother than it was before. (Improvement in the Lipschitzness sof the loss function)

stat3: If present smoothening effect over the optimization landscape is unique to a BatchNorm procedure.

stat4: All the normalization strategies  offer comparable performance to a Batch norm

stat5: ELU and SELU are two examples of non-linearities that have a progressively decaying slope instead of a sharp saturation and can be used as an alternative for BatchNorm**
- 1,2 and 5
- 2 and 3
- 2,4 and 5
- all statements
* `[ C ]`


---

**Q: Batch normalization:**
- The performance it shows is directly related to reducing the internal covariate shift
- Is the best method to smoothen the optimization landscape
- A noisy BatchNorm network performs better than a non-BatchNorm network
- The reason for BatchNorm good results is completely understood
* `[ C ]`


---

**Q: Consider following two statements:

\begin{enumerate}
	\item BatchNorm reduces the internal covariate shift.
	\item BatchNorm makes the optimization landscape smoother.
\end{enumerate}

Which one(s) are true or false?**
- 1 true and 2 true.
- 1 true and 2 false.
- 1 false and 2 true.
- 1 false and 2 false.
* `[ C ]`


---

**Q: After the authors of paper 9, the reason that BatchNorm makes training faster and more stable is ...**
- ... the internal-covariate shift.
- ... the internal distribution shift.
- ... that it introduces momentum.
- ... that it smooths the optimization landscape.
* `[ D ]`


---

**Q: What is the reason for BatchNorm improving training of neural networks?**
- Reduce internal covariate shift
- Smoothen loss landscape
- A & B
- None of the above
* `[ B ]`


---

**Q: Which of the following is most likely not the reason behinds Batch Normalisations gains: Batch Normalisation **
- leads to more favourable initialisations.
- leads to a more smooth optimisation landscape.
- leads to more reliable and predictive gradients.
- leads increased stability of layer input distributions.
* `[ D ]`


---

**Q: What seems to be the reason for the better performance in networks equipped with Batch normalization?**
- An increase in internal covariate shift.
- A decrease in internal covariate shift.
- Increased lipschitzness of the loss.
- Descreased lipschitzness of the loss.
* `[ C ]`


---

**Q: Why does BatchNorm increase training effectiveness of CNNs?**
- It eliminates noisy results at the beginning layers.
- It stops the gradient from vanishing.
- It makes the gradients more predictable.
- It stabilizes output values.
* `[ C ]`


---

**Q: Which of the following theoretical result is not inculded in this paper?**
- The effect of BatchNorm on the Lipschitzness of the loss
- The effect of BN to smoothness
- BatchNorm does less than rescaling
- Minimax bound on weight-space Lipschitzness
* `[ C ]`


---

**Q: Which of these statements is correct**
- Qualitatively the l_p–normalization
techniques lead to larger distributional shift than the unnormalized
networks, yet they still yield improved optimization performance.
- Qualitatively the l_p–normalization
techniques lead to smaller distributional shift than the unnormalized
networks, yet they still yield improved optimization performance.
- Qualitatively the l_p–normalization
techniques lead to larger distributional shift than the unnormalized
networks, but they don't yield improved optimization performance.
- Qualitatively the l_p–normalization
techniques lead to smaller distributional shift than the unnormalized
networks, but they don't improved optimization performance.
* `[ A ]`


---

**Q: which of these statements is true?
I: the existance of internal covariate shift is a good predictor for training performance
II: BatchNorm does not necessarily reduce the amount of internal covariate shift **
- only statement I is true
- only statement II is true
- both statements are true
- both statements are false
* `[ B ]`


---

**Q: According to the paper "How Does Batch Normalization Help Optimization", which of the following is not a benefit of batch normalization? **
- It provides robustness to hyperparameter setting.
- It contributes significantly in avoiding gradient explosion or vanishing.
- It helps the gradients used in training to be more predictive and well-behaved.
- It reduces the internal covariance shift (ICS).
* `[ D ]`


---

**Q: Which impact of BatchNorm on the training process is found?**
- We can control the change of the layers' input distribution during training
- We can reduce the internal covariate shift
- We can smoothen the optimization landscape
- None of the above
* `[ C ]`


---

**Q: Batch Normalization**
- stabilizes the input to a given network layer during training.
- augments the networks with additional layers that sets the mean and variance to 1 and 0.
- normalized inputs are scaled and shifted to preserve the model expressiveness.
- does all the above.
* `[ D ]`


---

**Q: Why is Batch Norm beneficial?**
- It enables to use larger learning rates which make training faster
- It helps to reduce learning time when the small learning rates are used
- It selects the best hyperparameters
- All the variants
* `[ A ]`


---

**Q: What is a more plausible explanation for the effectiveness of batch normalisation?**
- It reduces internal covariate shift.
- It smoothens the optimisation landscape.
- Both A and B.
- Neither A nor B.
* `[ B ]`


---

**Q: Which of the following statement is not correct about Batch Normalization:**
- Batch Normalization enable faster and more stable training of DNNs
- Batch Normalization causes internal covariate shift
- Batch Norm might not reduce the covariate shift
- it has a tendency to reduce generalization 
* `[ D ]`


---

**Q: How does Batch Normalization help the training process?**
- It reduces the Internal Covariate Shift (ICS)
- It reparametrizes the underlying optimization problem to make its landscape significantly more smooth
- It adds noise to the training data, therefore not making the network overfit
- It optimizes the learning rate for each batch
* `[ B ]`


---

**Q: Which of the following is a disproved results of Batch Norm from the previous approaches?**
- Reducing Internal Covariate Shift
-  Using Batch Norm as Regularization
- Making the Gradients less predictive
-  Lipschitzness of both the loss and the gradients are improved
* `[ A ]`


---

**Q: Which of the following is/are correct?

1. Batch normalization reduces the so-called internal covariate shift
2. Batch normalization smoothes the exploration landscape**
- Only 1 is correct
- Only 2 is correct
- Both are correct
- Both are incorrect
* `[ B ]`


---

**Q: How does BatchNorm impact network training:**
- By making the landscape of the optimization problem more smooth, gradients become more predictive allowing a larger range of learning rates.
- By making the landscape of the gradient descent more smooth, the gradients descent becomes more predictive allowing for faster network convergence.
- It does not.
- By reducing the internal covariate shift and minimizing the change in the distribution of layer inputs.
* `[ A ]`


---

**Q: Which of the following statements is correct?**
- Batch Norm prevents exploding or vanishing gradients
- Batch Norm provides robustness to different settings of hyperparameter
- Batch Norm keeps the activations away from saturation regions of non-linearities
- All of the above
* `[ D ]`


---

**Q: How Does Batch Normalization Help Optimization?**
- By reducing the internal covariate shift.
- By improving the statistical distribution of the input data, and hence avoiding the vanishing/exploding gradient problem.
- By reducing the effective $\beta$ smoothness, and improving the Lipschitzness of the loss function.
- By using a batch instead of a single input in the SGD, one improves the 'correctness' of the gradient significantly.
* `[ C ]`


---

**Q: Which of the following is results of Batch Norm is incorrect?**
- Reducing Internal Covariate Shift
- Using Batch Norm as Normalization technique
- Making higher gradient predictions thereby making quicker network convergence 
- Slower network convergence rate
* `[ D ]`


---

**Q: Which of the following statements is true?**
- The loss gradients are more smooth when using BatchNorm, such that more faster and more effective optimization is possible.
- The loss gradients are steeper when using BatchNorm, such that more faster and more effective optimization is possible.
- The success of BatchNorm is due to the reduction of internal covariate shift.
- The existence of internal covariate shift is a good predictor of training performance.
* `[ A ]`


---

**Q: In terms of training accuracy, which network model performs better and does it depend on covarience shift?  **
- Standard model(non- batchnorm) and doesn't depend on covarience shift
- Standard and BatchNorm, depends on co variance shift 
- Standard and noisy BatchNorm 
- Standard and BatchNorm, doesn't depend on covariance shift
* `[ D ]`


---

**Q: What is internal covariate shift (ICS)? as defined by S. Loffe and C. Szegedy**
- The natural shift in the hidden layers covariance when introducing time-shifts in RNN
- Changes in the input distribution for each neural layer cascading throughout the network causing need of lower learning rates and parameter initialization
- During random initialization of weights, ICS is the covariate surface of the networks entire loss landscape.
- When performing BatchNorm, ICS refere to the problem of the ever changing in batch size and how the covariate can be used to counteract this.
* `[ B ]`


---

**Q: Is the effectiveness of BatchNorm related to the smoothing effect?**
- Yes, smoothing helps the due to the loss change and improved Lipschitzness of the loss function.
- No, but smoothing creates an bigger coverance.
- Yes, smoothing is the generalisation of the problem.
- No, the effect of smoothing is determined by the L-norm used.
* `[ A ]`


---

**Q: How is batch normalisation best described in aiding the learning process?**
- Better estimates for Stochastic Gradient Descent (SGD).
- Reduction of internal covariate shift.
- Less erratic weight updates within the hidden layers.
- Stabilises the loss function curve.
* `[ A ]`


---

**Q: Why batch normalization work?**
- it decreasethe internal covariate shift(ICS)
- it stabilize the layer input distribution
- it reparameterizes the optimization problem to make it more stable
- the reason is still not know
* `[ C ]`


---

**Q: According to paper 9 on BatchNorm, what impact of BatchNorm on the training process is most key to its success?**
- prevention of exploding or vanishing gradients
- robustness to different settings of hyperparameters such as learning rate and initialization scheme
- keeping most of the activations away from saturation regions of non-linearities
- It reparameterizes the underlying optimization problem to make its landscape significantly more smooth
* `[ D ]`


---

**Q: Which of the following statements is false?**
- Internal covariate shift is not a good predictor of training performance
- BatchNorm makes the training process more stable
- BatchNorm is less robust to hyperparameter settings
- BatchNorm makes the training process more smooth
* `[ C ]`


---

**Q: why batch normalization is important**
- stabilize distribution of input during stabilize neural network training
- reduce internal covariate-shift
- all
- none
* `[ C ]`


---

**Q: What is the drawback of using batch nomalization**
-  Small batch leads to inaccurate estimation of the batch statistics
- Limits model design
- Limits other features such as temporal length
- All of the three
* `[ D ]`


---

**Q: How does batch normalisation help optimisation?**
- By making the optimisation landscape less smooth
- By decreasing the internal covariate shift
- By keeping the internal covariate shift stable
- By making the optimisation landscape more smooth
* `[ D ]`


---

**Q: Which of the statements regarding batch norm is correct?**
- Prevents exploding or vanishing gradients
- Robustness to different settings of hyperparameter
- Keeping the activations away from saturation regions of non-linearities
- All of the above
* `[ D ]`


---

