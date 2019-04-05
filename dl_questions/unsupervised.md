# Questions from `unsupervised` :robot: 

**Q: Which of the following is FALSE about contractive and denoising autoencoders?**
- Denoising autoencoders make the reconstruction function resist small perturbations of the input.
- Contractive autoencoders make the reconstruction function resist small perturbations of the input. 
- Contractive autoencoders have a regularizer that encourages the derivatives of the feature extraction function to be as small as possible.
- The reconstruction error alone would encourage the contractive autoencoder to learn an identity function.
* `[ B ]`


---

**Q: Which of the following about variational autoencoders is FALSE?**
- They are trained to copy the input to the output. 
- They do not require regularization for their learned encodings to be useful.
- They are helpful if one wants to sample from autoencoders.
- One can make use of the reparametrization trick to compute gradients.
* `[ A ]`


---

**Q: GANs are composed by:**
- two competing networks
- encoder and decoder
- convolutional layers followed by an LSTM
- None of the above
* `[ A ]`


---

**Q: Which of the following is FALSE, with regards to an autoencoder network?**
- can be used in an unsupervised setting
- being a binary classification problem, cross entropy loss is a sensible choice for the training setting
- the input can be seen as the label that the output of the network should match
- All of the above
* `[ B ]`


---

**Q: What is unsupervised learning used for?**
- compression to store large datasets
- pre-training for feature training
- density estimation
- all of the above
* `[ D ]`


---

**Q: What is the output of an autoencoder?**
- a compression of the input without loss
- a compression of the input with loss
- the input
- depends on use case
* `[ C ]`


---

**Q: Generative models can be evaluated by:**
- Comparing likelihoods.
- Using external evaluators.
- Performing a different task using unsupervised features and evaluating its performance.
- None of the above.
* `[ D ]`


---

**Q: The Frobenius norm of Jacobian refers to the:**
- Norm of first-order partial derivatives vector.
- Norm of first-order partial derivatives matrix.
- First-order partial derivatives vector.
- First-order partial derivatives matrix.
* `[ B ]`


---

**Q: What is unsupervised learning not good for?**
- pre-training for feature learning
- density estimation
- computation cost reduction
- generating new data samples
* `[ C ]`


---

**Q: What is true for a undercomplete hidden layer?**
- it compresses the input and it is useful for representation learning
- it compresses the input and it is useful for training data
- No compression is needed and it is useful for training data
- No compression is needed and it is useful for representation learning
* `[ B ]`


---

**Q: What is unsupervised learning?**
- Training the neural network without hyper parameters.
- Training the neural network without regularization.
- Training the network without using the ground truth label.
- None of the above.
* `[ C ]`


---

**Q: When the size of the hidden layer is smaller than the input layer, we call that:**
- Undercomplete
- Overcomplete 
- An autoencoder
- A Variational Autoencoder
* `[ A ]`


---

**Q: What could be the goal of an undercomplete hidden layer in an autoencoder?**
- Regularization
- Compression
- Translation
- Representation learning
* `[ B ]`


---

**Q: In Generative Adversarial Networks (GANs), the goal of the generator network is:**
- Classify the real images
- Classify the fake images
- Try to fool the discriminator by generating real looking images
- Try to discriminate between real and fake image
* `[ C ]`


---

**Q: Unsupervised Learning: Why should one use Regularization for an Autoencoder with an overcomplete hidden layer?**
- Since an overcomplete hidden layer has more neurons than the input layer, it could simply copy the input. One wants to avoid this with Regularization
- Since an overcomplete hidden layer has more neurons than the input layer, it never learn something useful without regularization
- Since an overcomplete hidden layer is always exactly one neuron more than the input layer, it could simply copy the input. One wants to avoid this with Regularization
- The asked statement is wrong, regularization should be avoided for overcomplete hidden layers
* `[ A ]`


---

**Q: What is the job of the Discriminator Network in a GAN? **
- Generate real looking images
- Try to discriminate between real images and images that were created by the Generator Network
- Try to classify images that were created by the Generator Network
- Try to discriminate between Images that were created by two different Generator Networks
* `[ B ]`


---

**Q: Which of the following statements about unsupervised learning are true?

statement1: Supervised learning requires training labeled data. Unsupervised learning, in contrast, does not require labeling data explicitly.

statement2: The two main unsupervised learning tasks  are clustering the data into groups by similarity and reducing dimensionality to compress the data while maintaining its structure and usefulness.

statement3: How much money will we make by spending more dollars on digital advertising? Will this loan applicant pay back the loan or not?  Are questions that could be nicely solved with unsupervised learning.

statement4: Unsupervised learning algorithms can perform more complex processing tasks than supervised learning systems. However, unsupervised learning can be more unpredictable than the alternate model**
- 1 and 4
- 1 2 and 3
- 1 2 and 4
- All statements are true
* `[ C ]`


---

**Q: Which statements about the autoencoder are true?

statement1: An autoencoder is a type of artificial neural network used to learn efficient data codings in an unsupervised manner.

statement2: An autoencoder learns to compress data from the input layer into a short code, and then uncompress that code into something that closely matches the original data.

statement3: Denoising autoencoders take a partially corrupted input whilst training to recover the original undistorted input. 

statement4: Contractive autoencoder adds an explicit regularizer in their objective function that forces the model to learn a function that is robust to slight variations of input values.**
- 1
- 1 and 4
- 2 and 3
- All statements are true
* `[ D ]`


---

**Q: What would not be a reason to apply unsupervised learning?**
- If there is more data available than labels.
- If obtaining labels is expensive.
- If obtaining labels is hard.
- If there is not a lot of data.
* `[ D ]`


---

**Q: Which of the following statements is not true with respect to Generative Adversarial Networks?**
- A generator network tackles the problem of sampling directly from a high dimensional training distribution.
- A generator network tries to generate labels for what is presented to it.
- During training, the goal is to make the independent networks perform better at their task.
- During training, the two networks acts as the opponent of the other.
* `[ B ]`


---

**Q: Consider an autoencoder. A feedforward network to produce its input at the output layer is defined as:

\begin{equation}
	g(f(x))=x
\end{equation}

Knowing that x is continuous, what loss function would you minimise to train g?**
- $$\sum_{i}\frac{1}{2}\left(g(f(x_{i}))-x_{i}\right)^{2}$$
- $$\sum_{i}\frac{1}{2}\left(g(f(x_{i}))+x_{i}\right)^{2}$$
- $$\sum_{i}\frac{1}{2} \sqrt{g(f(x_{i}))-x_{i}}$$
- $$\sum_{i}\frac{1}{2} \sqrt{g(f(x_{i}))+x_{i}}$$
* `[ A ]`


---

**Q: Consider following two statements:

\begin{enumerate}
	\item A generator network distinguishes between real and fake images.
	\item A discriminator network creates real looking but fake images.
\end{enumerate}

Which one(s) are true or false?**
- Both true.
- 1 true and 2 false.
- 1 false and 2 true.
- Both false.
* `[ D ]`


---

**Q: Generation of samples from an auto encoder is something you cannot do because it is an undefined range. So how does a variational autoencoder solve this problem?**
- A variational autoencoder takes all possible input values at the same time and create new images from them by combining them 
- A variational autoencoder takes from the original input samples The mean and the varience into two differnt parts in a layer. Which will be combined in the next layer in a sampled latend vector z. With $z = \mu + \sigma \odot \varepsilon $. From which you generate the new sample 
- A variational autoencoder takes from the original input samples the variance and applies them to the next input image which will be the new generated image
- None of the above answers is correct
* `[ B ]`


---

**Q: For a Contractive autoencoder you have the following formula: $\Omega(h) = \| \frac{\partial f(x)}{\partial (x)} \|_{F}^{2}$ Frobenius norm of Jacobian. What does $\Omega(h)$ mean in relation with x and h?**
- In case $\Omega(h)$ is a high value then the hidden layer will have a large influence on the output. Which has a complexity in terms of O(compute/memory)=O($y * h$) 
- In case $\Omega(h)$ is a high value then a small change in the input x will change h a lot. In case $\Omega(h)$ is small then if x changes then h does almost not change, which makes it a robust system. Which has a complexity in terms of O(compute/memory)=O($x * h$)
-  In case $\Omega(h)$ is a high value then a small change in the input x will not change h. In case $\Omega(h)$ is small value then a small change in the input x changes then h does almost not change, which makes it a robust system. Which has a complexity in terms of O(compute/memory)=O($x * h$)
-  In case $\Omega(h)$ is a high value then a small change in the input x will change h a lot. In case $\Omega(h)$ is small then if x changes then h does almost not change, which makes it a robust system. Which has a complexity in terms of O(compute/memory)=O($h * h$)
* `[ B ]`


---

**Q: In unsupervised learning, the under complete hidden layer does the following :**
- Prevents coping the input
- Compresses the input
- Copies the input
- Useful for representation learning
* `[ B ]`


---

**Q: How does denoising autoencoder increases robustness to noise?**
- Increasing the training data
- Using more weights
- Having deeper networks
- Input a noisy sample and optimise to reconstruct a noise free sample
* `[ D ]`


---

**Q: What holds true for generative model evaluation?**
- Higher likelihoods is not necessarily visually better than a lower one
- Just copying the training set will do very well
- We can use the unsupervised features to evaluate another task, but that does limit the use of generative models
- All of the above
* `[ D ]`


---

**Q: Which of the following is true?**
- Machine Learning has better theoretical bounds than Deep Learning
- Deep learning requires less programming
- Machine learning methods can be optimized more than Deep learning ones
- Machine learning requires a new architecture design for each new application
* `[ A ]`


---

**Q: What is a characteristic of unsupervised learning?**
- Learning on data which has no ground truth label
- Learning on data which does not exist
- Learning on data without a teacher or supervisor
- Learning on results from classifiers like SVM
* `[ A ]`


---

**Q: What is the purpose of a “generator network” in the field of generative adversarial networks?**
- Generating an image that looks “realistic”
- Supplying power using multiple power generators in a network
- Generating an efficient internet network
- Generating a neural network that fits the model best
* `[ A ]`


---

**Q: Which of the following statements about autoencoders is incorrect:**
- A denoising autoencoder regularizes by making reconstruction robust to noise.
- A contractive autoencoder penalizes variations orthogonal to the natural variations in the input data.
- A variational autoencoder maps input to latent distributions rather than attributes
- All of the above are correct. 
* `[ D ]`


---

**Q: Which of the following statements about Generative Adversarial Network (GAN) are true:

\begin{enumerate}
\item The Discriminator is trained to discriminate between real and fake data.
\item The Generator is trained to generate data that gets classified as real by the Discriminator.
\end{enumerate}**
- Only 1 is correct
- Only 2 is correct
- both 1 and 2 are correct
- Neither are correct
* `[ C ]`


---

**Q: For what operation is unsupervised learning NOT good for?**
- Density estimation
- Generating new data samples
- Labeling data
- Pre-training for feature learning
* `[ C ]`


---

**Q: Which operation works well with an overcomplete hidden layer?**
- Representation learning
- Clustering
- Compressing training samples
- Feature extraction
* `[ A ]`


---

**Q: How can you decrease the step size in stogastic gradient descent?**
- Increase the learning rate
- Decrease the learning rate
- Increase the regularisation parameter
- Increase the regularisation parameter
* `[ B ]`


---

**Q: What is not a way of preventing overfitting? **
- Using more data
- Increasing the learning rate
- Reducing the number of features 
- Applying early stopping
* `[ B ]`


---

**Q: Are there downsides to increasing the size of the decoder and encoder without using regularization?**
- No. The bigger their capacity, the more data they can gather, which is a major benefit of the autoencoder.
- Yes. The biggest downside is the increased computational time, but the autoencoder would perform better.
- Yes. Having, making their capacity too large can lead to the autoencoder to simply copy the input without extracting any useful features.
- The sizes are absolutely irrelevant.
* `[ C ]`


---

**Q: General Adversarial Networks (GAN) are often described as a "Two player game"? Which of the statements explaining this description is incorrect?**
- In GAN, two nets are performing opposite tasks and thus competing against each other.
- The two nets are a generator network and a discriminator network. 
- The generative network are responsible for mapping features onto labels (eg. Given that image A has features X, Y, Z, etc. image A must be fake)
- The generative network is responsible for outputting the expected features based on the label. (eg. Image A is fake, thus it must have features X, Y, Z, etc.)
* `[ C ]`


---

**Q: For a variational autoencoder, which of the following functions below can be used within the loss function as a regularisation term to ensure the latent representation layer follows a certain distribution?**
- Negative Log Likelihood
- Kullback-Leibler divergence
- Least Squared Loss
- Cross Entropy Loss
* `[ B ]`


---

**Q: For Generative Adversarial Networks, what is the purpose of the discriminatory network?**
- To directly compare the generated output distribution to a training sample distribution
- To determine if the generated output distributions come from some learned "true" distribution
- To learn a transformation of a dataset from a simple probability distribution to a more complex one
- To learn a latent representation of training samples and regularizing that representation to approximate a normal distribution. 
* `[ B ]`


---

**Q: Which statement is wrong?
Autoencoders …**
- Can be used to compress the input to a smaller dimensionality
- are forward networks if the encoder is deterministic
- are under certain circumstances equal to PCA
- which are contractive (= Contractive autoencoder) should be not be sensitive to rotation when the task is handwriting recognition
* `[ D ]`


---

**Q: Which statement is wrong?**
- Regularization prevents an undercomplete autoencoder from learning the identity function
- Variational Autoencoder describe latent attributes in probabilistic terms
- The sampled latent vector of a Variational Autoencoder is considered as a deterministic note
- A regular gradient decent approach cannot be used to update a Variational Autoencoder.
* `[ D ]`


---

**Q: What is NOT true about the size of the input $x$ and the size $h$ of the hidden layer?**
- If $h > x$ the hidden layer is called overcomplete.
- An overcomplete hidden layer is useful for representation learning.
- If $h < x$ the input will be compressed.
- An undercomplete hidden layer could just copy its input $x$.
* `[ D ]`


---

**Q: What is NOT true about Generative Adversarial Networks (GANs)?**
- GANs learn transformations from a simple distribution to the training distribution.
- The discriminator network tries to fool the generator network.
- The generator tries to minimize the objective function.
- Evaluating a GAN is different than evaluating deep convolutional networks.
* `[ B ]`


---

**Q: Which statement is true about unsupervised learning?**
- No ground truth label is available in unsupervised learning. 
- Ground truth label is known in unsupervised learning. 
- Nothing can be said in general  
- None of above are correct 
* `[ A ]`


---

**Q: Clustering algorithms are - **
- unsupervised learning.  
- supervised learning.  
- nothing can be stated  
- none of these  
* `[ A ]`


---

**Q: What is the effect of adding noise to the input of an autoencoder?**
- The network converges faster to the true error of the dataset
- The network becomes more robust to noisy samples
- The network's performance can increase due to skipping over local minima
- The network becomes better at predicting structure within the input
* `[ B ]`


---

**Q: What effect would a weak discriminator have the training of a GAN network?**
- The generator would be motivated to perform better
- The generator would be allowed to perform weakly
- The network has to do more iterations in order to converge
- The generator would supply samples of better quality, increasing the quality of the discriminator
* `[ B ]`


---

**Q: Q1: Regarding contractive auto-encoders, which of the following statements is true.**
- A contractive auto encoder generally obtains higher loss than regular auto encoders, in trade off for robustness.
- The numeric value of the encoding equation is considered in the objective function which the network is trained on.
- The hidden layer is smaller than the input layer.
- The output of the encoder is the same as the input of the encoder.
* `[ B ]`


---

**Q: Q2:
For variational auto encoders, which of the following is false:**
- Variational auto encoders can be used to generate extra training data.
- For latent features a normal distribution is learnt, with norms and standard deviations for each feature
- Without the “reparameterization trick” backpropagation is not possible.
- Adding noise to the mean of a distribution will improve model performance. 
* `[ D ]`


---

**Q: Which of the following cases gives the best scenario for training a Generative Adversarial Networks?**
- The discriminator maximizes it's error, while the generator minimizes it's error
- The discriminator minimizes it's error, while the generator maximizes it's error
- The discriminator maximizes it's error, while the generator maximizes it's error
- The discriminator minimizes it's error, while the generator minimizes it's error
* `[ A ]`


---

**Q: Which of the following networks applies unsupervised learning?**
- Convolutional Neural Network
- Generative Adversarial Network
- Recurrent Neural Network
- Feed-Forward Network
* `[ B ]`


---

**Q: Which of the following isn't something unsupervised learning is useful for?**
- Learning compression to store large datasets
- Pre-training for feature learning
- Generating new data samples
- Solving regression problems
* `[ D ]`


---

**Q: Review the following two statements about autoencoding in neural nets:
\begin{enumerate}
    \item Using noisy and clean examples of an image in training, an autoencoder can remove noise from images to improve quality
    \item In sparse autoencoders you have more hidden units than input units
\end{enumerate}
Which of the two statements are true?**
- Statements 1 and 2 are true
- Statement 1 is true and statement 2 is false
- Statement 2 is false and statement 1 is true
- Statement 1 and 2 are false
* `[ A ]`


---

**Q: What is the goal of an undercomplete autoencoder?**
- To classify images
- To classify text
- To encrypt messages
- To compress input
* `[ D ]`


---

**Q: What is the input of a generative autoencoder?**
- Images from the original dataset
- Random noise from N(0,1)
- Random noise from a learned U[0,I]
- Random noise from a learned N(0,I)
* `[ D ]`


---

**Q: Which of the following is true about over complete hidden layer?**
- It is useful for representation learning
- It is useful for reinforcement learning
- It is useful for reward learning
- None of the above
* `[ A ]`


---

**Q: Which of the following concepts is useful in unsupervised learning?**
- Regularization
- Normalisation
- Back propagation
- None of the above
* `[ A ]`


---

**Q: A contractive encoder is called so because**
- Its hidden layer is smaller in dimension than the input layer
- It has smaller weights compared to other types of neural networks
- It aims to optimize by reducing the squares of the gradient values of the hidden layer derivatives
- None
* `[ A ]`


---

**Q: A denoising autoencoder in a CNN setting is most likely to have what type of loss?**
- Cross-Entropy Loss
- Pixel-wise Loss (L1 norm)
- Pixel-wise Loss (L2 norm)
- None
* `[ C ]`


---

**Q: What is an Autoencoder?**
- Feed forward network to reproduce its input at the output layer
- An automatic encoder network
- A decoder that does not require any input
- none of the above
* `[ A ]`


---

**Q: What determines if an autoencoder is Overcomplete or Undercomplete?**
- The size of the hidden layer
- The size of the hidden layer, relative to the input size
- The size of the input
- The amount of hidden layers
* `[ B ]`


---

**Q: Where is unsupervised learning not good in.**
- Compression
- pre-training
- pattern recognition
- generating new data samples
* `[ C ]`


---

**Q: What can 't we say about the Frobenius norm of Jacobian used for the contractive autoencoder. \OHM(h) with input x**
- It measures how much h changes when x changes.
- When \OHm(h) is small the system is robust.
- The O(compute/memory) equals O(x)
- It penalizes unwanted variantions.
* `[ C ]`


---

**Q: Why is having no ground truth label sometimes a positive thing?**
- The dataset compresses better without layers
- Datasets with labels only work on feed forward networks
- It is impossible to know the labels on some samples
- Labels are often the most expensive thing to obtain
* `[ D ]`


---

**Q: Which of the following statements is correct:
 I The discriminator tries to fool the generator with fake data
II If, for a distribution of images, D = 1, is a real image and 0 a fake image then the discriminator aims to minimize D for real data**
- I and II are correct
- I is incorrect, II is correct
- I is correct II is incorrect
- I and II are incorrect
* `[ D ]`


---

**Q: Which of the following is not directly an application of unsupervised learning?**
- Learn compression to store large datasets
- Parallelize Computation
- Density Estimation
- Generating New Samples
* `[ B ]`


---

**Q: Why might one train a Generative Adversarial Network?
I: Learning how to generate more data
II: Detection of fake or malicious input into a network
III: Transforming random noise into a meaningful data point**
- I
- II & III
- I & III
- I, II, & III
* `[ D ]`


---

**Q: Which statement isn't true about the hidden (bottleneck) layer?**
- In an undercomplete layer (h < x) the input is compressed
- In an overcomplete layer (h > x) the input isn't compressed
- In an overcomplete layer we need regularization to prevent copying the input
- In an overcomplete layer we have to copy the input
* `[ D ]`


---

**Q: Which isn't true about a GAN?**
- It uses a generator to generate fake images
- It uses a generator to distinguish between real & fake images
- The generator uses random noise and learns the transformation from the random noise to the training distribution
- A GAN can only be used on a sequential input
* `[ D ]`


---

**Q: An auto encoder can be made robust to noise by...**
- Inputting a noisy sample and optimize the to reconstruct the noise free distribution.
- Adding an aditional layer which is focussed on removing the noise
- Using more training data
- Adding the auto encoder formula s(x) and inputting this in the loss function, which results in g(f(x)) = s(x)
* `[ A ]`


---

**Q: How can samples be generated from an auto encoder?**
- Draw them from an artificially created distribution
- By enriching the output with noise dimensions
- By comparing properties of different distributions to the auto encoder input using F-tests
- This is not possible due to the undefined range
* `[ D ]`


---

**Q: Why are developments in unsupervised learning important?**
- Because labels are often the most expensive to obtain
- Because then supervised deep learning will not be needed anymore soon
- Because unsupervised deep learning technologies are much faster than supervised deep learning technologies
- Because it allows us to learn without training data
* `[ A ]`


---

**Q: What are the objectives of the Generator network (GN) and the Discriminator network (DN)?**
- GN: Generating real looking images. DN: Discriminate between real and fake images
- GN: Generating real looking images. DN: Discriminate between different classes of the images
- GN: Generating new network architectures. DN: Discriminate between real and fake images
- GN: Generating new network architectures. DN: Discriminate between different classes of the images
* `[ A ]`


---

**Q: What is a motivation for unsupervised learning?**
- Ground truth labels are often expensive to obtain. These are not needed in unsupervised learning.
- Unsupervised learning gives labels to an unlabeled data set.
- Sequential modelling.
- Parallellization of your code.
* `[ A ]`


---

**Q: What does an autoencoder typically learn?**
- The identity operation
- To project an input on some manifold
- The smallest details of the input
- A hash function
* `[ B ]`


---

**Q: What is NOT true?**
- Back-propagation in variational autoencoders is the same as the other autoencoders
- The variational decoder samples from a probability distribution vector
- Compression methods should use the undercomplete auto-encoders
- Regularization is needed in overcomplete auto encoders
* `[ A ]`


---

**Q: What is true?**
- GANs belongs to unsupervised learning
- GANs typically train an adversary and a discriminator network
- GANs are optimized by a formula from game theory
- All of the above
* `[ D ]`


---

**Q: What is unsupervised learning?**
- Training without training data
- Training without labeled training data
- Testing without labeled testing data
- Validating without labeled validation data
* `[ B ]`


---

**Q: What is not a characteristic of an undercomplete layer?**
- It has less nodes than the input layer 
- It compresses the input
- Information is lost 
- It has less nodes than the output layer
* `[ D ]`


---

**Q: How can we achieve regularization in an autoencoder?**
- Making the decoder sensible to noise
- Training the autoencoder with noisy samples and optimize to reconstruct the noise-free samples
- Using only the noisy samples to train the autoencoder
- None of above
* `[ B ]`


---

**Q: What are Generative Adversarial Networks?**
- Simple networks to detect fake images
- Two networks that generate random images
- Two networks that compete to generate and detect fake images
- All the previous answers
* `[ C ]`


---

**Q: How could one prevent an overcomplete hidden (bottleneck layer) from simply copying its input?**
- Normalization
- Regularization
- None of the above
- Both of the above
* `[ B ]`


---

**Q: What is not an issue when working with GAN’s?**
- The generator network becomes much stronger than the discriminator network
- The discriminator network becomes much stronger than the generator network
- Training takes a lot of computation time.
- All of the above can be issues.
* `[ D ]`


---

**Q: What of the following statement describes an autoencoder?**
- An autoencoder is a neural network that is trained to attempt to copy its input to its output
- An autoencoder needs to have a hidden code dimension that is equal to the input
- An autoencoder needs to have a hidden code dimension that is larger than the input
- All of the above
* `[ A ]`


---

**Q: What is the objective of a Generative Adversarial Network?**
- Generator $G$ must learn correctly classify samples as real or fake while discriminator $D$ must learn to fool the classifier
- For fake data $z$, generator $G$ aims to maximize such that $G(z)$ is close to one
- For real data $x$, discriminator $D$ aims to minimize such that $D(x)$ is close to zero
- None of the above
* `[ D ]`


---

**Q: In Autocoder scenario, where Hidden layer is larger in size than the input/output layers,  what should be done to prevent overfitting the input?**
- Regularisation, by early stopping prematurely during gradient descent.
- Regularisation, by adding noise to the input features.
- Regularisation, by parameter sharing to significantly reduce weights. 
- Regularisation, by dropout-rescaling (p fraction) to avoid gradient explosion/vanishing.
* `[ B ]`


---

**Q: Which of the following is true about a Generative Adversarial Network? **
- Discriminator maximises its loss and Generator minimises its loss.
- Discriminator minimises its loss and Generator maximises its loss.
- Discriminator maximises the objective function and Generator minimises the objective function.
- Discriminator minimises the objective function and Generator maximises the objective function.
* `[ C ]`


---

**Q: Statement 1: if h (hidden layer) < x (input size): Overcomplete hidden layer. Which means no compression is needed and this is useful for representation learning. 
Statement 2: It is not possible to generate samples from an auto encoder. **
- Both statements are true
- Statement 1 is true
- Statement 2 is true
- None of the statements are true
* `[ C ]`


---

**Q: Statement 1: A solution to make sampling possible from high dimensional training distibution is to sample from a simple distribution, such as random noise, and learn transformation from noise to training distribution. 
Statement 2: For generative modeles higher likelihoods is not necessarily visually better than a lower one. **
- Both statements are true
- Statement 1 is true
- Statement 2 is true
- None of the statements are true
* `[ A ]`


---

**Q: Which of the following statements is false?**
- An autoencoder is a feed forward network that reproduces input at the output layer.
- An autoencoder consist of a decoder and an encoder.
- For an overcomplete hidden layer compression is needed
- None of the above.
* `[ C ]`


---

**Q: Which of the following statements about autoencoders is false?**
- The denoising autoencoder(DAE) is an autoencoder that receives a corrupted data point as input and is trained to predict the original, uncorrupted data point as its output.
- The contractive autoencoder applies the penalty term 'the Frobenius norm of Jacobian' to obtain a robust representation of the data which is less sensitive for variations of the data.
- A variational autoencoder (VAE) provides a probabilistic manner for describing an observation.
- None of the above
* `[ D ]`


---

**Q: What is not a real autoencoder?**
- Contractive autoencoder
- Denoising autoencoder
- Variational autoencoder
- All are examples of real autoencoders
* `[ D ]`


---

**Q: Statement 1: A GANs discriminator network is easily fooled by the generator network
Statement 2: Generative networks such as GANs can be easily evaluated**
- I & II are true
- Only I is true
- Only II is true
- I & II are false
* `[ D ]`


---

**Q: Which of the following applications is likely not a good fit for unsupervised learning approaches?**
- Classification tasks with easily obtainable labels
- Learning compression techniques
- Conducting pre-training
- Data augmentation
* `[ A ]`


---

**Q: Which of the following statements does not unequivocally hold?**
- A contractive autoencoder penalizes unwanted variations.
- In an overcomplete layer, no compression is needed.
- Making overcomplete layers robust to noise is typically done by providing noisy samples and expecting the originals as output.
- An undercomplete layer does not compress the input.
* `[ D ]`


---

**Q: Which of these is TRUE about autoencoder?**
- An Undercomplete autoencoder does not have a regulariation term
- A denoising autoencoder compresses the input
- the penality added to the loss function of the Contractive autoencoder is the same used in the Denoising autoencoder
- in the Denoising autoencoder loss function is adedd the Frobenius norm of Jacobian as penality term
* `[ A ]`


---

**Q: Which of these is FALSE about Generative Autoencoders?**
- they are composed by 2 network
- they usually create better images than then Variational autoencoders
- Discriminator network aims to fool the Generator network
- Generator network forces the output of the Discriminator network on generated images to be close to 1
* `[ C ]`


---

**Q: Take suppose that we want a GAN to generate images of kittens. In a basic setup, which, of the following statement is False:**
- The Discriminator tries to minimize D_\theta(x) for real data.
- The Generator will generate fake images
- Both Generator and Discriminator are trained through gradient descent
- None of the above.
* `[ A ]`


---

**Q: In the paper 'strike a pose' the author try to to solve the optimisation problem of determining which angle/translation most confuses the DNNs. What makes this hard?**
- When the hidden layer in an autoencoder is undercomplete it generally well suited for representation learning. 
- When the hidden layer in an autoencoder is overcomplete the data will be compressed.
- AutoEncoder noise is introduced by adding multiple hidden layers.
- None of the above
* `[ D ]`


---

**Q: Which of the following about the a Generative Adversarial Network (GAN) on image processing is TRUE?**
- It takes both fake images and real ones as input.
- The objective is to maximize the ability of generator.
- The objective is to let the generator win.
- Both B and C.
* `[ C ]`


---

**Q: Which of the following about autoencoder is WRONG?**
- The undercomplete hidden layer is able to compress the input and by doing that, it learns features (latent attributes) in the input.
- The overcomplete hidden layer does not compress the input but it can copy the input which then can be solved by regularization. 
- Both denoising autoencoder and contractive autoencoder take advantages of overcomplete layers.
- All of above.
* `[ D ]`


---

**Q: Which of the following statements is NOT true regarding the autoencoder?**
- Its hidden layer could undercomplete and overcomplete.
- Denoising autoencoder uses an overcomplete hidden layer.
- Contractive autoencoder uses an undercomplete hidden layer.
- Contractive autoencoder could capture the information of variations.
* `[ C ]`


---

**Q: Which of the following statements is NOT true regarding GANs?**
- It has a generator network which tries to fool the discriminator by generating real looking images.
- It has a discriminator network which tries to discriminate between real and fake images.
- GANs is like a two player minimax game.
- Its final goal is to make the discriminator successfully discriminate between real and fake images.
* `[ D ]`


---

**Q: Which one of the following choices of evaluating generative models is not true?**
- We cannot actually evaluate the log-probability of the data under the model, but can evaluate only an approximation.
- Generative modeling is different because changes in preprocessing, even very small and subtle ones, are completely unacceptable. 
- Some small and subtle changes in preprocessing for generative modeling is acceptable. 
- When benchmarking generative models on the MINIST dataset, it is essential to compare real-valued models only to other real-value models and binary-valued models only to other binary-valued models. 
* `[ C ]`


---

**Q: Which one of the following choices of the autoencoder is not true?**
- An autoencoder is a neural network that is trained to attempt to copy its input to its output.
- If an autoencoder succeeds in simply learning to set g(f(x))=x everywhere, it is especially useful.
- Autoencoders are restricted in ways that allow them to copy only approximately, and to copy only input that resembles the training data.
- The model of autoencoders is forced to prioritize which aspects of the input should be copied, it often learns useful properties of the data. 
* `[ B ]`


---

**Q: What could happen if using overcomplete hidden layer?**
- copy input
- compress the input
- bad for representation learning
- none of the above
* `[ A ]`


---

**Q: The objective for training GANs?**
- maximize discriminator and maximize generator
- maximize discriminator and minimize generator
- minimize discriminator and maximize generator
- minimize discriminator and minimize generator
* `[ B ]`


---

**Q: 1. Machine learning algorithms are application independent.
2. Deep Learning algorithms are application dependent**
- Both the statements are true
- Both the statements are false
- Statement 1 is true and Statement 2 is false
- Statement 2 is true and Statement 1 is false
* `[ C ]`


---

**Q: In GANs, Fake images are generated by generator network by taking ______ as input ?**
- Noise 
- Augmented images from the training data-set
- Average of images present in the training data-set 
- Affine transformed randomly sampled images from the data-set
* `[ A ]`


---

**Q: Which of the following statement is correct?**
- PixelCNNs define intractable density function.
-  PixelCNNs define tractable density function.
- VAEs define tractable density function.
- None of the above.
* `[ B ]`


---

**Q: Which of the following is the characteristic of Generative Adversarial Networks?**
- GANs don't work with an explicit density function.
- Learn to generate from training distribution through 2-player game.
- Can be unstable to train, as jointly training 2 networks is challenging.
- All of the above.
* `[ D ]`


---

**Q: True or false?
IAutoencoders are just feedforward networks. 
II. The same loss functions and output
unit types that can be used for traditional feedforward networks are also used for
autoencoders.**
- T,T
- F,T
- T,F
- F,F
* `[ A ]`


---

**Q: Any latent variable model pmodel(h, x) defines a stochastic encoder
pencoder(h | x) = pmodel(h | x) (14.12)
and a stochastic decoder
pdecoder(x | h) = pmodel(x | h). (14.13)**
- True, for decoding distribution
- False, for encoding distribution
- True, for encoding distribution
- False, for encoding distribution
* `[ C ]`


---

**Q: Which of following is to input the damaged data as input and train to predict the undamaged data?**
- Denoising autoencoder
- Contractive autoencoder
- Variational autoencoder
- Generative adversarial network
* `[ A ]`


---

**Q: What is the objective of discriminator and generator in function V(D,G)?**
- Discriminator tries to minimize V(D,G)
- Generator tries to minimize V(D,G)
- Discriminator tries to maximize D(G(x)) to 1
- Generator tries to maximize D(G(x))
* `[ B ]`


---

**Q: What is difference between autoencoder and variational autoencoder?**
- Autoencoder directly uses embedding, variational autoencoder samples from a distribution.
- Autoencoder samples from a distribution, variational autoencoder directly uses embedding.
- Autoencoder has less diverse (variation) embedding than variational autoencoder.
- Autoencoder has more diverse (variation) embedding than variational autoencoder.
* `[ A ]`


---

**Q: Why is it hard to evaluate generative models?**
- Higher model likelihood doesn't translate to better visual result.
- External evaluation has flaws and doesn't measure "originality" of generated image.
- Using unsupervised features to evaluate another task limits the use of generative models.
- All of the above
* `[ D ]`


---

**Q: What statement(s) is (are) not true for an overcomplete hidden layer in an autoencoder?**
- h<x
- No compression needed
- $g(f(x))=x$
- All of the above 
* `[ A ]`


---

**Q: What does a generator network use to generate fake images?**
- Images that the discriminator successfully classified as fake
- Noise
- Artist impressions
- Real images
* `[ B ]`


---

**Q: When the size of the hidden layer of an auto encoder is larger than the input, it could end up copying the input “as it is”. This would render the method unusable for representation learning. Which is a tool that could help fix this issue? **
- Regularization.
- Batch normalization.
- Random initialization of the network’s weights.
- Dividing the hidden layer in two and adding residual connections.
* `[ A ]`


---

**Q: What is the motivation behind the use of generative adversarial networks (GANs)?**
- Sample training data from the large dimensional space in some problems such as image classification.
- Improve the training data by learning a function that, among other things, can de-noise it.
- Complement an existing neural net by repeatedly feeding it the same inputs with small but relevant modifications.
- None of the above.
* `[ A ]`


---

**Q: What is unsupervised learning? **
- With ground truth label
- No ground truth label
- No full ground truth labels
- No semi-ground truth labels
* `[ B ]`


---

**Q: Why is unsupervised learning good?**
- Ground truth labels are not the most expensive to obtain
- Ground truth labels are the most expensive to obtain
- Ground truth labels are not important
- Ground truth labels are necessary only for supervised learning
* `[ B ]`


---

**Q: What best describes the functioning of a discriminator network?**
- Trying to fool the discriminator by generating real looking images
- Trying to discriminate between real and fake images
- Trying to fool the generator by generating real looking images
- Trying to generate real and fake images
* `[ B ]`


---

**Q: Which of the following topics are more related to 'Machine learning' than 'Deep learning' according to the graph on the final slide of lecture DL08?**
- Representation learning
- Architecture design
- Optimization
- Theoretical bounds
* `[ D ]`


---

**Q: What is unsupervised learning?**
- No ground truth label available
- Machine learning
- Deep learning
- AI
* `[ A ]`


---

**Q: What tries a denoising autoencoder to do?**
- Regularization by making reconstruction robust to noise. (Adds noise to input values)
- Removes noise from input values.
- Recodes input samples to remove noise
- Removing duplicate inputs to make sure network keeps seeing new data.
* `[ A ]`


---

**Q: Which of the following is true about the denoising autoencoders?

A Input – corrupted data point; output – original, uncorrupted data point 
B If the encoder is deterministic, the denoising autoencoder is a feedforward network**
- A
- B
- A and B
- None of the above
* `[ C ]`


---

**Q: Which of the following is true about generative adversarial networks?**
- There is no direct way to sample from high dimensional training distribution
- Training these networks can be done by generator network, discriminator network 
- There is no fixed method to evaluate these models
- All of the above 
* `[ D ]`


---

**Q: Which of the following statement is not true for unsupervised learning?**
- There is no ground truth label
- Labels are not often the most expensive to obtain
- Learn compression to store large datasets
- Density estimation
* `[ B ]`


---

**Q: Which of the following statement for the size of the hidden layer is not true?**
- h<x undercomplete hidden layer
- h<x compresses well for training samples
- h > x overcomplete hidden layer
- h > x the compression needed
* `[ D ]`


---

**Q: Is regularization useful when working with autoencoders?**
- It is indeed needed when using undercomplete autoencoders 
- It is indeed needed when using overcomplete autoencoders 
- It is needed for every type of autoencoder to work properly
- No, it is useless in this particular field of deep learning
* `[ B ]`


---

**Q: Which of the following architectures of autoencoders are used to make their unsupervised learning noise robust?**
- Contractive autoencoders
- Denoising autoencoders
- Both contractive and denoising autoencoders
- Contractive, denoising and undercomplete autoencoders
* `[ C ]`


---

**Q: Which one of the following makes the following sentence false? If the dimension of the latent representation is smaller than the input then...**
- it “compresses” the input.
- the autoencoder is forced to learn the most salient features of the training data.
- the autoenconder can learn to perform the copying task without extracting any useful information about the distribution of the data.
- the autoencoder is called undercomplete.
* `[ C ]`


---

**Q: What is not true about autoencoders?**
- They can be used for dimensionality reduction for data visualization.
- The only purpose is to copy the input to the output.
- A noisy image can be given as input to the autoencoder and a de-noised image can be returned as output.
- Autoencoders work by compressing the input into a latent-space representation, and then reconstructing the output from this representation.
* `[ B ]`


---

**Q: The contractive autoencoder:**
- Is an undercomplete autoencoder
- Is used to compress the input
- Is robust to small variations of the input
- Can copy the input if overtrained
* `[ C ]`


---

**Q: Which is true for GANS:**
- For a perfectly trained network, the discriminator should output 0.5
- During the training phase, the network relies as input only on random noise
- For a perfectly trained network, the discriminator should output 0
- For a perfectly trained network, the discriminator should output 1
* `[ A ]`


---

**Q: In an auto encoder, if the input x is continue, what loss function would you minimise in order to train the wanted result g(f(x)) =x, where h=f(x) is the hidden layer and r=g(h) is the output layer.**
- \sum_i 0.5*(g(f(x_i))-x_i)^2
- \sum_i 2*(g(f(x_i))-x_i)^2
- \sum_i 0.5*(f(g(x_i))-x_i)^2
- \sum_i 0.5*(f(x_i)-g(x_i))^2
* `[ A ]`


---

**Q: What is not a type of auto encoder?**
- Generative 
- Variational
- Denoising
- Contractive
* `[ A ]`


---

**Q: 1. An undercomplete hidden layer “compresses” the input. 2. An undercomplete hidden layer is useful representation learning.**
- Both statements are false
- Statement 1 is true and statement 2 is false
- Statement 1 is false and statement 2 is true
- Both statements are true
* `[ D ]`


---

**Q: How are a discriminator network, generator network and real images connected? **
- discriminator and real images are input to generator network
- the generator network and real images are input to the discriminator network
- generator network is input for the discriminator network, real images not involved
- real images are input for the generator network which is input for the discriminator network
* `[ B ]`


---

**Q: The sample generation from an auto-encoder is usually impossible. However, a given solution for this is:**
- Create a hidden layer with a distribution that can be easily sampled
- Create a hidden layer with a Gaussian distribution as it is the only one that can be easily sampled
- Adjust the input layer such that it can sample data from the beginning
- Adjust the input layer by adding a distribution that can be easily sampled
* `[ A ]`


---

**Q: GANS neural network is composed of two main neural networks: the discriminator and the generator. Which of the following statements is TRUE for a good functionality of GANS:**
- There should be a trade-off between the discriminator and the generator
- The discriminator should always outperform the generator
- The generator should always outperform the discriminator
- Both the generator and the discriminator should output 0.8
* `[ A ]`


---

**Q: What is an undercomplete hidden layer?**
- Does not compress the input and larger than the input layer
- Compresses the input and smaller than the input layer
- Compresses the input and larger than the input layer
- Does not compress the input and smaller than the input layer
* `[ A ]`


---

**Q: What is the purpose of a Discriminator network?**
- Try to discriminate between real and fake images
- Generate real looking images
- Generate fake looking images
- Generate fake and real looking images
* `[ A ]`


---

**Q: What would not be a good application of unsupervised learning?**
- Learn compression to store large datasets
- Pre-training for feature learning
- Weather forecasting based on last year's data
- Generating new data samples
* `[ C ]`


---

**Q: Which statement about GANS is correct?**
- The discriminator tries to fool the generator by creating real looking images
- The generator tries to find the difference between real and fake images so that it can generate only real images
- The generator tries to maximize the objective function so that it fools the discriminator
- The use of GANS gets limited when using the unsupervised features to evaluate other tasks
* `[ D ]`


---

**Q: Which is not an advantage of recurrent neural networks?**
- The input size of the learned model is independent of the sequence length
- The same transition function can be used at every time step
- Sharing statistical strength across different sequence lengths
- Images can be processed efficiently
* `[ D ]`


---

**Q: Which of the following is not a disadvantage of RNNs?**
- Gradients can explode or vanish (without additional modifications)
- They are computationally expensive
- They are memory-intensive
- They cannot learn long-term dependencies
* `[ D ]`


---

**Q: Where is unsupervised learning used for**
- Data is nowaday not the problem 
- Data has no classification
- All of the above
- None of the above
* `[ C ]`


---

**Q: Where are variation encoding good at **
- Creating robust fake images
- Creating noise on images
- All of the above
- None of the above
* `[ A ]`


---

**Q: What loss function does an autoencoder typically utilize? **
- $\sum_{i} \frac{1}{2} \sqrt{decoder(encoder(x_i)) - x_i}^2$
- $\sum_{i} \frac{1}{2} (decoder(encoder(x_i)) - x_i)^2$
- $\sum_{i} \frac{1}{2} \sqrt{encoder(decoder(x_i)) - x_i}^2$
- $\sum_{i} \frac{1}{2} (encoder(decoder(x_i)) - x_i)^2$
* `[ B ]`


---

**Q: If you want to generate samples with an autoencoder, what kind do you use?**
- Convolutional autoencoder
- Contractive autoencoder
- Variational autoencoder
- Denoising autoencoder
* `[ C ]`


---

**Q: Which of these statements about unsupervised learning is incorrect?**
- There is no ground truth label available during training
- Labels for the training and test set are expensive to obtain
- Unsupervised learning can be used to learn the compression of large datasets
- Unsupervised learning outperforms supervised training on large datasets
* `[ D ]`


---

**Q: Which of these statements concerning Generative Adversarial Networks (GANs) is incorrect?**
- It is impossible to generate samples using a variational auto encoder alone
- The discriminator network tries to discriminate between real and fake images
- The higher a likelihood of a generative network, the better the visual similarity of an image to a real image
- Learning to create real looking images is done by sampling from a simple distribution (e.g. random noise) and learning the high dimensional transformation to the training distribution associated with this simple distribution
* `[ C ]`


---

**Q: Decide if the following statements are true or false.
Statement 1: An undercomplete network may just copy its input to the ouput.
Statement 2: The main goal for an autoencoder is to minimize the output error as much as possible.**
- 1: true, 2: true
- 1: true, 2: false
- 1: false, 2: true
- 1: false, 2: false
* `[ D ]`


---

**Q: Decide if the following statements are true or false.
Statement 1: Regularization penalizes autoencoders with a high model capacity.
Statement 2: Both deep and single layer autoencoders can be userful in practice.**
- 1: true, 2: true
- 1: true, 2: false
- 1: false, 2: true
- 1: false, 2: false
* `[ C ]`


---

**Q: What is unsupervised learning, and what is it good for?**
- In unsupervised learning, the computer develops its own training strategy, as no human is involved this reduces costs
- In unsupervised learning, the model can freely obtain a solution, without a human limiting its feasible set, reducing costs
- There is no ground truth label. This is good because labels are computationally expensive to process
- There is no ground truth label. This is good because labels are most expensive to obtain.
* `[ D ]`


---

**Q: What are generator and discriminator networks used for?**
- In the training process of GANS, the generator tries to fool the discriminator by generating real looking images, the discriminator tries to discriminate between real and fake images
- In the training process of RNNs, the generator tries to fool the discriminator by generating real looking images, the discriminator tries to discriminate between real and fake images
- In the training process of GANS, the discriminator tries to fool the generator by generating real looking images, the generator tries to discriminate between real and fake images
- In the training process of CNNs, the generator tries to fool the discriminator by generating real looking images, the discriminator tries to discriminate between real and fake images
* `[ A ]`


---

**Q: Can you direcly generate samples from an auto encoder?**
- Yes, that for auto encoder is for.
- No, undefined range.
- It depend on the samples.
- Only Contractive autoencoder can.
* `[ B ]`


---

**Q: What is unsupervised learning?**
- Learning method with no ground truth label.
- Learning method with very large datasets.
- Learning method with very small datasets.
- None of the above.
* `[ A ]`


---

**Q: Which of the following statements are correct:
1: If the "hidden" layer in an unsupervised learning algorithm is smaller than the input layer, we call this an "undercomplete" layer

2: If the hidden layer isn an unsupervised learning algorithm is overcomplete, overfitting is impossible**
- Both are correct
- Both are false
- Only 1 is correct
- Only 2 is correct
* `[ C ]`


---

**Q: When having an overcomplete layer, there is the risk of overfitting. This can be prevented by applying regularisation. Which of the following are appropriate ways of regularisation?**
- Making reconstruction robust to noise
- Minimalize variance
- Both are correct
- Both are wrong
* `[ C ]`


---

**Q: Which of the following is a method for improving performance of GANs?**
- Alternative loss function implementation
- Using two timescale update rule
- Having gradient penalty
- All of the above.
* `[ D ]`


---

**Q: Which of the following is the disadvantage of using GANs?**
- the generator produces samples that belong to a limited set of modes, leading to model collapse
- Since the Generator loss improves when the Discriminator loss degrades (and vice-versa), we can not judge convergence based on the value of the loss function. 
- GAN does not represent the quality or the diversity of the output
- All of the above. 
* `[ D ]`


---

**Q: In case of a Variational Autoencoder, why do we need the “Reparameterization trick”?**
- To enable back propagation through a random node.
- In oder to create the sampled latent vector.
- To prevent copying of the input.
- None of the above.
* `[ A ]`


---

**Q: Which of the following statements is true?

1. Variational autoencoders are capable of both compressing data like an autoencoder and synthesizing data like a GANS.
2. GANS is based on two neural networks contest with each other in a zero-sum game framework.**
- Statements 1 is correct, statements 2 is incorrect
- Statement 1 is incorrect, statement 2 is correct
- Both statements are correct 
- Both statements are incorrect
* `[ C ]`


---

**Q: Which statement is false about unsupervised learning?**
- It can generating new data samples
- It can be used to initialize weights
- It can be used to learn compression to store large data sets
- None of the above
* `[ D ]`


---

**Q: What is true about undercomplete and overcomplete hidden layers?**
- Undercomplete hidden layer compresses the input and is suited for training
- Overcomplex hidden layer compresses the input and is suited for training
- Undercomplete hidden layer does not compress the input and is suited for representation learning
- Overcomplex hidden layer does not compress the input and is suited for representation learning
* `[ A ]`


---

**Q: How large should the hidden layer of an autoencoder be, if we wish to make a denoising autoencoder?**
- The same size as the input
- Larger than the input (overcomplete)
- Smaller than the input (undercomplete)
- Exactly twice as large as the input
* `[ B ]`


---

**Q: What is the main use of Generative Adverserial Networks (GANs)?**
- Compressing your input data
- Regularizing the training data
- Sampling from a high dimensional training distribution
- Generating deep learning architectures which work well in game theory
* `[ C ]`


---

**Q: What is the purpose of adding noise to the auto encoder input?**
-  To prevent the auto encoder from simply copying the input
- To augment the input data, generating more training samples
- To make the auto encoder learn to recognise a noisy image which can be useful for image quality inspection
- To ensure that the algorithm doesn’t get stuck in some kind of minima
* `[ A ]`


---

**Q: Which of the following auto encoder architectures is designed with the purpose of preventing the network from producing an exact copy of the input sample?**
-  GANs
- Contractive autoencoder
- Variational autoencoder
- Expansive autoencoder
* `[ B ]`


---

**Q: What is the function of the undercomplete hidden layer is in in autoencoder ?**
- Compress the input
- Decompress the input
- Expand the input
- Learn features from the input
* `[ A ]`


---

**Q: What is the main difference between unsupervised learning and supervised learning?**
- The testset of unsupervised learning is unlabeled
- The testset of unsupervised learning is labeled
- People do not need to supervise the trainning process for unsupervised learning
- No clear difference
* `[ A ]`


---

**Q: what is a generator network?**
- a nn used for generating samples
-  a nn used for deciding what to generate
- a nn used for discriminating between classes
- None of the above
* `[ A ]`


---

**Q: what is a consequence of a hidden layer smaller than the input layer**
- it compresses the input
- no compression is needed
- it is more sensitive to noise
- none of the above
* `[ A ]`


---

**Q: What is not true about an autoencoder?**
- An autoencoder is the combination of an encoder function and decoder function
- It can convert input data into a different representation
- It can convert the new representation back into the original format
- It aims to create a slightly different output each time it is run with the same input
* `[ A ]`


---

**Q: What is a deep belief network?**
- Generative models with several layers of latent variables
- Discriminative models with several layers of latent variables
- Generative models with a single layer of latent variables
- None of the above
* `[ A ]`


---

**Q: Which statement below is wrong?**
- Unsupervised learning does not need ground truth label
- Unsupervised could be used as pre-training for feature learning
- Unsupervised learning has limited advantages because labels are cheap to obtain anyway
- Unsupervised learning could also be used for generating new data samples
* `[ C ]`


---

**Q: Which statement below is wrong?**
- Generator network of GANS tries to fool the discriminator by generating real looking images
- Generator network of GANS tries to discriminate between real and fake images
- Deep learning is coupled to the application while machine learning is application independent
- In generative models, higher likelihoods is necessarily visually better than a lower one
* `[ D ]`


---

**Q: What of the following is NOT  a good use of unsupervised learning?**
- Learning compression
- Pre-training
- Density estimation
- Increasing computational efficiency
* `[ D ]`


---

**Q: What is a good way to denoise an autoencoder?**
- Optimize the network to reconstruct noise-free data
- Filter out noisy data in every hidden layer
- Increase the size of layers and/or depth of network
- None of the above
* `[ A ]`


---

**Q: What is the of unsupervised learning good for?**
- Learning compression for storing large datasets.
- Performing pre-training for feature learning.
- Generating new data samples.
- All of the above.
* `[ D ]`


---

**Q: Consider the problem of generating handwritten digits like the ones in the MNIST dataset (which are taken from the real world). What is the most suited deep learning approach to use considering this application?**
- Recurrent Neural Networks (RNNs)
- Convolutional Neural Networks (CNNs)
- Generative Adversarial Networks (GANs)
- Long-Short Term Memory Networks (LSTMs)
* `[ C ]`


---

**Q: What is not the benefit of unsupervised learning?**
- It can generate labels
- It can store large dataset
- It can generate new data sample
- It can pre-train
* `[ A ]`


---

**Q: Which technique could prevent copying from input?**
- Regularisation
- Normalisation
- Centralisation
- None all above
* `[ A ]`


---

**Q: Which statement is wrong about autoencoder ?**
- Autoencoders are designed to be able to learn to copy input to output. 
- It consists of two parts, an encoder function and a decoder function. 
- Autoencoders may also be trained using recirculation
- Non of them. 
* `[ A ]`


---

**Q: Which statement is wrong about denoising autoencoder (DAE) ? **
- DAE  is an autoencoder that receives a corrupted data point as input and is trained to predict the original, uncorrupted data point as its output.
- It learns something useful by changing the reconstruction error term of the cost function.
- It is an overcomplete model. 
- Non of them. 
* `[ D ]`


---

**Q: In an overcomplete autoencoder, which of the following can be achieved by adding a regularization term?**
- Prevent the hidden layer from just copying the input.
- Making the autoencoder noise-robust.
- Both A and B are correct.
- None of the above.
* `[ C ]`


---

**Q: Which of the following is False regarding the key difference between a conventional autoencoder and a variational autoencoder?**
- A convential autoencoder maps an input to a value, whereas a variational autoencoder maps an input to a distribution.
- A variational autoencoder can be sampled from easily, regardless of dimensionality of the training distribution.
- Backpropagation for a variational autoencoder, without certain tricks such as reparameterization, is more difficult than a conventional autoencoder.
- Variational autoencoders can be readily used to create interpolations in the latent space.
* `[ B ]`


---

**Q: What is unsupervised learning?**
- An scenario with labels that can be noisy
- An scenario with labels and extra information about the labels
- An scenario where there is not a ground truth label
- None of the above
* `[ C ]`


---

**Q: What two types of neural networks are involved in a GAN?**
- The adversarial network and the optimizer network
- The learning network and the competitive network
- The generator network and the adversarial network
- None of the above
* `[ C ]`


---

**Q: Undercomplete hidden layer**
- Compresses the input
- Expand the input
- Is useful for representation
- None of the above
* `[ A ]`


---

**Q: Why labels are problematic for Deep Leaning?**
- They require human labor
- They are expensive
- They can be noisy
- All of the above
* `[ D ]`


---

**Q: What is not a common application for unsupervised learning:**
- Compressing data
- Generating new data samples
- Clustering data samples 
- All are common applications
* `[ D ]`


---

**Q: Which statement is not true regarding denoising autoencoders**
- Denoising autoencoders always make use of undercomplete hidden layers
- Denoising autoencoders attempt to generate output $x$ from noisy input $\tilde{x}$
- Denoising autoencoders attempt to make the network more robust to small input variations
- Denoising autoencoders are a form of unsupervised learning
* `[ A ]`


---

**Q: Which of the following is/are a possible application of Auto-encoder?**
- Dimensionality Reduction of input samples
- Making the model robust to noisy input
- Learning key features from input samples
- All of above
* `[ D ]`


---

**Q: Which of the following is potential applications of Generative Adversarial Networks?**
- Generating a higher resolution image given an input image.
- Generating a perfect replica of an image.
- Given image, modifying the image to remove certain features of the image without any “noise” in the output image.
- None of above
* `[ A ]`


---

**Q: What is NOT a main use case for unsupervised learning?**
- pre-training in feature learning
- making a model robust to noisiness
- generating additional training data
- processing sequential data 
* `[ D ]`


---

**Q: Choose the FALSE statement regarding GANs:**
- training is a competition between the generator network and the discriminator network
- can be used for image generation to create adversarial data
- can be used for image generation to create additional training data
- can be used to predict the time-to-next-event based on a history of events 
* `[ D ]`


---

**Q: Which of the following is not correct?**
- unsupervised learning has no groud truth label
- regularization can help prevent copying input
- GAN has a generative network and a discriminator network
- method to evaluate generative model is mature.
* `[ D ]`


---

**Q: Which of the following statment is not correct**
- Encoder and decoder can be assembled from many different models freely based on problem
- encoder needs to compress input into a fixed length output, if input increase in length/amount the compression will cause more information loss and former information may be replaced by later information
- To some extend, attention model can solve the problem in B 
- Encoder and decoder can not be assembled from many different models freely based on problem
* `[ D ]`


---

**Q: What is not a common application of unsupervised learning?**
- Pre-training for feature learning
- Generating new data samples
- Determining variable learning rates 
- Learning compression to store large datasets
* `[ C ]`


---

**Q: What is true about under- and overcomplete hidden layers?**
- An overcomplete hidden layer compresses the input and is usefull for representation learning
- An overcomplete hidden layer doesn't compress the input and is usefull for representation learning
- An undercomplete hidden layer compresses the input and is usefull for representation learning
- An undercomplete hidden layer doesn't compress the input and is usefull for representation learning
* `[ B ]`


---

**Q: Which of the following statements hold true?**
- An contractive autoencoder is useful for compressing training data
- Regularisation has to be applied to contractive autoencoder to increase the quality of compression
- Regulaisation can make a contractive autoencoder more robust to noise
- All the above options are wrong
* `[ C ]`


---

**Q: Which of the following hold true?**
- A generator has to modify input images so as to fool the discriminator
- A discriminator determines whether an image generated is fake or not
- Both the options (a) and (b) are correct
- Both the options (a) and (b) are wrong
* `[ B ]`


---

**Q: What unsupervised learning is NOT good for?**
- Learn compression to store large datasets.
- Pre-training.
- Choosing the learning rate.
- Generating new datasamples.
* `[ C ]`


---

**Q: What is unsupervised learning?**
- No ground truth label on the dataset.
- No test dataset.
- No verification dataset.
- Adding noise to the training samples.
* `[ A ]`


---

**Q: Which of the following statement is incorrect:**
- undercomplete hidden layer will compress the inpu
- Regularization could be used to prevent copying the input  in the overcomplete networks
- Reconstructing the loss function based on the noise distribution as an input can make the network robust to noise
- It is possible to blindly take samples from an autoencoder because they have finite range
* `[ D ]`


---

**Q: Which of the following statement is correct;**
- Variational autoencoders have a hidden layer with a normal distribution
- it is possible to sample from the hidden layer of a variational encoder with the normal distribution
- GAN sample from a random ditribution and learn a transformation from the sample to the training distribution 
- The generative network in the GANs is trying to minimize the discriminator error
* `[ D ]`


---

**Q: What does small value of \Omega(h) mean for a contractive autoencoder?**
- Robust model
- h changes even when x does not change - Robust
- Wide deviation - bad model
- None of the above
* `[ A ]`


---

**Q: In a contractive autoencoder, what does \omega(h) ie., Frobenius norm of Jacobian measure?**
- How much h changes when x change.
- How much h deviates from the real value
- How h increases with x
- How h decreases with x
* `[ A ]`


---

**Q: In an Autoencoder, an object of dimension $n$ will be compressed in the encoder to dimension $m$ for which $m < n$ (with the help of ReLu activation functions). What does this process achieve?**
- A latent feature distribution 
- Reconstruction is easier
- A linear feature reduction
- A non-linear feature reduction
* `[ D ]`


---

**Q: The denoising autoencoder processes a noisy input and tries to reconstruct a noise free version of it. What does this autoencoder learn to cancel the noise in the input?**
- The lower dimensional manifold which describes the input data 
- The lower dimensional manifold which describes the noise
- The latent attribute distribution which describe the input data
- A regularization term which cancels the noise
* `[ A ]`


---

**Q: How can you improve the performance of GANs**
- By introducing gradient penalty
- Using timescale update rule
- Using alternative loss function implementation
-  all of them
* `[ D ]`


---

**Q: Which is a limitation of using GAN**
- Generator produces samples belonging to limited set of models
- Convergence cannot be judged based on loss function value
- Does not represent diversity of output
- All of the above
* `[ D ]`


---

**Q: Which of the following statements about autoencoders is valid:**
- Applying regularization to an undercomplete layer can lead to noise robustness in reconstruction of multi-dimensional data.
- Applying feature learning to an undercomplete layer can lead to noise robustness in reconstruction of sparsity.
- Applying compression to an overcomplete layer can lead to noise robustness in reconstruction of multi-dimensional data.
- Applying feature learning to an overcomplete layer can lead to noise robustness in reconstruction multi-dimensional data.
* `[ A ]`


---

**Q: About the objective of generative adversarial networks, which of the following statements hold:**
- The generator tries to minimize s.t. $D_{\theta}_D(G_{\theta}_G(x))$ approximates 1 whereas the discriminator tries to maximize s.t. $D_{\theta}_D(x)$ approximates 1, and $D_{\theta}_D(G_{\theta}_G(x))$ approximates 0.
- The generator tries to maximize s.t. $D_{\theta}_D(G_{\theta}_G(x))$ approximates 1 whereas the discriminator tries to maximize s.t. $D_{\theta}_D(x)$ approximates 1, and minimize s.t. $D_{\theta}_D(G_{\theta}_G(x))$ approximates 0.
- The generator tries to maximize s.t. $D_{\theta}_D(G_{\theta}_G(x))$ approximates 1 whereas the discriminator tries to maximize s.t. $D_{\theta}_D(G_{\theta}_G(x))$ approximates 1, and minimize s.t. $D_{\theta}_D(x)$ approximates 0.
- The generator tries to minimize s.t. $D_{\theta}_D(G_{\theta}_G(x))$ approximates 1 whereas the discriminator tries to maximize s.t. $D_{\theta}_D(G_{\theta}_G(x))$ approximates 1, and minimize s.t. $D_{\theta}_D(x)$ approximates 0.
* `[ A ]`


---

**Q: Which of the following statements is false?**
- A GANS network uses random noise to create real looking data
- A GANS network does not need real images to train itself
- In a GANS network, the discriminator tries to discriminate between real and generated images
- Generative models are hard to evaluate 
* `[ B ]`


---

**Q: A denoising autoencoder has a:**
- Undercomplete hidden layer
- Complete hidden layer
- Overcomplete hidden layer
- Absent hidden layer
* `[ C ]`


---

**Q: Which of the following is/are true?
I. Denoising autoencoders (DAE) try to minimize the the difference between the input and output created from the noise free input.
II. Using contractive autoencoders gives a robust function
III. Clustering is an application of unsupervised learning**
- I-II
- I-III
- II-III
- I-II-III
* `[ C ]`


---

**Q: Which one of the following is true?**
- Generator networks try to differentiate between real and fake images
- Variational autoencoders use numerical descriptiptions for latent attributes.
- When using generative models, an image with higher likelihood provides a better visualization compared to an image with a lower one.
- Regularizing the loss function can prevent the input data been copies to the hidden layer, if the layer has a bigger dimension than the input layer
* `[ D ]`


---

**Q: What is the difference between autoencoders with an undercomplete hidden layer and an overcomplete one, and what are they used for?**
- Undercomplete layer "compresses" the input,  has to throw away information while keeping the best possible reconstruction. Overcomplete layer can store everything and can be used for representation learning. 
- Overcomplete layer "compresses" the input,  has to throw away information while keeping the best possible reconstruction. Undercomplete layer can store everything and can be used for representation learning. 
- Undercomplete layer "compresses" the input,  can store everything and can be used for representation learning.  Overcomplete layer has to throw away information while keeping the best possible reconstruction.
- All of the above.
* `[ A ]`


---

**Q: Why is it not possible to sample from an autoencoder, and how can you adapt it to make sampling possible?**
- You can't sample from an autoencoder because the input range is undefined. Variational autoencoders aim to solve this by making the hidden layer a distribution which is easy to sample from.
- You can't sample from an autoencoder because they only learn to encode their training examples. Variational autoencoders aim to solve this by letting the encoder generalize.
- It is possible to sample from a normal autoencoder.
- You can't sample from an autoencoder because the input range is undefined. You can sample from them by using a GAN.
* `[ A ]`


---

**Q: Why do we need to regularize if we use an overcomplete hidden layer**
- Otherwise the network could just learn the identity function (not actually learning anything)
- Otherwise the network would train prohibitively slow
- Otherwise the network will never converge to a loss value
- Otherwise the network will throw away too much information
* `[ A ]`


---

**Q: Which of the following can NOT be said about evaluating GANS**
- It is a solved problem
- Likelihood evaluation is not a solution
- having External evaluators evaluate quality is not a solution
- evaluating unsupervised features as a proxy is very limiting
* `[ A ]`


---

**Q: The Unsurpervised learning is an efficient learning technique that’s used for :**
- Object detection 
- Density estimation  
- Data augmentation
- Both b and c
* `[ D ]`


---

**Q: What is true about the Generator network and Discriminator network:**
- Generator network try to fool the discriminator by generating real looking images. 
- Discriminator network Try to discriminate between real and fake images and this discrimination can be seen as a regression problem
- Discriminator network try to discriminate between real and fake images and this discrimination can be seen as a classification problem
- Both a and c
* `[ D ]`


---

**Q: What does loss function of an autoencoder minimize?**
- It minimizes the difference between the predicted label and the true label
- It minimizes the difference between the input data and the predicted label, because no true label is available
- It minimizes the difference between the input and reconstructed data
- It minimizes the difference between the reconstructed data and the predicted label
* `[ C ]`


---

**Q: Can autoencoders be used for generating new data and if yes, how?**
- Yes. Autoencoders give new outputs that is why it can be regarded as new generated data
- No, because the task of autoencoders is to reconstruct the input data but not to generate new. So autoencoders can never be used for generating new data
- Indeed, the task of autoencoders is to reconstruct the input data but not to generate new. However, if we introduce a hidden layer with distribution, we will easily generate new data.
- None of the variants
* `[ C ]`


---

**Q: The latent space used in GANs, where each coordinate of the vector follows a normal distribution,**
- is used as input to the generator, who should learn to map the latent space into the distribution of real samples.
- is used as input to the discriminator, who should learn to distinguish it from the real samples.
- is used as the target output of the generator, who should learn to map the latent space into the distribution of the real samples.
- is used as the target output of the discriminator, who should learn to distinguish it from the real samples.
* `[ A ]`


---

**Q: Generative Adversarial Networks, usually have a very complicated objective function, viz. $\arg \min_{\theta_G} \max_{\theta_D} \mathbb{E}_{x \sim p_{data}}[\log{D_{\theta_D}}(x)] + \mathbb{E}_{x \sim p(z)}[\log(1 - D_{\theta_D}(G_{\theta_G}(z)))]$. We can assume the discriminator $D$ outputs values in $[0, 1]$. In particular, the second term can be interpreted as:**
- the discriminator is trying to minimize the value $D(G(z))$, which in turn will maximize $1 - D(G(z))$, which is going to maximize the value of $\mathbb{E}_{x \sim p(z)}[\log(1 - D_{\theta_D}(G_{\theta_G}(z)))]$
- the discriminator is trying to maximize the value of $D(G(z))$, which in turn will minimize $1 - D(G(z))$, which is going to maximize the value of $\mathbb{E}_{x \sim p(z)}[\log(1 - D_{\theta_D}(G_{\theta_G}(z)))]$
- the discriminator is trying to minimize the value $D(G(z))$, which in turn will maximize $1 - D(G(z))$, which is going to minimize the value of $\mathbb{E}_{x \sim p(z)}[\log(1 - D_{\theta_D}(G_{\theta_G}(z)))]$
- the discriminator is trying to maximize the value of $D(G(z))$, which in turn will minimize $1 - D(G(z))$, which is going to minimize the value of $\mathbb{E}_{x \sim p(z)}[\log(1 - D_{\theta_D}(G_{\theta_G}(z)))]$
* `[ A ]`


---

**Q: What is the unsupervised learning in deep neural networks good for?**
- Learn compression to store large datasets.
- Pre-training for feature learning.
- Generating new data samples.
- Detecting image forgery.
* `[ D ]`


---

**Q: Which of the following statements about the Generative Adversarial Networks is false?**
- The generator network tries to fool the discriminator by generating real looking images.
- The discriminator network tries to discriminate between real and fake images.
- Typically, the generator is seeded with randomized input that is sampled from a predefined latent space.
- The generative network's training objective is to decrease the error rate of the discriminative network.
* `[ D ]`


---

**Q: In order to evaluate the performance of generative model we:**
- Evaluate likelihoods
- Let external evaluators evaluate model quality
- Use the unsupervised features to evaluate another task
- It is not clear yet how we can evaluate generate networks
* `[ D ]`


---

**Q: Given a discriminator D and generator G and input x. What is the relation between these variables?**
- D aims to minimize so that D(x) is close to 1 for real data and D(G(x)) is 0 for fake data.
- D aims to minimize so that D(x) is close to 1 for real data and D(G(x)) is 1 for fake data.
- D aims to minimize so that D(x) is close to 0 for real data and D(G(x)) is 0 for fake data.
- D aims to minimize so that D(x) is close to 1 for real data and D(G(x)) is 1 for fake data.
* `[ A ]`


---

**Q: Unsupervised learning is good for**
- Learn compression to store large datasets
- Pre-training for feature learning
- Generating new data samples
- All of the above
* `[ D ]`


---

**Q: Which of the following is false about a GAN**
- Training GANS is a two player game
- The generator takes random noise as input
- Generative models may be evaluated using likelihoods
- The discriminator will discriminate between real images and noise
* `[ D ]`


---

**Q: Which one is wrong?**
- An autoencoder neural network is an unsupervised learning algorithm
that applies backpropagation, setting the target values to be equal to the
input
- For the autoencoder. It’s not trivial to learn because it needs to represent a high-dimensional input with a lower-dimensional code vector, forcing it to learn a compact representation.
- If the autoencoder’s activation functions are linear, it is very similar to PCA method.
- None
* `[ D ]`


---

**Q: Which one is wrong**
- Higher likelihoods is necessarily visually better than a lower one
- Gan try to fool the discriminator by generating real looking image
- GANs have the problem that  the generator may produce artifacts such as a chicken with three legs or a cat with two heads. 
- None
* `[ A ]`


---

**Q: What is the advantage using variational autoencoders over normal encoders?**
- Variational autoencoders makes the hidden layer a distribution which is easy to sample from.
- Variational autoencoders can be used to interpolate results in the latent space since the distributions are smooth.
- None of the above
- Both A and B.
* `[ D ]`


---

**Q: What is the use of minimizing the Forbenius norm of the Jacobian of the hidden states with respect to the inputs?**
- It minimizes the adverse effect of noise in the inputs on the hidden states.
- It makes the hidden states robust to small variations in the input.
- All of the above
- None of the above.
* `[ C ]`


---

**Q: What is the not a reason to use unsupervised training?**
- The data provided by the user is not labeled.
- There is not enough data provided by the user.
- The data provided by the user must be compressed.
- None of the above.
* `[ B ]`


---

**Q: Which of the following statements about GANS is not true?**
- The Generator network tries to recognize generated images 
- The Generator network tries to generate real looking images.
- The discriminator network discriminates between real and fake images
- None of the above.
* `[ A ]`


---

**Q: Which of the following is NOT a benefit of unsupervised learning?**
- No labeling of training samples is required
- Enables storing large data
- Enables generating new data samples
- Does not require a loss function because of missing labels
* `[ D ]`


---

**Q: A hidden layer that is:**
- Undercomplete has more hidden layers than input layers
- Overcomplete compresses well for training
- Overcomplete can not be prevented from copying the input
- None of the above
* `[ D ]`


---

**Q: What is unsupervised learning?**
- Learning without training data
- Learning without ground truth label
- Learning without loss function
- Learning without humans
* `[ B ]`


---

**Q: Which two networks make up a GAN network?**
- Discriminator and generator
- Discriminator and generaliser
- Differentiater and generaliser
- Integrator and generator
* `[ A ]`


---

**Q: What is the name of the hidden layer of an auto encoder when the number of nodes of the hidden layer is smaller than the number of inputs?**
- a) under complete hidden layer
- b) over complete hidden layer
- c) complete hidden layer
- d) insufficient hidden layer
* `[ A ]`


---

**Q: In what way can you generate new samples from a variational auto encoder?**
- a) Make the hidden layer under complete
- b) Make the hidden layer over complete
- c) Make the hidden layer a distribution to sample from
- d) Is not possible as it is dependent on the input
* `[ C ]`


---

**Q: Consider the following two statements about Generative Adversarial Networks, which consist of a generator and discriminator network:
1.The generator network tries to fool the discriminator by generating real looking images and the discriminator network tries to discriminate between real and fake images. 
2.Generative Adversial Networks must be evaluated by external evaluators. 
Which of these statements is true?**
- Both are true
- Both are false
- 1 is true and 2 is false
- 1 is false and 2 is true
* `[ A ]`


---

**Q: Consider the following two statements about Recurrent Neural Networks
1.Regardless of the sequence length, the learned model always has the same input size, because sequence length is specified in terms of state transitions rather than a variable-length history of states.
2.The same transition function f with the same parameters can be used at every time step. 
Which of these statements is true?**
- Both are true
- Both are false
- 1 is true and 2 is false
- 1 is false and 2 is true
* `[ A ]`


---

**Q: What is unsupervised learning used for?**
- Learning compression to store large datasets
- pre-training for feature learning
-  generating new data samples
- All
* `[ D ]`


---

**Q: Regularisation Autoencoders- 
1. keep the encoder and decoder shallow with small code size
2. use a loss function that encourages the model to have other properties besides the ability to copy its input to its output
3. sparsity of the representation, smallness of the derivative of the representation, and robustness to noise or to missing inputs**
- 2&3 are true
- 1&2 are true
- All are true statements 
- None
* `[ A ]`


---

**Q: How can noise be removed from data?**
- Train a network to predict the original input give noisy data
- Regularize more, to remove effect of noise
- Average the data over all samples
- smooth it out using regression
* `[ A ]`


---

**Q: What is the purpose of a contractive auto-encoder?**
- reduce dimensional
- penalize unwanted variations
- Speed up training
- regularization
* `[ B ]`


---

**Q: Which statement about variational autoencoders is false?**
- It maps an input onto a distribution
- Bottleneck replaced with a mean and standard deviation vector
- Reparameterisation of the sampled latent vector is done with a uniform distribution
- We cannot sample from the mean and s.d. vector directly since backpropogation cannot be performed on sampling (instead using Reparameterisation)
* `[ C ]`


---

**Q: Which use for an autoencoder requires that it is undercomplete?**
- Compression of data
- Removing noise from images
- Ignoring unwanted variations
- All of the above
* `[ A ]`


---

**Q: An autoencoder network with an Undercomplete hidden layer can,...**
- ...never reconstruct the input data exactly, as the bottleneck layer is always encodes the data with a lower dimensionality.
- ...can reconstruct the input data exactly, as long as the number of distinct perpendicular feature vectors of the input data is smaller or equal to the number of neurons in the bottleneck layer.
- ...can always reconstruct the input data set, given a deep enough architecture.
- ...be in general compress data much more efficiently than common image compression algorithms (jpeg) for small datasets.
* `[ B ]`


---

**Q: A denoising autoencoder, is any architecture that: **
- Performs outlier detection on the training dataset.
- Reconstructs an original dataset from an input where parts have been removed.
- Is a special type of RNN that improves the signal to noise ratio of (electromagnetic) signals.
- Reconstruct an original dataset from an input that has been 'contaminated' with a noisy signal.
* `[ D ]`


---

**Q: Regularized autoencoders provide theability to do:**
- have a dimension equal to the input
-  dimension is lessthan the input dimension
- hoosing the code dimension and the capacity of the encoder and decoder based on the complexity of distribution to be modeled
- none of the above
* `[ C ]`


---

**Q: encoders where penalty Ω(h) is the squared Frobenius norm (sum of squared elements) of theJacobian matrix of partial derivatives associated with the encoder function.**
- contractive autoencoders
- denoising autoencoder
- sparse autoencoder
- none of the above
* `[ A ]`


---

**Q: Which of the following statements is not true?**
- For an under complete hidden layer,  the input should be compressed.
- For an over complete hidden layer, input compression is not required.
- Under complete hidden layer is useful for representation learning.
- Over complete hidden layer is useful for representation learning.
* `[ C ]`


---

**Q: Which of the following is not true about autoencoders?**
- They are trained to attempt to copy its input to its output with certain restrictions. 
- They learn useful properties of the input data. 
- They are used for dimensionality reduction and generative modelling. 
- Autoencoders cannot be trained using recirculation.
* `[ D ]`


---

**Q: What is unsupervised learning?**
- Making use of (partially) synthetically generated dataset
- Training without defining a loss function
- Training with unlabeled data
- Training without validation
* `[ C ]`


---

**Q: What does a contractive autoencoder do differently from a sparse autoencoder?**
- It penalizes the cost function
- it learns a function that does not change much when the input changes slightly
- It diminishes the impact of large input values
- All of the above
* `[ B ]`


---

**Q: Which is not true for autoencoder?**
-  A kind of neural network that aims to copy its input to output.
- It has a hidden layer that describes the code used to represent the input.
- PCA could be regarded as an overcomplete autoencoder.
- The network is consisted of two parts: encoder and decoder.
* `[ C ]`


---

**Q: What kind of autoencoder needs a regularization term in loss function?**
- Undercomplete autoencoder
- Overcomplete autoencoder
- Both A and B
- Neither
* `[ B ]`


---

**Q: Why is unsupervised learning advantageous for certain cases?**
- Because it is fancier.
- Because it is computationally faster.
- Because no labels are needed, which can be costly to create.
- Because no data is needed.
* `[ C ]`


---

**Q: What is the main difference in a denoising autoencoder and a contractive autoencoder?**
- A denoising autoencoder makes the reconstruction robust to noise while a contractive autoencoder makes the reconstruction robust to infinitesimal perturbations.
- A denoising autoencoder makes the reconstruction robust to infinitesimal perturbations while a contractive autoencoder makes the reconstruction robust to noise.
- A denoising autoencoder makes the reconstruction more sensible to noise while a contractive autoencoder makes the reconstruction more sensible to infinitesimal perturbations.
- A denoising autoencoder makes the reconstruction more sensible to infinitesimal perturbations while a contractive autoencoder makes the reconstruction more sensible to noise.
* `[ A ]`


---

**Q: Which option below can be regarded as unsupervised learning**
- Linear regression
- Given medical scanning images of cancer patients, build a model to estimate the stage of a new cancer patient.
- Given medical scanning images of cancer patients, cluster those images into several groups where images have similar features
- None of the three above
* `[ C ]`


---

**Q: Given a feed forward network x -> h=f(x) -> r=g(h),  if size(h) < size(x), what option below is incorrect?**
- The hidden layer compresses the input x
- The hidden layer is undercomplete
- It is useful for representation learning
- None the the three
* `[ C ]`


---

**Q: Which of the following statements is true for over-complete network?**
- NO compression required
- Is useful for representation learning
- Compresses well for training samples
- Both A) and B)
* `[ D ]`


---

**Q: Which of the following methods could prevent copying the input elements in unsupervised learning**
- Normalisation
- Regularisation
- Backpropogation
- Feed forward
* `[ B ]`


---

