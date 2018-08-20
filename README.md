# Journal :postbox:

I plan to fill this section with what I discovered today - - - AFAP <As Frequently as possible>! 

---
`August 15, 2018`

#### CS231n Convolutional Neural Networks

- A ConvNet architecture is in the simplest case a list of Layers that transform the image volume into an output volume (e.g. holding the class scores)
- There are a few distinct types of Layers (e.g. CONV/FC/RELU/POOL are by far the most popular)
- Each Layer accepts an input 3D volume and transforms it to an output 3D volume through a differentiable function
- Each Layer may or may not have parameters (e.g. CONV/FC do, RELU/POOL don’t)
- Each Layer may or may not have additional hyperparameters (e.g. CONV/FC/POOL do, RELU doesn’t)
- _Filter_ = _Receptive Field_
- In general, setting zero padding to be P=(F−1)/2 when the stride is S=1 ensures that the input volume and output volume will have the same size spatially.
- We can compute the spatial size of the output volume as a function of the input volume size (W), the receptive field size of the Conv Layer neurons (F), the stride with which they are applied (S), and the amount of zero padding used (P) on the border. How many neurons “fit” is given by (W−F+2P)/S+1
- **Parameter sharing** scheme is used in Convolutional Layers to control the number of parameters. It turns out that we can dramatically reduce the number of parameters by making one reasonable assumption: That if one feature is useful to compute at some spatial position (x,y), then it should also be useful to compute at a different position (x2,y2). Only D unique set of weights (one for each depth slice)
- It is worth noting that there are only two commonly seen variations of the max pooling layer found in practice: A pooling layer with F=3,S=2 (also called overlapping pooling), and more commonly F=2,S=2. Pooling sizes with larger receptive fields are too destructive.
- Rules of thumb CNN architecture:
	- The input layer (that contains the image) should be divisible by 2 many times.
	- The conv layers should be using small filters (e.g. 3x3 or at most 5x5), using a stride of S=1, and crucially, padding the input volume with zeros in such way that the conv layer does not alter the spatial dimensions of the input.
	- The pool layers are in charge of downsampling the spatial dimensions of the input. The most common setting is to use max-pooling with 2x2 receptive fields (i.e. F=2), and with a stride of 2 (i.e. S=2). Note that this discards exactly 75% of the activations in an input volume
- Keep in mind that ConvNet features are more generic in early layers and more original-dataset-specific in later layers.
- Transfer learning rules of thumb:
	- New dataset is small and similar to original dataset. _Use CNN codes (CNN as a feature descriptor)_
	- New dataset is large and similar to the original dataset. _We can fine-tune through the full network._
	- New dataset is small but very different from the original dataset. _It might work better to train the SVM classifier from activations somewhere earlier in the network._
	- New dataset is large and very different from the original dataset. _We can fine-tune through the full network with initialized weights from a pretrained network._
- It’s common to use a smaller learning rate for ConvNet weights that are being fine-tuned, in comparison to the (randomly-initialized) weights for the new linear classifier that computes the class scores of your new dataset.
---
`August 13, 2018`

#### Deep learning question bank

- The Bayes error deﬁnes the minimum error rate that you can hope toachieve, even if you have inﬁnite training data and can recover the true probability distribution.
- Stochastic Gradient Descent approximates the gradient from a small number of samples.
- RMSprop is an adaptive learning rate method that reduces the learning rate using
exponentially decaying average of squared gradients. Uses second moment. It smoothens the variance of the noisy gradients.
- Momentum smoothens the average. to gain faster convergence and reduced oscillation.
- Exponential weighted moving average used by RMSProp, Momentum and Adam
- Data augmentation consists in expanding the training set, for example adding noise to the
training data, and it is applied before training the network.
- Early stopping, Dropout and L2 & L1 regularization are all regularization tricks applied during training.
- To deal with exploding or vanishing gradients, when we compute long-term dependencies in a RNN, use LSTM.
- In LSTMs, The memory cells contain an element to forget previous states and one to create ‘new’
memory states.
- In LSTMs Input layer and forget layer update the value of the state variable.
- **Autoencoders** need an encoder layer that is of different size
than the input and output layers so it doesn't learn a one on one representation
	- _Denoising Autoencoder_: The size of the input is smaller than the size of the hidden layer
(overcomplete).(use regularization!)
	- Split-brain auto encoders are composed of concatenated cross-channel encoders. are able to transfer well to other, unseen tasks.

- **GANs**
- **Unsupervised Learning**: 
	- It can learn compression to store large datasets  
	- Density estimation
	- Capable of generating new data samples
- "Inverse Compositional Spatial Transformer Networks." ICSTN stores the geometric warp (p) and outputs the original image, while STN only returns the warped image (Pixel information outside the cropped region is discarded).
- Fully convolutional indicates that the neural network is composed of convolutional layers without any fully-connected layers or MLP usually found at the end of the network. The main difference with CNN is that the fully convolutional net is learning filters every where. Even the decision-making layers at the end of the network are filters.
- **Residual Nets** - Two recent papers have shown (1) Residual Nets being equivalent to RNN and (2) Residuals Nets acting more like ensembles across several layers. The performance of an network depends on the number of short paths in the unravelled view. 1.The path lengths in residual networks follow a _binomial distribution_.
- **Capsule** - A group of neurons whose activity vector represents the instantiation parameters of a specific type of entity.
- _Receptive field networks_ treat images as functions in Scale-Space
- _Pointnets_ PointNets is trained to do 3D shape classification, shape part segmentation and scene
semantic parsing tasks. PointNets are able to capture local structures from nearby point and the combinatorial interactions among local structures. PointNets directly consumes unordered point sets as inputs.
- Hard attention focus on multiple sharp points in the image, while soft attention focusses on
smooth areas in the image
- YOLO limitations: Inappropriate treatment of error, Generalizing errors, Prediction of objects
- Curriculum Learning - Easier samples come at the beginning of the training.

---
`July 23, 2018`

Nice [Numpy](http://cs231n.github.io/python-numpy-tutorial/) tricks.

- A * B in numpy is element wise and so is +, -, /, np.sqrt(A)   (It will _broadcast_ if shape not same)
- np.dot(A,B) = A.dot(B)   --> Matrix Multiplication
- Use Broadcasting wherever possible (faster)
	- If the arrays do not have the same rank, prepend the shape of the lower rank array with 1s until both shapes have the same length.
	- In any dimension where one array had size 1 and the other array had size greater than 1, the first array behaves as if it were copied along that dimension

#### CS231n 

- One Hidden Layer NN is a universal approximator
- Always good to have more layers. (Prevent overfitting by using Regularization etc.)
- **Initialization**: 
	- For Activation tanh = `np.random.randn(N)/sqrt(N)`
	- For RELU = `np.random.randn(n)*sqrt(2/n)`
	- Batch Norm makes model robust to bad initialization
- **Regularization**: (ADDED TO LOSS at the end)
	- L2 - Prefers diffused weight vectors over peaky weights = 1/2 (Lambda.W^2)
	- L1 - Makes weight vectors sparse (invariant to noisy inputs) (Only a subset of neurons actually matters)
	- Dropout(p=0.5) with L2(lambda cross validated) => In Practice
- **Loss** = Average of losses over individual runs(training points)
	- L = 1/N SUM(Li)
	- Hinge Loss - (SVM) => Li = SUM_j!=y(max(0, fj - fy + 1)). (Squared hinge loss also possible)
	- Cross Entropy - (Softmax) => Li = -log(e^fy / SUM(e^fj))
	- Large number of classes (Imagenet etc.) use Hierarchial Softmax.





