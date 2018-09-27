# Journal :postbox:

I plan to fill this section with what I discovered today - - - AFAP _(As Frequently as possible)_! 
These notes are best viewed with MathJax [extension](https://chrome.google.com/webstore/detail/github-with-mathjax/ioemnmodlmafdkllaclgeombjnmnbima) in chrome.

---
`September 26, 2018`

#### torch.nn
- If you share the weights across time, then your input time sequences can be a variable length. Because each time before backpropagating loss, you go over atleast a sequence.
	- Shared weights means fewer parameters to train.
	- IDEA! - For longer sequences, maybe share less weights across time.
- nn.LSTM: Suppose we have 2 layers. 
	- Input to L1 = input, (h1, c1)
	- Output from L1 = (h1_, c1_)
	- Input to L2 = h1_, (h2, c2)
	- Output from L2 = (h2_, c2_) ==> final output = h2_
- `tensor.size()` = `np.shape` || `tensor.view(_)` = `np.reshape`
- 

---
`September 25, 2018`

#### Recurrent Nets
- Recursive network that is going to be trained with very long sequences, you could run into memory problems when training because of that excessive length. Look at **truncated-BPTT**. Pytorch discussion [link](https://discuss.pytorch.org/t/implementing-truncated-backpropagation-through-time/15500).
- Ways of dealing with looooong sequences in LSTM: [link](https://machinelearningmastery.com/handle-long-sequences-long-short-term-memory-recurrent-neural-networks/)
	- TBPTT (in the point above)
	- Truncate sequence
	- Summarize sequence
	- Random sampling
	- Use encoder decoder architecture


---
`September 24, 2018`

#### Counting peaks/valleys in a 1D signal
- Tried to generate a sine wave and make a Neural Net predict the number of valleys in the wave (Could be a useful step while calculating the final count from the 1D signal in the matrix profile)
- I assumed a signal of a fixed length (100). I trained a simple MLP on it assuming 100 features in the input. (Overfits and fails to generalize -- as expected)
- I want to train an LSTM/GRU on it now. Since it learns to generate a sine wave (as some of the online examples show). I am hoping it will be able to learn counting.

#### Oh Py God
- `os.cpu_count()` gives the number of cores.
- threading.Thread vs multiprocessing.Process nice [video](https://www.youtube.com/watch?v=ecKWiaHCEKs)
- Use Threading for IO tasks and Process for CPU intensive tasks.
- Threading makes use of one core and switches context between the threads on the same core.
- Processing makes use of all the cores (Like a badass) 

---
`September 23, 2018`

#### Recurrent Neural Nets
- [This](https://www.youtube.com/watch?v=yCC09vCHzF8) video by Andrej Karpathy.
- Idea about the thesis project:
	- Karpathy uses a feature vector (one-hot vecotor for all characters possible)-->(feature vector from a CNN) for each timestep.
	- In the output layer, would it be better to have a one-hot vector representing the count instead of a single cell which will calculate the count ?
	- Should I pad the input sequence with 0s based on the video with the maximum number of points in the matrix profile ? (For a fixed `seq_length`) ?
	- To make the learning process for blade-counting online, need an RNN with 3 outputs, clockwise-anticlockwise-repitition
- `seq_length` in the RNN is the region where it can memorize.(size of input sequence (batch size of broken input))
- Gradient clipping for exploding gradients (because while backpropagating, same matrix($W_h$) is multiplied to the gradient several times ((largest eigenvalue is > 1)))
- LSTM for vanishing gradients (same reason as above (largest eigenvalue is < 1))
- LSTMs are super highways for gradient flow
- GRU has similar performance as LSTM but has a single hidden state vector to remember instead of LSTM's (hidden-state-vector and c vector)
- During training, feed it not the true input but its generated output with a certain probability p. Start out training with p=0 and gradually increase it so that it learns to general longer and longer sequences, independently. This is called schedualed sampling. [paper](https://arxiv.org/abs/1506.03099)



---
`September 19, 2018`

#### Everything Gaussian 
- When someone says random variable, it is a single dimension!
- _Central limit theorem_: If any random variable is sampled infinitely, it ends up being normally distributed
- Expectation values:
	- **Discrete**
		- $E[x] = \mu = \frac {1}{n}\sum_{i=0}^n x_i$
		- $E[(x-\mu)^2] = \sigma^2 = \frac{1}{n}\sum_{i=0}^{n}(x-\mu)^2$
	- **Continuous**:
		- $E[x] = \mu = \int_{-\infty}^{+\infty} x.p(x)dx$
		- $E[(x-\mu)^2] = \sigma^2 = \int_{-\infty}^{+\infty} (x-\mu)^2.p(x)dx$
- Multivariate Gaussian:
	- $p(x) = \frac{1}{\sqrt{2\pi\Sigma}}\exp\big{(}-\frac{1}{2}(x-\mu)^T\Sigma(x-\mu))\big{)}$
	- Covariance Matrix: $\Sigma$
		- $\Sigma = E[(x-\mu)(x-\mu)^T] = E\big{[}\begin{bmatrix} x_1 -\mu_1\\ x_2 - \mu_2 \end{bmatrix}\begin{bmatrix} x_1 - \mu_1 & x_2 - \mu_2 \end{bmatrix}\big{]} = \begin{bmatrix}\sigma_1^2 & \sigma_{12}\\ \sigma_{21} & \sigma_2^2 \end{bmatrix}$
		- Always symetric and positive-semidefinite (Hermitian) (All the eigen values are non-negative).
		- For a diagonal matrix, the elements on the diagonal are the eigen values.
		- You can imagine the distribution (for 2D features) as a hill by looking at the covariance matrix.
- Normally distributed classes:
	- Use formula $(x-\mu)^T\Sigma^{-1}(x-\mu) = C$ to get the equation of an ellipse(the iso curve that `seaborn.jointplot` plots).
	- The orientation and axes of this ellipse depend on the eigen vectors and eigen values respectively of the covariance matrix.
	- 


---
`September 17, 2018`

#### Siraj stuff
- Different parts of the image are masked separately
- Image segmentation
- Multi class classification inside a single image
- Data is value - Andrew Trask (OpenMind) [video](https://www.youtube.com/watch?v=qJ1rdVEcl5g)


#### Pytorch RNN - NameNet
This Recurrent net is a classifier and classifies names of people to its origins
- First pytorch RNN implementation using [this](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html) tutorial
- Only used linear (fully connected layers in this)
```
self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
self.i2o = nn.Linear(input_size + hidden_size, output_size)
self.softmax = nn.LogSoftmax(dim=1)

def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden
```
- The hidden component obtained after each output-calculation is fed back to the next input
- For this sequence of words, at each epoch, hidden layer is re-initialized to zeros (`hidden = model.initHidden()`) and model's gradients are reset (`model.zero_grad()`)
- Training examples are also randomly fed to it
- Negative Log Likelihood loss (`nn.NLLLoss`) is employed as it goes nicely with a LogSoftmax output (last layer).
- Torch `.unsqueeze(0)` adds a dimension with 1 in 0th location. (tensor(a,b) -> tensor(1,a,b))

#### Django ftw
- Use `read_only_fields` in the `Meta` class inside a serializer to make it non-updatable
- Views should contain all access-control logic



---
`September 12, 2018`

#### MathJax extension in Chrome for Github Markdown
- [Link](https://www.mathjax.org/) to MathJax's landing page.
- Download the chrome extension from [here](https://chrome.google.com/webstore/detail/github-with-mathjax/ioemnmodlmafdkllaclgeombjnmnbima).
- Bayes $P(w/x) = \frac{P(x/w). P(w)}{P(x)}$
- Normal (Univariate) $N(\mu, \sigma) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp{-\frac{(x - \mu)^2}{2\sigma^2}}$
- **Decision theory:**
	- An element(x) belongs to class1 if $P(w_1 | x) > P(w_2 | x)$ _(posterior probability comparison)_
	- i.e. $P( x | w_1) P(w_1) > P( x | w_2) P(w_2)$
	- where $P(x | w_1) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp{-\frac{(x - \mu)^2}{2\sigma^2}}$
	- Boom!
- Use `np.random.normal(self.mean, self.cov, self.n)` for univariate and `numpy.random.multivariate_normal` for multivariate data generation 


#### Seaborn for visualization in matplotlib
- Multivariate contours: `sns.jointplot(x="x", y="y",kind="kde", data=df);`
- Visualizing distributions 1D: `sns.distplot(np.random.normal(0, 0.5, 100), color="blue", hist=False, rug=True);`


---
`September 9, 2018`

#### Recurrent Neural Networks :dizzy:
Karpathy's [talk](https://skillsmatter.com/skillscasts/6611-visualizing-and-understanding-recurrent-networks) and ofcourse his unreasonable blog [post](http://karpathy.github.io/2015/05/21/rnn-effectiveness/).
- RNNs are freaking flexible:
	- one to many -> Image Captioning (image to sequence of words)
	- many to one -> Sentiment Classification (seq of words to positive or negative sentiment)
	- many to many -> Machine Translation (seq of words to seq of words)
	- many to many -> Video classification (using frame level CNNs)
- These carry **state** unlike regular NNs. Might not produce the same output for the same input
- Rules of thumb:
	- Use RMSProp, Adam (sometimes SGD)
	- Initialize forget gates with high bias (WHAT ?)
	- Avoid L2 Regularization
	- Use dropout along depth
	- Use clip gradients (to avoid exploding gradients) (LSTMs take care of vanishing gradients)
	- You can look for interpretable cells (Like one of the cell fires when there is a quote sequence going on)
- When using RNN with CNN, plug extra information(CNN's output) directly to a RNN's (green - recurrent layer)
- Use Mixture Density Networks especially when generating something.


---
`September 8, 2018`

#### Pytorch :snake:
- Earned basic badge in pytorch [forum](https://discuss.pytorch.org/).
- Finished plotting utility of train, test accuracy vs epochs and train vs test accuracy
- Finished plotting utility of loss vs epochs
- Finished plotting of Learning rate vs epochs
- To get reproducible results with torch:
	```
	torch.backends.cudnn.deterministic = True
	torch.manual_seed(1973)
	```
- Call `model.train()` before training and `model.eval()` after training to set the mode. (It matters when you have layers like Batchnorm or Dropout)
- Use `torch.nn.functional.<methods>` for non linear activations etc. where the model parameters like (training=True/False) doesn't matter. If using it with dropout make sure to pass arguments _training=False_ or use the corresponding torch.nn.<Module> (Layers).


---
`September 2, 2018`

#### Django :mango:
- Finished Routing
- **Permissions / Groups**
	- TODO 


---
`September 1, 2018`

#### Django unchained
- Django Rest Framework [tutorials](http://www.django-rest-framework.org/tutorial/1-serialization/) are amazing.
- _TokenAuthentication_ is needed for multipurpose API (But this doesn't support browsable APIs). Solution: Use SessionAuthentication as well (But this needs CSRF cookie set) Solution: Extend SessionAuthentication class and override *enforce_csrf* function and just return it. Boom! Browsable API with TokenAuthentication.
- Views can be Class based or function based. ViewSets are even a further abstraction over class based views (Class based views can make use of _generics_ and _mixins_).
- URLs are defined separately in a file usually called `urls.py`. (Use _Routers_ if you're using ViewSets).
- Model is defined by extending _django.db.models.Model_. All the field types are defined here: `code = models.TextField()`
- Serializer has to be defined for the model by extending *rest_framework.serializers.HyperlinkedModelSerializer / ModelSerializer* . Primary key relations etc are defined here. Serializer and Views are where the API is brewed.
- _project/project/settings.py_ contains all the project settings and resides in `django.conf.settings`.

Useful commands:
	- django-admin.py startproject projectname
	- django-admin.py startapp appname
	- python manage.py makemigrations
	- python manage.py migrate
	- python manage.py runserver

---
`August 28, 2018`

Amazing [video](https://www.youtube.com/watch?v=u6aEYuemt0M) by Karpathy. (Timing: 1:21:47)
- Convolutional net on the frame and the low-level representation is an input to the RNN
- Make neurons in the ConvNet recurrent. Usually neurons in convnets represent a function of a local neighbourhood, but now this could also incorporate the dotproduct of it's own or neighbours previous activations making it a function of previous activations of the neighbourhood for each neuron. Each neuron in the convnet - _Recurrent_!

** An idea for repitition estimation: Maybe look for activation spikes in the deepvis toolbox by Jason Yosinski and train a simple classifier on them. **


#### TenNet :hand:
- Used the `LeNet` architecture.
- Got `95%` test and `99%` train accuracy. Is it still an overfit ?


#### Uber AI labs (Jason Yosinski)
- **Coordconv layers** - for sharper object generation (GANs), Convolutional Neural Networks too and definitely Reinforcement learning. Paper [here](https://arxiv.org/abs/1807.03247)
- **Intrinsic Dimension** - Lower Dimension representation of neural networks (Reduces Dimensions). Paper [here](https://arxiv.org/abs/1804.08838)

---
`August 24, 2018`

#### PyTorch :fire:
- Implemented a Dynamic Neural Network using Pytorch's amazing dynamic Computation graphs and `torch.nn`.
- Fully Connected layers using `torch.nn.Linear(input_dimension, output_dimension)`
- Autograd is epic.
- Implemented a reusable save_model_epochs function to save model training state
- Cool Findings:
	- A fairly simple Dynamic net crushes the Spiral Dataset!
	- Tried the Dynamic Net (Fully connected) on [Sign Language Digits Dataset](https://www.kaggle.com/ardamavi/sign-language-digits-dataset) and it seems to overfit (Train: 99%, Test: 85%) as expected.
	- Will try to build a new net(`TenNet`) to crush this set now.

#### More Python :snake:
- Tried Spotify's Luigi for workflow development
- Tried Threading module and decorators and ArgumentParser.
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
	- $L = 1/N \sum{L_i}$
	- Hinge Loss - (SVM) => $L_i = \sum_{j \ne y}{max(0, f_j - f_y + \delta)}$. (Squared hinge loss also possible) ($\delta = margin$)
	- Cross Entropy - (Softmax) => $L_i = -log(e^{f_y} / \sum{e^{f_j}})$
	- Large number of classes (Imagenet etc.) use Hierarchial Softmax.





