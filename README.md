# Journal :postbox:

I plan to fill this section with what I discovered today - - - AFAP _(As Frequently as possible)_! 
These notes are best viewed with MathJax [extension](https://chrome.google.com/webstore/detail/github-with-mathjax/ioemnmodlmafdkllaclgeombjnmnbima) in chrome.

> "One day I will find the right words, and they will be simple." - Jack Kerouac

> "Youth is wasted on the young. Take control! NOW!" - George Bernard Shaw

---
`Nov 28, 2018`
#### PixelCNN and WaveNet
`msc`
- Reading the paper Conditional Image generation using PixelCNN Decoders. [Link](https://arxiv.org/pdf/1606.05328.pdf)
- PixelCNN employs the _Conditional Gated PixelCNN_ on portraits. It uses Flicker Images and crops them using a face detector!
	- Face Embeddings are generated using a triplet loss function which ensures that embeddings for one person are further away from embeddings for other people. (FaceNet [link](https://arxiv.org/abs/1503.03832))
- Use Linear Interpolation (between a pair of embeddings) to generate a smooth output from one to the next generation. Looks beautiful! DeepMind's generations.

---
`Nov 28, 2018`
#### LSTMs and WaveNet
`msc`
- Idea - find the sequence of images in the video that show the maximum similarity when super-imposed on the upcoming frames.
- Activation functions and derivatives:
	- `Sigmoid` (0 to 1): $f(x) = \frac {1}{1+e^{-x}}$ and $f'(x)=\frac{f(x)}{1-f(x)}$ (Useful for probabilities, though softmax is a better choice)
	- `TanH` (-1 to 1): $f(x)=\frac{e^x - e^{-x}}{e^x + e^{-x}}$ and $f'(x)=1 - f(x)^2$
	- `RELU` (0 to x): $f(x) = max(0, x)$ and $f'(x) = 1 if x>0 else 0$
![LSTM Un-rolled](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-chain.png)
- **LSTM**: [link](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)<br>
Notations: $h_t$ = output at each time step, $C_t$ = Cell State, $x_t$ = Input at a time step.
	- Forget Gate: I consumes the previous time step, merges it with current input, passes it through a sigmoid gate output(0 to 1) and pointwise multiplies the result with the cell state. (It specifies how much needs to be passed to the next state).<br>
	$f_t = \sigma (W_f[h_{t-1}, x_t] + b_f)$ (This square brackets means the h and x are concatenated and more weights for W_f to learn)
	- Cell state: It decides what needs to be saved in the current state. It takes the output from pointwise multiplication of forget gate and old cell-state and then adds (the tanh of current input and sigmoid of current input(for making it range from 0 to 1)) <br>
	$i_t = \sigma (W_i[h_{t-1}, x_t] + b_i)$ <br>
	$C_t^{-} =tanh(W_c[h_{t-1}, x_t] + b_c)$ <br>
	$C_t = C_t^{-} * i_t + f_t * C_{t-1}$ (Here * is pointwise multiplication) (This is the state stored for this cell that is forwarded to the next time-step)
	- Output: We still haven't figured out what to output ($h_t$). Now take this computed cell state, tanh it to make it span from -1 to 1 and pointwise multiply it with the sigmoid of current input.<br>
	$h_t = tanh(C_t) * \sigma (W_o[h_{t-1}, x_t] + b_o)$ This output is then sent upward (out) and also to the next time step.
	- Easy peasy :lemon: squeezy!
- **LSTM Variants**:
	- There is a variant which uses coupled forget and input gates: merges sigmoids for $f_t$ and $i_t$ and uses $1 - f_t$ instead of $i_t$.
	- There is a variant which uses peep-hole connections everywhere (Adds State value C_t everywhere in all W[]s)
	- **GRU**:
		- No Cell state variable to be forwarded through timesteps.
		- Only output is generated which is propagated up and to the next timestep.
		- Lesser parameters to train.
			- Uses the coupled forget and input gates idea.
			- $f_t = \sigma (W_f[h_{t-1}, x_t] + b_f)$ <br>
			  $c_t = \sigma (W_c[h_{t-1}, x_t] + b_c)$ <br>
			  $h_t^{-} = tanh (W_o[c_t * h_{t-1}, x_t] + b_o)$ <br>
			  $h_t = (1 - f_t)*h_{t-1} + f_t * h_t^{-}$ -> This is the output that is forwarded upwards and into the next time step.
		- Probably experiment with this as well, as it has lesser parameters to train for the small dataset we have.

---
`Nov 27, 2018`
#### Data Structures and Python
`algorithm`
- Making your implementation of (say a LinkedList) iterable in Python:
	- Declare a member variable which contains the current (`__current`) element (for iterations)
	- Declare function `__iter__` which inits the first element in `__current` and returns self!
	- Declare function `__next__` which calls `StopIteration()` or updates the `__current` and returns the actual item(node)!
	- :bomb: You can now do a `for node in LinkedList:` (elegant af)
- **Self-balancing trees** : (AVL Trees, Red-Black Trees, B-trees)
	- _B-Trees_ [link](https://medium.com/basecs/busying-oneself-with-b-trees-78bbf10522e7): Generalization of a _2-3 tree_ (Inorder traversal = sorted list. Node can have 1 (and 2 children) or 2 keys (and 3 children))
		- All leaves must be at the same level
		- Number of children = x is $B<= x < 2B$ (Note the < sign in upper bound)
		- If B = 2, It is a 2-3 tree (2 Keys, 3 children)
		- Insertion is okay, when overflows (more elements in a node than the allowed number of keys), move the middle element up. If keeps overflowing, keep going up till you reach the root.
		- Deletion - Trickier! Delete the node, Rotation!
		- Why? - Large datasets - On-disk data structure (not in-memory). It makes fewer larger accesses to the disk! - they are basically like maps with sorted keys (that can enable range operations)!
		- Databases nowadays usually implement B+ trees (store data in leaves and don't waste space) and not B-trees.


---
`Nov 26, 2018`
#### New Autoencoders
- Autoencoders: Lower dimensional latent space [link](https://www.youtube.com/watch?v=9zKuYvjFFS8)
	- Variational Autoencoders: Instead of bottleneck neurons, a distribution is learned (mean and variance) (Backpropagation Reparamatrized trick is used where the stochastic part is kept separate from mean and std. so gradients can flow back)
	- New Type: **Disentangled Variational Autoencoder** - Changing the latent variables leads to interpretable things in the input space (few causal features from a high dimensional space - latent variables (and understandable)).

---
`Nov 25, 2018`
#### Reading papers
`msc`
- Going through the [PixelRNN](https://arxiv.org/abs/1601.06759) paper as it is kind of a prerequisite for WaveNet.
- **Latent Variables**: [link](https://learnche.org/pid/latent-variable-modelling/what-is-a-latent-variable)
    - Latent variable is not directly observable by the machine (potentially observable - hypothesis - using features and examples)
    - In most interesting datasets, there are no/missing labels!
    - Principal Component Analysis / Maximum Likelihood estimation / Variational AutoEncoders. We use it when some data is missing! Who else uses this ? - Auto Encoders.
    - Latent variables capture, in some way, an underlying phenomenon in the system being investigated
    - After calculating the latent variables in a system, we can use these fewer number of variables, instead of the K columns of raw data. This is because the actual measurements are correlated with the latent variable
- Two Dimensional RNNs [link](https://arxiv.org/pdf/1506.03478.pdf) used for generating patterns in images. 
- Autoregressive - A value is a function of itself (in a previous timestep). AR(1) means the process includes instance of t-1.
- **Uber's pyro** - Probabilistic programming (Bayesian statistics) with PyTorch. Build Bayessian Deep learning models.
	- Traditional ML models like XGBoost and RandomForests don't work well with small data. [source](https://www.youtube.com/watch?v=7QlKZKbQa6M)
	- Used for Semi-supervised learning.
	- Variational inference models for time-series forecasting ? SVI ? (IDEA ? `msc` ?)
	

---
`Nov 14, 2018`
#### Pattern Recognition
`algorithm`
- Decision Trees: `Bagging` might not necessarily give better results but the results will have lower variance and will be more reliable / reproducible.
- Checking if a tree is a **BST**: Keep track of the min and max value in the subtree! That's all!


---
`Nov 12, 2018`
#### Causal Convolutions, WaveNet
`msc`
- Crazy reddit [post](https://www.reddit.com/r/MachineLearning/comments/7lvqay/d_future_of_lstm_and_gru_given_rise_of_causal/)
- Papers to read series: PixelRNN/PixelCNN/WaveNet/ByteNet
- ConvNets haven't been able to beat RNNs in question answering (Can't keep running hidden state of the past like RNNs)
- `IDEA: Try to generate a sine wave based on the number of repetitions in the duration specified! Minimize Loss somehow! How to deal with repetitions of varying lengths though ?`
- `IDEA: Think of a minimum repetition wave, try to minimize loss by varying the wavelength. The entire signal could be a combination of such wave-lets(scaled)`

---
`Nov 9, 2018`
#### Riemann Hypothesis
`Maths` `Numbers` `Primes`
- The $\zeta(s) = \frac{1}{1^s} + \frac{1}{2^s} + \frac{1}{3^s} + \frac{1}{4^s} ...$
- This is undefined for real numbers <=1 and is convergent for any values greater.
- Great [video](https://www.youtube.com/watch?v=d6c6uIyieoo). 
- Where is this function zero apart from the trivial(-2, -4, -6 etc.) ones. (On the strip between zero and 1 somewhere)
- Rieman's hypothesis = they lie on the line where the real-component = 1/2. This tells us something about the distribution of primes.
- Take away: How many primes are less than x ? $\frac{x}{ln(x)}$ :bomb: and the prime density is $\frac{1}{ln(x)}$

#### Thesis idea
`msc`
- How can I use the repeating properties of a sine wave for repetition counting ? Generate some features ?


---
`October 26, 2018`
#### Konigsberg Bridge problem 
`Algorithm` `Puzzle`
- The Graph needs to have all nodes with even degree and only zero or 2 nodes with odd degree for the _Eulerian Walk_ to be possible. (Same as being able to draw a figure without lifting the pencil or drawing on the same line again)


---
`October 25, 2018`
#### Cool tools
- _Tinkercad_ by **Autodesk** - Awesome for prototyping 3D models for 3Dprinting
- _Ostagram_ Style transfer on images.

#### Neural Style
[Pytorch tutorial](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html)
- Distances (Minimize both of these during optimization): 
	- $D_s$ - Style Distance
	- $D_c$ - Content Distance
- It is amazing how easy it was to run this :O (Loving PyTorch :fire:)

#### Torchvision useful functions
`msc`
- `torchvision.transforms` contains a bunch of image transformation options
- Chain them up using `transforms.Compose` like this: 
```
loader = transforms.Compose([
    transforms.Resize(imsize),  # scale imported image
    transforms.CenterCrop(imsize), # Center crop the image
    transforms.ToTensor()])  # transform it into a torch tensor
unloader = transforms.ToPILImage()
```
and use it like: 
```
def image_loader(image_name):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)
```
- Import a pre-trained model like: ([link](https://pytorch.org/docs/stable/torchvision/models.html))
```
cnn = torchvision.models.vgg19(pretrained=True).features.to(device).eval()
# In Pytorch, vgg is implemented with sequential modules containing _features_ and _classifier_(fully connected layers) hence the use of ".features"
```
These models expect some normalizations in the input
- Finished a wrapper around the neural style transfer tutorial code. :heart:

---
`October 24, 2018`
#### Video editing - Repetition counting
`msc`
- Preprocessing:
	- Using [Bresenhem's Line algorithm](https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm) to sample representative elements from a list.
		- Sample m items from n: `lambda m,n: [i*n//m + n//(2*m) for i in range(m)]`
		- Finally sampling 10fps for all videos using this technique
	- Making images Black&White
	- Resizing images to 64x64 for now (Toy Dataset)


---
`October 23, 2018`
#### HackJunction Budapest
- DCM files(from both MRI and CT scans) contain pixel_array for each slice.
- A different file contains information about the contours of tumour - corresponding to the slices.
- Learnings: 
	- **Fail Fast**,  **Move on**, **Don't try to protect your code**
	- Spend time on MLH challenges, win cool stuff like the Google home mini :wink:
	- Check out Algolia (Hosted search API) for quick prototyping

#### Algorithms Q1 New Year's chaos:
[Link](https://www.hackerrank.com/challenges/new-year-chaos/) to the problem.
- Solution: 
	- No one can move more than two positions backwards (2 bribes each) (Break if someone does)
	- Start from the back: See how many bribes the small number took to reach there. If the number at the end is 4, see from 4-2= _2 to end-1_ if there are numbers bigger than 4 and keep a count :happy:

---
`October 14, 2018`
#### WaveNet - Speech synthesis
- Two types of TTS(text to speech):
	- Concatenative TTS: Basically needs a human sound database (from a single human) (Ewww)
	- Parametric TTS: Model stores the parameters
- PixelRNN, PixelCNN - Show that it is possible to generate synthetic images one pixel at a time or one color channel at a time. (Involving thousands of predictions per image)
![WaveNet Structure](https://storage.googleapis.com/deepmind-live-cms/documents/BlogPost-Fig2-Anim-160908-r01.gif)
- Dilated convolutions support exponential expansion of the receptive field instead of linear
- Saves memory, but also preserves resolution.
- Parametrising convolution kernels as Kronecker-products is a cool idea. (It is a nice approximation technique - very natural) 
	- Reduces number of parameters by over 3x with accuracy loss of not over 1%.
- Convolutions arithmetic. [Link](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md)

---
`October 12, 2018`
#### Repetition counting
`msc`
- Things to try:
	- Discretizing the output like in [this](https://www.cs.tau.ac.il/~wolf/papers/repcounticcv.pdf) paper.
	- Look at [NEWMA](http://openaccess.thecvf.com/content_cvpr_2018/papers/Runia_Real-World_Repetition_Estimation_CVPR_2018_paper.pdf) as Jan suggested. It is fast and uses little memory instead of LSTM.
	- Can also look at [Wavenet](https://arxiv.org/pdf/1609.03499.pdf) and try out dilated convolutions ?

---
`October 11, 2018`
`msc`
#### PyTorch is back
- When you have bias in the number of classes in your training data:
	- Oversample smaller classes
	- Use `WeightedRandomSampler`
- [Transfer Learning](http://cs231n.github.io/transfer-learning/) is the way to go most of the times. Don't always freeze the pretrained convnet when you have a lot of training data
- Always try to overfit a small dataset before scaling up. :boom:


---
`October 10, 2018`
`msc`
#### RepCount 
- Discuss about IndRNN (Long sequence problems for Recurrent neural network)
- Plot activations in layers(one by one) over timesteps. Activation vs Timestep.
- NEWMA - online change point detection

#### Classifiers
- Logistic classifier vs LDA: LDA assumes $p(x/w)$ (class densities) is assumed to be Gaussian. It involves the use of marginal density $p(x)$ for the calculation of unknown parameters but for Logistic, $p(x)$ is a part of the Constant term. LDA is the better approach if Gaussian Assumption is valid.
- L1-distance = $\sum_p (V_1^p - V_2^p)$, L2-distance = $\sqrt{(\sum_p (V_1^p - V_2^p)^2)}$
- Square-root is a monotonic function (Can be avoided when using L2)
- KNN is okay in low dimensional datasets. Usually not okay with images.
- Linear Classifier:
	- Better than KNN because 
		- parameters need to be checked instead of all existing images.
		- Template is learned and negative-dot-product is used as distance with the template instead of L1, L2 distances like in KNN
	- The class score function has the form $Wx_i + b$. You get scores for each class.
	- If you plot a row of W, it will be a template vector for a class. Loss is a different thing be it SVM(hinge loss) or softmax (cross-entropy). 
	- And once you have the loss, you can perform optimization over the loss.
![svm-softmax](http://cs231n.github.io/assets/svmvssoftmax.png)
#### Constraint optimization
- *Lagrange Multipliers* - awesome MIT [video](https://www.youtube.com/watch?v=HyqBcD_e_Uw).
	- BOTTOM LINE - Helps find points where Gradient(first partial derivatives) of a function are parallel to the gradients of the constraints and also the constraints are satisfied. [post](https://medium.com/@andrew.chamberlain/a-simple-explanation-of-why-lagrange-multipliers-works-253e2cdcbf74)

---
`October 5, 2018`

#### Knapsack problem - Thief breaks into house and wants to steal valuable items (weight constrained)
- Brute-Force - $2^n$ (For each item, decide take/no-take)
- Greedy approach - Sort based on a criteria (weight, value or value/weight) - complexity = nlogn for sorting
	- You could get stuck @ a local optimum
	- these approaches often provide adequate/often not optimal solutions.
- Build a tree - Dynamic Programming - (Finds the optimal solution)
	- Left means take element and right means no-take
	- **Dynamic programming**:
		- Optimal Substructure
		- Overlapping subproblems
	- At each node, given the remaining weight, just maximize the value by chosing among the remaining items.
- Variants: `subset sum`, `scuba div` [link](https://www.spoj.com/problems/SCUBADIV/) etc.

---
`October 3, 2018`
`msc`
#### RNN for 1D signal repetition-counting
- Even the `nn.LSTM` implementation gives bad results. I suspect this could be because the sequence length is too huge? Trying to generate a sequence with a smaller length.
- Maybe look at some other representation of the 1D signal ? (Like HOG ?)
- PvsNP - What can be computed in a given amount of space and time ? (Polynomial vs Non Deterministic Polynomial)
	- P = Polynomial, NP = Non Polynomial but the answer can be checked in polynomial time.
	- NP-complete = Hardest problem in NP
		- Can prove a problem is np-complete if it is in NP and is NP-hard
	- NP complete problems can be used to solve any problems in NP :crown:
	- If A can be converted to B in Poly, A >= B

	> "If P = NP, then there would be no special value in creative leaps, no fundamental gap between solving a problem and recognising a solution once its found. Everyone who could appreciate a symphony would be Mozart and everyone who could follow a step by step argument would be Gauss!" - Scott Aronson



---
`October 2, 2018`

#### Interview PREP
- Python style [guide](https://github.com/google/styleguide/blob/gh-pages/pyguide.md)
- MIT Algorithm assignments [page](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-006-introduction-to-algorithms-spring-2008/).
- Example coding [interview](https://www.youtube.com/watch?v=XKu_SEDAykw)
- Interview prep blog [post](http://steve-yegge.blogspot.com/2008/03/get-that-job-at-google.html)
- Topics importante:
	- Complexity
	- Sorting (nlogn)
	- Hashtables, Hashsets
	- Trees - binary trees, n-ary trees, and trie-trees
		- Red/black tree
		- Splay tree
		- AVL tree
	- Graphs - objects and pointers, matrix, and adjacency list
		- breadth-first search
		- depth-first search.
		- computational complexity
		- tradeoffs
		- implement them in real code.
			- Dijkstras
			- A*
		- Make absolutely sure you can't think of a way to solve it using graphs before moving on to other solution types.
	- NP-completeness
		- traveling salesman 
		- knapsack problem
		- Greedy approaches
	- Math
		- Combinatorics
		- Probability [link](https://ocw.mit.edu/courses/mathematics/18-05-introduction-to-probability-and-statistics-spring-2014/assignments/)
	- Operating System
		- Threads, Processes
		- Locks, mutexes, semaphores and monitors
		- deadlock, livelock
		- context-switching
		- Scheduling
- Algo books
	- Algorithm Design Manual [link](http://mimoza.marmara.edu.tr/~msakalli/cse706_12/SkienaTheAlgorithmDesignManual.pdf)
	- Introduction to algorithms [link](https://labs.xjtudlc.com/labs/wldmt/reading%20list/books/Algorithms%20and%20optimization/Introduction%20to%20Algorithms.pdf)


---
`September 26, 2018`
`msc`
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
- From what I've found out, **batching** in pytorch gives a speedup when running on GPU. Not very critical while prototyping on toy-datasets.

---
`September 25, 2018`
`msc`
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
`msc`
- [This](https://www.youtube.com/watch?v=yCC09vCHzF8) video by Andrej Karpathy.
- Idea about the thesis project:
	- Karpathy uses a feature vector (one-hot vecotor for all characters possible)-->(feature vector from a CNN) for each timestep.
	- In the output layer, would it be better to have a one-hot vector representing the count instead of a single cell which will calculate the count ?
	- Should I pad the input sequence with 0s based on the video with the maximum number of points in the matrix profile ? (For a fixed `seq_length`) ?
	- To make the learning process for blade-counting online, need an RNN with 3 outputs, clockwise-anticlockwise-repetition
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
		- $\Sigma = E[(x-\mu)(x-\mu)^T] = E\big{[}\begin{bmatrix} x_1 -\mu_1\\ x_2 - \mu_2 \end{bmatrix}\begin{bmatrix} x_1 - \mu_1 & x_2 - \mu_2 \end{bmatrix}^T\big{]} = \begin{bmatrix}\sigma_1^2 & \sigma_{12}\\ \sigma_{21} & \sigma_2^2 \end{bmatrix}$
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
`msc`
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
`msc`
Amazing [video](https://www.youtube.com/watch?v=u6aEYuemt0M) by Karpathy. (Timing: 1:21:47)
- Convolutional net on the frame and the low-level representation is an input to the RNN
- Make neurons in the ConvNet recurrent. Usually neurons in convnets represent a function of a local neighbourhood, but now this could also incorporate the dotproduct of it's own or neighbours previous activations making it a function of previous activations of the neighbourhood for each neuron. Each neuron in the convnet - _Recurrent_!

** An idea for repetition estimation: Maybe look for activation spikes in the deepvis toolbox by Jason Yosinski and train a simple classifier on them. **


#### TenNet :hand:
- Used the `LeNet` architecture.
- Got `95%` test and `99%` train accuracy. Is it still an overfit ?


#### Uber AI labs (Jason Yosinski)
- **Coordconv layers** - for sharper object generation (GANs), Convolutional Neural Networks too and definitely Reinforcement learning. Paper [here](https://arxiv.org/abs/1807.03247)
- **Intrinsic Dimension** - Lower Dimension representation of neural networks (Reduces Dimensions). Paper [here](https://arxiv.org/abs/1804.08838)

---
`August 24, 2018`
`msc`
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





