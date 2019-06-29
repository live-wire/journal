# Questions from `paper_2` :robot: 

**Q: What are the four key ideas behind convolutional nets that take advantage of the properties of natural signals?**
- Shared weights, pooling, use of many layers, use of many nodes.
- Non-linear activation functions, shared weights, pooling, use of many layers.
- Non-linear activation functions, shared weights, local connections, pooling.
- Local connections, shared weights, pooling, use of many layers.
* `[ D ]`


---

**Q: Convolutional neural networks are based on:**
- The use of many layers
- Shared weights
- Pooling
- All of the above
* `[ D ]`


---

**Q: What type of neural network generalizes better in regard to networks with full connectivity between adjacent layers?**
- Convolutional
- Recurrent
- Long term short memory 
- Shallow 
* `[ A ]`


---

**Q: What is the purpose of unsupervised pre-training in a deep neural net?**
- initialize weights to random values
- avoid employing gradient descent
- initialize weights to sensible values
- none of the above
* `[ C ]`


---

**Q: What is the key advantage of neural networks in contrast to other classifiers?**
- The features needed for training can be learned automatically
- Classes that are not linearly seperable can now be seperated by neural networks
- Neural networks perform better than other classifiers when working with a low amount of data
- There is a lot of magic in neural networks
* `[ A ]`


---

**Q: Which statement about hidden layers is incorrect?**
- Units that are not in the input or output layer are called hidden units
- The hidden layers can be seen as distorting the input in a non-linear way so that categories become linearly separable in the last layer
- The weights of hidden layers are not updated for each back propagation because it would take too much time to calculate the gradients of the hidden units due to their non-linearity
- Hidden layers consist of a set of units that compute a weighted sum of their inputs from the previous layer and pass the result through a non-linear function
* `[ C ]`


---

**Q: Which of the following statements is true about Recurrent Neural Networks (RNNs)?**
- RNNs have been shown to surpass Convolutional Neural Networks in face recognition capabilities.
- RNNs use different parameters at each stage.
- For translation of sentences, RNNs are particularly unsuited, due to the complex challenges involved in Natural Language Processing.
- It is difficult to train RNNs to store information for very long.
* `[ D ]`


---

**Q: Supervised learning tasks are characterized by**
- training data sets containing the expected output for each data point
- training data sets not containing the expected output for each data point
- coming from real-world problems in healthcare
- being usually well solved by shallow neural networks
* `[ A ]`


---

**Q: In the convolutional layer in a convolutional neural network, units are organized in feature maps, within which each unit is connected to a local patch in the feature maps of the previous layer through a filter bank. Which of the following is most true:**
- All units in a feature map share the same filter bank.
- All feature maps in a layer share the same filter bank.
- All units in all feature maps have their own filter banks.
- None of the above answers are true.
* `[ A ]`


---

**Q: Which statement is incorrect:**
- A LSTM network is an augmentation of an RNN.
- ConvNets use backpropagation for function approximation.
-  Deep neural networks should use supervised learning.
- A RNN is a type of deep neural network.
* `[ C ]`


---

**Q: What is the difference between CNNs and RNNs?**
- RNNs cannot be used on images
- CNNs do not have recurrent connection of RNNs
- CNNs output single classifications but RNNs output vectors
- CNNs cannot take previous inputs into account
* `[ B ]`


---

**Q: There are four key ideas behind convolutional networks that take advantage of the properties of natural signals, which list sums up these four ideas?**
- Local connections, Shared weights, Pooling, the use of many layers
- Local connections, Shared weights, Cross validating, Rectified linear unit
- Cross validating, Rectified linear unit, Pooling, the use of many layers
- Gradient descent, Local connections, Pooling, Rectified linear unit
* `[ A ]`


---

**Q: Which Deep Learning Representation has the strongest ties with the backpropagation algorithm?**
- Neural networks
- Decision trees
- Graphical models
- Sets of rules
* `[ A ]`


---

**Q: What is the key advantage of deep learning over machine learning**
- Features can be extracted automatically from the data, preventing expensive manual feature creation.
- Deep learning is much faster
- Machine learning can not process images or multimedia content
- Deep learning always learns a consistent representation
* `[ A ]`


---

**Q: Given a Neural Network with 2 hidden layers, and training such a network on image data in order to classify an image, what representation of the data can we expect the 1st hidden layer to find?**
- Detection of arrangements of edges into particular motifs
- The presense or absense of particular edges at particular orientations or locations
- Assemblies of motifs into larger combinations
- Both A and B both occur simultaneously in the same layer
* `[ B ]`


---

**Q: What is one the most important advantages of deep nets?  **
- The usage of few inputs (about 30) 
- They can be combined to emulate a much larger network and model a social network
- The ability to give results insensitive to non-relevant variation of the input (orientation, position, illumination)
- The possibility to blindly transform the input into higher dimensional space where the classes are not separable
* `[ C ]`


---

**Q: Some problems of recurrent neural networks are/were:**
- Originally, gradients computed through time tended to explode or to vanish.
- It is not possible to pair them with other architectures such as convolutional networks.
- It is difficult to store information for a long time.
- A and C are correct.
* `[ D ]`


---

**Q: What is NOT an effective use of a deep neural network?**
- Solving a classification task with linear separability between the classes
- Predicting the next word of a sentence
- Correctly classifying two different breeds of dogs in different images
- Generating captions for images
* `[ A ]`


---

**Q: Which function does a rectified linear unit (ReLU) use, where z is the input to a node and c a parameter?**
- f(z) = tanh(z)
- f(z) = max(z, 0)
- f(z) = z + c
- f(z) = 1/(z)
* `[ B ]`


---

**Q: Which of the following is true?**
- When encountering a combinatorially large amount of saddle points the values of the objective function in these saddle points varies widely.
- It is often better to use RRN's for tasks that involve sequential input
- LSTM's have proven to be less effective than RNN's
- In the first stage convolutional neural networks consist of three types of layers: convolutional layers, pooling layers and gradient layers.
* `[ B ]`


---

**Q: Why is the ReLU a very popular non-linear function in deep learning?**
- It has nice differentiability properties
- The ReLU learns typically faster in huge networks
- It takes extremely less time to compute than a tanh(x) or any other non-linearity
- This is the only non-linearity that works
* `[ B ]`


---

**Q: When is a node in a deep neural network considered a hidden node?**
- Nodes that are kept hidden by the manufacturer
- Nodes that cannot be seen by the neural network
- Nodes that are turned off for a specific task
- Nodes that are not in the input or output layer
* `[ D ]`


---

**Q: Which of these given advantages are an advantage of deep nets over  classic learning algorithms


First, learning distributed representations enable generalization to new combinations of the values of learned features beyond those seen during training (for example, 2n combinations are possible with nbinary features)68,69. Second, composing layers of representation in a deep net brings the potential for another exponential advantage70(exponential in the depth).

Page 5 https://www.cs.toronto.edu/~hinton/absps/NatureDeepReview.pdf**
- 1
- 2
- Both
- None
* `[ C ]`


---

**Q: Which one is NOT the key ideas behind ConvNets that take advantage of the properties of natural signals?**
- local connections
- the use of many nets
- pooling
- shared weights
* `[ B ]`


---

**Q: Recurrent Neural Networks (RNNs) are particularly useful for:**
- Speech and text recognition;
- Object identification in images;
-  Web searches;
- All of the above.
* `[ A ]`


---

**Q: Which one of the following sentences is true about the convolutional neural networks?**
- In a convolutional neural network, the pooling layers are necessary to increase the number of extracted features
- In a convolutional neural network, there is always only one convolutional layer and it is the first layer of the network
- The main function of the convolutional layer in a convolutional neural network is to merge semantically similar features into one
- The main function of the convolutional layer in a convolutional neural network is to detect local conjunction of features from the previous layer 
* `[ D ]`


---

**Q: Why are recurrent neural networks problematic to train?**
- The RNN has a bias towards the first items used in training
- The RNN has a bias towards the last items used in training
- The backpropagated gradients either grow or shrink over time, so the steps either explode or vanish
- The RNN needs an extremely large dataset before it converges
* `[ C ]`


---

**Q: What is the role of a pooling layer (in a convolutional neural network)?**
- This layer detects the local conjunction from the previous layer
- This layer merges similar features into one.
- This layer initializes and updates the weights.
- This layer’s goal is to prevent overfitting
* `[ B ]`


---

**Q: What is an activation function good for in a neural network?**
- An activation function sums up all the input to a neuron until an activation threshold is reached
- It introduces nonlinearities in a network. Without the activation function, the network would behave like a single-layer network
- An activation function introduces linearity in a network. 
- An activation function introduces nonlinearities in a network. This allows to handle larger amounts of data.
* `[ B ]`


---

**Q: For a Sequential Input data, is it advisable to use a "Convolutional" or  a "Recurrent" Neural Network? Why?**
- Use ConvNN, Because a ConvNN best processes data in the form of multiple arrays.
- Use RNN, Because a RNN best processes data in the form of multiple arrays.
- Use ConvNN, Because a ConvNN best represents the temporal dynamic behaviour of the data.  
- Use RNN, Because a RNN best represents the temporal dynamic behaviour of the data. 
* `[ D ]`


---

**Q: Which kind of datasets are best used for recurrent neural networks?**
- Datasets consisting mainly out of partitioned data
- Datasets consisting mainly out of sequential data
- Datasets consisting mainly out of temporary data
- Datasets consisting mainly out of temporary partitioned data
* `[ B ]`


---

**Q: A main advantage of a convolutional neural network over a network with full conectivity between adjecent layers is:**
- detection of a pattern (motif, feature) can be made invariant of the location in the provided data sample.
- faster training.
- better generalization.
- all of the above.
* `[ D ]`


---

**Q: Recurrent Neural Networks (RNNs) are usefull for:**
- Convolutional inputs
- Sequential inputs
- Raw input
- Classified input
* `[ B ]`


---

**Q: What is a Major difference between conventional machine learning techniques and Deep learning methods?**
- Conventional machine learning needs to be done with supervised learning, meanwhile deep learning does not use supervised learning at all.
- With conventional machine learning the layer of features are designed by human engineers, meanwhile for deep learning the layer of features are learned from data using a general purpose learning procedure.
- Conventional machine learning is easier to use when you are working with large amount of data then when you are using deep learning.
- None of the above answers are true
* `[ B ]`


---

**Q: What is the main difference between Recurrent Neural Networks and Convolutional Neural Networks?**
- CNNs are only used for speech and language recognition, whilst RNNs can be used for image recognition only.
- CNNs maintain a "state-vector" that contains information about the history of the past elements, whilst a RNN does not.
- RNNs maintain a "state-vector" that  contains information about the history of the past elements, whilst a CNN does not.
- RNNs are only used for speech and language recognition, whilst CNNs can be used for image recognition only.
* `[ C ]`


---

**Q: What is the 4 key ideas for Convolutional Neural Network (CNN)**
- Shared weights, Non-linear activator functions, Detect local conjunctions and Exploration of non-hierarchical structures
- Local connections, Individual weights, Low feature space, Quasi-linear activator functions
- No lower bound, Use of many layers, Pooling and Finding global features
- Local connections, Shared weights, Pooling and Use of many layers
* `[ D ]`


---

**Q: "RNNs are very powerful dynamic systems, but training them has
proved to be problematic". What causes this problem?**
- The fact that most training algorithms have been developed for CNN's and are thus not really compatible with RNN's.
- The fact that gradient descent tends to get stuck in very bad local optima for RNN's.
- The fact that there are too much parameters to optimize
-  The fact that the backpropagated gradients either grow or shrink at each time step, so that over many time steps they typically explode or vanish.
* `[ D ]`


---

**Q: Which of the following are properties of recurrent neural networks(RNN)?**
- RNNs depend on the inputs from the neurons in the previous time steps.
- RNN is independent of the inputs from previous neurons which increases its efficiency.
- RNNs cannot be used for unsupervised learning.
- Convolutional RNNs always perform better than LSTM networks.
* `[ A ]`


---

**Q: Recurrent neural networks:**
- Process an input sequence one batch at a time
- Maintain a ‘state vector’ in their hidden units
- Are easy to train because the backpropagated gradients remain stable at each time step
- Are best used for binary classification
* `[ B ]`


---

**Q: what is a necessary requirement of shallow classifiers**
- lots of data
- well shaped features, allowing a split
- an automatic learner, allowing for good feature selection.
- none of the above
* `[ B ]`


---

**Q: Statement 1: The role of the convolutinal layer is to detect local conjunctions of features from the previous layer. \\
Statement 2: The role of the pooling layer is to merge semantically similar features into one.**
- Only statement 1 is true.
- Only statement 2 is true.
- Both statement 1 and 2 are true.
- None of the statements are true.
* `[ C ]`


---

**Q: The issue with recurrent neural networks is...**
- ... to train them by backpropagation.
- ... to make sure that they don't take over the world power.
- ... that they are not good at predicting the next word in a sequence.
- ... that they maintain in there hidden layers the future sequence of elements.
* `[ A ]`


---

**Q: If faced with a speech recognition task, what would be the appropriate neural network architecture to use? **
- Convolutional NN 
- LSTM
- Feed-Forward Neural Networks
- Multilayer perceptron
* `[ B ]`


---

**Q: What is one of the key differences between conventional Machine Learning and Deep Learning methods?**
- Deep Learning typically has to learn way fewer parameters than Machine Learning methods.
- Machine Learning usually requires an expert to choose a representation for the data, whereas Deep Learning is a representation-learning method.
- Machine Learning can be used for both supervised and unsupervised learning, whereas Deep Learning explicitly requires the data to be labeled.
- Deep Learning methods typically requiree way less data compared to Machine Learning in order to come up with a good model.
* `[ B ]`


---

**Q: One of the advantageous of  deep learning over the classic learning algorithms are  **
- The hidden layer of multi layer neural network learn  to represent the input network 
- Deep learning use distributed representations such as power combination and component structure 
- Deep learning based on counting frequencies of occurrences of short symbol sequences of length up to N (called N-grams)
- A and B 
* `[ D ]`


---

**Q: Which statement about deep learning is not correct?**
- Deep learning is proven to be very efficient in image and speech recognition.
- Deep learning is a synonym of machine learning.
- Deep learning uses multiple layers of which the output of a layer is used as the input of the following layer.
- A deep learning algorithm needs to be trained.
* `[ B ]`


---

**Q: Deep learning improved the state-of-the-art in speech recognition, visual object recognition, object detection and many other domains such as drug discovery and genomics. Natural language understanding is another area in which deep learning is poised to make a large impact. What type of neural networks currently considered as leading in natural language understanding?**
- Recurrent neural networks
- Convolutional neural networks
- Deep neural networks
- Feed forward neural networks
* `[ A ]`


---

**Q: Why deep learning is so special compared with machine lerning?**
- Obviously, because it is deep! The deeper the better
- It does not suffer from gettig stuck in local minima
- It can be learned to extract good features automatically
- It cannot be overfitted
* `[ C ]`


---

**Q: What is not part of the Convolutional Neural Network's architecture?**
- Pooling
- N-grams 
- Convolutional layers 
- Non-linearity 
* `[ B ]`


---

**Q: what is the wrong answer?**
- cnn its input is a matrix output and outputs a result and the weight aren't temporary.
- rrn has weights that are as well as tempory as the saved weigths that has been added.
- cnn can been used for picture mostly. 
- rnn performs well on langues or sequential data.
* `[ B ]`


---

**Q: Which of the following statements is True and what is an effective combination of methods for a learner to learn to 'translate' the meaning of an image into an English sentence?**
- The long short-term memory (LSTM) networks that use special hidden units, the natural behaviour of which is to remember inputs for a long time; CNN + RNN.
- RNNs have been found to be very bad at predicting the next character in the text or the next word in a sequence; CNN + RNN.
- Recent theoretical and empirical results strongly suggest that local minima are not a serious issue in general; Linear classifier.
- The pooling allows representations to vary very largely when elements in the previ- ous layer vary in position and appearance; SVM.
* `[ A ]`


---

**Q: For supervised learning, which statement is correct?**
- Data class lebels are present in training data
- Clustering is supervised learning because we specify number of clusters 
- data used to optimize the parameter settings of a supervised learner model is called verification
- none of the above are correct
* `[ A ]`


---

**Q:  One of the key ideas behind ConvNets is pooling. What is the role of pooling?**
- Detecting local conjuctions of features from the previous layer.
- Merging semantically similar features into one.
- Combating overfitting, as it is a regularization technique in which randomly selected neurons are ignored during training (i.e. their contribution is temporarily removed in the forward pass and weight updates are not applied to them in the backward pass).
- Generating more training examples by deforming the existing ones.
* `[ B ]`


---

**Q: Which of the following deep learning algorithm would you choose to predict the next word in a sequence of words ?**
- CNN
- Feed Forward Network
- RNN
- Self-organizing Map
* `[ C ]`


---

**Q: Why have convolutional neural networks become popular lately?**
- Efficient use of GPUs
- Use ReLUs instead of smoother non-linearities
- Techniques to generate more training examples by deforming existing ones
- All of the above
* `[ D ]`


---

**Q: Your tutor tells you to take a look at the features used by a convolutional deep network. Where should you take a look first?**
- In the training data
- The hidden layer
- Results
- None of the above
* `[ B ]`


---

**Q: What is not used in a recurrent neural network?**
- An input sequence
- State vector or memory
- Time-dependent weight matrices
- Time-dependent activation functions
* `[ C ]`


---

**Q: Which of the following is not an idea behind a convolutional neural network, that makes use of the properties of natural signals?**
- shared weights
- speed
- local connections
- pooling
* `[ B ]`


---

**Q: For small datasets, which pre-training method helps to prevent overfitting?**
- Supervised
- Unsupervised
- Semi-supervised
- Reinforcement
* `[ B ]`


---

**Q: Which of the following statement is not true for the convolutional neural networks(CNNs)?**
- Convolution layers detect local conjunctions of features from the previous layer
- Pooling layers merges semantically similar features into one
- Pooling allows different representation despite varying position/appearance
- Natural signals are hierarchies
* `[ C ]`


---

**Q: For tasks that involve sequential inputs, such as speech and language, most often ... is/are used?**
- Conditional Decision Trees
- Linear Regression
- Recurrent Neural Networks
- k-nearest neighbours
* `[ C ]`


---

**Q: Why are convolutional neural networks often used on multimedia signals (e.g. audio, images, and videos)?**
- They are designed to process data that come in the form of multiple arrays and learn from them.
- They work well with periodic signals.
- They are designed such that underlying structures are easily recognised.
- They work well with finite signals.
* `[ A ]`


---

**Q: What groundbreaking work did CIFAR perform to revive interest in Deep Feedforward Networks in 2006?**
- The researchers introduced unsupervised learning procedures that could create layers of feature detectors without requiring labeled data
- The researchers proved that local minimums in the objective function of sufficiently complex models had similarly low values
- The researchers created a framework to quickly construct parallelizable Deep Learning models, making it easier to create quickly training models
- The researchers created a digit recognition model that can perform as well as human subjects in classifying digits
* `[ A ]`


---

**Q: Which of the following statements are supported by the paper:
1. In the future we might be able to connect different types of Neural Networks.
2. Convolutional Neural Networks are at this point the best type of Neural Network we have. **
- Both are supported
- Only 1 is supported
- Only 2 is supported
- Both are NOT supported
* `[ D ]`


---

**Q: Backpropagation is:**
- a fully automated system for training which input values work best for a model.
- a method for interpreting the results of a neural network architecture.
- a training procedure that uses the error to set the model weights.
- a mechanism to reduce overfitting convolutional models.
* `[ C ]`


---

**Q: An RNN is called a "Recurrent" (neural network), because...**
- Because once trained, it can be reused in several applications without loosing it's accuracy.
- The outputs of the system are in turn used to generate more training data to further increase the accuracy.
- Certain sequences of layer & weight combinations are recurrently used in the architecture of the network.
- The outputs of the system are fed back into the network with a delay of (at least) 1 time step, to enable to network to exhibit temporal dynamic behavior.
* `[ D ]`


---

**Q: What negative gradient vector indicate during gradient descent learning method?**
- Direction of steepest descent in this landscape, taking it closer to a minimum
- Direction of steepest descent in this landscape, taking it closer to a maximum
- Direction of global minimum
- Error in our calculations
* `[ A ]`


---

**Q: When compared with traditional linear classifiers, deep learning does NOT have one of the following advantages**
- Eliminates the need to design feature extractors by hand
- Eliminates the need for any domain knowledge
- Reduces relevance of features which are not important for discrimination
- Increases relevance of features which are important for discrimination
* `[ B ]`


---

**Q: What is the difference between supervised learning and unsupervised learning?**
- Nu human in the loop
- Training data is treated differently
- No validation necessary, different techniques used (mostly clustering)
- Different test data
* `[ C ]`


---

**Q: What is not one of the key ideas behind Convolutional Neural Networks that take advantage of the properties of natural signals?**
- The use of shared weights
- The use of filter banks
- The use of pooling
- The use of local connections
* `[ B ]`


---

**Q: Which is NOT commonly used activation function?**
- ReLU
- Sigmoid
- Cos
- tanh
* `[ C ]`


---

**Q: Which of the following statements is false?**
- A convolutional neural network is not feedforward neural network.
- In practice, local minima are often not a problem when using backpropagation.
- Backpropagation can get stuck in local minima.
- All of the above statemetns are true.
* `[ A ]`


---

**Q: Why can't linear classifiers solve the selectivity-invariance dilemma by themselves?**
- Because they can divide the input space into very complex regions.
- Because they require a feature extractor to select the relevant features and ignore the non-relevant ones.
- Because there can't ever be enough data to train them.
- Because linear classifiers are generally too biased to reach a correctly selective solution.
* `[ B ]`


---

**Q: Which of the following statements are true:

statement 1: Machine Learning requires a lot of feature engineering by hand

statement 2: Deep Learning requires a lot of feature engineering by hand

statement 3: The SGD Stochastic gradient descent is called stochastic because each big subset of examples gives a noise estimate of the average gradient over all examples.

statement 4:  For the problems such as image and speech regognition it is important that input-output function is very sensitive to the variations in the input    (such as position, orientation or illumination)**
- 1 3 and 4
- 1
- 2
- 2 and 4
* `[ A ]`


---

**Q: For sequential/temporal data is better to use **
- Recursive Neural Network
- Recurrent Neural Network (RNN)
- Convolutional Neural Network (CNN)
- FeedForward Neural Network
* `[ B ]`


---

**Q: Why were neural nets and backpropagation ignored in the early days? **
- It was thought that that this was infeasible, because gradient descent algorithms tapped in local minima.
- The performance is heavily dependent on the input data.
- Too sensitive to noise.
- None of the above
* `[ A ]`


---

**Q: Which of the following is particularly good at vision related tasks?**
- Recurrent neural networks
- Autoencoders
- ReLUs
- Convolutional neural networks
* `[ D ]`


---

**Q: Which of the following is NOT a key idea behind Convolutional Neural Networks:**
- Usage of local connections
- Usage of shared weights
- Usage of linear activation functions
- Usage of pooling
* `[ C ]`


---

**Q: Which Neural network is associated with its correct application**
-  CNN - language processing
- MLP - image recogniton
- RNN - language processing
- CNN - Regression
* `[ C ]`


---

**Q: Which one of the following points is Not the advantage of deep learning?**
- Deep learning discovers intricate structure in large data sets.
- Deep learning uses the backpropagation algorithm to indicate how a machine
should change its internal parameters.
- Backpropagation algorithm has a complex mechanism to compute the representation in each layer
- Deep learning dramatically improve the state-of-the-art in speech recognition,
visual object recognition, object detection and many other domains such as drug discovery and genomics.
* `[ C ]`


---

**Q: What is the main reason to use hidden layers in a Neural Network?**
- When a linear activation function is used, the input space will be transformed to a new space which can be linearly separable.
- When a non-linear activation function is used, the input space will be transformed to a new space which can be linearly separable
- When a linear activation function is used, the input space will be transformed to a new space which can be non-linearly separable.
- When a non-linear activation function is used, the input space will be transformed to a new space which can be non-linearly separable.
* `[ B ]`


---

**Q: Which statement is true about Machine Learning and Deep Learning?**
- As the size of the data increases Machine Learning algorithms can show better results compared to Deep Learning
- Feature engineering is a crucial aspect in Deep Learning
- Convolutional Neural Networks(CNNs) are designed to process arbitrary timed-sequenced data
- CNNs does not perform well on spatio temporal(grid-like) data like patient information(gender, age, ...)
* `[ D ]`


---

**Q: What is the key advantage of Deep Learning Methods?**
- DL-Methods automatically learn good features
- DL-Methods do not need much training to be efficient
- DL-Methods do not need much computing power & memory
- DL-Methods automatically prevent overfitting
* `[ A ]`


---

**Q: Why do CNNs work so well on images?**
- Pooling is able to reduce the nonlinearity of the data 
- CNNs allow to create a representation without relying on a hierarchy of feature filters.
- CNN is particularly well suited for batch input which reduces the amount of computation for images
- They can create a hierarchy of increasingly complex filters that describe the input.
* `[ D ]`


---

**Q: For tasks that involve sequential inputs, what type of network is often best to use?**
- Conventional Neural Network (CNN)
- Recurrent Neural Network (RNN)
- Feed Forward (FF)
- Support Vector Machine (SVM)
* `[ B ]`


---

**Q: What is the role of the pooling layer?**
- To provide the output through classifying each sample.
- To detect local conjunctions of features from the previous layer.
- To merge semantically similar features into one.
- None of the above.
* `[ C ]`


---

**Q: Why is feature engineering NOT very important in deep learning?**
- Because deep neural networks can learn representations automatically
- Because the features will bias the neural network in the process of learning
- All of the above
- None of the above
* `[ A ]`


---

**Q: What "best" defines a good feature?**
- Low dimensionality
- Invariance to irrelevant stochastic processes in the capture of data
- One that works very well with linear classifiers
- One that does not have to be hand-designed and can be learnt
* `[ B ]`


---

**Q: I. Representation learning is a set of methods that allows a machine to
be fed with raw data and to automatically discover the representations
needed for detection or classification.

II.  Deep learning requires an engineer to design a feature extractor in order to transform the raw data into patterns which the classifier can detect**
- only I is correct
- only II is correct
- I and II are both correct
- Neither I or II is correct
* `[ A ]`


---

**Q: What is a purpose of 'pooling' layers in a Convolutional Neural Network?**
- Controlling over-fitting.
- Selecting important exact feature locations.
- Providing 'memory' to the network.
- Making back-propagation calculations more easy.
* `[ A ]`


---

**Q: In the late 1990s it was commonly thought that a simple gradient descent method would get trapped in poor local minima. What is the current vision on this?**
- The same is in the late 1990; the vision remained similar
- With the proper initial conditions, a solution of decent quality can be reached
- Regardless of the initial, solutions of very similar quality are almost always reached
- Experiments are still ongoing, there is no final conclusion yet
* `[ C ]`


---

**Q: Long short-term memory (LSTM) networks were introduced to solve the following problem:**
- RNNs process the input sequence one element at a time, so they need a memory to keep track of the history.
- RNNs have difficulty storing information for very long.
- RNNs process the input sequence one element at a time, so they need a good short-term memory to analyse the input sequence as a whole (once all elements are processed).
- RNNs have difficulty comparing the current sequence of elements to a previous input sequence. 
* `[ B ]`


---

**Q: What is the downside of using RNNs?**
- They have the tendency to in/decrease at times
- They are computationally more expensive than other methods
- There is a large risk of overfitting
- They are not suitable for more complex tasks
* `[ A ]`


---

**Q: Which of the following statement about convolutional neural networks is not true?**
- The architecture of a typical ConvNet is structured as a series of stages
- Units in a pooling layer are organized in feature maps
- All units in a feature map share the same filter bank
- Different feature maps in a layer use different filter banks
* `[ B ]`


---

**Q: What is wrong about the differences between a Convolutional NN and a Recurrent NN?**
- A recurrent NN uses information about the history, a convolutional NN does not
- A recurrent NN is a better choice for image recognition than a convolutional NN, since local patches of pixels together function as features and contain more information than individual pixels
- A convolutional NN has layers of neurons that filter the image by convolving it with a kernel
- A convolutional NN is the type of neural network that is used the most 
* `[ B ]`


---

**Q: Convolutional neural networks (ConvNets) are designed to process data that come in the form of multiple arrays, for example a colour image composed of three 2D
arrays containing pixel intensities in the three colour channels. 

What is not a property that ConvNets mainly take advantage of?**
- The use of many layers
- Shared weights
- Local connections
- Pre-processing
* `[ D ]`


---

**Q: Which one of the following is correct?**
- In supervised deep learning, we optimize weights by minimizing the loss function. Alternatively, we can optimize weights by minimizing the output error. Both methods lead to same results.
- In backpropagation, we compute the derivative/gradient of the objective function w.r.t. the input by working backwards w.r.t. the output and thus reduce computation.
- It is nearly infeasible to learn useful feature extractors with little prior knowledge, which can be solved by constructing deep neural network. However, when using simple gradient descent to adjust weights, it will get trapped in local optimum. 
- Unsupervised pre-training can help to prevent overfitting and improve the result. It is widely used in deep learning.
* `[ B ]`


---

**Q: 1. statement: Recurrent neural networks handle sequential input such as speech and language well.
2. statement: Convolutional neural networks are structured as series of stages.**
- Both statements are true.
- The first statement is true, the second is false.
- The first statement is false, the second is true.
- Both statements are false.
* `[ A ]`


---

**Q: The ReLU (Rectified Linear Units) activation function is:**
-  A linear function that is used as a normalization function of the feature maps values after the pooling layer
- A non-linear function that is used as a normalization function of the feature maps values after the pooling layer
- A non-linear function that is used as a normalization function of the feature maps after the convolutional layer
- A linear function that is used as a normalization function of the feature maps values before the convolutional layer
* `[ C ]`


---

**Q: Which of the following is false according to the paper?**
- The objective function is used to measure the error between the output from model and true label.
- To adjust weights in neural networks, gradient vector is needed and weighted will be moved in the direction of its gradient vector.
- The hidden layer can be seen as distorting the input in a non-linear way so that categories become linearly separable by the last layer.
- ReLU function can be used as a non-linear activation function in hidden layer.
* `[ B ]`


---

**Q: Mark the false sentence.**
- On supervised learning we collect a data set and it’s required to be labelled.
- The backpropagation procedure used to compute the gradient corresponds to a pratical application of the chain rule for derivatives.
- The hidden layers allow to change the input in a non-linear way so that categories become linearly separable by the last layer.
- None of the above.
* `[ D ]`


---

**Q: How many combinations of values of features are possible by using distributed representations with n features?**
- n
- 2n
- 4n
- $n^2$
* `[ D ]`


---

**Q: Which of the following is not true:**
-  Deep learning is very good at discovering intricate structures in high-dimensional data.
-  Supervised learning is the most common form of machine learning, including the deep learning.
- One can use generic non-linear features with kernel methods to make classifiers more powerful.
- Poor local minima is one of the main problem with large net-works in practice. 
* `[ D ]`


---

**Q: What is the greatest advantage of deep learning?**
- It allows to learn automatically which features are good using general porpoise learning procedure.
- It requires small computing resources.
- It performs best with small amount of data.
- Compared to machine learning it is more application independent.
* `[ A ]`


---

**Q: What is the advantage of the ReLU function for network training over other, smoother functions like tanh(z) or 1/(1+exp(-z))?**
- ReLU typically gives more accurate results.
- ReLU typically learns faster much in networks with many layers.
- ReLU is a newer technique
- ReLU is a linear function, which typically learns much faster
* `[ B ]`


---

**Q: Which of the following statements are correct?

1.Backpropagation is training algorithm used for multilayer neural network which indicate how a machine should change its internal parameters 
2.Gradient descent is basically an optimisation algorithm, which is used to learn the value of parameters that minimises the cost function **
- Statements 1 is correct, statements 2 is incorrect
- Statement 1 is incorrect, statement 2 is correct
- Both statements are correct 
-  Both statements are incorrect 
* `[ C ]`


---

**Q: Which of the following cannot be implemented by stochastic gradient descent (SGD) method?**
- Regression for prediction function.
- Access linear classifier y = wx +b.
- Assign binary label for two classes (eg, 0 and 1).
- Reach the lowest lost for a given loss function.
* `[ C ]`


---

**Q: What is the main reason rectified linear units are used as activation functions in neural networks?**
- They typically learn much faster in networks with many layers.
- Their complexity allows for much larger networks.
- They have been proven to be the overall best activation function.
- All of the above
* `[ A ]`


---

**Q: Which of the following is false about RNNs?**
- RNNs are most suitable for tasks that involve sequential inputs
- When training RNNs in a naive way, the gradients either explode or vanish
- RNNs can be seen as a very deep feedforward network, unfolded in time
- RNNs improve upon LSTM networks by storing information for longer time periods
* `[ D ]`


---

**Q: Which type of neural net is commonly used for tasks that involve speed and language or other sequential inputs?**
- MLPs
- CNNs
- RNNs
- SVMs
* `[ C ]`


---

**Q: Networked in a function chain f(x) = f^(3)(F^(2)(f(1)(x))). For f, what is the first layer, second, the output and depth?**
- First layer: f^(1), second f^(2), output: f^(3), hidden: f^(2), depth: 3
- First layer: f^(2), second f^(1), output: f^(3), hidden: f^(2), depth: 3
- First layer: f^(2), second f^(3), output: f^(1), hidden: f^(2), depth: 3
- First layer: f^(1), second f^(2), output: f^(3), hidden: f^(2), depth: 2
* `[ A ]`


---

**Q: What type of neural network would you use to process text databases?**
- Convolutional Neural Network
- Recurrent Neural Network
- Deconvolutional Neural Network
- Deep Residual Network
* `[ B ]`


---

**Q: In convolutional neural networks the process of merging semantically similar features into one is done:**
- In the first layer
- In the convolutional layer
- In the pooling layer
- In the hidden layer
* `[ C ]`


---

**Q: Which of the following is false:**
- Recurrent neural networks are used in speech recognition
- Convolutional neural networks can be used for classifying images
- Getting stuck on local minima is a serious problem for large networks
- Convolutional neural networks can be combined with recurrent neural networks
* `[ C ]`


---

**Q: In natural language processing we n-grams are used to represent text, what alternative representation does a neural network offer?**
- Neural networks can only perform classification tasks and thus cannot represent data in other ways.
- Neural networks can transform text into vectors which will allow for words that are semantically similar to be near each other.
- As text is of variable length neural networks won't be able to use text as input and thus not be able to create representations of text.
- Neural networks can only accept features in the form of number arrays, not text and thus are not able to create representations of text.
* `[ B ]`


---

**Q: A large amount of deep learning algorithms are Convolutional Neural Networks (ConvNets). These are known for their ability to understand languages and interpret images quickly and reliably. This is because of their exploitation of a specific property. Which one of the options is this property?**
- Their application is in a specific field, thus the amount of diversity in the input is limited.
- Natural Signals are compositional hierarchies.
- Natural signals are linear.
- Their input data is limited to 2D.
* `[ B ]`


---

**Q: Which of the following principles or “field wisdom statements” are not used in the field of machine learning?**
- More data beats a cleverer algorithm.
- Simplicity does not imply accuracy.
- Using stronger functions will results in stronger learners.
- Correlation does not imply causation.
* `[ C ]`


---

**Q: Which one of the statements is wrong?**
- ConvNets are designed to process data that come in the form of multiple arrays.
- Feedforward neural network learns to map a fixed-size input (for example, an image) to a fixed-size output (for example, a probability for each of several categories).
- Test sets serve to test the generalization ability of the machine — its ability to produce sensible answers on old inputs that it has seen during training.
- The most common form of machine learning, deep or not, is supervised learning.
* `[ C ]`


---

**Q: What is the key advantage of deep learning?**
- Good features can be learned automatically, it is not necessary to do manual feature extraction.
- It is faster than other machine learning methods
- It can take better advantage of heavy compute and large amounts of data
- Deep learning can learn very complex functions
* `[ A ]`


---

**Q: What is not a suitable application of a CNN?**
- Facial recognition
- Object detection in images
- Classifying an up or downtrend in a stock ticker
- Classifying points in a 2d plane
* `[ D ]`


---

**Q: What is the most common type of machine learning?**
- Supervised learning
- Unsupervised learning
- Deep learning
- Shallow learning
* `[ A ]`


---

**Q: Which of those statements is FALSE?**
- Deep nets usually can generalize new combination of the values of learned features beyond those seen from training
- Convolutional neural networks process data in the form of multiple arrays
- Using generic non-linear features make classifiers more powerful
- Using generic non-linear features improve prediction beyond those seen from training
* `[ D ]`


---

**Q: Which of these is one of the key idea of Convolutional Networks based on the  properties of natural signals?**
- Local Connections
- Shared  Weights
- Use of many Layers
- All of the above
* `[ D ]`


---

**Q: Given statement 1 & 2:
1 - Pooling layers in ConvNet group together features as detected by the convolutional layers.
2 - Convolution layers are based on convolution, where is checked how an input behave to its neighbor.**
- 1 is correct, 2 is false
- 2 is correct, 1 is false
- neither are correct
- both are correct
* `[ A ]`


---

**Q: name a network that can correct the difficulty a RNN has to store information for a long time.**
- LSTM
- Supervised learning
- CNN
- RNN
* `[ A ]`


---

**Q: In a CNN the sequence of steps is as follows**
- Non-linearity, Convolution, Pooling
- Convolution, Non-linearity, Pooling
- Convolution, Filtering, Pooling
- Filtering, Convolution, Pooling
* `[ B ]`


---

**Q: Which of the following is true according to the paper on deep learning?**
- Supervised learning is the most common form of machine learning
- In the 1990s, neural nets and backpropagation were popular among the machine learning communities.
- It is better to use RNN with non sequential input forms.
- Generally, human and animal learning is considered to be supervised
* `[ A ]`


---

**Q: Choose the false statement regarding the max-pooling layer of a convolutional neural network (ConvNet):**
- Reduces the dimensionality of the feature map
- Reduces the training time
- Provides invariance to small shifts
- Provides ConvNet with a memory module use in the encoder(CNN)-decoder(RNN) structure for “translating” the meaning of an image into textual representation
* `[ D ]`


---

**Q: What feature of convolutional neural networks is responsible for ensuring that representations of invariant characteristics do not vary much when elements in the previous layer vary in position and appearance?**
- Convolutional Layers
- Pooling Layers
- Filter Banks
- ReLU activation function
* `[ B ]`


---

**Q: Which is not the key idea behind ConvNets?**
- global connection
- shared weights
- pooling
- many layers
* `[ A ]`


---

**Q: Which of the following is true about the pooling layer in convolution neural nets ?**
- It detects local conjunctions of features from previous layer
- It reduces the dimension of representation and creates in-variances to small shifts
- It applies a non linearity to the output of previous layer
- None of the above
* `[ B ]`


---

**Q: Which statement is false?**
- With large neural networks, the problem occurs that the gradient descent gets trapped in poor local minima.
- As long as the modules of multilayer networks are relatively smooth functions of their inputs and of their internal weights, one can compute the gradients using the backpropagation procedure.
- The first few layers of a convolutional neural network are composed of two types, namely the convolutional layer and the pooling layer.
- Deep neural networks exploit the property that many natural signals are compositional hierarchies in which higher level features are obtained by composing lower level ones.
* `[ A ]`


---

**Q: What is the role of the convolutional layer in a convolutional neural network?**
- To detect local conjunctions of features
- To detect semantically similar features
-  To merge semantically similar features into one
- To merge local conjunctions of features
* `[ A ]`


---

**Q: Which is the expression of a ReLU?**
- f(x)=tanh(x)
- f(x)=1/(1+exp(-x))
- f(x)=max(x, 0)
- f(x)=dx/dt
* `[ C ]`


---

**Q: We would like to construct a neural network for sentiment analysis. Specifically, we aim to input speech extracts of a person whose duration is arbitrary. We expect the NN to conclude on the mood state of the person. We put emphasis on the relationship between the sequence of the words for the detection of possible patterns which will be indicative of the emotional state of the speaker. Which NN architecture is most suitable for the case in consideration.**
- A convolutional neural network (CNN)
- A recurrent neural network (RNN)
- Either architecture (CNN or RNN) will perform equally well if properly trained
- None of the above is suitable for this scenario
* `[ B ]`


---

**Q: When visualizing learned word vectors, what can be observed about semantically similar words or sequences of words?**
- They are mapped to nearby representations
- They are mapped to furthest representations
- They are clustered in groups of similar representation
- They are on the same axis
* `[ A ]`


---

**Q: Which of statements is true regarding the CNN and RNN?**
- ConvNet is much more complicated than the regular neural network because the gradients through the network are harder to compute.
- The role of pooling is to reduce the dimension of the representation thus help overcome overfitting problem.
- Training RNN would be problematic because the backpropagated gradients either grow or shrink at each time step, so over many time steps they typically explode or vanish.
- Conventional RNN could learn to store long-term information because it's a very deep neural network and every unit could store information of a specific time step.
* `[ C ]`


---

**Q: What is the most common form of machine learning?**
- Reinforced learning
- Unsupervised learning
- Feature learning
- Supervised learning
* `[ D ]`


---

**Q: Huge Success of Deep convolutional network in the 2012 Image Net competition brought ConvNets back to prominence.So which among the following attrributes lead to their success in the competition**
- Efficient use of GPU's
- ReLU activation function and Regularization using dropout
- Artificial data generation
- All of the above
* `[ D ]`


---

**Q: Which one is not a deep learning method?**
- Unsupervised
- Semi-supervised
- Supervised
- Controlled
* `[ D ]`


---

**Q: What is the reason for effective performance of LSTMs over conventional RNNs in tasks such as speech recognition systems?**
- LSTMs are actually less efficient than conventional RNNs in such tasks
- the presence of pooling layers that help group similar features 
- the computation of gradients is more complex for RNNs
- the presence of explicit memory,using special memory cell hidden layer units that connects to itself to store state values
* `[ D ]`


---

**Q: What is the role of pooling in convolutional networks?**
- Detect local conjunctions of features from the previous layer
- Involve sequential inputs
- Compute the error derivative
- Merge semantically similar features into one
* `[ D ]`


---

**Q: which statement is wrong? **
- Feedforward neural network architectures can only map a fixed-size input to a fixed-size output.
- Simple gradient descent is easy to get trapped in poor local minima — weight configurations for which no small change would reduce the average error.
- The hidden layers of a multilayer neural network learn to represent the network’s inputs in a way that makes it easy to predict the target outputs. 
- There are four key ideas behind ConvNets that take advantage of the properties of natural signals: local connections, shared weights, pooling and the use of many layers.
* `[ B ]`


---

**Q: What are disadvantages of machine learning with respect to deep learning
1 - Backpropagation gradients either grow or shrink at each timestep
2 - Machine learning relies on using pre-processed data with hand-engineered features**
- 1 only
- 2 only
- neither
- both
* `[ B ]`


---

**Q: When faced with limited data or smaller data sets, often the following technique can not only prevent over-fitting but also leads to better generalisation (LeCun et al.)?**
- Regularisation with dropout.
- Distributed representations.
- Recurrent neural networks.
- Unsupervised pre-training.
* `[ D ]`


---

**Q: Which of the following is NOT true:

I. Backpropagation is not used in deep learning.

II. ConvNet architectures are not really used in daily life.**
- I
- II
- Both
- Neither
* `[ C ]`


---

**Q: For which task would a RNN (recurrent neural network) be the obvious choice**
- Training a classifier to distinguish apples from pears
- Training a classifier to read handwritten letters
- Recognising several different elements in one picture 
- Predicting pedestrian behaviour from camera images
* `[ C ]`


---

**Q: Why are Recurrent Neural Networks (RNN) a good choice for tasks involving speech and language?**
- Because RNNs can easily learn new words.
- Because RNNs are good at dealing with sequential inputs.
- Because RNNs filter out low entropy information in text.
- Because RNNs are able to process sentences in parallel.
* `[ B ]`


---

**Q: To compute the gradient of an objective function with respect to the weights of a multilayer stack of modules, one needs to apply**
- Convolutional neural networks
- Recurrent neural networks
- Supervised learning
- Back propagation
* `[ D ]`


---

**Q: In convolutional neural networks, what is the role of pooling layer?**
- Detect local conjunction features from the previous layer
- Merge semantically similar features into one
- Make the representation variant to small shifts and distortions
- Keep the dimension of the representation constant
* `[ B ]`


---

**Q: Which of the following scenarios are true?**
- For Image recognition tasks, it is wise to use RNNs as they have memory accumulator networks to keep track of state of the world. 
- For tasks involving sequential inputs, it is recommended to use CNNs as they cut down the number of weights using local coherence in the input.
- For machine translation tasks, we use CNNs as theoretical & empirical evidence shows that it is difficult to learn to store information for long time using RNNs.
- For Speech and language processing tasks, RNNs can be used over CNNs as they process inputs, element-by-element per unit time and keeps track of the sequence in hidden layers.
* `[ D ]`


---

**Q: Which of the following is true for Backpropagation?**
- It is a learning algorithm for single layer feedforward neural network.
- It is a learning algorithm for training multi-layer architectures using stochastic gradient descent.
- It is nothing but a gradient descent based technique for maximizing the mean square difference between the actual and desired outputs.
- It is a feedback neural network.
* `[ B ]`


---

**Q: Recurrent neural networks are networks that deal very well with sequential input such as speech and language. Which of these following statements about recurrent neural networks is/are correct?

1. Training RNNs is hard due to an ever increasing or decreasing gradients at each time step.
2. Although the main goal of RNNs is to learn long term dependencies, it is actually very hard to learn to store information for very long.**
- Only 1 is correct
- Only 2 is correct
- Both are correct
- Both are incorrect
* `[ C ]`


---

**Q: Bias due to a simplistic learning algorithm can lead to**
- Underfitting
- Overfitting
- Training highly sensitive to small variations in training data
- Network won’t work if the input data is high dimensional
* `[ A ]`


---

**Q: What is true abour backpropagation?**
- It is a feedback neural network
- Actual output is determined by computing the outputs of units for each hidden layer
- Hidden layers output is not all important, they are only meant for supporting input and output layers
- None of above
* `[ B ]`


---

**Q: Deep neural networks expoloit several properties of patterns in nature. Indicate which of these are used by deeplearning to learn representations of data consisting of multiple layers of abstraction:**
- Infinite regression shown by several species of advanced primates is used to infer multiple representations with a single representation.
- Recursive syntactic pattern learning by songbirds is used to infer multiple representations from a single representation.
- Compositional hierarchies, such as local combinations of edges that form motifs, which ensemble into parts which form objects.
- Geometric replacement rules in fractals are applied to improve the accuracy and speed of representation learning.
* `[ C ]`


---

**Q: Below are three statements related to pooling layer in Convolution Neural Networks (CNN).
(a) Generally, pooling layer in CNNs reduce the image size.
(b) Pooling layer detects local conjunctions of features from the previous layer.
(c) Pooling layer is critical to the operation of CNNs.
Which of the following options are correct?**
- Option (a) is wrong, Options (b) and (c) are correct
- Option (a) is correct, Options (b) and (c) are wrong
- All three options are correct
- Options (a) and (c) are correct, Option (b) is wrong
* `[ B ]`


---

**Q: Why is RNN (Recurrent Neural Network) used for machine translation?**
- RNNs can be trained from an unsupervised learning problem environment.
- It is strictly more powerful than a Convolutional Neural Network (CNN) due to memory cell accumulators.
- It is applicable when the input/output is a sequence at specific time references.
- They use Long Short-Term Memories (LSTM), which is a useful back-propagation mechanism that helps in machine translation aids.
* `[ C ]`


---

**Q: In Deep Neural Networks:**
- Local minimum are the biggest problem when training large networks.
- Deep neural networks are based on obtaining low-level features from combinations of higher-level features.
- Position of motifs in an image don’t have to be determinant for the results of a neural network.
- None of the above.
* `[ C ]`


---

**Q: For tasks involving sequential inputs(like speech and language) use of which system is a better choice?**
- ConvNets
- RNNs
- Both are good choices
- None
* `[ B ]`


---

