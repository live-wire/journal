# Questions from `cnn` :robot: 

**Q: The weights in a trained CNN filter kernel function as:**
-  A feature selector 
- A data preprocessing step
-  A way to scale images to the required size
- A way to rotate images to a uniform position
* `[ A ]`


---

**Q: Which of the following is not a property of a convolutional neural network?**
- Insensitivity to the place of an object in an image.
-  Insensitivity to the rotation of an object in an image.
- Able to deal with images of various resolutions easily.
-  Requires images to be of at least a certain resolution, because convolution shrinks images. 
* `[ D ]`


---

**Q: In a CNN, after the spatial pooling has been carried out, what is done with the different feature maps that are output:**
- Paste them together side-by-side.
- Sum the feature maps pixel-wise.
-  Compute the max pixel-wise and then rescale de values.
- Put them in different channels (resembling for example an RGB image).
* `[ D ]`


---

**Q: Compared with feed-forward networks, convolutional ones are more sparse or have less parameters. The main benefit of this is:**
- The non existence of local minima is guarantied.
- It makes the network less susceptible to the course of dimensionality.
- It makes it much easier to use images as the input of the network.
- The number of layers, and therefore the training times are greatly reduced.
* `[ B ]`


---

**Q: The 3 x 3 Kernel [[0, 0, 0], [1, 1, 1], [-1, -1, -1]] ([first row], [second row], [third row]) detects**
- Left vertical edges
- Right vertical edges
- Top horizontal edges
- Bottom horizontal edges
* `[ D ]`


---

**Q: Why are Toeplitz matrix used in CNNs?**
- Because it many zeros (it is a very parse matrix)
- Because same parameters/values repeat within the matrix
- Because the eigenvalue of the matrix is zero
- Because it’s a symmetric matrix
* `[ B ]`


---

**Q: Function f() is equivariant to g() if**
- f(g(x)) = g(f(x))
- f(g(x)) = f(x)
- g(f(x)) = f(x)
- f(x) = g(x)
* `[ A ]`


---

**Q: Which of the following about pooling is FALSE?**
- Pooling is approximately invariant to local translations.
- Pooling is useful for image classification, because feature presence is often more important than feature location.
- Pooling increases the size of the receptive field linearly.
- To get a fixed sized output m for varying input size n, one can set the pool width equal to n/m.
* `[ C ]`


---

**Q: How should one combine the output of a pooling step to convert it to an input for the next convolution step?**
- Calculate the weighted average of the outputs.
- Concatenate them by putting the images side by side.
- Concatenate them on top of each other, using different colours.
- None of the above.
* `[ C ]`


---

**Q: Which of the following statements is true about the comparison between a Convolutional and a Feed Forward Neural Network?**
- The Convolutional Neural Network needs to train more parameters.
- The Feed Forward Neural Network can be trained with fewer samples.
- The Convolutional Neural Network should be used in every application.
- The Convolutional Neural Network is a limited version of the Feed Forward Neural Network.
* `[ D ]`


---

**Q: Are the kernels in representation learning required to have a specific shape?**
- The same as the object we search
- Only the periphery of the object 
- It will be learned by the algorithm
- None of the above
* `[ A ]`


---

**Q: How shrinking is prevented in deep convolutional networks?**
- Pruning
- Padding
- Depth increase
- Kernel change
* `[ B ]`


---

**Q: What is backpropagation?**
- An efficient algorithm to compute gradients
- An arithmetic way of calculating the outcome of a neural network
- A way to estimate the efficiency of a neural network
- A linear interpolation technique
* `[ A ]`


---

**Q: What is the benefit of the node-centered point of view in backpropagation?**
- Error signals do not propagate trough the network
- Due to modularity only derivatives with respect to its arguments  need to computed
- There are no messages passed through the network
- Nodes can have multiple outputs
* `[ B ]`


---

**Q: What is typically not performed by a spatial pooling layer?**
- Reducing the size of an input with 50% in every direction (for example x and y).
- Taking the maximum value of some subspace of the input to create an output.
- Remove all negative values.
- All of these actions are typically performed by a spatial pooling layer.
* `[ C ]`


---

**Q: Three size 3 convolution layers and one input image (bottom). Let layer 1 do a convolution with stride 2 (sub-sample). how much does the shaded neuron 'see' of the image?**
- 3
- 5
- 7
- 9
* `[ D ]`


---

**Q: If the stride for a convolutional layer is increased, …**
- the receptive field becomes smaller, but the network classifies faster
- the receptive field becomes smaller, thus more details are preserved
- the receptive field becomes larger
- the receptive field is not influenced at all
* `[ C ]`


---

**Q: Which statement about pooling is FALSE?**
- Pooling is approximately invariant to local translations
- Pooling summarizes the outcome over a region
- Pooling reduces the size of the feature map
- Pooling takes the average over a region in a feature map and passes this value on
* `[ D ]`


---

**Q: When performing a convolution, why would you use padding?**
- To make sure that the amount of nodes per layer does not shrink
- To make sure the pooling layers do not overfit
- To make sure that the used kernel generalizes well
- To make the network structure invariant to scaling
* `[ A ]`


---

**Q: For which functions $f(x)$ and $g(x)$ does it hold that $f(x)$ is equivariant to $g(x)$?**
- $f(x)=x^2$, $g(x)=2x$
- $f(x)=\sqrt{x}$, $g(x)=x+1$
- $f(x)=2x$, $g(x)=5x$
- $f(x)=2x$, $g(x)=x+1$
* `[ C ]`


---

**Q: What is true about convolution over an image:**
- It is equivariant to rotation
- It is equivariant to translation
- It is equivariant to scaling
- None of the above
* `[ B ]`


---

**Q: What is true about convolution a convolutional neural network?**
- A convolutional neural network is a form of a feed-forward network.
- A convolutional neural network 
- has less parameters when compared to a regular feed-forward network of the same size.
- All of the above.
* `[ D ]`


---

**Q: What is the main difference between using a smaller kernel and a larger kernel for CNN? (Given the same image size)**
- A larger kernel generates more general features compared to a smaller kernel
- A larger kernel is less resistant to noise compared to a smaller kernel
- A larger kernel generates better features for complex images
- A larger kernel is more resistant to noise compared to a smaller kernel
* `[ A ]`


---

**Q: Which is not true about a Toeplitz matrix?**
- The matrix is always sparse
- The matrix is always invertible
- The non-zero values always occur close to each other
- The matrix contains repeating values.
* `[ B ]`


---

**Q: Reducing the size of your convolution filter is the same as:**
- Increasing the size of your image
- Reducing the size of your image
- Taking a small part of the filter
- Reducing the resolution of your image
* `[ A ]`


---

**Q: Pooling an image 2x2 max, 2x2 subsample:**
- Reduces the number of features of the image by 75%
- Takes the maximum value of groups of 4 pixels and saves only that one
- Reduces the width and height of an image by 50%
- All of the above
* `[ D ]`


---

**Q: Which kind of kernel function do we need when input a smaller image and want it to be same with normal output?**
- larger kernel
- smaller kernel
- same kernel
- it depends on the type of the kernel function, like whether it is a Gaussian distribution.
* `[ A ]`


---

**Q: According to the knowledge of Convolutional and feed forward, what is the result of following formula: [10; 11; 9; 20; 19; 21] ★ [-1; 0; +1]?**
- [10, 9, -1, 1]
- [-1, 9, 10, 1]
- [9, 10, -1, 1]
- [-1, 10, 9, 1]
* `[ B ]`


---

**Q: What you can say about a Toeplitz matrix?**
- shares parameters
- local
- sparse  
- all of the above
* `[ D ]`


---

**Q: How does convolution and pooling increase the receptive field (RF)?**
- both increase the RF linearly
- convolution increases RF linearly, pooling increases RF multiplicatively
- convolution increases RF multiplicatively, pooling increases RF linearly
- both increase the RF multiplicatively
* `[ B ]`


---

**Q: Pooling .... **
- increases RF multiplicatively
- decreases RF multiplicatively
- increases RF linearly
- decreases RF linearly
* `[ A ]`


---

**Q: What is a downside of padding?**
- Less parameters to train
- Random or unrelated information is added to the original input
- It is not possible to classify speech anymore
- Translation at the input becomes vulnerable
* `[ B ]`


---

**Q: Which of the following statements about representation learning is FALSE?**
- Kernel weights are feature detectors.
- Smaller input image = larger kernel.
- Convnet learns the feature representation.
- Learning weights = Learning feature.
* `[ B ]`


---

**Q: Which of the following about Equivariance in convolutional opertion is False?**
- If the input shifts to the left, the output shifts to the left
- Let g be a translation, and f be a convolution, then first translating and then convolving, is the same as, first convolving and then translating.
- In images, camera position is important, objects may be at specific area.
- We can add prior knowledge (convolution) to deep nets, then we will get  huge gain in params and compute
* `[ C ]`


---

**Q: What should be the weights to a moving neighbourhood average filter? Assume that the constant “a” is derived from experiments.**
- a) a * [1 1 1; 1 1 1; 1 1 1]
- b) a * [-1 0 1; -1 0 1; -1 0 1]
- c) a * [1 0 -1; 1 0 -1; 1 0 -1]
- d) a * [0 0 0; 0 1 0; 0 0 0]
* `[ A ]`


---

**Q: A convolutional neural network is –by design- a limited parameter version of a feed forward network. Although less parameters means less flexibility, why is this still a good thing?**
- a) parameters have to be set by hand, less parameters means less work
- b) the less parameters, the better the neural network
- c) each parameter has to be learnt from data, so less parameters have to be learnt
- d) less parameters are not a good thing
* `[ C ]`


---

**Q: The number of channels (values per pixel) of an image output by a convolutional layer depends on**
- the number of layers used prior to this layer.
- the number of filters used.
- the number of channels of the input image.
- the non-linearity measure used.
* `[ B ]`


---

**Q: Which property does not always apply to convolutional matrices?**
- sparsity
- locality
- square dimensions
- parameter sharing
* `[ C ]`


---

**Q: What would be the result of applying a convolution with the kernel below on a certain image:

$ \begin{bmatrix}
0 & 0 & 0 \\
0 & 0 & 1 \\
0 & 0 & 0 
\end{bmatrix}  $**
- The image gets translated to the left
- The image gets translated to the right
- The image's edges are activated
- The image gets rotated
* `[ A ]`


---

**Q: What is the main purpose of a Pooling layer:**
- Merge different convolution results into a single image
- Introduce non-linearity into the network
- Reduce the amount of trainable parameters
- None of the above
* `[ C ]`


---

**Q: What is true for fully connected layers and not for sparsely connected layers in a convolutional network?**
- Every node from one layer is connected to all nodes of the other layer
- Every node is connected to only a few nodes of the other layer
- Every node is full of convolutions
- Every layer is full of convoluting nodes
* `[ A ]`


---

**Q: What is padding in the context of convolutional networks?**
- Adding nodes to the borders/boundaries of a layer
- giving every node a padding and a margin
- adding delays to the network
- None of the above, padding is only used in CSS.
* `[ A ]`


---

**Q: From the Convolution of a network you go through multiple steps. Convolution network structure. First you take the image. From the image you pass through certain learned filters. From these filters you obtain the convolution or otherwise called feature maps. Then you go to a non-Linearity function. With after doing some Non-linearity function you do spatial pooling. After obtaining the Spatial pooling what do you need to do before going to the next layer of the network?**
- You have the three values outputs of the spatial Pooling. Each on you assign a weight and then you take the sum of them together in a single image.
- You concatenate the images, but not next to each other but on top of each other with giving each of image there own color.
-  Make all the imagess to a vector. Flatten them and concatenate all the images together. Put all the images next to each other. (then link (things) together in a chain or series) 
- You put another neural network on all the outputs
* `[ B ]`


---

**Q: You have the following 2 size 3 convolution layers. With numbers being a node from left to right. \\
1 2 3 4 5 6 7 \\ 
1 2 3 4 5 6 7 \\
1 2 3 4 5 6 7 \\
From the middle (4) node if you have stride 2 on layer 1 how many nodes does it exist the output from the input. **
- 5
- 7
- 3
- 1
* `[ B ]`


---

**Q: What are the most popular/mostly used distinct layers in Convolutional network?**
- CONV, FC (Fully connected), RELU, POOL
- CONV, RELU, FC
- CONV, FC
- CONV, RELU
* `[ A ]`


---

**Q: Consider Convolutional neural network architecture. Which statement is correct?**
- Each layer of a network may or may not have parameters
- Each layer has to contain convolutional operation
- Each layer outputs two times smaller image
- Each neuron in each layer is fully connected
* `[ A ]`


---

**Q: Why is it an advantage that CNN has less parameters than feed forward?**
- Easier to code
- It needs less time to train
- Less dimensions generally result in a lower classification error
- The space complexity is lower
* `[ C ]`


---

**Q: What is the result of [10, 5, 2, 3, 8, 1] * [-1, 0, 1]?**
- [-10, 0, 2, -3, 0, 1]
- [-8, -2, 6, -2]
- [5, -8, -2, 6, -2, -8]
- [10, 5, -8, -2, 6, -2, -8, -1]
* `[ B ]`


---

**Q: Which of the following layer are not used in the CNN？**
- convolutional layer
- pool layer
- fully connected layer
- LSTM layer
* `[ D ]`


---

**Q: Two size 3 convolution layers and one input image (bottom). Let layer 1 do a convolution with stride 2 (sub-sample). How much does the shaded neuron 'see' of the image?**
- 7 pixels
- 6 pixels
- 5 pixels
- 4 pixels
* `[ A ]`


---

**Q: What is NOT true about convolutional neural networks (CNNs)?**
- CNNs always use non-linearity's
- CNNs are a limited parameter version of feed forward networks
- CNNs are deep neural networks using a variation of multilayer perceptrons
- CNNs typically have different weight vectors across different neurons
* `[ D ]`


---

**Q: What is NOT true about pooling?**
- Reducing memory size is an advantage of pooling
- Larger pooling filters reduce the representation size
- Pooling is approximately invariant to local translations
- Pooling summarizes the outcome over each sub-region
* `[ B ]`


---

**Q: What is not a characteristic of the convolution matrix**
- Non-zero values occur next to each other
- Sharing parameters
- Positive definiteness
- Sparsity
* `[ C ]`


---

**Q: What is the reduction factor if one subsamples with stride 2 over a 4-dimensional object?**
- 1/2
- 1/4
- 1/8
- 1/16
* `[ D ]`


---

**Q: What step in a convolutional network generates the feature maps?**
- the pooling layer
- the non-linearity layer
- the convolution layer
- the connection between two layers
* `[ C ]`


---

**Q: What is the benefit of padding in a convolutional network?**
- prevent the network from shrinking
- counter the loss of information through convolution
- keep the network symmetric
- increase connectivity between layers
* `[ A ]`


---

**Q: Learning a Convolution Kernel weight in a CNN achieves, **
- Noise Reduction.
- Control Over-fitting.
- Feature Extraction
- All of the Above.
* `[ D ]`


---

**Q: What is the task of a POOLING layer in a CNN ?**
- Downsampling - Summarise the outcome over a local region.
- Reduce computation time.
- Control Over-fitting.
- All of the Above.
* `[ D ]`


---

**Q: Say a $1D$ max-pooling layer is applied to the signal $(\cdots,4,2,3,4,3,2,4,\cdots)$ giving an output of $(\cdots,4,4,4,4,4,4,4,\cdots)$ and say that each output neuron of the max-pooling layer is connected to $n$ contiguous input neurons. Then what is the minimum value possible for $n$, based on the input/output given?**
- $n = 2$
- $n = 3$
- $n = 4$
- $n = 7$
* `[ B ]`


---

**Q: In a convolutional layer, several feature maps can be convolved with the input image, giving several output images that we want to feed into another convolutional layer. Those intermediate outputs are usually**
- stacked up on top of each other to create a $3D$ object, creating a channel for each intermediate output image (much like the RGB scheme has 3 channels)
- flattened out into line vectors and then put on top of each other, creating a $2D$ object
- summed with a weighted average to create an intermediate $2D$ object
- put together side-by-side, as if creating a tiling, to make a bigger $2D$ object
* `[ A ]`


---

**Q: How do you prevent shrinking in convolution networks?**
- By applying padding on both sides of every layer in the network.
- By making the network fully connected.
- By reusing neurons of two or more layers back.
- None of the above.
* `[ A ]`


---

**Q: Which of the following statements are true with regards to CNNs: I: Using a bigger image with a bigger kernel is equivalent to using a smaller image with a smaller kernel. II: Kernels are especially useful in finding patterns that span the entire inputsize.**
- II & I
- II
- I
- Neither
* `[ C ]`


---

**Q: In an image, how can we remove noisy pixels?**
- Replace each noisy pixel by its neighborhood average.
- Replace each noisy pixel by the average pixel value of the image.
- Decrease the opacity of each noisy pixel.
- Set each noisy pixel to white.
* `[ A ]`


---

**Q: Which of the statements below are true for convolutional neural networks (CNNs) when applied to image analysis?**
- Pooling layers reduce the spatial resolution of the image.
- A CNN can be applied to an unsupervised learning task.
- They have more parameters than fully-connected networks with the same amount of layers and the same amount of neurons in each layer.
- None of the above.
* `[ A ]`


---

**Q: What is on the ends of a computational graph of the backpropagation of a neural network?**
- The weights of the network and the Loss function.
- The derivative of the weights and the weights.
- The activation function.
- The loss function.
* `[ A ]`


---

**Q: What is computed in a forward pass?**
- Loss
- derivatives
- both A and B
- none of the above
* `[ A ]`


---

**Q: The dimensions of feature to next layer is decided by?**
- The size of input images
- The number of learned filters
- The dimensions of the input images
- The size of convolution kernel
* `[ B ]`


---

**Q: How to prevent image shrink?**
- Pooling
- Convolution
- Padding
- Not all above, should be the another technique not mentioned above
* `[ C ]`


---

**Q: Which of the following is not a property of a Toeplitz matrix?**
- The values of the matrix are sparse
- The non-zero values occur next to each other
- The same values are repeated throughout the matrix
- They are always diagonalizable
* `[ D ]`


---

**Q: Which of the following kernels creates a "moving neighbourhood average" when convoluted over an image?**
- \frac15\begin{bmatrix}
0 &1  &0\\ 
 1&  1& 1\\ 
 0& 1 &0 
\end{bmatrix}
- \frac19\begin{bmatrix}
1 &1  &1\\ 
 1&  1& 1\\ 
 1& 1 &1 
\end{bmatrix}
- \begin{bmatrix}
0 &0  &0\\ 
 1&  0& -1\\ 
 0& 0 &0 
\end{bmatrix}
- \begin{bmatrix}
0 &0  &0\\ 
 0&  0& 1\\ 
 0& 0 &0 
\end{bmatrix}
* `[ B ]`


---

**Q: If input image dimension is 7x7, filter size is 3, stride is 2, and 1 layer of zero padding is done around the borders. Then what is the size of the output after one convolution?**
- 3
- 4
- 5
- 6
* `[ B ]`


---

**Q: If 6 filters of size 5x5x3 are applied to an image of size 32x32x3, with stride 1. Then what is the size of the image after one convolution layer?**
- 28x28x3
- 32x32x3
- 28x28x6
- 32x32x6
* `[ C ]`


---

**Q: Given the following 3x3 kernel,

0 0 0 
0 0 1
0 0 0

what are the effects of convolution between a kernel and a given input image?**
- Move image to the left
- Move image to the right
- Find its edges (Edge detector)
- Nothing
* `[ A ]`


---

**Q: Which of the following statements are true?

1. Convolution increases RF multiplicatively.
2. Pooling increases RF linearly.**
- Statements 1 is correct, statements 2 is incorrect
- Statement 1 is incorrect, statement 2 is correct
- Both statements are correct
- Both statements are incorrect
* `[ D ]`


---

**Q: Which of the following is false about backpropgation?**
- It is an efficient algorithm to compute gradients
- It is used to train a huge majority of deep nets
- It an an engine that power "end-to-end" optimization and representation learning
- It is the inverse of forward propagation
* `[ D ]`


---

**Q: Which of the following statements is false?**
- A topological ordering is a linear ordering of vertices so for every directed edge uv, vertex u comes before v in the ordering
- In backpropagation, each node aggregates the error signal from its children
- The Bar notation is primarily used because it is less cluttered and emphasizes value re-use
- Backpropagation is "just" a clever application of the product rule of calculus
* `[ D ]`


---

**Q: A node can not be:**
- A matrix
- A vector
- A Fourier transformation
- A tensor
* `[ C ]`


---

**Q: Can you take a node with the connections to its children and parents out of the network for evaluation purposes? **
- No, you need the input of its parents to properly evaluate the node
- Yes, but this way you can't compute the derivative with respect to its arguments
- No, you don't know how the error propagates to the nodes' children
- Yes, you can compute the derivative with respect to its arguments
* `[ D ]`


---

**Q: If the function f is equivariant to the function g, then what property holds?**
- f(g(x)) = g(f(x))
- f(x) = g(x)
- argmax(x, f) = argmax(x, g)
- f(f(f(...f(x)...)) = g(x)
* `[ A ]`


---

**Q: Which of the following is not a property of all Toeplitz Matrices?**
- it has a constant diagonal
- It is square
- It is sparse
- Parameters are shared
* `[ B ]`


---

**Q: What is a valid convolution?**
- all pixels in the output are a function of the same number of pixels inthe input
- convolution that result is non-zero
- convolution over real valued functions
- convolution that result is finite
* `[ A ]`


---

**Q: By definition convolution is an operation on...**
- two functions of real valued data
- complementary functions
- time-continuous functions
- two discrete functions
* `[ A ]`


---

**Q: What is more useful in image classification?**
- Detect the feature location in the image
- Detect the size of the feature in the image
- Detect a feature traslation in the image
- Detect a feature presence in the image
* `[ D ]`


---

**Q: What is the function of the ReLu layer in a convolutional neural network?**
- To set all the negative values to 0
- To change the sign of all the negative values
- To detect the negative value
- None of the above
* `[ A ]`


---

**Q: Why are convolutional neural networks simpler to train with respect to feed forward neural network?**
- Because they can work on matrix (images) instead that on vectors
- Because they need less paramenters to be learned by the training dataset
- Because they are not so deep and therefore it is faster to train every level and its parameters
- They are not faster to train
* `[ B ]`


---

**Q: One of the last operations that occur in a CNN layer is called "pooling". Why is it used?**
- It it used to pass the results of the current layer to the next one
- It is used to shift the images obtained,highlighting the most important features
- It is used to reduce the size of the obtained outputs
- It is used to select the filters that have to be used in the next layer
* `[ C ]`


---

**Q: When convolving an input image with a certain kernel (for instance, a Laplacian), which one of the following statements is true?**
- With a smaller kernel, it is more likely for the network to detect smaller blobs
- With a smaller kernel, it is more likely for the network to detect bigger blobs
- Nothing changes if we change the size of the kernel, because the convolution is insensitive to scaling
- None of the above
* `[ A ]`


---

**Q: Which of the following statements is correct about CNNs?**
- Because of shared parameters, CNNs have more weights to learn than feedforwards networks
- The number of filters used in the previous layer is important for determining the “depth” of the filters in the next layer.
- CNNs typically learn, during training, the number of convolutional layers which are needed to solve the problem
- The receptive field in a CNN is the number of filters in the first layer.
* `[ B ]`


---

**Q: Back propagation is an efficient algorithm to:**
- compute the gradient of the error function with respect to a networks weights
- compute the error of a gradient with respect to a networks weights
- compute the gradient of the weights with respect to a networks error function
- compute the absolute error of a network
* `[ A ]`


---

**Q: Which of the following statements is false?**
- When using back propagation, you can often re-use pre-computed values
- The back propagation algorithm first has a forward pass, then a backward pass over the network
- When using back propagation, each node aggregates the error signal from its parents
- None of the above is false
* `[ C ]`


---

**Q: What is the objective of backpropagation algorithm?**
- to develop learning algorithm for multilayer feedforward neural network
b) 
- to develop learning algorithm for single layer feedforward neural network
-  to develop learning algorithm for multilayer feedforward neural network, so that network can be trained to capture the mapping implicitly
- none of the mentioned
* `[ C ]`


---

**Q: What is true regarding backpropagation rule?**
- It is a feedback neural network
- actual output is determined by computing the outputs of units for each hidden layer
- hidden layers output is not all important, they are only meant for supporting input and output layers
- none of the mentioned
* `[ B ]`


---

**Q: What is NOT a reason for applying a non - linear activation function like ReLU in between two convolutional layers?**
- It is difficult to work with negative numbers so the ReLU function is applied
- because having a linear function in between two convolutional layers is the same as having one convolutional layer due to the linearity property of convolution
- non-linear functions make the gradient sparse and easier to compute
- It makes sure that two edges in opposite directions don't cancel
* `[ A ]`


---

**Q: Why are convolutional neural networks (CNN's) better suited for some tasks than fully connected layers?**
- Because it is possible to tune more parameters with convolutional layers by choosing a larger kernel
- Because in some cases fully connected neural networks have too many degrees of freedom making the optimization unpractical
- Because convolutional layers are more efficient
- Because convolution is invariant to small translations and scaling of the input data
* `[ B ]`


---

**Q: Choose the Kernel that can be used for edge detection**
- [0 0 0; 1 0 1; 0 0 0]
-  [0 0 0; 0 1 0; 0 0 0]
-  [0 0 0; 0 0 1; 0 0 0]
- [0 0 0;1 0 -1;0 0 0]
* `[ D ]`


---

**Q: What is the effect of Pooling on Receptive field?**
- Increases linearly
- Decreases linearly
- Increases multiplicatively
- Does not change
* `[ C ]`


---

**Q: When passing a 32x32x3 image input into the first layer in a Convolutional Neural Network which convoles that image with a single 5x5x3 filter with a stride of 1, how many dimensions would the resulting output from this layer be?**
- 28x28x3
- 32x32x1
- 32x32x3
- 28x28x1
* `[ D ]`


---

**Q: In order to stop the representation of our input from shrinking as it passes through multiple convolutional layers, what approach do we take to counter this?**
- Only use 3x3 filters
- Add zero padding to the representation before beginning the convolution
- Add zero padding to the representation after the convolution
- Limit the number of convolution layers in the architecture
* `[ B ]`


---

**Q: Because of equivariance , convolution and translation operations **
- Can be done in any sequence, the output image will be the same
- Sequence decide what the output image will be
- Nullify their effect and output image is same as input image
- Don’t give any important information when applied to the input image
* `[ A ]`


---

**Q: Convolution requires less training compared to feed forward network because**
- Initial parameters chosen in CNN are better
- Has lesser number of parameters to be trained because , they are shared 
- Has lower number of layers in the network
- More efficient
* `[ B ]`


---

**Q: What is NOT an advantage of introducing a pooling layer in a Deep Convolutional Neural Network architecture?**
- Reduces the memory by decreasing the size of the feature maps
- Makes the network capable of accepting variable size input by adapting pool width on the input size 
- Increases the network receptive field
- Makes the exact pixel localization of an object in an image more accurate
* `[ D ]`


---

**Q: Consider the following consecutive operations: Conv2D(kernel_size = [3,3], stride = 1, zero_padding = [1,1]) where zero_padding of [X, Y] means adding X rows and Y columns of “0”,  MaxPool2D(pool_size = [2,2]) where pool_size is the size of the pooling kernel. Starting from a gray scale image with size [216, 216], what will be the dimensionality of the resulted feature map?**
- [216, 216]
- [72, 72] 
- [108, 108] 
- [432, 432]
* `[ C ]`


---

**Q: Which kernel would you use to highlight horizontal edges on an image?**
- \begin{bmatrix}
-1 &  0& 1\\ 
-2 &  0& 2\\ 
-1 &  0& 1
\end{bmatrix}
- \begin{bmatrix}
-1 &  -2& -1\\ 
0 &  0& 0\\ 
1 &  2& 1
\end{bmatrix}
- \begin{bmatrix}
0 &  0& 0\\ 
1 &  1& 1\\ 
0 &  0& 0
\end{bmatrix}
- \begin{bmatrix}
1 &  0& 0\\ 
0 &  1& 0\\ 
0 &  0& 1
\end{bmatrix}
* `[ A ]`


---

**Q: How would you calculate the output size of a 2D rectangular feature map after a convolution layer with one kernel?
W_{i} : input width
K : convolutional kernel size
P : padding
S : stride**
- $$W_{o} = \frac{(W_{i} - K + 2P)}{S}$$
- $$W_{o} = \frac{(W_{i} - K + 2P)}{S + 1}$$
- $$W_{o} = \frac{(W_{i} - K + P)}{S + 1}$$
- $$W_{o} = \frac{(W_{i} - K + P)}{S}$$
* `[ B ]`


---

**Q: What is fact is true about?**
- Pooling is (perfectly) invariant to local translations
- Pooling does not reduce memory, it is subsampling that does
- Pooling increases the receptive field multiplicative 
- When pooling, feature location is more important that feature presence
* `[ C ]`


---

**Q: Which answer is true?**
- When pooling, if your input size n, is larger than your desired output size m, you should decrease the pool width.
- An odd sized convolution results in an equal boarder when padding
- The dimension of output, after convolution is applied to input is: n-m, where n is input dimension and m is output dimension.
- The benefit from pooling is that memory is reduced because summary statistics are unbalanced
* `[ B ]`


---

**Q: How would one get rid of noisy pixels in an image?**
- We take the moving average of neighbouring pixels
- We take the continuous convolution of neighbouring pixels
- We take the integral over neighbouring pixels
- We take the average of all the pixels in the image
* `[ A ]`


---

**Q: How does deep learning differ from traditional machine learning?**
- In traditional machine learning we only use hand-programmed decision rules to classify while deep learning uses feature extraction
- In  traditional  machine  learning  we  learn  useful  features  by  representation learning, in deep learning we select these by hand
- In traditional machine learning we select useful features by hand.  In deep learning we learn them by representation learning
- In  deep  learning  we  use  hand  programmed  decision  rules  to  classify  while traditional machine learning uses representation learning
* `[ C ]`


---

**Q: Convolution can be used to:
- detect Edges
- Detect certain objects (example Waldo)**
- Both
- Edges can be detected but no objects
- Edges can not be detected but objects can
- None
* `[ A ]`


---

**Q: Pooling what is true**
- Is a convolution
- Pooling is approximately invariant to local translations.
- Does padding
- More output nodes than input nodes
* `[ B ]`


---

**Q: Which of the following statements is false?**
- Toeplitz matrix (or diagonal-constant matrix) is the same as convolution.
- A Toeplitz matrix has many zeros (sparse), non-zero values that occur next to each other (local) and same values that repeat (sharing parameters)
- Less parameters means less flexibility, therefore more parameters are better. 
- A CNN is (by design) a limited parameter version of feed-forward. 
* `[ C ]`


---

**Q: Which of the following statements is false?**
- First translating, and then convolving, is the same as first convolving and then translating.
- Stride controls how the filter moves around the input volume.
- Convolution increases the receptive field multiplicatively and pooling increases the receptive field linearly. 
- Convolving an image ensures that the image will shrink. Padding can be used to prevent the shrinking.
* `[ C ]`


---

**Q: Using backpropagation, in the forward pass..**
- We compute high losses for predictions with high confidence
- We compute low losses for predictions with low confidence
- Neither
- Both
* `[ D ]`


---

**Q: Using backpropagation, when does the training phase end?**
- When the average gradient value converges
- When we reach a minimum
- When the average gradient is equal to 0
- None of the above
* `[ A ]`


---

**Q: The dimension of the output image of a convolutional layer is equal to:**
- the number of learned filters.
- the dimension of the input image.
- the dimension of the input of the convolutional layer.
- three, because an image has three color channels.
* `[ A ]`


---

**Q: Which of the following options is not an (important) idea that is leverages by convolution that can help machine learning systems?**
- Equivariant representations
- Parameter sharing
- Sparse interactions
- Full connectivity
* `[ D ]`


---

**Q: What kind of a prior do we use in convolutional neural networks?**
- We assume that images contain redundant spatial information and hence use rectangular kernels which are learnable for feature extraction
- We assume that images contain redundant spatial information and hence use subsampling layers to encode information
- We assume that images contain redundant spatial information andhence use activation function like sigmoid or Relu to constrain theoutpu
- None
* `[ A ]`


---

**Q: If you had to spot ”Wally”, a cartoon character in an image cluttered with various other cartoon characters, why would one use CNNs?**
- CNNs contain a prior that makes feature presence more important than feature location
- CNNs are good at learning weights for their filters
- CNNs  have  large  amount  of  learning  capacity  due  to  them  being”deep”
- None
* `[ A ]`


---

**Q: Choose the right option with respect to the advantages of having sparse interaction in Convolution Neural Networks**
- They make the network memory efficient
- The statistical efficiency of the network is improved
- There are fewer floating point operations 
- All of the above
* `[ D ]`


---

**Q: Choose a statement which is not applicable for CNNs**
- In deep CNN, units in the deeper layer have indirect interaction with only a small portion of the image
- Convolution and Pooling can cause underfitting if the prior knowledge implied by them is not compatible with the task they have to perform
- Pooling helps the representation to be invariant to small translations
- Convolution creates a 2-D map of where certain features appears in the input
* `[ A ]`


---

**Q: In which step does a neural network learn (update)**
- In the forward pass
- While computing Loss
- In the backward pass
- None of the above
* `[ C ]`


---

**Q: Which of the following statements is correct
1. Backpropogation is based on the derivative product rule
2. Backpropogation goes from the result nodes to the input nodes**
- Both are correct
- Only 1 is correct
- Only 2 is correct
- Both are NOT correct
* `[ C ]`


---

**Q: What is/are the main advantage(s) of pooling?**
- Approximatly invarian to small changes in the input
- Reduced memory usage
- Improvement in statistical efficiency of the network 
- all of the above
* `[ D ]`


---

**Q: What is the main advantages of concolution?**
- Sparse interactions
- parameter sharing
- equivariant representations
- all of the above
* `[ D ]`


---

**Q: Which of the following statements is true about the ReLU function applied after convolution layer in CNN ?**
- It removes all the negative values
- It removes all the noise in the image
- It reduces the size of feature map
- All of the above
* `[ A ]`


---

**Q: Unequal padding on the border is necessary when**
- Kernel size is even
- Kernel Size is odd
- Kernel is bigger than image
- Kernel size doesn’t matter
* `[ A ]`


---

**Q: The performance of a convolution filter is NOT affected by**
- the position of a feature to be detected on the input image.
- the size of a feature to be detected on the input image.
- the rotation of the feature to be detected on the input image.
- algorithm used to handle the borders of the input image.
* `[ A ]`


---

**Q: Which of the following statements is correct?**
- Engineers have to add specific constraints to the parameters of the convolutional layers to make learning more effective.
- Deep neural networks usually learn that convolution is beneficial for image processing.
- Feed forward networks are a subset of convolutional networks.
- Pooling reduces the computations with costs of additional memory consumption.
* `[ A ]`


---

**Q: Convolving a 7x7 image with a 3x3 filter with stride 2 produces  the output of size**
- 7x7
- 3x3
- 5x5
- 4x4
* `[ B ]`


---

**Q: Max pooling reduces computational cost.
Max pooling helps in making the system translation invariance**
- Statement 1 and 2 both correct
- Statement 1 and 2 both wrong
- Statement 1 is correct and Statement 2 is wrong
- Statement 1 is wrong and Statement 2 is correct
* `[ A ]`


---

**Q: Suppose an image with 128x128 pixels is fed into a convolutional layer of 5x5 matrix, what happens to the output image size?**
- 123x123
- 125x125
- 124x124
- 124x128
* `[ C ]`


---

**Q: Which one of the following about the effect of convolution and pooling on receptive field (RF) is correct?**
- Convolution allows to quickly see more of the input layer.
- Convolution increases RF multiplicatively, Pooling increases RF linearly.
- Convolution increases RF linearly, Pooling increases RF multiplicatively.
- Both of them increase RF linearly.
* `[ C ]`


---

**Q: What statement is not true?**
- Convolving an image that is scaled by .5 with a filter that is scaled by .5 produces the same result as the same image with scale 1 convolved with a filter with scale 1.
- Padding needs to be done only at the end of a conv net
- The nonlinear step in a conv net removes negative values
- Multiple layers can be passed to the next layer in a conv net
* `[ B ]`


---

**Q: What of the following statements is true:
1. A convolutional NN has more parameters than a feed forward design
2. The equivariance relation means that translation and convoluting do not commute**
- 1
- 2
- both
- neither
* `[ D ]`


---

**Q: What is the main feature of deep learning?**
- Feature extraction + decision rules
- Feature extraction + classifier
- Representation learning
- None of the above
* `[ C ]`


---

**Q: What is the definition of pooling?**
- Summarizing the outcome over a region
- Producing a set of linear activations
- Making the representation more variant to small translations
- None of the above
* `[ A ]`


---

**Q: Which values fill in kernel H that obtain an image G, identical to F translated one pixel to the top?
F(x, y) * H(x, y) = G(x, y)

3x3 kernel H
[ 0        a        0 ]       
| b        0        c |
[ 0        d        0 ]  **
- a = 1; b = 0; c = 0; d = 0
- a = 0; b = 1; c = 0; d = 0
- a = 0; b = 0; c = 1; d = 0
- a = 0; b = 0; c = 0; d = 1
* `[ D ]`


---

**Q: Mark the false justification. What is the motivation for pooling in convolutional neural networks?**
- To increase dimensionality.
- To avoid overfitting.
- Because detecting feature presence is more important than feature location, in image classification.
- To make the representation become approximately invariant to small translations of the input.
* `[ A ]`


---

**Q: When applying a convolution filter to an image, where in the image would a strong response be expected?**
- In a region with large gradients
- In a region that resembles the filter
- In a region with a lot of strong colors
- In the center of the image
* `[ B ]`


---

**Q: What does the equivariance property of convolution mean?**
- If the input shifts to the left, the output shifts to the right
- (f*g) = (g*f)
- If the input shifts to the left, the output shifts to the left 
- The values in the filter have to sum up to 1
* `[ C ]`


---

**Q: In analysis of images we have to manage with noisy data. Which of the following solutions is widely used? **
- Replace noisy pixel with one color
- Replace noisy pixel neighborhood with average of noisy pixel and neighborhood pixel
- Replace noisy pixel by neighborhood average
- Remove noisy pixel
* `[ C ]`


---

**Q: While applying to a black and white image, what is the effect of applying kernel(3x3): 
[0, 0, 0; -1, 0, 1; 0, 0, 0]**
- Image is shifted to the left
- Image is mirrored
- Image has changed color and right side of the object’s edges are white and left side of the object’s edges are black
- Image has changed color and right side of the object’s edges are black and left side of the object’s edges are white
* `[ D ]`


---

**Q: Convolution leverages three important notions that can help improve a machine learning system. Which of the following is not one of these notions.**
- Sparse interactions
- Parameter sharing
- Equivariant representations
- Pooling
* `[ D ]`


---

**Q: A typical Convolutional Neural Network layer comprises three basic components.
Which of the following is the last component of the CNN layer?**
- Pooling
- Representation
- Detector
- Convolution
* `[ A ]`


---

**Q: Using the following kernel for cross correlation, what will happen to a picture? 
Kernel: H(u,v)=[1 1 1
                           1 1 1
                           1 1 1]**
- The picture will be averaged with the pixels around a given pixel.
- The picture will be averaged with pixels around a pixel and it will get lighter (higher values)
- The picture will be averaged with pixels around a pixel and it will get darker (lower values)
- The picture will stay the same
* `[ B ]`


---

**Q: How can one store information from nodes in a layer in one picture using spatial pooling?**
- Take the averages of the nodes.
- Give each node its own color.
- Multiply each pixel of the nodes with each other.
-  Just use all nodes for the next layer. 
* `[ B ]`


---

**Q: What of the following is generally NOT a layer in convolutional networks?**
- Convolution
-  Non-Linearity
- Spatial Pooling
- Augmentation
* `[ D ]`


---

**Q: Regarding image sizes and kernel sizes, which of the following is equivalent when doing convolution?**
- Increasing image size and increasing kernel size
- Decreasing image size and decreasing kernel size
- Decreasing image size and increasing kernel size
- None of the above
* `[ C ]`


---

**Q: The kernel dimensionality of layer $k$ depends on**
- the amount of kernels used in layer $k-1$
- the kernel dimensionality of layer $k-1$
- the dimensionality of the individual feature maps of layer $k-1$
- the amount of 'colors' in the input images
* `[ A ]`


---

**Q: Which of the following statements is true: \begin{enumerate} \item Subsampling increases a neurons receptive field. \item Pooling without subsampling reduces the size of the feature maps. \end{enumerate}**
- Only 1 is true
- Only 2 is true
- Both are true
- Neither are true
* `[ A ]`


---

**Q: If the shape which I want my kernel to detect is too large, what can be done to fix this?**
- Increase the size of the kernel or increase the size of the image.
- Increase the size of the kernel or decrease the size of the image.
- Decrease the size of the kernel or increase the size of the image.
- Decrease the size of the kernel or decrease the size of the image.
* `[ B ]`


---

**Q: What's the key difference between Feed-Forward Neural Nets (FFNN) and Convolutional Neural Nets (CNN) ?**
- A CNN limits the number of parameters used in a FFNN to avoid the Curse of Dimensionality
- FFNN share parameters, whereas CNN don't
- Convolutional Neural Networks will perform much better on extremely large datasets.
- None of the Above.
* `[ A ]`


---

**Q: Choose the correct option for the given Statements:
A)Convolution is invariant to Translation of image
B)Pooling is invariant to translation of image**
- A) is correct
- B)is Correct
- Both A) and B) are Correct
- Both the statements are incorrect
* `[ C ]`


---

**Q: Choose the correct option for the given Statements:
A)Convolution Increases Receptive field 
B)Pooling Increases Receptive field multiplicatively
C)Increasing receptive field allows us to see more of the image from a given hidden layer**
- A) is correct
- B) is correct
- B) and C) is correct
- A),B) and C) all are correct
* `[ D ]`


---

**Q: What can you say about Convolutional Neural Networks (CNN) and Feed Forward Neural Networks**
- They are the same
- CNNs are always better for images 
- CNNs have less parameters (by design) than Feed Forward Neural Networks
- None of the above
* `[ C ]`


---

**Q: What is a Kernel (or Filter)?**
- It is an operator that is convolved with the input image that helps the feature extraction
- Is a matrix that can be used to identify lines in an image
- Is a vector that stores compact information of the image
- None of the above
* `[ A ]`


---

**Q: In a convolution operation between an image and a kernel, in order to detect the small objects from the image, one should:**
- Use a smaller kernel
- Use a bigger kernel
- Use a non-symmetric kernel
- None of the above
* `[ A ]`


---

**Q: In the first layer of a convolutional neural network having 3 learning filters will result in:**
- Not having any feature map because a bigger number of learning filters is requires
- Having 3 feature maps
- Having 2^3 feature maps
- Having only one feature map
* `[ B ]`


---

**Q: Which is not advantage of pooling?**
- Reducing memory
- Reduce computation
- Speed up convergence
- Make the model more accurate
* `[ D ]`


---

**Q: What is the difference between feed-forward and convolutional networks?**
- CNN is by design a limited parameter version of feed forward
- Feed-forward is much better
- Convolutional networks is much better
- Non of them is true
* `[ A ]`


---

**Q: When applied to an image, what is the effect of the following kernel: [[0,0,0], [0,0,1], [0,0,0]]?**
- It blurs the input image.
- It detects edges in the input image.
- It shifts the image to the left.
- It has no effect, the input image stays the same.
* `[ C ]`


---

**Q: When it comes to the principles and mechanisms behind Convolutional Neural Networks, which of the following statements is FALSE?**
- Pooling is approximately invariant to local translations.
- Convolutional Neural Networks care about the exact location of an object in the image.
- Convolutional Neural Networks are by design a limited parameter version of feed forward networks.
- One of the advantages of using pooling is the memory reduction.
* `[ B ]`


---

**Q: Suppose we feed a convolutional neural network with a 32x32 image, and the first layer presents 7 filters 3x3 with stride 1 and padding 1; what dimensions will the output of the first layer have?**
- 32x32
- 32x7
- 30x30x7
- 32x32x7
* `[ D ]`


---

**Q: Which of the following statements about the Toeplitz matrix (or diagonal-constant matrix) is FALSE?**
- It is sparse
- It is local (non-zero values occur next to each other)
- It is always invertible
- The non-zero values are always the same parameters repeating in different positions
* `[ C ]`


---

**Q: How many border pixels would be lost (in total) during a convolution, if using a 5x5 filter on a 100x100 px image?**
- 400
- 800
- 784
- 396
* `[ C ]`


---

**Q: What is the usual ordering of operations in a convolutional layer?**
- Non-Linear Activation -> Convolution ->  Spatial Pooling
- Non-Linear Activation -> Convolution ->  Sub-Sample -> Spatial Pooling
- Convolution -> Sub-Sample -> Non-Linear Activation -> Spatial Pooling
- Convolution -> Non-Linear Activation -> Spatial Pooling
* `[ D ]`


---

**Q: What do the weights of a kernel in a CNN learn/represent?**
- Lines in an image (a linear approximation of edges)
- Edges in an image (including curved lines)
- Features in an image at various levels of abstraction
- A smoothening filter to improve quality of the image 
* `[ C ]`


---

**Q: What are the benefits of subsampling or pooling?**
- It increases the receptive field multiplicatively (of a neuron in a higher layer so that it can see more of the input image).
- It increases the receptive field linearly (of a neuron in a higher layer so that it can see more of the input image).
- Reduces memory requirement in addition to Option A.
-  Reduces memory requirement in addition to Option B.
* `[ C ]`


---

**Q: If we apply a convolution to an image with a 3x3 filter with all zeros and a 1 in the (3, 2) position (middle right) what would be the resulting image?**
-  the image will not change 
- The image will be shifter by one pixel to the left
- The image will be blurred 
- The image will be shifter by one pixel to the right 
* `[ B ]`


---

**Q: If the first layer of a convolutional NN contains 3 convolution filters, what will be the number of “channels" encoded into the image produced at the output of this layer?**
- 1
- 6
- 3
- 4
* `[ C ]`


---

**Q: Given an expected accuracy, how does the typical parameter budget in Feed Forward Network compare to Convolutional Neural Network?**
- Parameter Budget in CNN > Parameter Budget in Feed Forward Network
- Parameter Budget in CNN < Parameter Budget in Feed Forward Network
- Parameter Budget in CNN = Parameter Budget in Feed Forward Network
- Cannot be generalised
* `[ B ]`


---

**Q: Convolution improves on the feed forward network, via**
- Sparse Interactions
- Parameter Sharing
- Equivariant representation
- All of above
* `[ D ]`


---

**Q: In representation learning, what are kernels used for?**
- Kernels are used to crop images to preferred size
- Kernels are used a feature detectors.
- Kernels are used to convolve images
- Kernels are not used in representation learning
* `[ B ]`


---

**Q: What is not a result of the equivariance principle?**
- If the input shifts to the left, the output shifts to the left
- First convolving and then translating, is the same as first translating and then convolving.
- Object detection is invariant to changes in position
- The outcome is summarized over the region
* `[ D ]`


---

**Q: What is the difference between padding and striding?**
- With padding you add equal data so the kernel can look at extreme edges, where with striding you reduce the size of the next layer so the output values in the activation will be more independent of neighboring values.
- With padding you reduce the size of the next layer so the output values in the activation will be more independent of neighboring values, where with striding you add equal data so the kernel can look at extreme edges
- With padding you add random data so the kernel can look at extreme edges, where with striding you increase the size of the next layer so the output values in the activation will be more independent of neighboring values.
- With padding you increase the size of the next layer so the output values in the activation will be more independent of neighboring values, where with striding you add random data so the kernel can look at extreme edges
* `[ A ]`


---

**Q: What does the term equivariance mean?**
- When a transformation is applied to the input of a function a, the same transformation is applied to the output of function b.
- When a transformation is applied to the input of function a, the same transformation is applied to the output of function a.
- It means that 2 functions are equivariant if they have the same output, given the same input.
- It means that 2 functions are equivariant if the have the same output, given any input.
* `[ B ]`


---

**Q: If you have 24x24x8 input image and you apply maxpooling with stride 2 and filter size 2. What will be the output dimension**
- 12x12x8
- 12x12x4
- 16x16x8
- 16x16x4
* `[ A ]`


---

**Q: What will be the result of applying convolution to a 32x32x8 image with 32 filters of size 2 and stride 2**
- 16 x 16 x 32
- 24 x 24 x 8
- 16 x 16 x 8
- 24 x 24 x 32
* `[ A ]`


---

**Q: What is the result of applying a kernel to an image?**
- Identifies parts of the image that resemble the kernel
- Removes parts of the image that resemble the kernel
- Separates the image into its different shades
- Replaces sectors of an image with the maximum value within that sector
* `[ A ]`


---

**Q: Which statement is not true about max-pooling?**
- Pooling summarizes the outcome over a region
- Keeps the most important features of the image
- Increases the resolution of an image by up-sampling
- Approximately invariant to local translations (Identifies features, rather than position)
* `[ C ]`


---

**Q: Which of the following is true?**
- Convolution involves flipping the kernel while cross-correlation doesn’t
- Both convolution and cross-correlation involve flipping the kernel
- Convolution doesn’t involve flipping the kernel while cross-correlation does
- Both convolution and cross-correlation doesn’t involve flipping the kernel 
* `[ A ]`


---

**Q: Which of the following statements about pooling is/are true?

A. A pooling function replaces the output of the net at a certain location with a summary statistic of the nearby outputs.
B. Pooling helps to make the representation approximately invariant to small translations of the input.**
- A
- B
- A and B
- None of the above
* `[ C ]`


---

**Q: Given an image of 100x100 pixels , what it would be its size if we apply a 2x2 window for the spatial pooling layer?**
- 50x50 pixels
- 75x75 pixels
- 100x100 pixels
- 25x25 pixels
* `[ A ]`


---

**Q: Which of the following statements about the receptive field of a neural network is true?**
- Convolution increases the receptive field linearly
- Pooling increases the receptive field linearly
- Convolution increases the receptive field multiplicatively
- None of the above
* `[ A ]`


---

**Q: If we convolve a 7x7 image with a 3x3 kernel. Assume that the stride = 1 and we are padding the image only one time. What is the image output dimensions?**
- 7x7
- 6x6
- 5x5
- 8x8
* `[ A ]`


---

**Q: How many hyperparameter ha a max pooling layer?**
- None
- 1
- 2
- 3
* `[ D ]`


---

**Q: By using a kernel with positive values on the left and equally negative values on the right, we can..**
- Shift the image
- Scale the image
- Highlight borders
- None of the above
* `[ C ]`


---

**Q: What is the difference between feed-forward and convolutional networks?**
- CNN allows for more parameters
-  Feed-forward networks use an activation function to the result
-  CNN have a backward link to the neurons in the previous layer 
- None of the above
* `[ D ]`


---

**Q: what is a property of kernel weights**
- they are not an abstraction of a moving neighbourhood average
- they have an associated color
- they are feature detectors
- none of the above
* `[ C ]`


---

**Q: what is the purpose of spatial spooling in a convolutional network?**
- to pass a learned kernel over the original input
- to remove negative values from the kernelized processed original input
- to allow combining multiple channels into one output
- None of the above
* `[ C ]`


---

**Q: In which order can Deep Learning best be represented?**
- 1) Input, 2) Representation learning, 3) Output
- 1) Input, 2) Feature extraction, 3) Decision rules, 4) Output
- 1) Input, 2) Feature extraction, 3) classifier, 4) Output
- 1) Input, 2) classifier, 3) Feature extraction, 4) Output
* `[ A ]`


---

**Q: What does equivariance mean? Let g be a translation and f be a convolution.**
- When equivariance occurs, either translation or convolution happens, but never both at the same time
- First translating and then convolving is the same as first convolving and then translating
- First translating and then convolving is NOT the same as first convolving and then translating 
- First convolving and then translating is the one and only correct order
* `[ B ]`


---

**Q: Given is a convolutional neural network that adds a padding in each layer: 5 values are added on the left side and 4 are added on the right side. What is the width of the convolutional kernel?**
- 11
- 10
- 9
- 8
* `[ B ]`


---

**Q: Given is that convolution is written as g(x) and translation as f(x). Which of the following mathematical statements is correct about equivariance? **
- f(g(x)) = g(f(x))
- f(g(x)) = g(x) * f(x)
- f(g(x)) = f(x) - g(x)
- f(g(x)) = f(g(x)) - g(x)
* `[ A ]`


---

**Q: What is relationship between kernel size and field of view of neuron in CNN?**
- No relationship
- Higher kernel size, greater field of view
- Higher kernel size, smaller field of view
- Non-monotonic relationship
* `[ B ]`


---

**Q: What is the difference between effect of convolution vs pooling on receptive field size?**
- Convolution increases receptive field linearly, pooling increases receptive field multiplicatively
- Convolution increases receptive field multiplicatively, pooling increases receptive field linearly
- No difference, both increase receptive field size linearly/multiplicatively
- Both don't affect receptive field size
* `[ A ]`


---

**Q: The convolutional neural network:**
- Is fully connected
- Is more flexible than a feed forward network
- Is a version of feed forward with fewer parameters
- Is not robust to different positions of the same object in an image
* `[ C ]`


---

**Q: Pooling:**
- Reduces the number of outcomes from a region
- Increases the reception field linearly
- Is independent on the input size
- Does not affect the size of the original image
* `[ A ]`


---

**Q: A company, wind rose, has asked us to create a system that recognizes roses in pictures. We are given a lot of images that are labelled with either 'roses' and 'not roses'. After using these images for training it turns out we clearly don't have enough samples! What can we do in order to improve the results?**
- Re-use the training images by applying transformations such as rotation and shifting on the images, this will generate new training data.
- Adding noise to the training images would help possibly improve the error in the validation set.
- Both A and B
- Neither A or B
* `[ C ]`


---

**Q: Which 3 by 3 kernel will, when applied to an image, yield the exact same image**
- The 3x3 Identity Matrix
- A 3x3 matrix with all 1's
- A 3x3 matrix with all 0's
- A 3x3 matrix M with such that M(2,2) = 1 (the center cell is equal to 1)
* `[ D ]`


---

**Q: What is the dimension of the convolved image if the size of initial image is (n x n) and filter size is (m x m)?**
- m x m
- (n+m) x (n+m)
- (n-m+1) x (n-m+1)
- (n-m) x (n-m)
* `[ C ]`


---

**Q: Why is it better to use odd size of the kernel (filter)?**
- padding becomes equal
- padding becomes unequal
- the learning process becomes quicker
- helps not to shrink the image
* `[ A ]`


---

**Q: What’s the importance of the Rectified Linear activation function (ReLu)?**
- Detects valuable information (features) about the image
- Is used before the pooling layer and transform the CNN to a classifier very good results for binary linearly separable classification problems
- All the values are limited using a threshold
- All the negative values are set to zero, something that make the back propagation algorithm robust about the gradient calculations
* `[ D ]`


---

**Q: CNNs response well when the classifier:**
- Deals with sequential data
- Detects specific objects in a video
- Is used to align two images when the second image is an affine transformation
- None of the above
* `[ B ]`


---

**Q: What is meant with the term End-to-End learning?**
- That with the use of the forward step the output can be calculated from the input
- That one uses the representation of a picture as input in a network
- That even though each layer has a different output the network still finds a prediction in the end
- That the output of a network is used to learn the feature extraction
* `[ D ]`


---

**Q: What is not a feature of the Toeplitz matrix?**
- It has sharing parameters
- It is PSD
- It is local
- It is sparse
* `[ B ]`


---

**Q: Which of these statements about the learning rate is incorrect?**
- The learning rate is arguably the most important parameter
- It it is not useful to have different learning rates on different axes
- Decay is reducing the learning rate over time
- A low learning rate might take a long time to find the instantiation
* `[ B ]`


---

**Q: Which of these statements concerning convolutional neural networks is incorrect?**
- A convolutional neural network is a limited parameter version of a feed-forward network
- Smaller images will result in larger kernels
- Adding padding to an input image will decrease the size of the output image after convolution
- Reducing memory management is an advantage of pooling
* `[ C ]`


---

**Q: When applying a moving neighborhood average, what problem occurs?**
- The edges of your training data get removed
- Rounding errors
- This average is not the true average over the observed patch
- More parameters need to be optimised
* `[ A ]`


---

**Q: A convolution of an image with the matrix [[0 0 -1] [0 0 0] [1 0 0]] will:**
- Mirror the image diagonally
- Give high output for all diagonal edges
- Give low output for all diagonal edges
- None of the above
* `[ D ]`


---

**Q:     What is the kernel of a cnn?**
- The programming language it is written in
- A vector describing the layout of the network
- A filter that is moved across a image
- An optimization method that decreases the amount of required nodes
* `[ C ]`


---

**Q: What is the main use of a cnn?**
- Deciding whether an object is present in an image
- Where an object is in an image
- Fixing damaged images
- Making images more machine readable
* `[ A ]`


---

**Q: 1.	[51, 48, 32, 44, -88] * [-1,1] =**
- [99, 80, 76,-44]
- [-3, 16, -12, 132]
- [-3, -16, 12, -132]
- [-51, -48, -32, -44, 88]
* `[ C ]`


---

**Q: 2.	[1, 2 , 6, 2, 3, 3, 1 ,5] * [1, 0, 0, -2], If this convolution is written as a matrix multiplicitation, how many rows and columns would the matrix have?**
- 4 rows and 4 columns
- 8 rows and 4 columns
- 4 rows and 8 columns
- 8 rows and 8 columns
* `[ B ]`


---

**Q: Which of the following statements holds true for the convolutional neural networks?**
- Parameter sharing is mainly a depth-wise property in the sense that a deeper convolutional layer may share some parameters with the a more shallow convolutional layer
- The depth of the output of a convolutional layer depends on the size of the convolution kernels
- The receptive field of a neuron in a deeper convolutional layer is larger than the receptive field of a neuron in a preceding (more shallow) layer
- Convolutional layers may scale or rotate the original image
* `[ C ]`


---

**Q: A set of 32x32x3 training images is drawn from CIFAR-10. We construct a shallow convolutional neural network. The network comprises a Convolutional Layer with 8 filters, 4x4 kernels, zero padding size of 2 and stride 2. This layer precedes a 1x1 FC ReLU unit and 4x4 max pooling layer. The output of the polling layer ends in a FC layer which classifies the image among 10 classes. Which are the dimensions of the output volume that enters the final FC layer?**
- 16x16x4
- 7x7x4
- 7x7x8
- 16x16x10
* `[ C ]`


---

**Q: What is the purpose of a pooling layer?**
- A pooling layer is used to smooth out pictures
- A Pooling layer is used to reduce the size of an image
- A pooling layer is used to cancel outlier behaviour in images
- A pooling layer is used to improve the qualuty of a picture
* `[ B ]`


---

**Q: What is not an advantages of a CNN over a normal neural network**
- A CNN uses filters which can detect features likes edges, circles etc...
- A CNN uses less computational memory then a full neural network
- A CNN converges faster when sampling images than a neural network
- A CNN is doesnt scale as good as a neural network does.
* `[ D ]`


---

**Q: What is the main use of convolution kernels**
- Highighting / detecting the occurrence of patterns in an image
- Subsampling the image to reduce computational load
- Speeding up computation of derivatives
- Subsampling the image to spread local patterns to deeper layers
* `[ A ]`


---

**Q: How does convolution differ from correlation?**
- The kernel is mirrored in the convolution operation
- Convolution is differentiable
- Convolution does not shrink the input image
- Convolution has a smaller time complexity
* `[ A ]`


---

**Q: What is a valid gausian blur kernel (3x3) (not normalized)**
- {0,0,0;0,1,0;0,0,0}
- {1,2,1;1,2,1;1,2,1}
- {1,2,1;2,4,2;1,2,1}
- {0,-1,0;-1,5,-1;0,-1,0}
* `[ C ]`


---

**Q: Why can less parameters be a good thing.**
- less flexibility
- Curse of dimensionality (each parameter has to be learned)
- underpreforming
- better tunable 
* `[ B ]`


---

**Q: Which statements about pooling are true?
I. The size is reduced by subsampling stride to the power of dimensionality.
II. Adapt pool can be used to get a fixed size output with a varying input size.**
- Only statement I is true
- Only statement II is true
- Both statements are false
- Both statements are true
* `[ D ]`


---

**Q: For a two convolution layers, where the second layer has size 3 and the first layer has size 5 with stride 3. How much neurons of the first layer are used for 1 output neuron?**
- 5
- 15
- 11
- 12
* `[ C ]`


---

**Q: Statement 1: For convolution to generalize to a kernel the weights variables need te be summed per pixel and then multiplied. 
Statement 2: Kernel weights are feature detectors**
- Both statements are true
- Statement 1 is true
- Statement 2 is true 
- None of the statements are true
* `[ C ]`


---

**Q: Statement 1: A CNN consists of: input, learned filters, convolution, spatial pooling and then the next layer.
Statement 2: CNN is by design a limited parameter version of feed forward.**
- Both statements are true
- Statement 1 is true
- Statement 2 is true
- None of the statements are true
* `[ C ]`


---

**Q: What is the final layer of a ConNet?**
- Convolutional layer
- Pooling layer
- Fully Conncected layer
- Normalization layer
* `[ C ]`


---

**Q: In a ConvNets what happens to the image volume after the RELU layer?**
- increases
- remains constant
- decreases
- depends by the image
* `[ B ]`


---

**Q: Which of the following are true about back propagation?**
- It is the whole learning algorithm for multi layer neural networks
- It refers to the method for computing the gradient and can compute derivatives of any function
- It is speciﬁc to multi-layer neural networks
- None
* `[ B ]`


---

**Q: What does the variable in a node of a computational graph indicate?**
- Scalars and vectors
- Matrix
- Tensor
- Any of the above
* `[ D ]`


---

**Q: What happens if you multiply a certain picture with kernel H?

\begin{equation}
	H = \begin{bmatrix} 0&0&0\\0&0&1\\0&0&0 \end{bmatrix}
\end{equation}**
- It moves to the right.
- It moves to the left.
- It moves up.
- It moves down.
* `[ B ]`


---

**Q: How can one prevent the shrinking phenomenon during a convolution?**
- Padding.
- Subtracting.
- Adding.
- None of the above.
* `[ A ]`


---

**Q: Which following statement about CNN is false?**
- Convolutional networks usually have sparse connectivity because the kernel is smaller than the input.
- Convolutional networks could leverage parameter sharing to reduce computational costs and improve efficiency. This is done by learning only one set of parameters instead of separate sets.
- Convolutional networks are always equivalent to transformations.
- Convolutional networks are designed for processing grid-like data.
* `[ C ]`


---

**Q: Which following statement about pooling is false?**
- Pooling replaces the output of the net at a certain location with a summary statistic of the nearby outputs.
- Pooling helps to make the representation become approximately invariant to small translations of the input. 
- Pooling is essential for handling inputs of varying size. 
-  When doing pooling with downsampling, the rightmost units of current layer could have a smaller pooling size than others and could be omitted.
* `[ D ]`


---

**Q: How do convolutional neural networks gain some invariance to translations of their input?**
- By using max pooling
- By using convolutional layers
- Due to the fully connected layer at the end of the network
- They learn to undo translations
* `[ A ]`


---

**Q: What is meant by the "receptive field" of a neuron in a CNN?**
- All the pixels from the input image that have an effect on this neuron
- The neurons in the layer below this one that are connected to it
- The moving "square" on the input image when doing convolution
- A convolutional filter
* `[ A ]`


---

**Q: Which of the following is not an advantage of pooling in a Convolutional Neural Network?**
- It provides a speed up since receptive field can see more of the image
- It reduces memory requirements since dimensionality is reduced
- It is computationally less expensive
- None of the above
* `[ D ]`


---

**Q: "Pooling is invariant to local transformations in the image". For image classification why is this NOT a disadvantage**
- It is easier to ignore local transformations as it reduces computational intensity
- In image classification, presence of features is more important than their location
- It is a disadvantage since information is potentially lost.
- None of the above
* `[ B ]`


---

**Q: Select statements  that motivate the usage of the CNN instead of Regular Neural Networks;

Statement 1 Regular Neural Nets don’t scale well to full images.

Statement 2 Convolutional Neural Networks take advantage of the fact that the input consists of images and they constrain the architecture in a more sensible way.

Statement 3 They are great for capturing local information (eg neighbor pixels in an image or surrounding words in a text).

Statement 4 CNN Needs more samples than simple neural network so it reduces chance of overfitting.**
- All statements are true
- 2 3 and 4
- 1 and 3
- 1 2 and 3
* `[ D ]`


---

**Q: In typical architecture of the CNN three hyperparameters control the size of the output volume: the depth, stride and zero-padding. We can compute the spatial size of the output volume as a function of the input volume size (W), the receptive field size of the Conv Layer neurons (F), the stride with which they are applied (S), and the amount of zero padding used (P) on the border. For example for a 7x7 input and a 3x3 filter with stride 1 and pad 0 we would get output of size?**
- 3x5
- 3x7
- 7x7
- 5x5
* `[ D ]`


---

**Q: which of the following statements is wrong?**
- Convolution is an operation on two functions of a real-valued argument.
- Convolution has commutative property. 
- Cross-correlation is different with  convolution since it is with ﬂipping the kernel. 
- It is rare for convolution to be used alone in machine learning, instead convolution is used simultaneously with other functions
* `[ C ]`


---

**Q: which of the following statements is wrong?**
- Convolutional networks with sparse interactions are accomplished by making the kernel smaller than the input.
- A pooling function replaces the output of the net at a certain location with asummary statistic of the nearby outputs.
- Pooling helps to make the representation approximatelyinvariantto small translations of the input. 
- Because pooling summarizes the responses over a whole neighborhood, it is not possible to use fewer pooling units than detector units
* `[ D ]`


---

**Q: What is true about spatial pooling?**
- You combine the average in spatial pooling.
- You add the feature maps to eacht other.
- You multiple weights and add them together.
- Concatenate different colors (layers) to a image with colors. 
* `[ D ]`


---

**Q: What is true about the feedforward vs. convolution neuron networks?**
- Feedforward is used when the location is important.  
- Feedforward is less data needed then a convolution neuron network.
- Feedforward uses the property shared parameters.
- Feedforward use sparcial matrix to compute the convolutions.
* `[ A ]`


---

**Q: Which of these steps does belong to a deep CNN?**
- Feature extraction
- Classifying based on decision rules 
- Representation Learning
- Classifying with the help of a Support Vector Machine
* `[ C ]`


---

**Q: What is the biggest benefit of pooling in combination with subsampling?**
- The amount of computational power needed is greatly reduced
- The amount of memory needed is greatly reduced
- A CNN that uses pooling with subsampling is more reliable
- Pooling in combination with subsampling provides a network with a sparser connectivity
* `[ B ]`


---

**Q: Backpropagation is ...**
- ... a mathematical operator.
- ... a software package in linux. 
- ... an algorithm.
- ... something to predict the trajectory of your grandmother.
* `[ C ]`


---

**Q: Given variable w, dependent on x,y,z. The derivative of function f(w) with respect to w df/dz is equal to:**
- df/dw*dw/dx*dx/dy*dy/dz
- df/dw*dw/dz
- df/dx
- df/dw*dw/dz*dz/dx
* `[ B ]`


---

**Q: Which of the below mentioned statements are true for chain rule?**
- Chain rule reduces the computation efficiency of the Loss function.
- Chain rule makes the calculations easier by dividing the derivative into multiple independent components and combining them post calculations.
- Both A and B
- None of them
* `[ B ]`


---

**Q: Which of the statements is true about back-propagation?**
- Each node aggregates the error from it's parent.
- Each node aggregates the error from it's children.
- Each node aggregates the error from both it's parent and children.
- None of the above.
* `[ B ]`


---

**Q: Back propagation is used to calculate**
- Gradients
- Products
- Sums
- Integrals
* `[ A ]`


---

**Q: Back propagation is based on **
- Linear algebra
- Product rule
- Fundamental theorem of calculus
- Bayes statistics
* `[ B ]`


---

**Q: Which of following statements is FALSE regarding the convolution talked in the lecture?**
- Convolution is able to get rid of noisy pixels.
- A image convolved by an identity filter would remain the same.
- Convolution is able to represent edges using some specific filters.
- Convolution is usually use the end of a CNN.
* `[ D ]`


---

**Q: Which of following statements is FALSE regarding the techniques used in CNN?**
- Convolution has less parameters to learn than standard neural network thus it is not as powerful as a full connected neural network.
- Convolution will not outperform a fully connected neural network if there's infinite number of training data.
- Pooling is approximately invariant to local translations.
- Reducing memory is a great advantage of pooling.
* `[ A ]`


---

**Q: Suppose we got a network with two convolutional layers filter size 3 and an input layer (Such that cnn2(cnn1(input))). What happens to the receptive field of cnn2 if we increase the stride of cnn1 **
- Nothing will change
- The receptive field will increase
- The receptive field will decrease
- None of the above
* `[ B ]`


---

**Q: Which of the following statements is correct:**
- Increasing the filter size can affect the receptive field on neuron future layers
- Stride determines the filter size
- Max Pooling selects the average value
- None of the above
* `[ A ]`


---

**Q: Which of the following is not an idea on how convolution improves a machine learning system?**
- Sparse interactions
- Equivariant representations
- Parameter sharing
- Less significance of the curse of dimensionality
* `[ D ]`


---

**Q: Pooling over spatial regions provides invariance to?**
- Rotation
- Scale
- Translation
- None of the above
* `[ C ]`


---

**Q: Which of the following is not used in the CNN?**
- Convolutional
- Pooling
- All
- none
* `[ C ]`


---

**Q: What is the important of pooling in CNN**
- summarizes output
- classify
- Extract features
- all
* `[ A ]`


---

**Q: What is the result of the [1 0 -1]t (vertical vector) convolution on an image?**
- Enanchement of the horizontal borders, direction left to right
- Enanchement of the horizontal borders, direction right to left
- Enanchement of the vertical borders, direction bottom to top
- Enanchement of the vertical borders, direction top to bottom
* `[ C ]`


---

**Q: What is usually a not a feature of the Toeplitz matrix?**
- Sparse (many zeros)
- Local (non-zero values occur next to each other)
- Sharing parameters (same values repeat)
- Square (same vertical and horizontal dimension)
* `[ D ]`


---

**Q: Which statement is wrong?**
- CNN is a limited parameter version of feed forward
- CNN is less flexible than feed forward in terms of parameters
- Less parameters could be beneficial because of the curse of dimensionality
- CNN has less parameters than feedforward by accident
* `[ D ]`


---

**Q: Which statement is true?**
- Pooling averages the outcome over a region
- Pooling is approximately invariant to local translations
- Pooling is important for image classification because feature location is more important than feature presence
- Convolution increases RF multiplicatively, Pooling increases RF linearly
* `[ B ]`


---

**Q: Complete the following sentence. In a "1-d" gradient descent for a multi-layer network:**
- The a local minimum is the global minimum and there are multiple coordinates for each bias in all the layers.
- The a local minimum is the global minimum and there are multiple coordinates for each weight only of the some layers.
- The a local minimum is not necessarily the global minimum and there are multiple coordinates for each bias in all the layers.
- The a local minimum is not necessarily the global minimum and there are multiple coordinates for each bias in some of the the layers.
* `[ A ]`


---

**Q: What is the application of backpropagation in deep neural networks?**
- Efficient computation of the quotient rule in error propagation.
- Transferring the output of a deep neural network back into the input layer.
- Adaptation of the neuron thresholds in the hidden layer
- Efficient computation of the error gradients.
* `[ D ]`


---

**Q: Which step does end-to-end learning skip compared to normal machine learning approaches?**
- Feature extraction
- Data preprocessing
- Data collecting
- None of the above
* `[ A ]`


---

**Q: What are the advantages of pooling?**
- Care about feature presence more than feature location
- Invariant to feature location difference between images
- Reduce memory
- All of the above
* `[ D ]`


---

**Q: What is a downside of removing noisy pixels by using the neighbourhood average?**
- Lower resolution due to losing the boundary of the data.
- Possible errors due to division by 0.
- Computationally unmanageable. 
- This method does nothing to remove noisy pixels.
* `[ A ]`


---

**Q: What is a possible solution to the loss of the boundary when using the neighbourhood average method?**
- Mirroring the neighbour values.
- Copying the border from the unaltered data.
- Neither option works.
- Options A and B could work.
* `[ D ]`


---

**Q: Which of the following statements about backpropagation is false?**
- The idea behind backpropagation is to share the repeated computations of the calculation of the derivatives using the chain rule wherever possible.
- The computation graph of the backpropagation is similar to the network architecture. The nodes correspond to values that are computed, rather than to units in the network. 
- The backpropagation algorithm is modular: it is broken down in small chunks that can be reused for other computations. 
- The  backprop method for the univariate neural net can be used in a similar way for a multivariate neural net. 
* `[ B ]`


---

**Q: What is the right order of the steps of the backpropagation algorithm to compute gradients of node n_n . The steps are (random itemized):
1.	Create topological ordering of graph
2.	For i = N-1,…1: \bar{n}_i = \sum_{n_j \in Children(n_i)} \bar{n_j} \frac{\partial n_j}{\partial n_i}
3.	\bar{n}_N = 1 (gradient of a node with respect to itself is 1; \frac{df}{df}=1)
4.	For i=1,2,..,N: Evaluate n_i using its function f^(i)(n_i)**
- 1,2,3,4
- 2,3,1,4
- 1,3,4,2
- 1,4,3,2
* `[ C ]`


---

**Q: What is better?**
- More info needed for all of the below
- Forward propagation
- Backward propagation
- FW+bw combined
* `[ A ]`


---

**Q: How well does the chain rule preform in higher dimensions?**
- ++
- +
- -
- --
* `[ A ]`


---

**Q: When a function is equivariant and the input is shifted to the right…**
- The output is shifted equally to the left.
- The output is shifted equally to the right.
- The output is shifted twice as far to the left.
- The output is shifted twice as far to the right.
* `[ B ]`


---

**Q: What is NOT a benefit of pooling?**
- Pooling greatly reduces memory.
- The receptive field size increases multiplicatively.
- Pooling allows to ‘see’ more of the image.
- By pooling the convolutional network can handle more complex features.
* `[ D ]`


---

**Q: What is the goal of pooling in Convolutional Neural Networks?**
- Increase the total number of parameters for higher accuracy.
- Introduce slightly more overfitting in return for much more efficient computation.
- Reduce overfitting and lowering the computational complexity of the CNN model.
- Increasing the amount of available data for the training process.
* `[ C ]`


---

**Q: What is the effect of increasing the stride of a convolutional filter on the receptive field?**
- It increases the receptive field, but decreases the amount of data available after applying the CNN.
- It decreases the receptive field, but increases the amount of data available after applying the CNN.
- It increases the receptive field, and increases the amount of data available after applying the CNN.
- It decreases the receptive field, and decreases the amount of data available after applying the CNN.
* `[ A ]`


---

**Q: What is a typical order of layers for a CNN (from input to output)?**
- spatial-pooling, convolution, non-linearity, learned-filters
-  convolution, non-linearity, spatial-pooling, learned-filters
- learned-filters, convolution, non-linearity, spatial-pooling
- CNN nets don't contain convolution layers
* `[ C ]`


---

**Q: CNN's are a limited parameter version of feed forward networks. Why is this an advantage?**
- Curse of dimensionality
- Avoiding bias
- less parameters meen a better result
- back-propagation is not required
* `[ A ]`


---

**Q: How does a convolution differ from a simple 'moving neighborhood average'?**
- A convolution has more parameters  (than only window size) and can have any values as kernel matrix.
- A convolution uses a gaussian kernel to compute the averages.
- A convolution also works in higher dimensions (whilst the moving neighborhood average is limited to 2 dimensions).
- A convolution is a neighborhood average that does not move (moving neighborhood averages use a sliding window technique).
* `[ A ]`


---

**Q: How does a convolutional neural network deal with the problem of keeping spatial information as compared to a standard feed-forward network?**
- The network connects each pixel of the input image to a distinct node in the next layer, giving a 1-to-1 mapping.
- The nodes in the convolutional layer are not fully connected to the previous layer, instead they only receive localized information.
- Spatial information is not needed in the neural network because the network will learn the spatial information autonomously.
- Each pixel is stored with its original (x,y) coordinates and this information is given to the network during the first forward pass.
* `[ B ]`


---

**Q: Which one is true regarding a convolution with size 5?**
- Kernel is even, unequal border
- Kernel is odd, unequal border
- Kernel is even, equal border
- Kernel is odd, equal border
* `[ D ]`


---

**Q: Which one can the convolution operation can handle? 
I. Scale II. Location III. Color**
- I-II
- I-III
- II-III
- I-II-III
* `[ C ]`


---

**Q: When applying a 5x5 moving neighbourhood average to a picture of 10x10 pixels, how many pixels remain in the filtered image?**
- 100
- 81
- 64
- 36
* `[ D ]`


---

**Q: What is the implication of equivariance while transformating?**
- Transformation of the input is detectable in the output
- Transformation of the input is not detectable in the output
- Transformation of the input can or cannot be detectable in the output
- -
* `[ A ]`


---

**Q: Three size 3 convolution layers and one input image (bottom). Let the last layer do a convolution with stride 2 (sub-sample), then how much does one output 'see' of the image**
- 6
- 7
- 8
- 9
* `[ D ]`


---

**Q: [100;101;99;200;199;201]*[-1;0;+1] =**
- [-1;99;100;1]
- [1;99;100;1]
- [-1;99;100;-1]
- [1;99;100;-1]
* `[ A ]`


---

**Q: The second derivative of a quadratic function is calculated. Which of the following is not true?**
- If the second derivative is zero, there is no curvature.
- If it is negative, the function curves downward.
- If it is positive, the function increases with decrease in the variable.
- None of the above
* `[ C ]`


---

**Q: In multiple directions the there is a second derivative for each direction at a single point. The condition number of the Hessian measures how much the second derivatives differ from each other. When the Hessian has a poor condition number**
- The gradient increases slowly in all directions.
- The gradient increases rapidly in all directions
- The gradient increases rapidly in one direction and slowly in the other
- None of the above
* `[ C ]`


---

**Q: Consider a single output neuron and before that, two size three convolution layers with a stride of one. how many pixels could the output neuron "see"?**
- 9
- 27
- 7
- 5
* `[ C ]`


---

**Q: What is the output to the following convolution in a cnn: [10, 11, 9, 20, 19] * [-1, 1] **
- [1, -2, 11, -1]
- [-1, 2, -11, 1]
- [1, -11, 2, -1]
- [-1, 11, -2, 1]
* `[ A ]`


---

**Q: Which one is wrong**
- With backprop, the early layers train faster than the later ones, making the early layers incapable of accurately identifying the pattern building blocks needed to decipher the full pattern.
- If all weights are equal, nodes will learn the same thing during backpropagation, this limits the capacity	
- To do backpropagation, it needs to keep the output values of each layer, until you get to the backpropagation step.
- If you want to learn a relationship between an input and an output, you need to optimize jointly the iteration that takes in the output, and the iterations that give the output.
* `[ A ]`


---

**Q: which one is wrong**
- The idea behind backpropagation is to share the repeated computations wherever possible
- Backpropagation  is a way of computing the partial derivatives of a loss function with respect to the parameters of a network
- With backprop, the early layers train slower than the later ones, making the early layers incapable of accurately identifying the pattern building blocks needed to decipher the full pattern.
- the number of iterations you can optimize jointly, is not restricted by the computer memory
* `[ D ]`


---

