# Questions from `paper_6` :robot: 

**Q: In the case of bio-medical data, when little image data is available to train on, which of the following is false:**
- Rotating the images can improve performance significantly.
- Applying elastic deformation on training images to generate more training data can improve performance significantly.
- Reducing the size of the pooling filter can improve performance significantly 
- Using smaller/less feature channels can improve performance significantly.
* `[ C ]`


---

**Q: Suppose you have the task of classifying pictures fruits. In particular you want to differentiate between peaches, apples, citrons and lemons. As you don’t have enough data you are going to perform data augmentation. Which of the following augmentation strategies would not be appropriate for this problem?**
- Rotation.
- Noise.
- Color.
- Obstruction.
* `[ C ]`


---

**Q: What is Data Augmentation?**
- Set weights to both image recognition algorithms (usually a CNN) and training samples (images) in a way that forces the model to concentrate on observations that are difficult to correctly recognize.
- Instead of training the model repeatedly, select images the model incorrectly labeled and train the model specifically on this data. To avoid overfitting the model is afterwards trained on all training images again.
- Transformations on images to improve the generalization of a model and to overcome the problem of a small amount of training data.
- None of the answer above
* `[ C ]`


---

**Q: Which of the following about the U-net architecture is FALSE?**
- The architecture consists of an expansive and a contracting path.
- The number of feature channels is doubled at each upsampling step. 
- The large number of feature channels in the upsampling part enables the network to propagate context information to higher resolution layers
- The network does not have any fully connected layers.
* `[ B ]`


---

**Q: The term localization in the context of image processing refers to:**
- Assigning a class label to each pixel.
- Focusing on specific parts of the image.
- Finding the part of the image where the cost function is at a local optimum.
- Classifying the whole image as one class.
* `[ A ]`


---

**Q: Why is data augmentation essential in the case of few samples?**
- To teach the network the desired invariance and robustness properties
- To make it resilient to noise
- To decrease the computational complexity
- None of the above
* `[ A ]`


---

**Q: Why is the overlap-tile strategy important to apply the network to large images?**
- Otherwise does the network not have fully connected layers
- Otherwise it becomes harder for the network to learn invariance
- Otherwise the network is subject to segmentation problems
- Otherwise the Image resolution is limited by the GPU memory
* `[ D ]`


---

**Q: What do the authors of “U-Net: Convolutional Networks for Biomedical Image Segmentation” consider as important steps when creating a neural network for biometical image segmentation?**
- In the upsampling part they use a large number of feature channels, which allow the network to propagate context information to higher resolution layers. 
- As for our tasks there is very little training data available, it can be helpful to use excessive data augmentation by applying elastic deformations to the available training images because this allows the network to learn invariance to such deformations.
- It can be important to make use of a weighted loss, where the separating background labels between touching cells obtain a large weight in the loss function. (this is done to improve the separation of touching objects of the same class)
- All of the above
* `[ D ]`


---

**Q: How can a U-shaped neural network architecture help to segment images?**
- A u-shaped architecture uses pooling layers in combination with up-convolutions to improve image segmentation
- A u-shaped architecture uses  two different input layers in parallel, where the output of these two networks is combined into a single segmentation
- A u-shaped architecture combines low-resolution features from different convolutional layers with the output from pooling layers. 
- A u-shaped architecture combines high-resolution features from different convolutional layers with the output from up-convolutions. This helps to assemble a more precise segmentation
* `[ D ]`


---

**Q: In biomedical image processing, what is one of the main complications compared to for example normal image classification?
(1)	The output  should include localization (each pixel should get a class label)
(2)	Large input datasets are often beyond reach in biomedical tasks**
- Only (1) is correct
- Only (2) is correct
- Both (1) and (2) are correct
- Both (1) and (2) are incorrect
* `[ C ]`


---

**Q: When is data augmentation useful?**
- When there are few training samples available.
- When you want a network to work on larger images than it was trained for
- When you have plenty of data available, but it is all unlabelled.
- None of the above.
* `[ A ]`


---

**Q: Which is not one of the reasons why the u-net architecture is suitable for biomedical data**
- It requires a very small dataset to train
- It can classify sections of images
- It has a good performance on biomedical data
- It uses semi-supervised learning
* `[ D ]`


---

**Q: What is the benefit of data augmentation?**
- It can prevent overfitting
- It is a way to generate more data when little data is available
- Both A and B
- Neither A or B
* `[ C ]`


---

**Q: What is the strength of the strategy that the network does not have any fully connected layers and only uses the valid part of each convolution?**
- This strategy allows the network to learn invariance to such deformations.
- This strategy allows the seamless segmentation of arbitrarily large images by an overlap-tile strategy
- This strategy allows the network to propagate context information to higher resolution layers
- This strategy allows the network to see only little context
* `[ B ]`


---

**Q: What is the use of data augmentation?**
- avoid overfitting
- increase invariance
- create a bigger training set
- all of the above
* `[ D ]`


---

**Q: In biomedical image processing, thousands of training images are usually beyond reach. What is a way to still train a deep convolutional network?**
- Replacing pooling operators by upsampling operators
- By drawing the initial weights of the network from a Gaussian distribution
- Applying elastic deformations to the available training images
- Use a more expensive GPU
* `[ C ]`


---

**Q: Which one of the following properties does Not U-net structure have?**
- The network uses both fully connected layers and the valid part of each convolution.
- It's suitable for situation in which very little training data is available.
- It's a fully convolutional network.
- A large number of feature channels in the upsampling part allow the network to propagate context information to higher resolution layers.
* `[ A ]`


---

**Q: One advantage of the fully convolutional U-net architecture is that it requires very few training samples to yield reasonable precise segmentations by presenting the network the same samples but every time it sees the sample, it is slightly altered (deformed, rotated, etc.). What is this technique called?**
- a) Data Amplification
- b) Data Augmentation
- c) Data Enlargement
- d) Data Multiplication
* `[ B ]`


---

**Q: Which of the following statements (if any) on the U-net segmentation architecture is false?**
- Speed is a major drawback of the traditional sliding-window approach, due to the network needing to be run separately for each patch of an image.
- In the traditional approach there is a trade-off between location accuracy and the use of contextual information.
- The U-net architecture consists of a contracting path and an expansive path.
- All of the above are true.
* `[ D ]`


---

**Q: What is not a reason that justifies applying data augmentation?**
- Only a small dataset is available for training
- Attempting to reduce overfitting
- Generate more test set examples for model evaluation
- Attempting to increase the robustness of the model
* `[ C ]`


---

**Q: A deep convolutional netwerk with different layers can have large numbers of feature channels. What is the purpose of those feature channels?**
- Propagate context information
- They learn the feature model
- They converge to zero
- None of the above, its not a feature it is a bug.
* `[ A ]`


---

**Q: How does data augmentation work for images in machine learning?**
- You take the original data images. Then you duplicate these images multiple times and add them to the original images.
- You take the original data images. Then you take a copy of a image. This image you can rotate the original image, change lighting conditions, crop it differently. Then do this for all the images from the original data images and combine them together.
- You take the original data images. Which you then remove images from that does not include what you want to recoginize with your model.
- You take the original data images. Then you put them through the first model for machine learning. After which you combine the results of this model and the original data images for the next model.
* `[ B ]`


---

**Q: In deep networks with many convolutional layers and different paths through the network, a good initialization of the weights is extremely important. Why?**
- Otherwise, some parts of the network might give excessive activations, while other parts never contribute
- Otherwise, computing backpropagation becomes very hard
- Otherwise, gradient descent tends to get stuck in local minima
- Otherwise, it takes a lot of time to train a deep net.
* `[ A ]`


---

**Q: What is not a needed for microscopical images to get a good classification result for small training sets?**
- Difference in color
- Rotation invariance
- Robustness to deformations
- Shift invariance
* `[ A ]`


---

**Q: Which of the following statement for the U-Net is not true?**
- Data argumentation is applied due to very little training data
- The separation of touching objects of the same class is another challenge.
- For the training, the softmax and the cross entropy loss function are applied.
- The weight map for each ground truth segmentation do not need to be pre-computed.
* `[ D ]`


---

**Q: Why is data augmentation essential when only few training samples are available?**
- To teach the neural network robustness properties
- To teach the neural network invariance properties
- To teach the neural network invariance and robustness properties
- To decrease the training time of the neural network
* `[ C ]`


---

**Q: What is the main improvement the authors of the paper made with their architecture**
- It was de first CNN
- It works with very few training images and yields more precise segmentations
- It was the first "fully convolutional network"
- This was the first network that used GPU acceleration
* `[ B ]`


---

**Q: What allows the U-Net CNN to work well on a small amount of training samples?**
- The network is small by design, allowing for faster convolution
- A strong use of data augmentation, creating slightly modified duplicate training data
- Multiple networks combining to one large network that are trained on the same data
- duplication of training data for multiple training runs
* `[ B ]`


---

**Q: Which of following is NOT a key motivation for U-Net: Convolutional Networks ? **
- Improve the poor apparent error produced by general convolution networks in Bio-Medical Image Data.
- To present a network and training strategy that does not rely strongly on the requirement of a large input training set and  use the available annotated samples more efficiently.
- To include class labels for localised pixel or set of pixels rather than a class label for an input image as a whole.
- None of the above (Everything above is one of the key motivation)
* `[ A ]`


---

**Q: You need to train a model that should be robust with respect to some transformations, like rotations or elastic deformations, but your training data has a small size. The article on medical image segmentation suggests that**
- a possible solution could include augmenting the data with the transformations that we wish our model to be robust to
- it won't be possible to train the model, as robustness to transformations is only possible with very large data sets
- trained models are not able to learn elastic deformations, even when clever data augmentation is performed
- data augmentation is too expensive and thus it is better to just use the few available training examples.
* `[ A ]`


---

**Q: Which of the following statements is not a drawback of thesliding-window setup proposed by Ciresan et al.?**
- It is quite slow.
- There is a lot of redundancy in its execution.
- More max-pooling layers are required for larger patches, this reduces localization accuracy.
- A lot of training images are needed, many more than the data in terms of patches.
* `[ D ]`


---

**Q: What is data augmentation?**
- Synthetically modifying data based on a variety of conditions (such as rotation, scaling etc.)
- Randomly leaving out data at different iterations.
- Copying the data to have input.
- None of the above.
* `[ A ]`


---

**Q: What is a good initialization of weights important for a deep network with many convolutional layers and different paths through the network?**
- Each feature map in the network has unit variance.
- All weights are the same.
- The standard deviation of the weights is less than 1.
- None of the above.
* `[ A ]`


---

**Q: In U-nets, the expansive path is more or less () to the contract path?**
- Exclusive
- Complementary
- Independent
- Symmetric
* `[ D ]`


---

**Q: \begin{document}
Review the following two statements about
convolutional neural networks:
\begin{enumerate}
    \item A good initialization of the weights is extremely important in deep networks with many convolutional layers and different paths through the network 
    \item When only a few training samples are available, data augmentation is unnecessary to teach the network the desired invariance and robustness properties
\end{enumerate}
Which of the statements are true?**
- Both statements are true
- Statement 1 is true and statement 2 is false
- Statement 1 is false and statement 2 is true
- Both statements are false
* `[ B ]`


---

**Q: Read the following statements.\newline
A. Detecting low contrast structures and fuzzy membranes are one of the difficulties in biomedical image segmentation.\newline
B. Elastic deformation is one of the data segmentation technique.\newline
C. As number of training images are very less in case of biomedical tasks, so we use data augmentation technique to increase the number of training images.\newline
D. Data augmentation helps to teach network the desired invariance and robustness properties.\newline
Select the correct options.**
- B, C, D are correct
- A, B, C, D are correct
- A, C, D are correct
- A, B, C are correct
* `[ C ]`


---

**Q: Which of the following statements are true?

1. Data augmentation is essential to teach the network the desired invariance and robustness properties, when only few training samples are available.
2. To minimise overhead and make maximum use of the GPU memory, we favour large input tiles over a large batch size.**
- Statements 1 is correct, statements 2 is incorrect
- Statement 1 is incorrect, statement 2 is correct
- Both statements are correct 
- Both statements are incorrect 
* `[ C ]`


---

**Q: Which of the following statements is false?**
- In biomedical image processing, a class label is generally supposed to be assigned to each pixel
- A weighted loss function can be used to separate touching objects of the same class
- The exact initialization of weights in deep networks with many convolutional layers and different paths is not very important
- The tiling strategy allows the authors to apply their network to large images, even when the amount of GPU memory is limited
* `[ C ]`


---

**Q: How many incoming nodes does one neuron in a 3x3 convolutional network with 64 feature channels in the previous layer have?**
- 9
- 64
- 192
- 576
* `[ D ]`


---

**Q: Which of the following is NOT a challenge that makes biomedical image classification uniquely difficult?**
- There are not a lot of images available for training in the desired applications 
- Local classification is more important than classification of the whole image
- Images collected for biomedical purposes are of lower quality and contain more noise than everyday images
- It's hard to collect images in some classification scenarios because of legal barriers
* `[ C ]`


---

**Q: What is the reason for introducting upsampling operators in the U-net?**
- to increase the output resolution
- to increase the number of training samples
- to increase the number of features 
- to decrease the number of used convolutional layers
* `[ A ]`


---

**Q: In the U-net architecture presented for Biomedical Image segmentation, why are pooling operators replaced by upsampling operators?**
- To increase the resolution of the output
- To reduce the size of the necessary training sets
- To avoid the reduction of the localization accuracy caused by max-pooling layers
- All the previous answers
* `[ D ]`


---

**Q: Data augmentation allows to get a larger training set starting from the given one, to obtain a better trainig, and it is particularly useful when dealing with feature recognition in images. Which of the following is not a technique adopted in it?**
- Colour scaling of the input images
- rotation of the images
- deformation of the images
- overlapping of the images
* `[ D ]`


---

**Q: When dealing with biomedical data, data augmentation is …**
- not effective, as the deformations of the tissues are unpredictable
- very effective, as the deformations of the tissues can be simulated
- it is proved to be completely useless
- None of the above
* `[ B ]`


---

**Q: A U-net does not contain: **
- Convolution layer
-  Fully connected layers
- ReLU units
- Ìt contains all of the above
* `[ B ]`


---

**Q: what is true about UNet architecture**
- u-net is convolutional network architecture is used for fast and precise segmentation of images. 
- u-net is convolutional network is used to build complex computing networks
- both a and b are correct  
- None of above is correct
* `[ A ]`


---

**Q: What is data augmentation**
- A method used to teach a Neural Network invariance properties by augmenting the available training data
- Filtering the available training data such that it is easier to learn for the neural network
- Augmenting the test data such that the neural network achieves a higher accuracy
- Splitting the training data into several smaller sub-data samples so that the neural network has more training data
* `[ A ]`


---

**Q: Select the incorrect option about U-net architecture.**
- Is computationally efficient 
- High accuracy given proper training, adequate data set and training time
- Preferable for bio-medical applications, as number of annotated samples are usually limited
- None of the above
* `[ D ]`


---

**Q: In the U-Net Architecture, which of the following is NOT an element of the architecture or it's training pipieline?**
- Upsampling layers 
- A deep fully-connected layer
- A weighted loss function where background pixels get weighted more
- Extrapolation of input images by mirroring
* `[ B ]`


---

**Q: The following are characteristics of U-Net**
- The network uses many fully connected layers
- Data augmentation is used for learning invariance
- Pooling is used instead of upsampling operators
- Class label is assigned to an image instead of every pixel
* `[ B ]`


---

**Q: What is the role of a Convolutional layer with [1,1] kernel (like the one at the end of U-Net)?**
- It changes the dimensionality in the filter space 
- It increases the receptive field of the network
- It produces feature maps with a larger size compared to the input of the layer
- It produces feature maps with a smaller size compared to the input of the layer
* `[ A ]`


---

**Q: What is the advantage of using fully convolutional networks for image segmentation?**
- Faster training due to less number of layers.
- It can be used for arbitrary sized inputs.
- The upsampling layers reduce the chance of underfitting.
- The lack of fully-connected layers reduce the chance of overrfitting.
* `[ B ]`


---

**Q: Which of the following is true**
- Larger patches require more max-pooling layers, allowing to see little context
- Upsampling operators decrease the resolution of the output
- Data augmentation can be achieved by applying elastic deformations to training images.
- In CNN with many layers, good initialization of weights is not essential when trained using drop out.
* `[ C ]`


---

**Q: Which of the following is true?**
- To prevent excessive activations or parts that never contribute good initializations of weights is imperative
- When a lot of data samples are available,  it is necessary to augment data in  order  to  teach  a  network  the  desired  amount  of  invariance  and  robustness properties
- A u-net architecture achieves a low performance on very different biomedical segmentation applications
- Random elastic deformations of the training samples are not relevant concepts to train a segmentation network with very few annotated images
* `[ A ]`


---

**Q: U-net uses and requires**
- A lot of annotated data and it uses convolutions
- A lot of annotated data and it doesn't uses convolutions
- A few annotated data samples and it uses convolutions
- A few annotated data samples and it doesn't uses convolutions
* `[ C ]`


---

**Q: Which of the following statements is false?**
- The u-net architecture does not perform well on biomedical segmmentation applications.
- When only few training samples are available, data augmentation is essential to teach the network invariance and robustness properties.
- The U-net architecture described in the paper, consist of a contracting path and an expansive path.
- Large patches (local regions) require more max-pooling layers that reduce the localization accurancy. 
* `[ A ]`


---

**Q: What are advantages of a so-called "fully convolutional network"?**
- It requires a lot of input images, but also is faster in processing them.
- It is very fast and has low redundancy.
- It only needs one convolutional layer.
- It works with very few training images and yields more precise segmentations.
* `[ D ]`


---

**Q: Which of the following options is not a characteristic of the U-Net?**
- It relatively needs only few annotated images.
- It has a very reasonable training time.
- It has very good performance, in any domain.
- It has potential to be applied in many more tasks.
* `[ C ]`


---

**Q: In the U-Net architecture, if there are 64 channels in a preceding hidden layer, followed by 64 channels in the successive layer, how many learnable filters do we have between these layers?**
- 1 learnable convolutional filter applied on all 64 channels of the input hidden layer to yield 64 channels in the output hidden layer
- 64 learnable convolutional filters applied on all 64 channels of the input hidden layer to yield 64 channels in the output hidden layer
- 64 learnable subsampling layers between hidden and output layers
- None
* `[ B ]`


---

**Q: Which of the following statements does not hold good with respect to concepts discussed in the paper - "U-Net: Convolutional Networks for Biomedical Image Segmentation"**
- Higher localisation accuracy is achieved by increasing max pooling layers.
- Context information is propagated by a number of feature channels to successive layers in the upsampling part of U-net
- Data augmentation techniques are used to simulate realistic deformations which in turn increase the generalization capability of the network
- None of the above
* `[ A ]`


---

**Q: What property of images is made advantage of using the architecture described in the paper**
- The fact that all parts of an image have the same resolution
- The fact that parts of an image can be classified separately
- The fact that images always show a single concept
- The fact that images generally have more blue than red in their pixels
* `[ B ]`


---

**Q: What is the drawback of the approach used by Ciresan et al.?**
- Quite slow (running each patch seperatly)
- overlapping patches
- trad-off between localization accuracy and the use of context
- all of the above
* `[ D ]`


---

**Q: Which of the methods overcomes the problem of having less training data?**
- Regularisation
- Normalisation
- Data Augmentation
- None of the above 
* `[ C ]`


---

**Q: Which of the following statements is correct?**
- The outputs of multiple layers are required to localize features on an image.
- Upsampling images is usually a bad practice, because it increases the likeliness of overfitting.
- Data augmentation is used to filter noisy in inputs and make them suitable for processing
- The values of the weights in a deep neural networks after teaching is irrespective of the initial weights.
* `[ A ]`


---

**Q: What are the inherent property of optimal input**
- Sinusoid;sensitive to translation
- Sinusoid;insensitive to translation
- Sinc; sensitive to translation
- Gaussian; insensitve to translation
* `[ B ]`


---

**Q: In the study of CNN for Biomedical Image Segmentation, which one of the following is WRONG?**
- Data augmentation helps to learn the invariance to deformations in the original image.
- Data augmentation is important for CNNs when there are limited data available.
- Data augmentation results in better performance of convolutional nets while it leads to massive increase in the processing time. There's a trade-off between performance efficiency and time efficiency.
- It requires thousands of annotated training samples to train deep networks successfully.
* `[ C ]`


---

**Q: What statement is not true?**
- Localization in a task is impossible using CNNs
- Data augmentation can improve results from training on a small training set
- Localization in a task is possible using CNNs
- Biomedical tasks usually do not have a lot of training images available
* `[ A ]`


---

**Q: Which of the following will double the number of feature channels?**
- Application of 3x3 convolution and 2x2 max pooling operation with stride 2
- Upsampling of the feature map followed by a 2x2 convolution
- Application of two 3x3 convolutions and a 2x2 max pooling operation with stride 2
- None of the above
* `[ C ]`


---

**Q: Mark the false sentence. Data augmentation is used: **
- When only few training samples are available.
- By applying elastic deformations on the training samples.
- To decrease the training time.
- To teach the network the desired invariance and robustness properties.
* `[ C ]`


---

**Q: What kind of padding is used in the network proposed in this paper?**
- The missing context is take as zero
- The missing context is extrapolated by mirroring the input image
- There is no padding, regions with missing context are ignored
- The missing context is extrapolated, using the local gradient of the image
* `[ B ]`


---

**Q: Which of the following layer is NOT used in network architecture?**
- Rectified linear unit
- 2x2 Max pooling layer
- 3x3 Average pooling layer
- 3x3 Convolutional layer
* `[ C ]`


---

**Q: Data augmentation is NOT necessary:**
- When we want to teach a network to deal with desired robustness properties.
- When we need a shift and rotation invariance.
- When there is a significant amount of data samples.
- When we need to elastic deformations of the training samples.
* `[ C ]`


---

**Q: Which of the following statements is true?
I. The UNet network combines upscaled features from contracted layers with high resolution pictures to find high resolution features.
II. The Unet network uses multiple instances (altered) of the same imager to gather more data to train on. **
- Only I is true.
- Only II is true.
- Both statements are true.
- None of the statements is true.
* `[ C ]`


---

**Q: What is one countermeasure to not having enough training data?**
- Data augmentation
- Convolution
- Random sampling
- Spatial Pooling
* `[ A ]`


---

**Q: Which of the following are \emph{not} modifications of the fully convolutional network, resulting in the so-called U-net:**
- There are a large number of feature channels in the upsampling step
- Only the valid part of each convolution is used
- Features from the contracting path are combined with the features of the expansive path
- None of them are.
* `[ C ]`


---

**Q: What is one of the major difficulties in classifying biomedical images?**
- The data is very noisy: individuals vary a lot.
- There is very limited data available.
- Realistic deformation can not be simulated realistically.
- All of the above.
* `[ B ]`


---

**Q: Which of the following statements regarding the use of U-Net for Biomedical imaging are correct:
A)U-Nets have implicit data Augmentation in the network architecture itself.
b)The output of U-Net for Biomedical imaging also gives localization.**
- A) is correct
- B) is correct
- A) and B) both are correct
- None of the above
* `[ C ]`


---

**Q: Why is data augmentation important?**
- To teach the neural network the robustness properties (e.g. rotation)
- It makes the neural network learn faster
- Only augmenting with scaling is necessary
- None of the above
* `[ A ]`


---

**Q: In deep networks with many convolutional layers weights should be properly initialized. In the ideal case, the weights should be:**
- Adapted such that the last feature map has unit variance
- Adapted such that the first feature map has unit variance
- Adapted such that each feature map has  very high variance
- Adapted such that each feature map in the network has approximately unit variance
* `[ D ]`


---

**Q: In which field does UNet widely used?**
- Medical
- Military
- Education
- Industry
* `[ A ]`


---

**Q: As the authors point out in the “U-Net: Convolutional Networks for Biomedical Image Segmentation” paper, one of the challenges in cell segmentation tasks is the separation of touching objects of the same class. What is the authors’ proposed method to deal with this issue?**
- Performing data augmentation by applying elastic deformations to the training images.
- Extrapolating context by mirroring the input images.
- Using a weighted loss.
- Excluding any fully connected layers.
* `[ C ]`


---

**Q: The use of excessive data augmentation by applying elastic deformations to the available training images is proved to be really important because**
- It allows the network to learn invariance to such deformations
- It brings less noise to the dataset
- In this way the result is scale invariant
- None of the above
* `[ A ]`


---

**Q: The authors elegantly use Data Augmentation techniques in this work. Why?**
- Only to generate more training samples.
- To provide their network with the correct robustness and invariance.
- To clearly separate the background from the foreground in their input data.
- In order to have enough data so as to have sufficient empirical evidence of their networks superiority
* `[ B ]`


---

**Q: What was the purpose of the expansive part of the U-Net by Ronneberger et al.  where up-convolution was implemented?**
- To improve the symmetry of the network because symmetry was observed to increase the performance of the network.
- To generate higher quality images from low quality images for faster manual segmentation. 
- To carry forward the segmentation information in the inner layers and mark the corresponding segments in the original image.
- To reduce the number of channels in the output image.
* `[ C ]`


---

**Q: Which of the following is not a component of the U-net architecture?**
- Convolution
- Exp-convolution
- Up-convolution
- Max pooling
* `[ B ]`


---

**Q: Which of the following techniques can be used for data augmentation?**
- Color jittering
- Cropping the subject
- Introducing Gaussian Noise
- All of above
* `[ D ]`


---

**Q: What is not a drawback of a sliding window setup to predict the class label of each pixel?**
- The network must be run separately for each patch
- There is a lot of redundancy due to overlapping patches
- There is a trade-off between localization accuracy and use of context
- A successive convolution layer learns to assemble a more precise output
* `[ D ]`


---

**Q: Why is a good initialization of the weights in deep networks with many convolutional layers and different paths through the network important?**
- Otherwise, the network might lose it's invariance or robustness properties
- Otherwise, none of the layers may actually contribute
- Otherwise, parts of the network might give excessive activations, while other parts never contribute
- Otherwise, it may get stuck converging to a local minimum
* `[ C ]`


---

**Q: For U-Net architecture, which of the following is a reason for having output image smaller than input image:**
- Unpadded convolutions 
- Cropping in the expansive path
- Excessive max pooling 
- Overlapping tile strategy used in training
* `[ A ]`


---

**Q: Which statement about UNET architecture is false:**
- The contraction path involves convolution and max pooling
- With the contraction path, the size of the image decreases
- The expansion path involves the transpose convolution of the down-sampled image
- With the expansion path, the depth of the network (context ) increases
* `[ D ]`


---

**Q: The network architecture consists of two paths namely contracting and expansive. Which of the following statements is true about these paths?**
- Cropping is required for expansive path.
- Cropping is required for contracting path.
- Cropping is required for both paths.
- Cropping is not required by both paths. 
* `[ A ]`


---

**Q: Which is the reason of applying an excessive data augmentation by elastic deformations to the available training images instead of any other data augmentation technique , when the available training data are very little?**
- Makes the network to learn invariance to elastic deformations
- It is easier to implement and requires less computational cost, while the results are equivalent
- Has proved to achieve the best performance in comparison with the other techniques
- None of the above 
* `[ A ]`


---

**Q: What is the meaning of the 'u' in the u-architecture**
- Just a funny name, it has no special meaning.
- It's an end to end architecture
- The feature maps are first reduced and afterwards increased in size
- N.A.
* `[ C ]`


---

**Q: What is not a reason why data augmentation is essential in biomedical image processing?**
- To improve the learning rate
- To learn invariance to deformation
- To learn robustness properties
- None of the above
* `[ D ]`


---

**Q: What can a nn with a large enough single hidden layer of sigmoid units approximate**
- an accuracy of 1
- any decision boundary
- all of the above
- none of the above
* `[ C ]`


---

**Q: What is NOT true about data augmentation?**
- Essential when only few training samples are available
- Shifting and rotation of images
- Random elastic deformation of images
- Interchanging parts of multiple images to make an entire new image
* `[ D ]`


---

**Q: The U-Net is a convolutional network that consist of a contracting and expansive path. Why are these networks more suitable for image segmentation compared to a traditional convolutional network? **
- High resolution feature maps from the contracting path can be combined with the low resolution feature maps from the expansive path to give localization information
- Low resolution feature maps from the contracting path can be combined with the high resolution feature maps from the expansive path to give localization information
- The contracting path will capture all feature maps that below to the different segments and this localization information is used in the expansive path
- An U-Net is not more suitable for image segmentation than a traditional convolutional network
* `[ A ]`


---

**Q: What characterizes U-Net's architecture?**
- Fully convolutional network with skip connections
- Convolutional network with segmentation
- Use of recurrent neurons
- Use of attention mechanism 
* `[ A ]`


---

**Q: Which of the following is true:**
- The output image size decreases after unpadded convolution
- To better adapt to the network the weights initialization should be random from the standard normal distribution
- Data augmentation assigns weights to the training samples
- Applying CNN to biomedical images cannot lead to both good localization and the use of context
* `[ A ]`


---

**Q: Segmentation can be done with a 'sliding window approach', what is a notable drawback of this method?**
- As you are using patches, you are introducing a lot of bad training data.
- A sliding window is not able to localize.
- As the we are limited by a sliding window, it is possible to miss out on crucial context.
- A sliding window is very fast and therefore as a trade-off has less accuracy.
* `[ C ]`


---

**Q: What is the ideal criterion for the adaptation of initial weights in deep networks?**
- The feature map has approximately unit average
- The feature map has approximately unit average and variance
- The feature map has approximately unit z-score
- The feature map has approximately unit variance
* `[ D ]`


---

**Q: The key concept to train a segmentation network for Biomedical images:**
- The augmentation of the samples for the purpose of using large number of data samples to train the CNN
- To produce image samples that are translated and rotated in respect with the original image
- To produce image samples with elastic deformations
- The use of more max-pooling layers to increase the localisation accuracy
* `[ C ]`


---

**Q: What is not a good reason to use data augmentation? **
- One uses data augmentation to make the training more invariant
- One uses data augmentation to decrease the variance of the training set.
- One uses data augmentation because he has a small training set.
- One uses data augmentation to improve the robustness of the training
* `[ B ]`


---

**Q: Which of these statements about convolutional networks for biomedical segmentation is incorrect?**
- Data augmentation is essential to teach the network the desired invariance and robustness properties, when only few training samples are available
- Initialization of the weights is extremely important for deep networks with many convolutional layers and different paths through the network
- Random elastic deformations of the training samples are the key concept to train a segmentation network with few annotated images
- Running the network separately for each patch of an image will increase the speed of the algorithm
* `[ D ]`


---

**Q: In deep networks with many convolutional layers and different paths through
the network, a good initialization of the weights is extremely important. Why?**
- Parts of the network might give excessive activations
- Otherwise it takes ages to train the network
- The statement is not true
- Otherwise the network overfits the training data
* `[ A ]`


---

**Q: What are the limitations to medical image segmentation?**
- Low quality cameras are used in hospitals
- There is very few images available
- All bodies look different so the network can't train
- Images are not allowed to be used due to privacy concerns
* `[ B ]`


---

**Q: Statement 1: A U-net architecture network needs a lot of training data. Statement 2: A U-net architecture has no convolutional layers**
- Both statements are true
- Both statements are false
- Statement 1 is true and statement 2 is false
- Statement 1 is false and statement 2 is true
* `[ B ]`


---

**Q: What is the particular advantage of U-nets and their dual path approach?**
- The dual path approach allows for an increase in the receptive field of the neurons
- Deeper nets in the contracting path stack feature maps in multiple channels 
- The ability to identify  the semantic and spatial context of the input features
- The use of ground-truth images for training and validating the accuracy of the classifiers
* `[ C ]`


---

**Q: which statement(s) is/are true
1) Data augmentation is used to create artifical data if not enough real data is available
2) articifical data is equal to original data when training CNN's**
- 1) is True
- 2) is True
- Both are true
- both are false
* `[ A ]`


---

**Q: What technique can be used to process images larger than available in GPU / system memory, but without losing information**
- Tiling
- Downsampling
- Using shallower convolutional nets
- Using a larger convolution kernel
* `[ A ]`


---

**Q: Where was the U-net developed for. **
- biomedical image segmentation
- pattern recognition
- speech recognition
- Mnist dataset
* `[ A ]`


---

**Q: Which of the following things are required for localizing an object?**
- Pooling
- Convolution
- Upconvolution
- All of the above
* `[ D ]`


---

**Q: Statement 1: Data augmentation is essential to teach the network the desired invariance and robustness properties, when only few training samples are available. 
Statement 2: To maximize the overhead and make maximum use of the GPU memory, large input tiles are favored over large batch size and hence reduce the batch to a single image. **
- Both statements are true
- Statement 1 is true
- Statement 2 is true
- None of the statements are true
* `[ A ]`


---

**Q: What is semantic segmentation?**
- understand an image at pixel level
- identify each entity in the image
- identify one specific entity in the image
- understand an image at pixel level and each entity separately
* `[ A ]`


---

**Q: Why is data augmentation essential in microscopical images?**
- To teach the network robustness to deformation and grey value variation.
- To teach the network shift and rotation invariance
- Both A and B
- Data augmentation is not essential
* `[ C ]`


---

**Q: What is one of the reasons that the success of convolutional networks was limited in the beginning of its existence?**
- The programmers did not know that convolutional networks could be applied in deep learning software. It was used for other applications so therefore it was not used by deep learning programmers and thus not successful.
- Windows was used instead of Linux.
- The size of the available training sets at that time.
- None of the above answers is correct.
* `[ C ]`


---

**Q: Which following statement is not true according to paper?**
- Data augmentation could help training networks more robust to deformation.
- In biomedical image processing the classification task is performed on each pixel rather than the whole image.
- The initialization of weights are not so important because they will be updated later.
- Data augmentation could be used when there are few available data.
* `[ C ]`


---

**Q: Why does the U-Net use an overlap-tile strategy?**
- To be able to work on arbitrarily sized images
- To save on gpu memory
- To allow the network to focus on smaller parts of the image, increasing accuracy
- To augment the training data.
* `[ A ]`


---

**Q: In the paper: "U-Net: Convolutional Networks for Biomedical
Image Segmentation", the authors use elastic deformations on the training data for the following reasons EXCEPT:**
- There are not as many training data samples hence elastic deformations on existing images increases the sample size.
- The elastic deformations allows the model to learn to be invariant to the deformations which typically occur in actual image samples in the biomedical domain
- The deformations are used to provide a control set with which the model can then be verified with
- None of above.
* `[ C ]`


---

**Q: The U-Net is an algorithm that is based on the fully convolutional network. Its architecture was modified and extended to work with fewer training images and to yield more precise segmentations. Which of the following statements about the U-net algorithm are true?

Statement 1:The network does not have any fully connected layers and only uses the valid part of each convolution, i.e., the segmentation map only contains the pixels, for which the full context is available in the input image.

Statement 2: To allow a seamless tiling of the output segmentation map it is important to select the input tile size such that all 2x2 max-pooling operations
are applied to a layer with an odd x- and y-size.

Statement 3: We use a high momentum such that a large number of the previously seen training samples determine the update in the current optimization step.

Statement4: Data augmentation is essential to teach the network the desired invariance and robustness properties, when only few training samples are available. 

Statement5: In case of microscopical images we primarily need shift and rotation invariance as well as robustness to deformations and gray value variations. Especially random elastic deformations of the training samples seem to be the key concept to train a segmentation network with very few annotated images.**
- All statements are true
- 1 2 and 3
- 3 4 and 5 
- 1 3 4 5 
* `[ D ]`


---

**Q: which of the following statements is wrong about U-Net?**
- A successive convolution layer can then learn to assemble a more precise output.
- The initial weights should be adapted such that each feature map in the network has approximately unit variance.
- Data augmentation can be used when few training samples are available.
- The use of u-net architecture is very limited which is only suitable for small training data set.
* `[ D ]`


---

**Q: Why is the stocastic gradient descent used in deep learning **
- you randomly select features
- Clean noise from the data
- Upsamplening for getting the localminimum 
- None of the above 
* `[ A ]`


---

**Q: Why is a typical CNN usually not suitable for biomedical image processing **
- The output should contain detailed information about localization and generally, not a lot of training images are available.
- Medical standards are so high, that it is unlikely that it can ever be proven that a CNN will never make a mistake
- A doctor will always outperform a CNN in reading X-ray pictures
- Since no two people are the same, it is simply impossible for a CNN to find any correlation between for example tumors in different people 
* `[ A ]`


---

**Q: What is the advantage of a u-net in image localization?**
- A u-net is fast because it does not need to be run for every patch.
- A u-net has faster backpropagation.
- A u-net is smaller and therefore faster.
- A u-net can localize your grandmother.
* `[ A ]`


---

**Q: Which of the statements below are true about a residual network?**
- The impact of dropping individual modules from a residual network on performance is lesser than dropping a layer in a plain network.
- The training error always decreases with an increase in the number of layers in residual networks
- Both A and B.
- None of the above.
* `[ C ]`


---

**Q: What is an advantage of the fully convolutional method used in the paper?**
- It's more intuitive 
- It easier adapts to high dimensionality
- It needs but a small amount of training data
- It can use incomplete data
* `[ C ]`


---

**Q: Which of the following statements is FALSE regarding U-Net?**
- U-Net is built upon the so-called "fully convolutional network".
- The upsampling part of U-Net allows the network to propagate context information to high resolution layers.
- U-Net has a very good performance even if the size of data set is small.
- U-Net uses a weighted loss to deal with the problem of separation of touching objects of the same class.
* `[ C ]`


---

**Q: U-Net: Convolutional Networks for Biomedical Image Segmentation describes a way to train a CNN for image segmentation using very few examples, what made it possible to only use a few images?**
- Their new "U" architecture
-  Data augmentation
- Adaptive learning rate
- Per pixel sliding window during training
* `[ B ]`


---

**Q: Which of the following is true about the u-net architecture?

1. One of the advantages is that the u-net needs very few annotated images
2. The architecture however suffers from a significant downgrade in training time.**
- Only 1 is correct
- Only 2 is correct
- Both are correct
- Both are incorrect
* `[ A ]`


---

**Q: 4. The training of very deep network benefit**
- introducing short path
- vanishing gradient problem
- All
- none 
* `[ C ]`


---

**Q: How many fully connected layers are in the U-net? **
- 0
- 1 in the input
- 1 in the output
- 2, 1 in the IN and 1 in the OUT
* `[ A ]`


---

**Q: Which statement is wrong?**
- Using the same number of parameters as the deep models they mimic, it's possible that shallow feed-forward nets can learn the complex functions previously learned by deep nets
- Deep convolutional nets are significantly more accurate than shallow convolutional models, given the same parameter budget
- Model compression works by passing unlabeled data through the large, accurate teacher model to collect the real-valued scores it predicts, and then training a student model to mimic these scores.
- A shallow net has to have less hidden units in each layer to match the number of parameters in a deep net
* `[ D ]`


---

**Q: Please complete the following sentence. In deep networks with many convolutional layers and different paths through the network.**
- Random initialization of weights allows for a uniform variance for each featuremap.
- The initialization of weights is idealy such that each featuremap has weighted variance.
- The initialization of weights is ideally such that each featuremap has approximately unit variance.
- The initialisation of weights from a gaussian distribution would lead to a large variation in the feature map variance.
* `[ C ]`


---

**Q: What special problems does the proposed method solve about biomedical images?**
- That trained networks need to be shift and rotation invariant
- That trained networks need to be robust to deformations
- That trained networks need to be robust to gray value variations
- All of the above
* `[ D ]`


---

**Q: What is the benefit data augmentation in deep learning?**
- Makes computation times faster.
- Helps prevent overfitting.
- Effectively increases the data size.
- B and C are correct.
* `[ D ]`


---

**Q: Ciresan trained a network in a sliding-window setup to predict the class label of each pixel by providing a local region around that pixel as input. First, this networ can localize. Secondly, the training data in terms of patches is much larger than the number of training images. 
Consider the following statements about the strategy in Ciresan et al.:
1.	The strategy in Ciresan et al. is quite slow because the network must be run separately for each path, and there is a lot of redundancy due to overlapping patches. 
2.	The strategy in Cresan et al. has a trade-off between localization accuracy and the use of context. Larger patches require more max-pooling layers that reduce the localization accuracy, while small patches allow the network to see only little context. 
Which of the statements is true?**
- None
- Both
- 1 is false, 2 is true
- 1 is true, 2 is false
* `[ B ]`


---

**Q: The U-net stands for/is an?**
- U shaped network
- Fully convolutional network
- Has an large number of feature channels in the upsampling part.
- All of the above
* `[ D ]`


---

**Q: Why does the U-net from the paper up-sample when first having down-sampled?**
- To reduce the computations done by the GPU.
- To create the output segmentation map from the down-sampled contracting path.
- To further increase the number of feature channels.
- To compensate for the lost border pixels during down-sampling.
* `[ B ]`


---

**Q: Why does data augmentation explain part of the success of U-Net on biomedical images?**
- Why does data augmentation explain part of the success of U-Net on biomedical images?
- Biomedical images benefit from deformational invariance.
- U-Net performs signficantly better on large annotated training sets.
- Convolutional filters don't have invariance by default.
* `[ B ]`


---

**Q: What is an important characteristic of a U-net?**
- It needs few labeled data to train
- In contrast with its name, the U-net has a W-shape.
- none of the above
- all of the above
* `[ A ]`


---

**Q: What does the paper state about 'random elastic deformations of the training samples'?**
- They seem to correct the imperfections in the training data.
- They seem to speed up the training time if the deformations are along the primary axis.
- They seem to be key to training a segmentation network with very few annotated images.
- They seem to reduce the chance of overfitting by adding noise to the training data.
* `[ C ]`


---

**Q: What is data augmentation?**
- Creating new data points using extrapolation from existing data points
- Changing the dimensionality of data
- Normalizing the data
- Getting rid of the noise
* `[ A ]`


---

**Q: What are statements about patching (Providing the area round a pixel as input) are true?**
- Generally, less context is used when patching 
- There is a lot of redundancy due to overlapping patches
- Both are true
- Both are false
* `[ B ]`


---

**Q: What is the drawback of the strategy in Ciresan to solve localization problem in biomedical images, (which o predict the class label of each pixel by providing a local region (patch) around that pixel)**
- redundancy due to overlapping patches
- the network must be run separately for each patch
- there is a trade-off between localization accuracy and the use of context
- all of the three
* `[ D ]`


---

**Q: In biomedical segmentation deformation is a common variation in tissue. Which of the following helps in accounting this for the learning networks?**
- Applying plastic deformation for excessive data augmentation
- Having a large number of differently deformed images
- Both a and b
- None of the above
* `[ A ]`


---

**Q: x**
- x
- x
- x
- x
* `[ A ]`


---

**Q: Which one is wrong **
- This paper propose a novel interpretation of residual networks showing that they can be seen as a collection of many paths of differing length
- Residual networks avoid the vanishing gradient problem by introducing only paths which can carry gradient throughout the extent of very deep networks
- Residual networks avoid the vanishing gradient problem by introducing short paths which can carry gradient throughout the extent of very deep networks
- Residual networks avoid the vanishing gradient problem by introducing short paths which can carry gradient throughout the extent of very deep networks
* `[ A ]`


---

