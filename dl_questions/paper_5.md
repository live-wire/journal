# Questions from `paper_5` :robot: 

**Q: To compress a neural network model using distillation techniques, one needs to:**
- Remove nodes that do not affect the target outcome
- Give more weight to the training samples which were incorrectly predicted
- Mimicking the real valued output scores of a teacher model, with unlabeled data. 
- Use a teacher model to create new one-hot labeled data from unlabeled data.
* `[ C ]`


---

**Q: Imagine you have 2 fully connected feed-forward networks. The first one (deep) has 4 input units, 2 hidden layers with 3 and 2 units and an output layer with 1 unit. The second network (wide) has the same number of input and output units and only has one hidden layer with 5 units. Taking this into account which one could take longer to train and why?**
- Deep because it has more parameters to train. 
- Wide because it has more parameters to train.
- Deep because it has less parameters to train.
- Wide because it has less parameters to train.
* `[ C ]`


---

**Q: Can shallow nets (student) be trained to be as accurate as deep convolutional models (trainer) using a similar number of parameters in the deep and shallow model (according to Urban et al. (2017))?**
- Yes
- No
- We cannot tell as we do not know if the shallow nets are trained well enough
- We cannot tell as we do not know if the teacher model is as accurate as possible
* `[ B ]`


---

**Q: Which of the following about linear bottleneck layers in neural networks is FALSE?**
- A linear bottleneck layer speeds up learning by reducing the number of parameters that must be learned.
- The linear terms can be absorbed back into the non-linear weight matrix after learning.
- Adding a linear bottleneck layer makes the model deeper. 
- One reason for applying a linear bottleneck layer is that training wide, shallow mimic models with back-propagation has been found to be slow.
* `[ C ]`


---

**Q: Which of the following is true about student mimic models?**
- Mimic networks perform better when using dropout.
- Student models overfit and need extra regularization methods.
- Student mimic models use the real-valued labels provided by the teacher model.
- All of the above.
* `[ C ]`


---

**Q: Where is the biggest increase in accuracy for convolutional vs not convolutional and deep vs shallow**
- When we increase the depth
- When we decrease the depth
- When we go from convolutional to not convolutional
- When we go from not convolutional to convolutional
* `[ C ]`


---

**Q: What is an advantage of shallow convolutional models over deep convolutional nets with a similar parameter budget?**
- They are more accurate
- They allow for better backpropagation
- They are faster
- They are more subject to noise
* `[ C ]`


---

**Q: Can shallow models be as accurate as deep convolutional models given the same parameter budget? **
- It’s not possible to say that shallow models can’t be as accurate as deep convolutional models because one does not know if the she shallow nets are trained ‘well enough’
- Yes, shallow models can be as accurate as deep convolutional models given the same parameter budget?
- No, deep convolutional models will always have a bigger accuracy compared to the shallow models because they lack convolution, something which is critical in machine learning.
- Yes, this is because of the “dark knowledge” in the soft targets, the extended distillation and the use of simple bases.
* `[ A ]`


---

**Q: Which factor is the most important one for a good performance of a neural network? **
- high number of hidden layers
- include convolutional layers
- a large layer width
- As many parameters as possible
* `[ B ]`


---

**Q: In the paper “DO DEEP CONVOLUTIONAL NETS REALLY NEED TO
BE DEEP AND CONVOLUTIONAL?” which of the following has the largest impact on model performance?**
- The amount of parameters used (parameter budget)
- The amount of convolutional layers
- The kernel used for non-linearity
- The amount of pooling layers
* `[ B ]`


---

**Q: What is true about model compression?**
- It is a lossless compression method in which a large neural network can be stored in a more efficient manner.
- It is training a small neural network to mimic the behaviour of a large neural network. The advantage is that the smaller neural network does not need to train on the original dataset anymore.
- It is training a small neural network to mimic the behaviour of a large neural network. This results in the smaller neural network performing better than the larger neural network
- None of the above
* `[ D ]`


---

**Q: How do deep convolution nets compare to shallow convolution models, given the same parameter budget?**
- They are equally accurate
- Depending on the dataset, either the shallow or the deep model may be more accurate.
- The deep net is significantly more accurate 
- The deep net is slightly more accurate
* `[ C ]`


---

**Q: Considering convolutional neural networks with a fixed number of parameters:**
- Deep CNN’s generally perform better than shallow CNN’s
- Deep CNN’s generally perform worse than shallow CNN’s
- Deep CNN’s generally perform as well as shallow CNN’s
- No comparisons can be made between deep and shallow CNN’s
* `[ A ]`


---

**Q: In learning rate schedule, which of the following is the stop condition when training?**
- the learning rate is increased by a factor of more than 2000 in total
- the error do not drop for 30 epochs in a row
- the validation error does not drop for ten epochs in a row
- the learning rate is reduced by a factor of more than 1000 in total
* `[ B ]`


---

**Q: Do deep convolutional nets really need to be deep and convolutional?**
- neither deep nor convolutional
- deep, but not convolutional
- not deep, but convolutional
- both deep and convolutional
* `[ D ]`


---

**Q: What could be a form of data augmentation for image datasets?**
- Adding mirrored images from the original dataset
- Take original dataset images and shift hue
- Take original dataset images and change saturation
- All of the above
* `[ D ]`


---

**Q: Which of the following statement about training a shallow network and the necessity of convolutional network to be deep enough is wrong?**
- A network with a large enough single hidden layer of sigmoid units can
approximate any decision boundary.
- Deep convolutional nets do, in fact, need to be both deep and convolutional, even when trained to mimic very accurate models via distillation.
- There are many steps required to train shallow student models to be as accurate as possible: train state-of-the-art deep convolutional teacher models, form an ensemble of the best deep models, collect and combine their predictions on a large transfer set, and then train carefully optimized shallow student models to mimic the teacher ensemble.
- The key idea behind model compression is to train a compact model to predict the exactly same functionlearned by another larger, more complex model.
* `[ D ]`


---

**Q: Rank the neural networks in their best performing order: (1) CNN with 1 convolutional layer, (2) MLP with 1 hidden layer, (3) CNN with 2 convolutional layers, (4) MLP with 2 hidden layers.**
- a) 1, 2, 3, 4
- b) 1, 3, 2, 4
- c) 2, 4, 1, 3
- d) 3, 1, 4, 2
* `[ C ]`


---

**Q: Which of the following statements about the depth of deep learning models is false?**
- With the same budget of parameters, deep models are generally superior in accuracy to shallow models.
- Model compression decreases the depth of a model needed to achieve the same accuracy as their uncompressed counter-parts.
- The results of a validation on one state-of-the-art benchmark dataset can suffice to validate a machine learning strategy.
- The term "dark knowledge" refers to the relative probabilities of outputs for unselected (incorrect) classes.
* `[ C ]`


---

**Q: Which statement is correct regarding the student-teacher learning architecture:**
- The teacher model's output is used as the student model's input
- Student models try to learn a relation between the input data and the output of the teacher's model
- Classification accuracy of the teacher model is not related to the accuracy of the student model
- Student models generally have more learnable parameters than teacher models
* `[ B ]`


---

**Q: Which statement is true?**
- Deep convolutional neural networks generally perform better than Shallow convolutional neural networks
- Shallow convolutional neural networks generally perform better than deep convolutional neural networks
- Shallow convolutional neural networks generally perform as good as deep convolutional neural networks
- None of the above
* `[ A ]`


---

**Q: The key idea behind model compression is to train a compact model to approximate the function learned by another larger, more complex model. So how does model compression work?**
- Model compression works by making the function of the larger, more complex model more simple.
- Model compression works by passing the unlabeled data through the large, accurate teacher model to collect the real-valued scores it predicts and then training a student model to mimic these scores.
- Model compression works by passing the unlabeled data through the large, accurate teacher model to collect the real-valued scores it predicts and then doing the same for the compact model. With finally combining the real-valued scores of the two models into a new real-valued score.
- None of the above answers are correct
* `[ B ]`


---

**Q: Consider deep and shallow neural networks. What can be considered as a dark knowledge?**
- A knowledge passed from the teacher network to the student network
- A knowledge that shallow network somehow infers from a training set
- A knowledge that deep neural network gains by training
- None of the above
* `[ A ]`


---

**Q: Which of the following methods can not be a part of training shellow nets to mimic deeper convolutional nets?**
- Bayesian hyperparameter optimization
- Using a super teacher
- Data augmentation
- Training student models to mimic shellow nets
* `[ D ]`


---

**Q: Which of the following step is not included correctly in training shallow nets to mimic deeper convolutional networks?**
- Model compression and distillation
- Mimic learning via L1 regression on logits
- Bayesian hyperparameter optimatization
- Learning-rate schedule
* `[ B ]`


---

**Q: Do deep convolutional neural networks really need to be deep and convolutional?**
- Yes, the students need a single convolutional layer to learn
functions of comparable accuracy as the deep convolutional teacher.
- Yes, the students need multiple convolutional layers to learn
functions of comparable accuracy as the deep convolutional teacher.
- Yes, the students need multiple pooling layers to learn
functions of comparable accuracy as the deep convolutional teacher.
- Yes, the students need a single pooling layer to learn
functions of comparable accuracy as the deep convolutional teacher.
* `[ B ]`


---

**Q: Which of the the deep learning models below works in general best for image classification problems compared to the others stated?**
- A deep model
- A convolutional model
- A shallow model
- A model with few parameters
* `[ B ]`


---

**Q: What is meant with the term "dark knowledge" in teacher/student networks?**
- the relative probabilities of output classes of the network
- pre-learned extra information on the data set
- untracable learned parameters of the network
- the information contained within the hidden layers of a neural network
* `[ A ]`


---

**Q: Which of the following is FALSE about the process, Distillation ?**
- The key idea of distillation is to train a compact model to approximate the function learned by another larger, more complex model
- Process of Distillation - A Student model (Less complex, deep)  is trained using the optimal weights derived by the Teacher model (Deep and complex) on the labelled dataset.
- Distillation is a process by which Deep Learning model compression is achieved.
- Distillation on a Deep Leaning model can give us a student model (Less complex, deep) which can match the performance of the teacher model with usage of same data and hyper parameters.
* `[ B ]`


---

**Q: After training an ensemble of models (the "teacher" ensemble) to solve a classification task, a "student" model can be trained - not to learn how to solve the original classification task, but - to mimic the previously trained ensemble of models. (Pick the odd one out)**
- this allows for smaller models (less parameters) to be trained with an accuracy comparable to that of the teacher ensemble
- student models can be made more shallow (less deep) because mimicking the teacher's outputs gives more information to the student than just predicting $0/1$ labels
- student models with the same number of parameters as the teacher models can outperform the teaching models
- student models need to be at least as deep as the deepest model of the teacher ensemble
* `[ D ]`


---

**Q: Which of the following is correct with regards to training accurate models on CIFAR-10 with CNNs? I: The student models do have to be as deep as the teacher model they mimic. II: The students need multiple convolutional layers to learn functions of comparable accuracy as the deep convolutional teacher.**
- II & I
- II
- I
- Neither
* `[ B ]`


---

**Q: Is there a significant reason to have convolutional networks be deep and convolutional?**
- Generally no, shallow convolutional networks can provide a level of accuracy that is similar to deep networks.
- Sometimes, on some datasets complex functions previously learned by deep networks can be learned by shallow networks.
- Yes, shallow networks perform magnitudes worse than deep networks.
- None of the above.
* `[ B ]`


---

**Q: How does a shallow net match the number of parameters of a deep net?**
- More units in each layer.
- More bias.
- Both A and B
- None of the above
* `[ A ]`


---

**Q: Do deep convolution nets really need to be both deep and convolution?**
- They have to be both deep and convolution
- They have to be deep but not necessarily to be convolution
- They have to be convolution but not necessarily to be deep
- They do not necessarily to be either deep or convolution
* `[ A ]`


---

**Q: Review the following two statements about about student models trained to mimic a high-accuracy ensemble of deep
convolutional models:
\begin{enumerate}
    \item In general, increasing the amount of parameters in student nets increases the accuracy.
    \item In the case of image recognition, the effect of increasing the depth of the convolutional layers is less important than increasing the amount of parameters in student nets.
\end{enumerate}
Which of the statements are true?:**
- Both statements are true
- Statement 1 is true and statement 2 is false
- Statement 1 is false and statement 2 is true
- Both statements are false
* `[ B ]`


---

**Q: For deep and shallow models, which of the following statement is valid?**
-  Shallow models are less accurate than deeper models, if the number of parameters are same.
- Model with 1 layer and large number of parameters is less accurate than the model with 2 layers and less number of parameters.
-  To train the model more accurately, extra layers are required.
- All of the above.
* `[ D ]`


---

**Q: What is the definition of Knowledge Distillation, in specific for deep learning?**
- Knowledge distillation is a model compression method in which a small model is trained to mimic a pre-trained larger model.
- Knowledge distillation is a method for extracting useful and strong features form a given dataset in order to increase the accuracy.
- Knowledge distillation is a model compression method which reduces the size of a given dataset in order to speedup the execution time.
- Knowledge distillation is a method for identifying outlier features in order to reduce the loss of a given CNN. 
* `[ A ]`


---

**Q: Which of the following statements is false?**
- Models trained via distillation can provide extremely accurate models
- Shallow neural networks perform worse than deep neural networks for high-dimensional data
- Several layers of convolutions significantly can improve the accuracy for highly complex data
- A network with a single hidden layer cannot approximate every decision boundary
* `[ D ]`


---

**Q: Which of these models is likely to perform best after training on the CIFAR-10 dataset?**
- An MLP with 1 hidden layer
- An MLP with 3 hidden layers
- An MLP with 5 hidden layers
- A CNN with one convolutional layer
* `[ D ]`


---

**Q: Which of the following conclusions were NOT made from emperical findings in "Do Deep Convolutional Neural Networks Really Need to Be Deep and Convolutional?"**
- Relatively smaller, convolutional models outperform larger, MLP models
- Deep convolutional neural networks need to be deep and convolutional
- Student models learned more from soft classification targets than binary classification targets
- Student models need to be approximately as large as the teacher model to perform well
* `[ D ]`


---

**Q: What are the soft targets provided by a teacher to a student model?**
- logits - the output to the last softmax layer 
- weights for CNN layer parameters
- unlabelled data
- hyperparameters
* `[ A ]`


---

**Q: Regarding the comparison of the accuracy of one shallow model and a deep convolutional model, is it true that increasing the number of hidden layers the accuracy increases?**
- Yes, for both shallow and deep convolutional model
- Yes, but only in the deep convolutional model
- Yes, but only in the shallow model
- No, the accuracy decreases after two hidden layers
* `[ B ]`


---

**Q: Generally speaking, given a fixed number of parameters that can be learned, deep convolutional neural network and shallow non-convolutional network have the same learning capacity?**
- No, shallow networks are better
- Yes, unless the training set is too little
- No, deep networks are better
- Yes, there are no known differences
* `[ C ]`


---

**Q: Which of the following statements about deep and shallow nets is true?**
- If the two nets have the same number of parameters, the shallow net typically needs to be “wider” than the deep net
- A shallow student net can achieve results comparable to its deep net teacher model
- Distillation can be used to “compress” an ensemble model into a smaller model
- All of the above
* `[ D ]`


---

**Q: Which of the following statements is true?**
- Dark knowledge is false knowledge that students learn from their teacher
- A student model copies hidden layers in a trained teacher model and uses it to train itself
- A bottleneck layer reduces the number of parameters that must be learned
- Shallow nets contain few parameters, but a lot of hidden layers
* `[ C ]`


---

**Q: What is true about A shallow net?**
- A shallow network has less number of hidden layers
- A shallow network has more number of hidden layers
- Number of hidden layers do not play important role in this net
- None of above is correct
* `[ A ]`


---

**Q: I. Deep neural networks are always more accurate than shallow neural networks of the same architecture
II. Shallow student neural networks can outperform Deep teacher neural-networks of the same architecture**
- Only I is true
- Only II is true
- I and II are both true
- I and II are both false
* `[ A ]`


---

**Q: Choose the method to train most accurate shallow models on CIFAR-10**
- Distillation
- Hyperparameter Optimization
- Bayesian Optimization
- Either a or b
* `[ C ]`


---

**Q: The paper "Do Deep Convolutional Nets Really Need to be Deep and Convolutional?" discusses distillation and model compression. During distillation, the student network mimics the teacher network by...**
-  Training on the original labelled dataset
- Training on the labels predicted by the teacher network on the original dataset
- Training on the real-valued scores predicted by the teacher network
- Copying the weights of "important" nodes in the teacher network
* `[ C ]`


---

**Q: A shallow net can achieve same accuracy as a deep convolution models , if **
- It has 1 or 2 convolution layers and same number of parameters are used.
- it has no convolution layers and has a single fully connected non linear layer 
- Uses less convolution layers with more number of parameters.
- It can never achieve the same efficiency 
* `[ C ]`


---

**Q: According to the paper “Do deep convolutional nets really need to be deep and convolutional?”, what is the TRUE statement?**
- Soft targets generated by the teacher network used for training of the student network have a lower information density compared to one-hot targets
- A larger student network with one convolutional layer performs better than a smaller student network with more convolutional layers (large/small in term of number of parameters)
- The bottleneck layer significantly speeded up the learning for the student networks with more than 2 convolutional layers
- The most accurate student models with no convolutional layers have the least hidden layers 
* `[ D ]`


---

**Q: How shallow networks compare in accuracy to CNNs?**
- Shallow convolutional models are significantly more accurate than deep convolutional nets, given the same parameter budget.
- Shallow convolutional nets and CNNs achieve appriximately the same accuracy given the same parameter budget.
- Deep convolutional nets are significantly more accurate than shallowconvolutional models, given the same parameter budget.
- They are not comparable because of completely different application domains.
* `[ C ]`


---

**Q: Which of the following is true**
- Mimic networks perform consistently better when trained using dropout
- Non convolution student models mimic deep convolutional models better than convolutional student models
- A deep net has to have more hidden units in each layer to match the number of parameters in a shallow net
- Distillation helps when training models that are limited by architecture 
* `[ D ]`


---

**Q: Indicate the true statement**
- A distilled model derived from a deep convolutional model performs as well as the original model
- A network with a single fully-connected non-linear layer and no convolutional layers are able to learn functions as accurate as deeper convolutional models.
- Shallow convolutional models are significantly more accurate than their deep counterparts
- Shallow  nets  that  contain  only  1-2  convolutional  layers  also  are  unable  to achieve accuracy comparable to deeper models if the same number of parameters are used.
* `[ D ]`


---

**Q: Can shallow networks learn the same as deep networks?**
- No 
- Yes but only by using convolution in the shallow networks
- Yes
- Yes but it requires that the shallow network is trained by a deep network.
* `[ A ]`


---

**Q: Which of the statements is false?**
- The bottleneck layes speeds learning by reducing the number of parameters but it does not make the model deeper.
- A shallow net has less hidden units in each layer than a deep net.
- In model compression, unlabeled data is passed through a large, accurate teacher model to collect scores and then a student model is trained to mimic the scores.
- Distillation often allows smaller and/or shallower models to be trained that are nearly as accurate as larger, deeper models they are trained to mimic.
* `[ B ]`


---

**Q: Shallow nets are able to achieve accuracy comparable to deeper models, but only..**
- If the same number of parameters are used in the shallow and deep models.
- If many more parameters are used in the deep models.
- If many more parameters are used in the shallow models.
- Shallow nets are unable to achieve accuracy comparable to deeper models.
* `[ D ]`


---

**Q: Based on the paper, which of the following statements best answers the question: do deep convolutional nets really need to be deep and convolutional?**
- Yes, they are significantly more accurate than shallow convolutional networks.
- No, models can heavily be reduced.
- They should at least be deep.
- They should at least be convolutional.
* `[ A ]`


---

**Q: Which of the following are false**
- Student models are better trained on the final logits of the teacher model, rather than the softmax outputs
- Student  models  are  better  trained  on  the  softmax  outputs  of  the teacher model, rather than the last logits
- Distilled models having fewer convolutional layers compared to their teacher models allows for the production of models with comparable accuracy as their teachers
- None
* `[ B ]`


---

**Q: As discussed in the paper - 'Do Deep Convolutional Nets Really Need to be Deep and Convolutional?', what do the student models learn from their teacher models?**
- Represent weights of teacher model in an alternative combinations in different layers.
- Learn function mapping the input to output
- Learn smaller nuances in the training set data missed by the teacher model
- None of the above
* `[ B ]`


---

**Q: Which of the following statements is supported by the paper
1. Convolution will always increase accuracy
2. With the same number of parameters multiple layers will always outperform a single layer network**
- Both are supported
- Only 1 is supported
- Only 2 is supported
- Both are NOT supported
* `[ D ]`


---

**Q: What kind of state-of-the art algorithm was used to teach the shallow nets?**
- SNN-ECNN-mimic 128 convolutions
- 3 conv. layers, 3 max-pool
- Ensemble of 16 CNN
- Emsemble of 4 CNN
* `[ C ]`


---

**Q: If a deep neural net has m layers and n neurons in each layer, a shallow neural net designed to mimic the deep net should have?**
- More layers than the deep net
- More neurons in each layer than the deep net
- Same number of layers and neurons as the deep net
- Less number of neurons in each layer as the deep net
* `[ B ]`


---

**Q: Which of the following statements is correct?**
- Shallow models trying to mimic deep models usually outperform shallow models trained on raw input data.
- Shallow models with slightly higher number of parameters can outperform deep models.
- Shallow models usually outperform the mimicked deep models.
- Adding more convolutional layers without changing the number of parameters usually makes performance worse.
* `[ A ]`


---

**Q: In case of biomedical application artificial deformations are added to input image**
- It makes the system invariant to tissue structural changes
- It reduces the need of number of  annotated images needed to train the system
- It makes the system much more robust 
- All of the above
* `[ D ]`


---

**Q: Which technique most likely WON'T help increase the output accuracy of a CNN?**
- Adding more convolutional layers.
- Using model compressing.
- Adding extra regularization (e.g. dropout).
- Building a ensemble of multiple CNNs.
* `[ C ]`


---

**Q: What statement is true:
For complicated tasks, ______ performs as good as a Deep CNN
1. A shallow convolutional NN with the same number of parameters
2. A shallow NN without convolutional layer.**
- 1
- 2
- both
- neither
* `[ D ]`


---

**Q: Given the same parameter budget, which of the following nets provide the highest accuracy?**
- Small neural nets
- Single fully-connected non-linear nets
- Shallow convolutional nets
- Deep convolutional nets
* `[ D ]`


---

**Q: Mark the false sentence.**
- Distillation helps when training models that are limited by architecture or number of parameters.
- The student models have to be as deep as the teacher model they mimic, in order to learn functions of comparable accuracy as the deep convolutional teacher.
- Deep convolutional nets do need to be both deep and convolutional, even when trained to mimic very accurate models via distillation.
- Deep convolutional nets are significantly more accurate than shallow convolutional models, given the same parameter budget.
* `[ B ]`


---

**Q: What is the key idea behind model compression?**
- Trying to get as many parameters that are the same
- Sacrificing performance for speed
- Efficiently storing the parameters of a large model
- Training a compact model to approximate the function
learned by another larger, more complex model.
* `[ D ]`


---

**Q: In paper for construction of deep neural networks appears the layer linear bottleneck layer. Which statement about this layer is TRUE?**
- It is differently called combination of max pooling layer and fully connected layer
- It is layer which performs convolution on filter 4x4
- Aim of this layer is to fasten the training process
- Acts like RELU layer
* `[ C ]`


---

**Q: Do Deep Convolutional Nets Really Need to be Deep and Convolutional? Which of the following statements is FALSE?**
- Distillation helps when training models that are limited by architecture and/or number of parameters.
- If one controls for the number of learnable parameters, nets containing a single fully-connected non-linear layer and no convolutional layers are not able to learn functions as accurate as deeper convolutional models. 
- Deep convolutional nets are as accurate as shallow convolutional models, given the same parameter budget.
- Shallow nets that contain only 1-2 convolutional layers are unable to achieve accuracy comparable to deeper models if the same number of parameters are used in the shallow and deep models. 
* `[ C ]`


---

**Q: Which of the following networks would perform best according to the paper?**
- A CNN with 4 convolutional layers with 10 million parameters. 
- A CNN with 1 convolutional layers with 10 million parameters.
- A student model trained to mimick an ensamble of 16 CNNs with 10million parameters
- An ensamble of 16 CNNs with multiple configurations. 
* `[ D ]`


---

**Q: What is Model Compression?**
- Training a compact model to approximate the function of a more complex model
- Decreasing the size of a model by removing random parts of it.
- Changing the outputs of a model to a smaller fixed possible set of outputs.
-  None of the above
* `[ A ]`


---

**Q: Model Compression is the act of**
- compressing the final output dimension of a model
- compressing the output dimension of every layer in a model
- compressing the domains of the outputs of a model on every layer
- training a compact model to approximate the function learned by a larger model
* `[ D ]`


---

**Q: According to the authors, how could you improve a student model, that is based on an ensemble of CNN's, which currently has 10 million parameters and no convolutional layer?**
- Increase the number of parameters.
- Add a convolutional layer.
- Increase the amount of hidden layers from 1 to 2.
- All of the above.
* `[ D ]`


---

**Q: Student models that mimic the deep neural network is trained as a regression problem on which of the following output labels?**
- Actual classes
- Softmax output of the deep net
- Log probabilities before softmax
- None of the above
* `[ C ]`


---

**Q: Why is the architecture of a Neural Network important?**
- The architecture itself has features that can help with the performance given that provides internal properties (e.g. frequency selectivity)
- Because Convolutional Neural networks always outperform other type of architectures
- None of the above
- All of the above
* `[ A ]`


---

**Q: The main purpose of data augmentation in deep learning is to:**
- Generate large tests sets 
- This is a technique used only in machine learning, not in deep learning
- Generate large training sets
- Apply filters on noisy images
* `[ C ]`


---

**Q: What is the function of teacher model when we want to design a small model with less parameters?**
- Label the unlabled data
- Reduce the variance of data
- Speed up convergence
- Make the mean value of data zero
* `[ A ]`


---

**Q: One of the key steps for training shallow networks to mimic deeper convolutional networks presented in the “Do deep convolutional nets really need to be deep and convolutional?” paper is model compression. How does model compression work?**
- Pass the unlabeled data through the large, accurate teacher, collect the scores it predicts and train a student model to mimic these scores.
- Pass the labeled data through the large, accurate teacher, collect the score it predicts, add these as new features to the data and fit the student model on this new data.
- Tune the hyperparameters of the large, accurate teacher on the labeled data and build the student model using these hyperparameters.
- Tune the hyperparameters of the large, accurate teacher on the labeled data and build the student model using the same architecture as the teacher, but removing the convolution layers.
* `[ A ]`


---

**Q: Regarding the training of a single-layer fully-connected network with NO convolution, how is the technique used in the paper "Do Deep Convolutional Nets Need To Be Deep and Convolutional"? called?**
- Back propagation
- Distillation
- Segmentation
- SGD
* `[ B ]`


---

**Q: What are the student networks trained on and why?**
- The 1-hot CIFAR-10 dataset. Because this provides the network with the best hard-targets for learning.
- The output of the softmax layer of the teacher network. Because this allows the student to learn from the teacher outputs.
- The logits before the softmax layer. Because this helps provide the "dark knowledge" the student needs.
- On all outputs of all layers in the network. Because the student can learn and simplify the relationships that the teacher learns this way.
* `[ C ]`


---

**Q: Consider a classifier designed to distinguish a football, a chessboard and a basketball. Which of the following desired outcome of the network (for a football) captures the essential benefit of using a deep neural network to train smaller networks compared to the use of hard-labelled data set. (A football has the same shape as a basketball and almost the same coloured pattern as that of a chess board.)**
- Football(0.9), Chessboard(0.3), Basketball(0.5)
- Football(0.5), Chessboard(0.5), Basketball(0.5)
- Football(1.0), Chessboard(0.0), Basketball(0.0)
- Football(0.1), Chessboard(0.3), Basketball(0.5)
* `[ A ]`


---

**Q: which of the following is true?**
- Sometimes, shallow networks that have 1 or 2 convolutional layers can learn the same function learned by a deeper network if the number of parameters are the same.
- Sometimes, by controlling the number of parameters, networks containing a single layer and no convolutional layer can learn the same function learned by a deeper convolutional network.
-  Sometimes, shallow FF nets can learn the same functions learned by deeper networks using the same number of parameters.
- Shallow networks can never achieve the same performance as a deeper network.
* `[ C ]`


---

**Q: Given an expected accuracy, how does the typical parameter budget in Deep Convolutional Neural Network compare to Shallow Convolutional Neural Network?**
- Parameter Budget in Deep CNN > Parameter Budget in Shallow CNN
- Parameter Budget in Deep CNN < Parameter Budget in Shallow CNN
- Parameter Budget in Deep CNN = Parameter Budget in Shallow CNN
- Cannot be generalised
* `[ B ]`


---

**Q: When comparing a deep net with a shallow net, using the same budget of parameters, what is the difference between them regarding performance?**
- Shallow nets have significantly higher performance than deep nets.
- Deep nets have significantly higher performance than shallow nets.
- The performance of deep nets and shallow nets is almost the same.
- The performance of deep nets is slightly better than shallow nets.
* `[ B ]`


---

**Q: What is model compression and how does it work?**
- Model compression works by passing unlabeled data through the teacher model to collect the real-valued scores it predicts, and then training a student model to mimic these scores.
- Model compression works by passing specific labeled data through the teacher model to collect the real-valued scores it predicts, and then training a student model to mimic these scores.
- Model compression works by applying Principal Component Analyses on the teacher model to create a smaller but similar student model 
- Model compression works by training multiple teacher models with different parameters which at some point converge to a smaller student model
* `[ A ]`


---

**Q: Which of the following is NOT true about introducing a linear bottleneck layer between the input and non-linear layer of the shallow nets?**
- bottleneck layer speeds learning by reducing the number of parameters that must be learned
- bottleneck layer does not make the model deeper because the linear terms can be absorbed back into the non-linear weight matrix after learning.
- bottleneck layer makes the shallow nets have more hidden units in each layer as compared to a deep net.
- None of the above
* `[ D ]`


---

**Q: What method to increase the accuracy of a shallow neural net, as opposed to a deep and convolutional neural net, has the greatest effect?**
- Train the shallow net to mimic the deeper net
- Add convolution layers to the shallow net
- Train the shallow net using distillation
- Add more parameters to the layers of the shallow net
* `[ B ]`


---

**Q: A linear bottleneck is used between the input and non-linear layers to speed up the learning. It speeds up the learning by**
- reducing number of parameters but doesn’t make the model deeper.
- reducing number of parameters but makes the model deeper.
- increasing number of parameters but doesn’t make the model deeper. 
- increasing number of parameters but makes the model deeper. 
* `[ A ]`


---

**Q: According to the experiments conducted on this paper,  the students models that were trained on the soft-targets from a teacher:**
- Do not need any extra regularization method as they were provided a significant regularization from the teacher
- Require extra regularization as the regularization from the teacher is not provided to them
- Require some regularization but not as much as the teacher model, as the regularization from the teacher is partially provided to them
- Did not provide any insight about the need or not of regularization
* `[ A ]`


---

**Q: What is true about shallow networks?**
- They try to mimic another deep network architecture
- They learn independently
- They have many layers
- They use the often used softmax-entropy functions at the end of the network
* `[ A ]`


---

**Q: Paper 5 provides the first empirical demonstration that deep
convolutional models really need to be both deep and convolutional. They do so by training student models to mimic deep trained nets, and showing a higher accuracy through certain modifications to the student model. What is not one of these modifications shown to improve accuracy?**
- Adding a convolutional layer
- Adding depth, but without convolutional layers
- Adding more than one convolutional layer
- None, all of the above modifications improve accuracy
* `[ B ]`


---

**Q: do more layers usually give more accuracy?**
- yes, although the difference can be tiny
- no, shallow student convolutional networks can be trained that are exactly as accurate as their deeper teachers
-  yes, the accuracy scales with a constant factor
- no, a shallow network exists that can outperform any deep network.
* `[ A ]`


---

**Q: What represents "model compression" best?**
- Train a compact model to approximate the function learned by another, larger complex model
- Compressing the coded model to reduce file size
- Leaving out ReLU and/or pool layers to compress the model
- Train a large model to approximate the function learned by another, more compact model
* `[ A ]`


---

**Q: Given is a student-teacher learning model where the student use soft targets, instead of 0/1 hard targets, as labels. The student is a CNN and the teacher an ensemble of CNN's. Which of the following statements is NOT true about this way of learning?**
- The soft targets place an emphasis on the relationships learned by the teacher across all of the outputs. 
- The soft targets have a high information density and provide regularization by reducing the impact of brute-force memorization.
- The accuracy of the student trained on soft targets will likely be higher than the same model trained on hard targets.
- The convolution learned by the student will always be the same as the convolution learned by the teacher. 
* `[ D ]`


---

**Q: What is distillation in machine learning?**
- Reduce large dataset to smaller dataset
- Training small model using larger model
- Training large model using smaller model
- Using an ensemble of models for a more accurate result
* `[ B ]`


---

**Q: Which of the following is false:**
- Model compression means training a simpler model to mimic a more complex model
- Models trained through distillation perform better than similar models trained on the original dataset
- On CIFAR-10 dataset student models with convolutional layers achieve worse accuracy than ones without but deeper
- For the same number of parameters a shallow model cannot have a higher accuracy than the deeper model
* `[ C ]`


---

**Q: What is not true regarding model compression?**
- When using teacher and student models it is important to apply regularization to the student models otherwise the student model easily overfits.
- Teacher models will yield better accuracy than student models trained with said teacher models.
- Models with significantly (orders of magnitude) less parameters can still be as accurate as models with more parameters.
- Student models are trained with un-labeled data as opposed to labeled data.
* `[ A ]`


---

**Q: What is the better model if the number of parameters is fixed?**
- Compressed shallow model because it enables to achieve high accuracy and have fewer convolutional layers
- Deep convolutional model because they always perform better than shallow nets given the same number of parameters
- Compressed shallow model because deep convolutional nets always require more parameters than shallow models for better accuracy
- Shallow models because their implementation is easier
* `[ B ]`


---

**Q: The best shallow model trained by using:**
- A linear bottleneck layer while it is constructed with multiple max pooling layers
- The dropout method
- The distillation method
- One convolutional layer
* `[ A ]`


---

**Q: Which statement about student and teacher models is not correct? **
- The teacher uses unlabeled data to give probabilities as outputs
- Student models can mimic ensembles of teachers
- Student models are more shallow than the teacher model they try to mimic
- Student models can turn out to be more accurate than their teacher
* `[ C ]`


---

**Q: Which of these statements about deep convolutional networks is incorrect?**
- Deep convolutional nets are significantly more accurate than shallow convolutional models no matter the parameter budget
- Model compression allows accurate models to be trained that are shallower and have fewer convolutional layers than deep convolutional architectures
- Depth-constrained student models trained to mimic high-accuracy ensemble of deep convolutional models perform better than similar models trained on the original hard targets
- An ensemble of multiple deep convolutional neural nets, each trained with different hyperparameters, and each seeing slightly different training data results in a more accurate model
* `[ A ]`


---

**Q: What reason would you have to distill a deep, convolutional network to a shallower one, with similar number of parameters?**
- To speed up implementation of the network
- To speed up training of the network
- To increase training accuracy
- To give insight in how the network learns
* `[ A ]`


---

**Q: What is the time complexity of a layer of a feed forward neural network with 2 layers of n and m perceptrons and a linear bottleneck layer of k perceptrons?**
- O(n*m*k)
- O(n*m+k)
- O(n*k + m*k)
- O((n*m)^k)
* `[ C ]`


---

**Q: Assuming all models are trained with the same teacher and have the same number of parameters, which of the following models is most accurate?**
- CNN with 1 convolutional layer
- CNN with 2 convolutional layer
- MLP with 1 hidden layer
- MLP with 4 hidden layer
* `[ B ]`


---

**Q: A key concept underlying distillation and model compression techniques in neural networks is the propagation of “dark knowledge”. Which of the following concepts better describe this notion?**
- A Bayesian hyperparameter optimization for the student network is based on the hyperparameters of the hidden layers of the teacher netwrok
- Soft targets contain valuable information on how the teaching network works
- The data augmentation that is used on training the teacher neural network defines the predictability of the student neural network
- The ensemble set of trained NNs outperforms any single NN implementation 
* `[ B ]`


---

**Q: What is the benefit of a shallow CNN vs a deep CNN**
- A shallow CNN computes accurate results faster than a deep CNN
- A shallow CNN computes results faster than a deep CNN
- a shallow CNN computes more accurate results
- a shallow CNN requires less computational power than a deep CNN
* `[ D ]`


---

**Q: What is Dark knowledge?**
- Information contained in the relative probabilities provided by a teacher algorithm
- Information from the darker parts of an image
- Contributions from the deeper layers in the network to the probabilities
- Hyperparameters chosen by the authors, akin to the term "black art"
* `[ A ]`


---

**Q: what is not a correct step to train shallow student models**
-  train state-of-the-art deep convolutional teacher models,
- form an ensemble of the best deep models,
- collect and combine their predictions on a large transfer set
- make the network each iteration bigger
* `[ D ]`


---

**Q: What is FALSE about using a linear bottleneck?**
- The benefit is that it speeds up the training.
- The linear terms can be absorbed back into the non-linear weight matrix after learning.
- Reduces the numbers of parameter that must be learned and make the model deeper.
- None of the above, all statements are true.
* `[ C ]`


---

**Q: Statement 1: A network with a large enough single hidden layer of sigmoid units can approximate any decision boundary. 
Statement 2: The bottleneck layer speeds learnign by reducing the number of parameters that must be learned, but does not make the model deeper. **
- Both statements are true
- Statement 1 is true
- Statement 2 is true
- None of the statements are true
* `[ A ]`


---

**Q: What is true about the performance of shallow models and deep models?**
- shallow models perform better than deep models
- shallow models' performance is better if trained with trained data of deep models
- shallow models' performance is better if trained to mimic deep models
- shallow models always perform worse than deep models
* `[ C ]`


---

**Q: Given that the same number of parameters are used, which of the following are true?**
- Nets with no convolutional layers learn functions as accurately as deeper convolutional models.
- Shallow nets with 1-2 convolutional layers learn functions as accurately as deeper models.
- Deeper convolutional nets are significantly more accurate than the shallow convolutional models.
- None
* `[ C ]`


---

**Q: Consider following statement: "Deep convolutional nets are ... shallow convolutional models, given the same parameter budget." If there is a correct one, which answer should be on the empty dotted space?**
- equally accurate as
- significantly less accurate than
- significantly more accurate than
- None of the above answers is correct.
* `[ C ]`


---

**Q: Which following statement is not true about hard targets and soft targets mentioned by Hinton ?**
- Both hard targets and soft targets could be used for transferring knowledge from teacher model to student model.
- Soft targets are probabilistic distributions of different categories when training a classification model.
- Soft targets are similar to 'one hot' encoding. Distributions are deterministic, one entry be 1 and the rest be 0.
- Soft targets have higher entropy than hard targets and could provide more information.
* `[ C ]`


---

**Q: Why are more layers needed to learn accurate models from one hot encoded vectors as opposed to the output of a teacher network?**
- This is currently an unanswered question
- The model pre-extracts certain data and encodes it as "dark information" 
- One-hot encoded vectors require more regularization
- This is not necessary.
* `[ A ]`


---

**Q: In the paper: "DO DEEP CONVOLUTIONAL NETS REALLY NEED TO
BE DEEP AND CONVOLUTIONAL", which of the following is shown empirically by the authors**
- Shallow student models with convolution layers performed better than shallow student models with no convolution layers even though they had the same depth
- There was very little difference in accuracy between student models with and without a convolutional layer
- Shallow student models without convolution layers performed better than student models of similar depth but with a convolution layer.
- None of the above
* `[ A ]`


---

**Q: In the paper ``Do Deep Convolutional Nets Really Need to be Deep and Convolutional?” to answer the question the accuracy of shallow replicas is compared to the deep networks. Which of the following statements are true?

Statement1 The key idea behind model compression is to train a compact model to approximate the function learned by another larger, more complex model.

Statement2 Adding depth to the student MLPs (Multilayer Perceptrons) without adding convolution does not significantly close this “convolutional gap”.

Statement3 Comparing student CNNs with 1, 2, 3, and 4 convolutional layers, it is clear that CNN students will not benefit from multiple convolutional layers.

Statement4 We are able to compress deep convolutional models to shallow student models without significant loss of accuracy.**
- All statements are true 
- 1 2 and 3
- 2 3 and 4
- 1 and 2
* `[ D ]`


---

**Q: which of the following statements is wrong?**
- Shallow nets that contain 1 or 2 convolutional layers are unable to achieve accuracy comparable to deeper models with the same parameters.
-  Mimic networks perform consistently worse when trained using dropout.
-  Deep convolutional nets are more accurate than shallow convolutional models, given the same parameter budget.
-  it is not possible to gain increases in accuracy in models with few parameters by training deeper, thinner nets
* `[ D ]`


---

**Q: if the same number of parameters are used in the shallow and deep learning models what is true about those next statements.
1: Deep convolution  nets are significantly more accurate that shallow netwoks 
2: Mode compression allows accurate models to be trained that are shallower and fewer convolutional layers than deep convolutional architectures.**
- I
- II
- I & II 
- None
* `[ C ]`


---

**Q: Which of the following statements is true?
I: Deep convolutional nets are significantly more accurate than shallow convolutional models.
II: Model compression can not lead to shallower nets with similar performance.**
- Only statement I is true
- Only statement II is true
- Both statements are true
- Both statements are false
* `[ A ]`


---

**Q: Distillation is ...**
- ... a chemical process to decrease the alcohol percentage.
- ... a way to make improve backpropagation.
- ... a process that helps in model augmentation.
- ... a process that helps in model compression.
* `[ D ]`


---

**Q: The performance of a neural network is dependent on**
- the feature learning algorithms
- the architecture of the network
- Both A and B
- None of the above
* `[ C ]`


---

**Q: When shallow nets are trained by deep nets**
- They learn functions equally accurate as on training data
- They learn functions more accurate as on training data
- They learn functions less accurate as on training data
- Shallow nets cannot be trained by deep nets
* `[ B ]`


---

**Q: Which of following techniques is NOT used in the experiments.**
- Use a linear bottleneck to speed up training.
- Use Bayesian hyperparameter optimization to explore hyperparameters.
- Make use of weight decay and dropout to optimize the networks.
- Generate extra large training sets from the original images.
* `[ C ]`


---

**Q: Do Deep Convolutional Nets Really Need To Be Deep And Convolutional' describes how shallow models can be trained using the student-teacher model by training a student as:**
- A regression model of the teacher weights.
-  A regression model of the teacher hyperparameters.
-  A CNN trained on the teacher weights.
- A CNN trained on the output of the teacher.
* `[ A ]`


---

**Q: Which of the following is false about shallow (convolutional) nets vs deep convolutional nets?**
- Single fully-connected non-linear layer is not able to learn functions as accurate as deeper convolutional models
- Deep convolutional nets are significantly more accurate than shallow convolutional models (given the same amount of parameters).
- Training shallow nets that contain 1-2 convolutional layers are able to achieve accuracy comparable to deeper models
- Model compression allows accurate models to be trained that are shallower and have less convolutional layers than the deep convolutional ones learned from the original training data
* `[ C ]`


---

**Q: The random weights do well in certain architectures object recognition tasks because **
- Certain convolutional pooling work better in a random weights
- Random weights less error
- Random weights are convergence fast
- none
* `[ A ]`


---

**Q: Which of these parameters is NOT optimized in the training of Shallow Student Models? **
- initial learning rate
- momentum
- hidden units in each layer
- dropout
* `[ D ]`


---

**Q: Which statement is wrong?**
- Excessive data augmentation by applying elastic deformations to the available training images is very useful in biomedical augmentation, because deformation used to be the most common variation in tissue and realistic deformations can be simulated efficiently.
- Data augmentation is essential to teach the network the desired invariance and robustness properties, when only few training samples are available.
- Better performance with fewer training images is realized by supplementing a usual contracting network by successive layers, where pooling operators are replaced by upsampling operators.
- In many cell segmentation tasks, the separation of touching objects of the same class is quite easy
* `[ D ]`


---

**Q: Please complete the following sentence. Adding a linear bottleneck layer between the input and non-linear layers of a shallow mimic model is used for:**
- Increasing the shallow net accuracy.
- Speeding up the learning of shallow networks.
- Allowing non-linear feature mapping.
- Hyperparameter optimization with Bayesian optimisation.
* `[ B ]`


---

**Q: Which of the following is TRUE about the linear bottleneck layer?**
- It speeds up training
- It does not make the model deeper
- It reduces the number of parameters that must be learned
- All of the above
* `[ D ]`


---

**Q: If you have a fixed number of learning parameters, and two nets: one containing a single, fully connected, non-linear layer and no convolutional layers and a second deeper convolutional model. Which would you expect to learn the required functions more accurately?**
- The first model.
- The second model.
- Absolutely impossible to say.
- They would be equally as good due to the fixed number of learning parameters.
* `[ B ]`


---

**Q: Which of the following statements is false?**
- If one controls for the number of learnable parameters, nets containing a single fully-connected
non-linear layer and no convolutional layers are not able to learn functions as accurate as deeper
convolutional models.
- Shallow nets that contain only 1-2 convolutional layers also are unable
to achieve accuracy comparable to deeper models if the same number of parameters are used in
the shallow and deep models.
- Deep convolutional nets are just as accurate than shallow
convolutional models, given the same parameter budget
- There is evidence that model compression allows accurate models to be trained that are shallower and have fewer convolutional layers than the deep convolutional architectures needed to learn high-accuracy models from the original 1-hot hard-target training data.
* `[ C ]`


---

**Q: Regarding deep or shallow convolutional nets, what statements is true?**
- Deep convolutional nets are significantly more accurate than shallow
convolutional models, given the same parameter budget.
- Deep convolutional nets are significantly more accurate than shallow
convolutional models, given a specific deep parameter budget.
- To learn high-accuracy models from the original 1-hot hard-target training data, the depth is not influential to its accuracy.
- To learn high-accuracy models from the original 1-hot hard-target training data, a shallow depth does suffice.
* `[ A ]`


---

**Q: Which of the following statements is true?**
- The paper also shows that adding additional depth to student MLP’s WITHOUT adding convolution DOES NOT close the gap between CNN’s and MLP’s for image recognition tasks.
- The paper also shows that adding additional depth to student MLP’s WITH adding convolution DOES NOT close the gap between CNN’s and MLP’s for image recognition tasks.
- The paper also shows that adding additional depth to student MLP’s WITHOUT adding convolution DOES close the gap between CNN’s and MLP’s for image recognition tasks.
- The paper also shows that adding additional depth to student MLP’s WITH adding convolution DOES close the gap between CNN’s and MLP’s for image recognition tasks.
* `[ A ]`


---

**Q: Which model architecture is expected to perform the best on a computer vision image classification task?**
- A fully connected 3-layer network with 10M parameters.
- A fully connected 3-layer network with 1M parameters.
- A convolutional 8-layer network with 10M parameters
- A convolutional 8-layer network with 1M parameters
* `[ C ]`


---

**Q: Why is Teacher-Student method more efficient in machine learning training?**
- The model has been used in human education for years, thus proven to be working efficiently.
- The correct label given by the teacher algorithm is not a single class as is the case with human labelled data. Consequently, the student learns subtleties of the input data.
- This method is not more efficient at all
- none of the above
* `[ B ]`


---

**Q: What is the model compression method described in the paper?**
- Once trained models are compressed into prediction models that cannot be trained anymore but have very fast prediction computation time.
- A teacher model is split into several student models that each only train with a subset of the training data, to avoid over-fitting. 
- Student models are trained to mimic the results as a parent model does on unlabeled data.
- Uninformative parameters are removed to reduce the memory needed for the model to train.
* `[ C ]`


---

**Q: According to the paper about the deep convolutional networks, which one is true?**
- Training the shallow networks(student) from the convolutional model(teacher) produces more accurate results
- Training the shallow networks(student) from the convolutional model(teacher) is not as efficient as using the main data
- Shallow networks need less parameters to be as accurate as convolutional networks
- Student models do not need to be convolutional to perform good
* `[ A ]`


---

**Q: To what extent can a single layer approximate a complex dataset/boundary?**
- Both in theory and practice, single layers often represent complex datasets/boundaries
- In theory, a large enough single layer can represent a complex boundary, however, it has been shown empirically that this is difficult to achieve in practice
- Single layers can never accurately represent a complex boundary
- Theoretically, single layers cannot approximate a complex dataset/boundary, however, in practice, the results are often very good.
* `[ B ]`


---

**Q: If you want to mimic a deep model using a deep model using a shallower model, what is the most possible sequence of those steps below: a.train deep convolutional teacher models b. form an ensemble of the best deep models c.collect and combine predictions on a large transfer set d.optimize shallow
student models**
- abcd
- cbad
- bcad
- adbc
* `[ A ]`


---

**Q: Which of the following is/are part of the training steps for a shallow student networks?**
- Training state-of-the-art deep convolutional teacher models.
- Forming an ensemble of the best deep models.
- Combining the predictions on a larger data set.
- All the above.
* `[ D ]`


---

**Q: Which of the following neural networks is most accurate, given that the same number of parameters are used in all of them?**
- A single, fully connected, non-linear layer.
- A shallow net with a few convolutional layers.
- A deep net, with many convolutional layers.
- They all perform the same when working with the same number of parameters.
* `[ C ]`


---

**Q: Which one is wrong **
- Certain convolutional pooling architectures can be inherently frequency selective and translation invariant, even with random weights
- The random-weight networks with convolutional pooling architectures indicates that architecture is important in feature representation for object recognition
- It is sure that architecture search will improve the performance of state-of-the-art systems
- This paper studying the basis of the good performance of these convolutional pooling object recognition systems
* `[ C ]`


---

