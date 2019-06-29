# Questions from `paper_14` :robot: 

**Q: What are out-of-distribution samples?**
- Samples that are not from the training distribution.
- Inputs specifically crafted by an adversary to cause a target model to misbehave.
- Samples that are neither in the training nor in the testing distribution.
- a) and b). 
* `[ D ]`


---

**Q: What is the problem with having a 3D renderer which includes non-differentiable operations?**
- it is computationally too expensive to use, if the training set is big
- feature maps cannot be visualised
- It is not possible to optimize the parameters of the render by means of standard gradient descent
- Allows only the use of gradient descent but not stochastic gradient descent.
* `[ C ]`


---

**Q: Neural networks have problems with out-of-distribution inputs. But does it matter?**
- mainly theoretical problem
- somewhat practical problem
- real problem with minor consequences
- real with problem with possible catastrophic consequences
* `[ D ]`


---

**Q: How does the rendering speed of differentiable and non-differentiable renderers compare?**
- Both are equally fast.
- Differentiable renderers are faster.
- Non-differentiable renderers are faster.
- Depends on the dataset.
* `[ C ]`


---

**Q: What is the axis that needs the smallest change that would lead to misclassification?**
- Yaw
- Roll
- Pitch
- All three
* `[ C ]`


---

**Q: Which statement is true?**
- The reason for misclassification an object can be due to that the ImageNet training-set images themselves may contain a strong bias towards common poses, omitting uncommon poses.
- The reason for misclassification an object can be due to that the models themselves may not be robust to even slight disturbances of the known, in Distribution poses.
- One of the most effective methods for defending against OoD examples has been adversarial training.
- All of the above statements are true.
* `[ D ]`


---

**Q: Deep neural networks that classify images can be easily fooled when the targets in that image are 3d rotated and/or translated. What could be a way to create a more robust classifier (more invariant to translation and rotation of the input)?**
- Generalize more by using (smaller) parts of the training set
- Generalize more by adding adversarial examples to the training set
- Specialize more by doing more training iterations
- Specialize more by using a TCN
* `[ B ]`


---

**Q: Neural Networks are easily fooled by strange poses of familiar objects. How could that be improved?**
- Use data augmentation to create a training set distribution that covers more diverse object poses
- Train the network for a longer period of time
- Use a larger training set. The data distribution within the training set can be the same as before
- Avoid strange object poses during testing under any circumstances
* `[ A ]`


---

**Q: Which of the following statements are true? (Based on the paper "Strike (with) a Pose: Neural Networks Are Easily Fooled by Strange Poses of Familiar Objects")

statement1: In sum, our work shows that state-of-the-art DNNs perform image classification well but are still far from true object recognition. 

statement2: One of the most effective methods for defending against
OoD examples has been adversarial training, i.e. augmenting
the training set with AXs.

statement3: Adversarial examples are inputs specially crafted by an adversary
to cause a target model to misbehave. 

statement4: Despite excellent performance on stationary test sets, deep neural networks (DNNs) can fail to generalize to out-of-distribution (OoD) inputs, including natural, nonadversarial ones, which are common in real-world settings.**
- 2 and 3 
- 1 and 4
- 1 2 and 3
- All statements are true
* `[ D ]`


---

**Q: In order to tackle the naive understanding of DNNs with respect to objects, the paper does not propose:**
- To address the biased data.
- Classify arbitrary poses.
- Use strong 3D geometric priors.
- Use data augmentation as a tool.
* `[ B ]`


---

**Q: Complete the following statement: "Deep neural networks are ... sensitive to pose perturbations." Tick the most correct answer.**
- highly
- quite
- only a little
- not at all
* `[ A ]`


---

**Q: What is one of the major weaknesses deep neural networks?**
- That deep neural networks is an architecture that cannot be adapted to new problems relatively easily, like speech but only for images.
- That they can fail to generalize to out-of-distribution (OoD) inputs, which can come from real-world settings.
- That once you use a different training and test set, deep neural networks only work for a specific image set.
- That deep neural networks need to be deep in terms of amount of layers to work.
* `[ B ]`


---

**Q: Which of the following is not true about ML models with OoD inputs:**
- Adversarial training of DNNs , helps them well to held out objects
- Out of distribution inputs can be obtained by 3D translations and 3D rotations
- DNNs misclassification is uniform over the whole pose space
- A misclassified object by DNN can be misclassified as many different labels
* `[ A ]`


---

**Q: How does the paper propose to solve the problem of strange poses of familiar objects?**
- Pose transformation using a 3D 3x3 rotation matrix 
- Using constrained random search
- Doing gradient descent with finited inference in the cross-entropy loss
- All of the above
* `[ D ]`


---

**Q: What are out-of-distribution (OoD) inputs?**
- unfamiliar data that deviate systematically from those that the model saw during training, and which may cause adverse and unpredictable outcomes.
- Inputs drawn from different distributions to fool the network
- inputs from a faulty distribution, e.g.: a biased coin/dice
- There is no such thing as OoD.
* `[ A ]`


---

**Q: Which of the following is \emph{not} true about the classification of Deep Neural Networks on objects with out of distribution poses:**
- Miss-classification uniformly covers the pose space
- An object can be miss classified as many different labels
- correct classificiations are not localized in the rotation and translation landscape
- A large number of labels are shared across lighting settings.
* `[ C ]`


---

**Q: Which rotation easily confuses neural networks?**
- Yaw
- Pitch
- Roll
- All of the above
* `[ D ]`


---

**Q: Which statement is true? **
- The batch normalization error increases rapidly when the batch size becomes smaller
- The batch normalization error decreases rapidly when the batch size becomes smaller
- The batch normalization error does not depend upon the batch size 
- The batch normalization error converges to 0 when increasing the batch size to infinity 
* `[ A ]`


---

**Q: What alterations to an image consisting of a main object placed on a background image do you expect would not cause a DNN image recognition architecture to classify the altered image incorrect?**
- Rotation of the object within the image.
- Severe change of the lighting of the image.
- Major alterations to the background image.
- Translating the main object in the image.
* `[ D ]`


---

**Q: In the paper "Strike (with) a Pose: Neural Networks Are Easily Fooled by Strange Poses of Familiar Objects", what is considered an Out-of-Distribution sample?**
- A naturally occurring difference between the real data and the sample data
- A sample created by an adversarial network in order to fool the discriminator
- A sample which is not represented in the distribution of the training data
- A sample which has been augmented with random noise
* `[ C ]`


---

**Q: Which of the following statements are false?
(1) When an autopilot car failed to recognize a white truck against a bright-lit sky and therefore crashes into the truck is considered as a Type 1 classification errors
(2) (Deep Neural Network) Classifiers are easily confused by small object rotations and translations
(3) Adversarial training is an efficient method for defending (Deep Neural Network) Classifiers against out-of-distribution (classes not found in the training set) inputs.**
- -1
- -2
- -3
- None of the statements is false
* `[ A ]`


---

**Q: What is NOT true about deep neural networks (DNNs) and out-of-distribution (OoD) inputs?**
- In general, Machine Learning models are vulnerable to both adverserial and natural OoD inputs.
- DNNs robustness can be improved through adversarial training with more 3D objects such as rotations and translations.
- DNNs are robust to object rotations and translations.
- DNNs are robust to different lightning settings.
* `[ C ]`


---

**Q: Classification on 3D objects can be done based on existing 2D object theory **
- True  
- False 
- Nothing can be said in general  
- none of them 
* `[ C ]`


---

**Q: What is a method of allivating the adversarial pose problem?**
- Adressing biased data by using data augmentation
- Adding strong 3d geometric priors to the model
- Generate data using 3D renders
- All of the above
* `[ D ]`


---

**Q: Paper 14:
Which of the following is not a result found in the paper:**
- Adversarial training data improves classification on poses included in the data set.
- Classification was sensitive to a change in rotation.
-  Classification was sensitive to a change in illumination. 
- Misclassification of 'adversarial poses' generalizes across different classifiers.
* `[ C ]`


---

**Q: How did the authors in the paper "Strike (with) a Pose: Neural Networks Are Easily Fooled by Strange Poses of Familiar Objects" manage to improve the performance of the Deep Neural Networks in the out-of-distribution circumstances?**
- By using a Convolutional Neural Network that makes use of self-attention to learn all kind of properties about the image
- By using a Convolutional Neural Network that processes 3D images and classifies them based on the angles of the object
- By using a Generative Adversarial Network that generates images based on the images in the given dataset and classifies these images with the discriminator to help the image classifier
- By using an Adversarial Network that generates images based on the images in the given dataset in multiple poses
* `[ D ]`


---

**Q: Review the following two statements about deep neural networks trained on images:
\begin{enumerate}
    \item Deep neural networks can perform very badly for out of distribution inputs.
    \item Deep neural networks have shown to be better at object classification than at image classification.
\end{enumerate}
Which of the two statements are true?**
- Statement 1 and 2 are true
- Statement 1 is true and statement 2 is false
- Statement 1 is false and statement 2 is true
- Statement 1 and 2 are false
* `[ B ]`


---

**Q: What change in pose had a negative effect on the performance of the model?**
- Rotation
- Translation in the X dimension
- Translation in the Y dimension
- All of the above
* `[ D ]`


---

**Q: Which of the following methods can be used to optimise a non-differentiable function?**
- Gradient descent with finite difference
- Random search
- Both a and b
- Bi-conjugate gradient method
* `[ C ]`


---

**Q: How could we make geometric and texture perturbations look similiar**
- Make the perturbations quite small
- Make the perturbations quite large
- Change camera parameters
- None
* `[ A ]`


---

**Q: What is the main problem addressed in this paper ?**
- The fact that DNN's are bad at doing human pose estimation
- Pose estimation with kinect
- The fact that the classification of an object strongly depends on it's pose 
- None of the above
* `[ C ]`


---

**Q: What can not be a problem with a image dataset captured by persons.**
- The dataset is biased, i.e. persons like to capture certain object in a certain way. (from the front and in the middle of the frame)
- The data set does not include "weird" poses, and therefore the network cannot classify this.
- The photos are all take in the same weather conditions and therefore the network includes the lightning and background in the classification.
- Because people make the photo's they can be non realistic. 
* `[ D ]`


---

**Q: Which of the following is true concerning pose changing of images?**
- An object normally is only misclassified to a low amount of labels
- In general some poses cause more misclassifications than others
- A property of AX models is that they transfer across machine learning models, allowing for black box attacks
- Adversarial  training with AX models  does  not  work  very  well  for  defending  against  out  of pose examples
* `[ C ]`


---

**Q: In the "Strike (with) a Pose" paper, what changes in the image rendering were found to change the classification of a given image?
I: Brightness of lighting
II: distance from the perspective of the viewer
III: rotations upon an axis (roll, pitch, yaw)
IV: position in 2D frame**
- I, III, & IV
- II, III, & IV
- II & III
- III & IV
* `[ C ]`


---

**Q: Which statement isn't true?**
- DNNs can only label a small subset of the entire pose space of a 3D-object
- ML model often wrongly classify OoD-examples with high confidence
- DNNs are highly sensitive to slight pose perturbations
- All the statements above are true
* `[ D ]`


---

**Q: What is an effective method to correctly classify out-of-distribution examples?**
- Enlarging the training set
- Stepping away from deep learning and using a nearest mean classifier
- Adversarial training
- Resizing all images to the same size
* `[ C ]`


---

**Q: What is NOT a reason that state-of-the-art DNNs cannot cope with objects in a slightly different pose?**
- Photographs that are used for these networks are taken by people, so they reflect aesthetic tendencies of their captors
- A network learns representations, not 3D objects
- These networks are not trained on every rotation objects
- The lighting changes too drastically
* `[ D ]`


---

**Q: The paper proposes a method for testing a computer vision DNN. What is the key idea to test this pose sensitivity?**
- Back propagate the gradient through a differentiable renderer to evaluate the sensitivity
- Use the CIFAR-10 data set to test upon
- Learn the error through yet another network
- Correlating the effect of dropout to the pose sensitivity
* `[ A ]`


---

**Q: Which transformation affects the performance of a deep neural network the most?**
- Translation by the x axis
- Translation by the y axis
- Translation by the z axis
- Rotation by the x axis
* `[ C ]`


---

**Q: Which of these statements is false**
- There is no need to have prior information on the geometry of objects you want to recognise with a DNN
- A DNN has difficulties identifying objects in an unexpected pose
- Lighting influences a DNN's ability to identify an object
- You could combat misclassification of objects in an unexpected pose with strong 3D geometric priors
* `[ A ]`


---

**Q: Why deep neural networks (DNNs) can fail to generalize to out-of-distribution (OoD) 3D inputs?**
- DNNs are very sensitive to disturbances along the depth, pitch, and roll and datasets are normally are constructed from photographs taken by people with the object in a classical pose
- DNNs are very sensitive to Lighting changes and datasets are normally are constructed from photographs with very different light intensity 
- DNNs are very sensitive to the changing of the background of an image and there is often a reality gap between a render and a real photo
- None of Above
* `[ A ]`


---

**Q: What is different in the “Strike with a pose” article when compared to other, similar, articles?**
- The authors of strike with a pose hand craft the poses in such a way that they will result in very high wrong confidences
-  The authors of strike with change the image without using a 3D renderer, thus reducing unnatural rendering errors.
- The authors of strike with a pose do not hand craft the poses.
- None of the above
* `[ C ]`


---

**Q: What is an effective method to defend against out of distribution (OoD) examples in neural networks?**
- Parameter augmentation
- Adversarial training
- Random search
- None of the above
* `[ B ]`


---

**Q: Which of the following could help DNN models correctly classify out-of-distribution inputs (familiar inputs with strange poses)?**
- Adversarial training on 3D space objects.
- Strong 3D geometric priors.
- Regularisation
- Both A and B.
* `[ B ]`


---

**Q: Statement 1: For real-world technologies, such as self-driving cars, autonomous drones and search-and-rescue robots, the test distribution may be non-stationary, and new observations will often be out-of-pose (OoP). 
Statement 2: To understand natural Type II classification errors, you could use 6D poses (3D translations and 3D rotations) of 3D objects that caused DNNs to misclassify. **
- Both statements are true
- Statement 1 is true
- Statement 2 is true
- None of the statements are true
* `[ C ]`


---

**Q: Which of the following statements is false?**
- Deep neural networks (DNNs) can fail to generalize to out-of-distribution (OoD) input.
- ML models are vulnerable to natural OoD examples.
- It is shown that image classifiers may
be able to correctly label an image when foreground objects are removed (i.e., based on only the background content).
- None of the above
* `[ D ]`


---

**Q: Which statement is not true?**
- Generating targeted adversarial poses cannot be done by optimization methods yet
- Neural networks are easily confused by object rotations and translations
- Common object classifications are shared across different lighting settings
- Correct classifications are highly localized in the rotation and translation landscape
* `[ A ]`


---

**Q: Which of the following do Alcorn et al. _not_ conclude the following in their paper on the influence of poses on network classification performance:**
- One can easily confuse and drastically reduce the performance of a neural network by rotrating objects in a scene.
- Adversarial poses are not transferable to different classifiers, making the generation of these poses a highly NN-dependent task.
- It is feasible to generate adversial poses which expose the NNs weakness for uncommon poses.
- Changes in the lighting of a scene does not alter the susceptability of a NN to being fooled by irregular poses.
* `[ B ]`


---

**Q: Which of these parameters has the LESS influencial in the DNN ability to classify (Alcorn et al., 2018) ?**
- Depth
- Pitch
- Roll
- Yaw
* `[ D ]`


---

**Q: In the paper 'strike a pose' the author try to to solve the optimization problem of determining which angle/translation most confuses the DNNs. What makes this hard?**
- The images are extremely high resolution because of the added 3d models.
- It was hard to take derivative of the loss function with respect to the orientation position parameters w since the renderer contains many non-differentiable functions. 
- Imagenet (network used in paper) cannot classify 3d scenes by default so it was impossible to determine this angle.
- None of the above. 
* `[ B ]`


---

**Q: Which of the following is WRONG?**
- Even the state-of-art DNNs can perform extremely worse on out-of-distribution inputs.
- DNNs are highly sensitive to slight pose perturbations. 
- Training DNNs on both normal and out-of-distribution inputs helps to generalize the input.
- Out-of-distribution input can happen in real life and sometimes leads to severe consequences. 
* `[ C ]`


---

**Q: Which of the following methods is NOT used to demonstrate that common pose transformations can largely affect the results?**
- The center of the transformed image is not constrained to be inside a subvolume of the camera viewing frustum.
- Instead of iteratively following some approximated gradient to solve the optimization problem,  a new pose is randomly selected in each iteration.
- Using a z_\delta -constrained random search (ZRS) procedure both as an initializer for gradient-based methods and as a naive performance baseline.
- Calculating the first-order derivatives via finite central differences and performing vanilla gradient descent to iteratively minimize the cross-entropy loss L for a target class.
* `[ A ]`


---

**Q: Which one is not the advantage of the Transformer based on attention mechanisms ?**
- In terms of computational complexity, self-attention layers are faster than recurrent layers when the sequence length n is smaller than the representation dimensionality d. 
-  Self-attention mechanism of the Transformer could yield more interpretable models. 
- The transformer allows for significantly more parallelization. However, the recurrent models precludes parallelization within training examples. 
-  To improve computational performance for tasks involving very long sequences, self-attention could be restricted to considering only a neighborhood of size r in the input sequence centered around the output position. This wouldn't increase the maximum path length. 
* `[ D ]`


---

**Q: Select the response of deep neural networks easily to strange poses of familiar objects.**
- fail to generalize to out-of-distribution inputs
- sensitive to slight pose perturbations
- adversarial poses transfer across models
- all of the above
* `[ D ]`


---

**Q: In order to make geometric perturbations to resemble texture perturbations we should  **
- Make the perturbations very large
- Make the perturbations very small
- Change the camera setting
- None of the above
* `[ A ]`


---

**Q: What could be a reason that machine learning (ML) models frequently assign wrong labels with high confidence in real-world technologies?**
-  ML models are vulnerable to natural out of distribution examples
- ML models are vulnerable to adversarial out of distribution examples
- Many image datasets are biased due to the aesthetic tendencies of their captors
- All of the above
* `[ D ]`


---

**Q: Standard 3D renderer include many non-differentiable operations and cannot be inverted
What approaches can be used?
(1)differentiable renderer and performing gradient descent
using its analytical gradients
(2)non-differentiable renderer and approximating the gradient
via finite differences
(3)differentiable renderer and applying an kernel to linearly approximate the finite differences**
- 1&2
- 2&3
- 1&3
- 1&2&3
* `[ A ]`


---

**Q: In methods part, which of the following methods is NOT compared?**
- random search 
- gradient descent using finitedifference (FD) approximated gradients 
- gradient descent using the DR gradients
- non-gradient methods
* `[ D ]`


---

**Q: Why is the classifier network misclassifying objects with unusual poses?**
- They are not seen in the training data.
- Objects in unusual poses look inherently different.
- The training data distribution does not match real world distribution.
- Unusual poses are hard to classify and are near the decision boundary.
* `[ A ]`


---

**Q: Why are neural networks sensitive to the pose of an object for classification?**
- Some poses never occur in real life
- Some objects are completely unrecognisable from certain angles
- Some objects look very similar to other objects from certain angles
- The Network is not trained on all the poses an object can have
* `[ D ]`


---

**Q: Which is a way in which DNNs could be enhanced to reduce the negative effect of out of distribution attacks?**
- Adversarial training (adversarial examples).
- Data augmentation.
- Enlargement of the dataset to include more examples.
- All of the above.
* `[ D ]`


---

**Q: What does DNN often fail to estimate and where are these inputs common?**
- DNNs can fail to generalize to adversarial inputs, which are common in real-world settings
- DNNs can fail to generalize to non-natural inputs, which are common in real-world settings
- DNNs can fail to generalize to out-of-distribution (OoD) inputs, which are common in synthetic settings
- DNNs can fail to generalize to out-of-distribution (OoD) inputs, which are common in real-world settings
* `[ D ]`


---

**Q: Which of the following methods was not used in this paper for the main experiments?**
- Pose transformations
- Random search
- k-nearest neighbor
- Gradient descent with finite-difference
* `[ C ]`


---

**Q: What can be come an issue with neural networks?**
- Slightly altering images to attack the network to generate a total different outcome.
- Required computation power
- Required dataset
- Labeling of datasets
* `[ A ]`


---

**Q: Which of the following statements is/are true about the 3D object dataset?

A In order to maximize the realism of the rendered images only 3D models that have high-quality 2D textures are used. 
B Reality gap is evaluated by, quantitatively evaluating the DNN predictions on the renders followed by evaluating the renders by comparing them to real photos. **
- A
- B
- A and B
- None of the above 
* `[ C ]`


---

**Q: Which of the following approach is not included in the method of this paper?**
- Pose transformations
- Random search
- 3D dataset
- Gradient descent with finite difference
* `[ C ]`


---

**Q: Neural networks are very easily fooled by familiar objects seen in a non-classical confiuration. Which could be a reason for this?**
- Classical image datasets are biased on certain object positions
- Different light intensity tricks classification
- Convolution is not so effective when objects change shape
- Even humans can be tricked, so we should accept this from the networks
* `[ A ]`


---

**Q: Which of the following options, according to the background and empirical results in this paper, is false?**
- DNNs are highly sensitive to slight pose perturbations.
- The state-of-the-art DNNs perform both image classification and true object recognition well.
- With the framework presented, they are generated unrestricted 6D poses of 3D objects and studied how DNNs respond to 3D translations and 3D rotations of objects.
- DNNs could also benefit from strong 3D geometry priors.
* `[ B ]`


---

**Q: Which of the following is true:**
- Usually, the misclassified out of distribution objects from a class tend to be classified with the same label
- DNN is robust to disturbances in the rotation and the translation of the object
- Adversarial training is a solution for enhancing the ability of the classifier to correctly classify out of distribution examples
- For an image out of distribution there is no real life correspondence
* `[ C ]`


---

**Q: State-of-the-art DNNs perform image classification well, but are still far from true object recognition. Why can this conclusion be drawn?**
- Because they achieve low accuracy on new datasets
- Because they correctly label only a small portion of the 6D pose space of 3D objects
- Because their object recognition is based on 2d images
- Because training on adversarial poses helps DNNs generalize to held-out objects from the same class
* `[ B ]`


---

**Q: What is not suggested by the paper to minimize the misclassification of out-of-distribution (Ood) inputs?**
- adversial training
- Utilisation of models with strong 3D geometric priors.
- Using a non-differentiable renderer
- Training data augmentation by harnessing images from 3D renderings
* `[ C ]`


---

**Q: One of the problems of machine learning models is that they are vulnerable to natural out-of-distribution examples. This is problematic especially for real data in which the existence of the out-of-distribution examples cannot be avoided. For example, a classifier cannot distinguish between the sky and a car with the same color. One possible solution for a better classification in this case is:**
- Extracting only useful insights from the images
- Training only with images in which the background is ignored
- Training only with images of a single object
- Testing only on images with a single object
* `[ A ]`


---

**Q: What does OoD (Out of Distribution) mean?**
- Not from the training distribution
- Not from the test distribution
- Larger value than in the normal distribution 
- Your distribution is too large
* `[ A ]`


---

**Q: Which statement is true concerning the performance of DNNs on classifying real world objects?**
- Classifying an arbitrary pose of a real world object is relatively easy for a DNN
- Neural networks are easily confused by object rotations and translations
- The correct classifications are lowly localized in the rotation and translation landscape
- Common object classifications are not shared across different lighting settings
* `[ B ]`


---

**Q: Which of the following statements is false?**
- The classification performance on generated adversarial examples depends a lot on the classifier used
- State-of-the-art deep neural networks are still far from true object recognition
- ImageNet is biased because it's constructed from photographs taken by people
- The rotation of an object doesn't influence the classification performance on that object a lot
* `[ D ]`


---

**Q: What is true about Deep-learning**
- DNN are good in image classification
- DNN are good with object recognition
- All true
- All false
* `[ A ]`


---

**Q: The models studied were very sensitive to changes in the $z_{\eps}$ (depth) dimension. Speculate why this might be.**
- The lighting conditions were shown to have large effect in the predictions.
- CNNs are not scaling invariant.
- CNNs are scaling invariant.
- The lighting conditions were shown to have small effect in the predictions.
* `[ B ]`


---

**Q: Which of these statements concerning image classification is incorrect?**
- Datasets like ImageNet are biased because their images were taken by humans
- Many deep neural networks are overfitting to certain poses of objects
- Out of Distribution (OoD) examples are generated synthetically
- Augmenting a training set with adversarial training samples can improve a deep learning network's performance on Out of Distribution (OoD) samples
* `[ C ]`


---

**Q: Decide if the following statements are true or false.
Statement 1: The primary cause of the performance drop in the recognition of object in unnatural poses on artificially generated models is the deviation in small details between the two datasets.
Statement 2: After appropriately retraining a usual deep neural network on dataset of artificially generated objects in unnatural poses, the accuracy is comparable to the accuracy of the algorithm acting on the canonical data.**
- 1: true, 2: true
- 1: true, 2: false
- 1: false, 2: true
- 1: false, 2: false
* `[ D ]`


---

**Q: In general, what is the issue when using DNN's on image classification and what is the approach for solving this?**
- DNN's only capture context, causing important details to get lost
- DNN's don't look at images on pixel level, causing important details to get lost
- DNN's are not sensitive to slight pose perturbations, meaning it is often classifying wrong when poses are altered. By introducing the different view angles of a 3D model, the problem can be reduced
- DNN's are higly sensitive to slight pose perturbations, meaning it is often classifying wrong when poses are altered. By introducing the different view angles of a 3D model, the problem can be reduced
* `[ D ]`


---

**Q: What type of objects in image can DNN mismach?**
- Stacionary objects.
- Out-of-distribution object. 
- Flying object.
- Grey-scale object.
* `[ B ]`


---

**Q: When translating and transforming objects (such as cars on images), the performance of image classification had the following adverse effect**
- Recognized almost no object in some orientations
- Had a slight reduction in classification performance
- Was proven to be translation invariant
- Showed to have a boost in performance.
* `[ A ]`


---

**Q: For differentiable renders, in order to enable back propagation through the non-differentiable rasterization process which of the following is an appropriate method?**
- replace the discrete pixel color sampling step with a linear interpolation sampling scheme that admits non-zero gradients.
- replace the discrete pixel color sampling step with a tri-cubic interpolation sampling scheme for non-zero gradients.
- filter and use the discrete pixel color sampling step with a linear interpolation sampling scheme that admits non-zero gradients.
- None of the above
* `[ A ]`


---

**Q: Deep Neural Networks can be easily fouled by:**
- Pose modification
- Texture modification
- Lighting modification
- All of the above
* `[ D ]`


---

**Q: 2D images created by a 3D render cannot:**
- Gain insight what is happening in the DNN.
- Test how the 6 degree of freedom has influence on the classification
- Optimize for lightning
- None of the above
* `[ D ]`


---

**Q: Which of the following is FALSE?**
- State-of-the-art Deep Neural Networks perform image classification well, but are still far from true object recognition
- Small changes in the pose of an object might result in wrong classifications, even for state-of-the-art solutions
- Due to the properties of convolutional neural networks, these networks are not susceptible to rotations and translations of objects
- Rotating 3D models to make classifiers fail to classify the object correctly is a valid way to generate adverserial samples to use in an adverserial learning setup
* `[ C ]`


---

**Q: Why do neural network fail to correctly classify an object in an image when the object is in an unusual pose?**
- The image of the object in a strange pose comes from a different distribution
- The neural network has never seen the object in that pose before. 
- Most datasets contain images of objects in their usual pose.
- all of the above.
* `[ D ]`


---

**Q: What is out-of-distribution (OoD) for nueral network?**
- The distribution of test set are different from the trainning set
- The distribution of test set are the same as the trainning set
- The test set are the same as the trainning set
- The test set are part of the trainning set
* `[ A ]`


---

**Q:  Do neural networks tend to generalize to already seen object in a different position?**
- Yes, always
- Yes, if specific precautions are taken
- No, never
- None of the above
* `[ B ]`


---

**Q: Which of the below solutions is valid to prevent datasets to only reflect the aesthetic tendencies of their captors?**
- Generate 3D models
- Shift the luminance strategically
- Firstly identify the main subjects
- None of the above
* `[ A ]`


---

**Q: Which statement below is wrong?**
- Object detectors can be more robust to adversarial attacks than image classifiers
- One of the most effective methods for defending against out-of-distribution (OoD) examples has been adversarial training
- To quantitatively evaluate the DNN’s sensitivity to rotations and translations, the tests should investigate how it responds to multiple parameter disturbances at the same time
- Deep neural networks (DNNs) can fail to generalize to out-of-distribution (OoD) inputs, including natural, non-adversarial ones, which are common in real-world settings
* `[ C ]`


---

**Q: Why have DNN’s generally not been trained on data with 3D rotations/translations as is created in the paper?**
- There is no gain to training on such data.
- Generating the data is costly and labor-intensive
- This data never reflects real life.
- Because it contains unlabeled data
* `[ B ]`


---

**Q: In the paper “Strike (with) a pose: neural networks are easily fooled by strange poses of familiar objects” the authors specify the fact that one of the most effective methods for defending ML models against OoD examples has been adversarial training. What is the idea behind adversarial training? Consider the concrete case of a dataset consisting of images when answering this question.**
- Augmenting the training set with additional examples from the training distribution.
- Augmenting the training set with additional examples specifically crafted by an adversary in order to cause a target model to misbehave. 
- Augmenting the training set by applying pose transformations to the already existing images in the training set.
- None of the above.
* `[ B ]`


---

**Q: 3D translations and 3D rotations is?**
- 6D
- 5D
- 4D
- 3D
* `[ A ]`


---

**Q: Which statement is wrong? **
- DNNs can fail to generalize to out-of-distribution (OoD) inputs. 
- image classifiers may be able to correctly label an image when foreground objects are removed 
-  Neural networks are easily confused by object rotations and translations。 
- non of them 
* `[ D ]`


---

**Q: Which of the following is False about the findings of this paper?**
- Simply changing the lighting conditions of an image already has a drastic affect on classification accuracy.
- Adverserial poses are highly specific to each classifier: a 3D rotation of an object poses little threat to an object detector like YOLOv3.
- It is yet unclear whether augmenting the data with a lot of adversarial 3D objects can improve DNN generalization to held-out objects, as obtaining such datasets is costly.
- None of the above.
* `[ B ]`


---

**Q: What is adversarial training?**
- Training the model with generated out of distribution examples 
- Training a model with complicated examples using the training at hand (by rotating, blurring, scaling, etc…)
- Training with examples that have the wrong label so the model robustness improves
- - None of the above
* `[ A ]`


---

**Q: Is this the end?**
- yes it is
- no it isn't
- maybe
- see you later
* `[ A ]`


---

**Q: Which statement is false ragarding object classificaton on OoD (Out-of-Distribution) inputs:**
- Misclassifications occur uniformly over the transformed inputspace
- Objects get misclassified as many different labels
- Changes to lighting settings affect misclassification stronger than transformation of object poses
- All of the above
* `[ C ]`


---

**Q: In context of a classifier model that classifies a certain object with very high accuracy of over 99%, which of the following transformations to a sample test dataset would most likely cause the model to mis-classify?**
- 3d transformations (Rotation / Translation)
- Mild variation in lightings (Brightness, Hue, Saturation, etc)
- Both a and b
- None of above
* `[ A ]`


---

**Q: Identify the FALSE statement according to the paper:**
- adversarial poses transfer across models, with the poses misclassified rate transfer being the highest for the state-of-the art YOLO object detector 
- training on adversarial poses did not improved the ability of DNNs to generalize well to held-out adversarial poses data
- all the images used were rendered against a plain background obtained from the RGB mean of AlexNet images
- the ability of Google's Inception Network to correctly classify images is highly localized in the rotation and translation parameter space
* `[ A ]`


---

**Q: Which of the following is not right?**
- Training based on snthetic data may not be able to generalize to real-world data.
- Simulated+Unsupervised learning can add realism to the simulator while preserving the annotations of the synthetic images
- The local adversarial loss removes the artifacts and makes the generated image significantly more realistic
- The local adversarial loss will not make the image more realistic
* `[ D ]`


---

**Q: Which of these statements is/are true?
1: DNN's can correctly classify most objects irrespective of the object's pose. 
2: Random Search is a fairly complicated and imprecise way to generate samples**
- Only statement 1 is correct
- Only statement 2 is correct
- Both statements are correct
- Both statements are false
* `[ D ]`


---

**Q: Which of the following hold true?**
- Changing the camera setting will assist in making the geometric perturbations resemble the texture perturbations
- Making the perturbations large will assist in making the geometric perturbations resemble the texture perturbations
- Both the options (a) and (b) are correct
- Both the options (a) and (b) are wrong
* `[ B ]`


---

**Q: What is a convinient and accurate way to test NNs in how prone are they in recognizing objects in different poses.**
- Rotate 2D images
- Mirror 2D images.
- 3D models.
- Flip 2D images
* `[ C ]`


---

**Q: Which of the following statements is incorrect:**
- Neural networks often fail to generalize on out of distribution input data
- one way of data augmentation is harnessing images generated from 3D renderers
- Modern GL is a from of non-differentiable renderer
- In the Random search procedure, they iteratively follow some approximated gradient to solve the optimization problem
* `[ D ]`


---

**Q: Which of the following statement about 3D poses and neural networks is not true?**
- Correct classiﬁcations are highly localized in the rotation and translation landscape
- Lighting changes alleviates DNN’s vulnerability towards adversarial poses
- Optimization methods can effectively generate targeted adversarial poses 
- None of the above
* `[ B ]`


---

**Q: In the paper Strike with a pose, we saw that DNN's are easily fooled when a object had a strange pose. What does this tells us about convolution?**
- Convolution is not naturally equivariant some transformation, such as changes in scale or rotation of the object.
- Convolution is naturally equivariant some transformation, such as changes in scale or rotation of the object.
- That the learned filters of convolution in the DNN are hard to interpreted and 3D scene simulation can help with this 
- Convolution is not suitable for object recognition
* `[ A ]`


---

**Q: Which of the following is a way to enable back propagation through non differentiable rasterization process**
-  replace the discrete pixel color sampling step with a linear interpolation sampling scheme that admits non-zero gradients
-  replace the discrete pixel color sampling step with a bicubic interpolation sampling scheme that admits non-zero gradients
-  replace the discrete pixel color sampling step with a linear interpolation sampling scheme that admits zero gradients
-  replace the discrete pixel color sampling step with a tricubic interpolation sampling scheme that admits non-zero gradients
* `[ A ]`


---

**Q: Which of the following statements accuratly describes the current (2019) accuracy dependencies of deep neural networks (DNN's)?**
- DNN robustness to object orientations and translations are dependent of lightning for high confidence missclassifications and on similarities to the 3D object exapmles in ImageNet.
- DNN robustness to object orientations and translations are independent of lightning for high confidence missclassifications and dependent on similarities to the 3D object exapmles in ImageNet.
- DNN robustness to object orientations and translations are independent of lightning for high confidence missclassifications and of similarities to the 3D object exapmles in ImageNet.
- DNN robustness to object orientations and translations are dependent of lightning for high confidence missclassifications and independent of similarities to the 3D object exapmles in ImageNet.
* `[ A ]`


---

**Q: Which of the following statements is FALSE?**
- Common object classifications are shared across different lighting settings
- Neural networks are easily confused by object rotations and translations
- False classifications are highly localized in the rotation and translation landscape
- Optimization methods can effectively generate targeted adversarial poses
* `[ C ]`


---

**Q: Which one of the following is not used in the experiment?**
- Random Search
- Pose Transformation
- Gradient Descent
- Variable Autoencoder
* `[ D ]`


---

**Q: DNN classification accuracy can be highly affected by pose transformations. How come they are so sensitive to these kinds of transformations?**
- They are trainined using pictures of objects which are strongly biased perspective wise. Eg. Most training pictures of an object will be from a similar angle, so only this viewpoint is learned.
- They don't learn 3d representations of objects, so have trouble generalizing to different poses.
- Most pose transformations cause objects to be OoD, even if they are from known classes.
- All of the above.
* `[ D ]`


---

**Q: The ability of current DNN's to correcly classify objects in images is**
- Highly localized in rotation and translation parameter space
- Highly localized in rotation parameter space, but not translation parameter space
- Highgly localized in translation parameter space, but not rotation parameter space
- Insensitive to both rotation and translation
* `[ A ]`


---

**Q: The authors of this paper discuss and proof that DNNs:**
- Dont perform well enough in tasks of object detection in an image
- Dont perform well when in tasks of object detection in an image when the object had artificially rotated and translated
- But perform better under specific light conditions 
- Both b and c
* `[ B ]`


---

**Q: Non differentiable rendere can cause problems during gradient descent step. But what was the primiry rationale to try it?**
- Its rendering speed is higher 
- If we compute approximate gradients for it, they are less noisy
- Both variants
- None of the variants
* `[ A ]`


---

**Q: State-of-the-art image classifiers can achieve outstanding accuracy for data that follows the distribution of the training data. However, there are many real-world applications (such as self-driving cars) where these models cannot yet be employed. One of the reasons is that**
- the training data generally does not contain images of the objects in all possible positions, and so there is a variety of positions in which the objects will be incorrectly classified; this in turn can lead to dangerous situations.
- the models that achieve state-of-the-art accuracy are too computationally expensive and cannot be embedded in the systems that would like to make use of them.
- international laws prohibit DL models to be used in software that is available to the public.
- the average user will not know how to run the backpropagation algorithm.
* `[ A ]`


---

**Q: Which of the following is not a conclusion drawn from the paper?**
- The state-of-the-art DNNs perform image classification well but are still far from true object recognition.
- It is possible to improve DNN robustness through adversarial training with many more 3D objects.
- The authors hypothesize that future ML models capable of visual reasoning may benefit from strong 3D geometry priors.
- Training on adversarial poses generated by the 30 objects (in addition to the original ImageNet data) helped DNNs generalize well to held-out objects in the same class.
* `[ D ]`


---

**Q: What, according to the authors, plays a role in the poor performance when "strange poses" are created?**
- The biased datasets created by humans make "strange poses" hard to predict as they do not behave like as imagery created by humans.
- Neural networks are simply not fit for image classification.
- Both A & B
- None of the above
* `[ A ]`


---

**Q: Which of the following is true?**
- ImageNet data set is biased because it reflects the aesthetic tendencies of their captor 
- ML models are vulnerable to adversarial out-of-distribution (OoD) examples by not natural OoD examples
- Type II classification error is when a DNN fails to classify adversarial OoD examples
- A DNN is highly sensitive to all single parameter disturbances, but it is especially sensitive to yaw and roll
* `[ A ]`


---

**Q: Which one is wrong?**
- In this project, after trained on Inception-v3 and ImageNet and found that the model was highly sensitive to pitch and roll perturbations.
- The authors suggest that one solution is to load up ImageNet with lots of adversarial examples. 
- This paper states that  the state of the art neural networks are good at "classifying" things in pictures, but   not really recognizing objects, in the true sense of that expression.
- None
* `[ D ]`


---

**Q: What is the advantage of using  3D models and 3D renderers in augmenting data-sets for training DNNs?**
- 3D objects can be rendered in non-conventional poses so that the networks can learn how to generalize to “out of distribution” examples.
- 3D models of fictional objects can be used to teach the networks.
- 3D models embed more depth information that the networks can learn.
- All of the above.
* `[ A ]`


---

**Q: What is not a problem for object recognition when the picture is altered?**
- Rotation
- Translation
- Lighting
- None of the above
* `[ D ]`


---

**Q: Which of the following statements are TRUE?**
- In Generative Adversarial models the role of the generator is to provide inputs specially crafted to cause a target model to misbehave
- When test distributions are non-stationary, new observations will often be out-of-distribution (OoD) which is a serious issue in real-world autonomous systems
- Adversarial training does not always improve Deep Neural Networks to held-out objects in the same class
- All of the above
* `[ D ]`


---

**Q: Neural network are easily fooled by**
- rotations but not translations
- translations but not rotations
- translations and rotations
- neither translations nor rotations
* `[ C ]`


---

**Q: Which of the two following statements is correct, regarding DNN?
1)	DNNs are not sensitive to slight pose perturbations of objects they try to classify
2)	DNNs can fail to generalize to out-of-distribution (OoD) inputs**
- a) Both are correct
- b) Statement 1 is correct, Statement 2 is false
- c) Statement 1 is false, Statement 2 is correct
- d) Both are false
* `[ C ]`


---

**Q: Consider the following two statements about Ood errors in computer vision models:

1.For real-world technologies, such as self-driving cars the test distribution may be non-stationary, and new observations will often be out-of-distribution. Inputs crafted by an adversary cause a target model often to misbehave. 
2.In the paper a framework for finding out-of-distribution errors in computer vision models in which iterative optimization in the parameter space of a 3D render is used to estimate changes that cause a target Deep Neural Network to misbehave. 

Which of theses statements is true?**
- Both are true
- Both are false
- 1 is true and 2 is false
- 1 is false and 2 is true
* `[ A ]`


---

**Q: Which of the following about State-of-the-art DNNs is true?**
- Very good at both image classification and object detection
- Very good image classification but bad object detection
- Bad at image classification but good at object detection
- Both image classification and object detection are not upto the mark
* `[ B ]`


---

**Q: Why do cnns have dificulty recognizing objects in different settings**
- Training data can be biased so the network learns features that don't belong to the object
- The networks are good at learning geometries that may change in a different pose
- As long as you zoom in far enough, at one point the network will see a totally different object
- All of the above
* `[ D ]`


---

**Q: What statement about misclassifying objects due to position/pose changes is false?**
- This is a type-II error
- This does not happend with out of distribution inputs
- This means that deep neural networks are good image classifiers but not object recognisers
- A rotation of an object by a 20 degrees can change the classification
* `[ B ]`


---

**Q: Deep neural networks classify images by: Extracting a general feature representation, ...**
- ...that includes high level representatiaons of the underlying concept of the object to be classified; and are hence insensitive to deviations of translational or rotational position.
- ...that includes high level abstraction of only the training data sample; and are hence insensitive to deviations of translational or rotational position.
- ...that includes low level abstraction of the underlying concept of the object to be classified; and are hence sensitive to deviations of translational or rotational position.
- ...that includes high level abstraction of only the training data sample; and are hence sensitive to deviations of translational or rotational position.
* `[ D ]`


---

**Q: what is the effect ofout-of-distribution poses of well-known object to DNN? **
- minor perfomance dip (2-3 percent)
- huge performance dip (50 percent)
- 97 percent incorrectly classified
- only effect on training, not on validation
* `[ C ]`


---

**Q: DNN is sensitive to which of the following transformations?**
- Depth
- Pitch 
- Roll
- All of the above. 
* `[ D ]`


---

**Q: What is one of the key points the authors of the paper try to make?**
- Object recognition is very dependent on the background of the image
- A larger dataset would benefit any object detector to detect out-of-distribution inputs
- Unrealistic out-of-distribution inputs are extremely hard to classify in object recognition
- deep neural networks can fail to generalize to out-of-distribution inputs that are are common in real-world settings
* `[ D ]`


---

**Q: Which techniques could be used to help neural networks recognize familiar objects in strange position?**
- Use data augmentation technique such as rotating images
- Generate unbiased images from 3D renders 
- Use 3D scene
- All of above
* `[ D ]`


---

**Q: What is the takeaway message of paper 14?**
- NNs know everything!
- NNs understand objects they classify quite naively.
- NNs understand objects they classify quite good.
- The best image classifiers can correctly classify many objects in the whole 6D object space.
* `[ B ]`


---

**Q: By which option you may fool the general image recognition system while recognizing familiar objects**
- Add some texture to the images
- Use images of objects with unusual poses
- Use images of objects which are spinned
- All of the three
* `[ D ]`


---

**Q: How many parameters needs to be tuned to estimate the pose of the 3D object**
- 8
- 9
- 6
- 3
* `[ B ]`


---

