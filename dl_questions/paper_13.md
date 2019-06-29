# Questions from `paper_13` :robot: 

**Q: Which of the following is NOT a goal of Simulated + Unsupervised (S+U) learning?**
- Improve the realism of synthetic images from a simulator using unlabeled real data.
- Preserve the annotation information of the synthetic images.
- Generate images without artifacts, since machine learning models can be sensitive to artifacts in the synthetic data.
- Reinforce drifting so that the discriminator becomes more robust to small changes in its input. 
* `[ D ]`


---

**Q: What is the goal of the “Refiner” network that the authors propose?**
- To generate several realistic images starting from a single synthetic one
- To generate several synthetic images starting from a single real image
- To make a synthetic image more realistic
- To attach a label to an unlabeled image
* `[ C ]`


---

**Q: The goal of the paper is to improve the realism of synthetic images, how is this done?**
- improving the simulator
- using labeled real data
- using unlabeled real data
- all of the above
* `[ C ]`


---

**Q: Which of the following statements does not constitute a challenge of synthetic image generation?**
- Manually annotating the generated images.
- Avoid introducing artifacts.
- Generating realistic images.
- Maintaining the data distribution of real images.
* `[ A ]`


---

**Q: What needs to be done to add realism to the synthetic image?**
- Noise needs to be added
- Reduce the gap between distributions of synthetic and real images
- Noise needs to be removed
- Increase the gap between distributions of synthetic and real images
* `[ B ]`


---

**Q: Which stamen is false?**
- The lack of realism may cause models to overfit to ‘unrealistic’ details in the synthetic images.
- To add realism to the synthetic image, the gap between the distributions of synthetic and real images needs to be bridged.
- Another key requirement for the refiner network is that it should learn to model the real image characteristics without introducing any artifacts.
- Adversarial training is preferable because it only focuses on the last image and not on the older images, preventing artifacts.
* `[ D ]`


---

**Q: Why would one apply self-regularization to a refiner neural network when handling synthetic images?**
- To learn useful representations of the synthetic images.
- To minimize the image difference between the synthetic and the refined images.
- To find a new label for the refined image.
- To classify all local image patches separately. 
* `[ B ]`


---

**Q: Both GANs and Simulated + Unsupervised learning (S+U) aim at creating realistic images. What is one of the main differences between the two methods?**
- GANs take a random vector as an input, S+U uses synthetic images s as input
- GANs take synthetic images as input, S+U uses a random vector as an input
- a GAN is a two-player system where one network tries to fool the other one. S+U is only a one-player system
- an S+U is a two-player system where one network tries to fool the other one. GAN is only a one-player system
* `[ B ]`


---

**Q: Which of the following statements are true? (Based on "Learning from Simulated and Unsupervised Images through Adversarial Training")

statement1: The the goal of (S+U) learning is to improve the realism
of synthetic images from a simulator using unlabeled real data. The improved realism enables the training of better machine learning models on large datasets
without any data collection or human annotation effort.

statement2: To add realism, we train our refiner network using an adversarial loss, similar to Generative Adversarial Networks (GANs). The GAN framework requires training two neural networks with competing goals, which is known to be unstable and tends to introduce artifacts.

statement3: To add realism to the synthetic image, we need to bridge the gap between the distributions of synthetic and real images. An ideal refiner will make it impossible to classify a given image as real or refined with high confidence.

statement4: When we train a single strong discriminator network, the refiner network tends to over-emphasize certain image features to fool the current discriminator network, leading to drifting and producing artifacts.**
- 1 and 3
- 2 and 4
- 2 3 and 4
- All statements are true
* `[ D ]`


---

**Q: Which of the following statements is not true with respect to the proposed SimGAN method?**
- It uses regularization to preserve the annotation information.
- Because of the introduction of a local adversarial loss, the introduction of artifacts is reduced.
- Applying local adversarial loss is sufficient in itself with respect to its goal.
- The output of the discriminator is included into the loss function of the refiner.
* `[ C ]`


---

**Q: Where does S+U learning stands for?**
- Simulated + Unencoded learning. 
- Simulated + Unlearned learning.
- Simulated + Unsupervised learning.
- None of the above answers is correct.
* `[ C ]`


---

**Q: As the use of synthetic images is becoming more popular in the field of deep learning. How are these synthetic images used?**
- They are used to obtain more data that can be used for learning a network. They do this by obtaining the synthetic images from a simulator that creates them from previous real images. 
- They are used to create large labeled training datasets which take a lot of time to make. They do this by obtaining the synthetic images from a simulator and then trying to make it as real as possible to create a dataset which is labeled and can be used for learning a network. 
- They are used to create data that is false but looks closely to the images that needs to be detected. They do this by obtaining the synthetic images from a simulator that changes the real images in such a way there will be more data to train false results on.
-  None of the above answers is correct.
* `[ B ]`


---

**Q: Refiner network in simulated unsupervised learning does the following:**
- Avoids introducing spurious artifacts
- Adds realism to images from simulator
- Distinguishes between simulated and real images
- Operates on pixel level and preserves the global structure
* `[ B ]`


---

**Q: In the paper "Learning from Simulated and Unsupervised Images through Adversarial Training" what method had the least distance from the ground truth?**
- Synthetic Data partially trained
- Synthetic Data utilizing the entire training set
- Refined Synthetic Data partially trained
- Refined Synthetic Data utilizing the entire training set
* `[ D ]`


---

**Q: What is the purpose of the discriminator in GAN learning**
- Deciding whether an input image is real or fake (e.g. created by a generator)
- Finding the discriminant of the layer matrix of a network
- discriminating good and bad networks by their nature
- The use of discriminators stopped in 2001 because they don’t serve a purpose anymore
* `[ A ]`


---

**Q: Which of the following is \emph{not} a characteristic of the SimGAN network for S+U learning:**
- It trains with an adversarial loss
- Its loss function contains a self-regularization term
- The discriminator computes loss over local patches
- the discriminator trains using only the most recent refined images
* `[ D ]`


---

**Q: Which advantage does the SimGAN algorithm NOT have over the default GAN algorithm?**
- Preserves annotations
- Reduces complexity
- Avoids artifacts
- Stabilizes training
* `[ B ]`


---

**Q: What is not a property of batch normalization? **
- Reduction in internal covariate shift
- Prevention of exploding or vanishing gradients
- Increasing robustness to different settings of hyperparameters
- Bringing most of the activations closer to saturation regions of non-linearities
* `[ D ]`


---

**Q: How can the annotations of a simulated image be preserved when the image is refined using real unlabeled data?**
- By implementing a local discriminator network to model the real images.
- By implementing a global discriminator network to model the real images.
- By having the discriminator network only focus on the latest refined image.
- By implementing a regularization loss which minimized per-pixel differences between the real and synthetic images.
* `[ D ]`


---

**Q: In the paper "Learning from Simulated and Unsupervised Images through Adversarial Training", which types of loss does SimGAN NOT take into account when training?**
- Adversarial loss 
- Self-regularization loss
- KL-Divergence of simulated vs real data distributions
- None of the above (i.e All are valid)
* `[ C ]`


---

**Q: Which of the following statements are correct?

(1) The GAN (Generative Adversarial Network) approach can be seen as a two-player game, where a third network automatically evaluates the performance of the two antagonistic networks.
(2)The output of a discriminator is always a single number between 0 and 1
(3) A fully connected encoder network never holistically modifies an image content and therefore preserves the global structure.**
- -1
- -2
- -3
- None statment is correct
* `[ D ]`


---

**Q: What is NOT true about Simulated+Unsupervised (S+U) learning on synthetic images?**
- The goal is to improve the realism of synthetic images from a simulator.
- The learning task requires labeled real data.
- The learner can preserve annotation information for training of machine
learning models.
- Using a Generative Adversarial Network enables the generation of more realistic images.
* `[ B ]`


---

**Q: Noise distribution in data affects unsupervised learning **
- True  
- FALSE
- nothing can be stated  
- none of them 
* `[ A ]`


---

**Q: What is the reason the discriminator network classifies on local patches instead of the full image?**
- The per-pixel reach is the same for patches as a global image, thus patches provide more data but with the same quality
- Computation of smaller patches is more time efficient
- The refiner network tends to over-emphasize certain image features, leading to artifacts and drifting
- A probability map instead of a score give better insight on which parts to focus on for the refiner network
* `[ C ]`


---

**Q: Which of the following is not a characteristic of the proposed S + U network?**
- A discriminator network is used to classify individual patches of an image.
- A refiner network is used to generate realistic images from synthetic data.
- In order to preserve annotations, a self-regulation loss is added to the objective function.
- The receptive field of the discriminator is limited by increasing the stride of the ConvNet.
* `[ D ]`


---

**Q: What makes the Generative Adversarial Network presented in the paper: "Learning from Simulated and Unsupervised Images through Adversarial
Training" perform better than regular GANs?**
- The GAN framework makes use of a CNN in combination with self-attention to learn properties about the images
- The GAN framework makes use of the 'Visual Turing Test' to improve the discriminator and reduce the total loss
- The GAN framework refines it's synthetic images to add more 'realism' to the generated images
- The GAN framework makes use of unsupervised learning to train it's network
* `[ C ]`


---

**Q: Which of the following statements about supervised+unsupervised learning is true?**
- S+U learning tends to create artifacts
- The S+U network needs a minimum of one neural net to work properly
- The S+U discriminator's receptive field is set to the whole image 
- Some of the modifications applied to S+U make it more stable than GAN
* `[ D ]`


---

**Q: What problem is attempted to be solved by using a history of previously refined images in the model?**
- Re-introducing artifacts to the model
- Overfitting on a single image
- A lack of training images in the original datset
- A lack of noise in the model
* `[ A ]`


---

**Q: What does the generator network do?**
- Maps a random vector to a realistic image
- Distinguishes the generated from the real images
- Improves the stability of the network
- All of the above
* `[ A ]`


---

**Q: Which of the following methods can be used to improve ML model performance via dataset generation (or augmentation) in a particular task domain?

A - Train a GAN network to generate synthetic images of the domain of interest and perform domain adaptation
B - Perform affine transformations on the existing dataset
C - Generate more supervised training data
D - Train a GAN network to generate photo-realistic images of the domain of interest **
- A and D
- A and B and C and D
- B and C
- A and C
* `[ B ]`


---

**Q: What is the main goal of S+U learning?**
- To be able to create/augment a data set without human contribution
- To be able to to generate the most beautiful images
- To improve the current state of image generation
- To improve the speed of network training
* `[ A ]`


---

**Q: What are key features of an S+U Learning with SimGan network? (according to the paper)**
- To bridge the gap between the distribution of synthetic and real images.
- It should learn to model the real image characteristics without introducing any artifacts.
- It should be able to learn on noisy images with improper labels.
- The discriminator should be able to classify any refined image generated by the refiner network as fake.
* `[ C ]`


---

**Q: Which of the following is true?**
- An adversarial network focuses on all refined images
- A key requirement for a refiner network is that is should learn to model thereal image characteristics without introducing any artifacts
- The goal of S+U learning is to use a set of labeled real images $y_i\in Y$ to learn a refiner $R_\theta(x)$ that refines a synthetic image $x$ where $\theta$ are the function parameters.
- And ideal refiner makes it possible to classify a given image as real or refinedwith high confidence
* `[ B ]`


---

**Q: In training adversarial networks for sample generation, why might one reuse samples produced by the generator when training the discriminator?**
- Ensure that the discriminator is training well by giving it more data
- Show samples to the discriminator which has old artifacts so that the discriminator doesn't forget those artifacts
- Generating samples with a generator is computationally expensive
- All of the above
* `[ B ]`


---

**Q: Which isn't a key modification to the standard GAN algorithm?**
- Adding of a self-regularization term
- Use of local adversarial loss
- Updating the discriminator using a history of regined images
- Using a generator
* `[ D ]`


---

**Q: Why will synthetically created images usually not result in the desired performance when training a model on them?**
- They can not be created before there is knowledge in the model, which results in overfitting
- There is a gap between the distribution of real images and synthetic images
- The technology in graphics is not advanced enough
- Neural networks are not fit to classify synthetic images
* `[ B ]`


---

**Q: Why does the technique as described in the paper limit the discriminator's receptive field to local regions instead of the whole image?**
- To avoid drifting and introducing spurious artifacts
- To make the learning process quicker
- To make sure that the generating network converges
- In that way the loss is easier to calculate
* `[ A ]`


---

**Q: Which is not a key modification to the GAN algorithm proposed by the paper**
- a self-regularization term
- local adverserial loss
- using history of refined images
- using a discriminator
* `[ D ]`


---

**Q: What is true?**
- The input of the discriminator has some added noise
- Discriminator task is telling real from fake/refined images
- The Refiner task is telling real from fake/refined images
- The Discriminator tries  to fool the Refiner
* `[ B ]`


---

**Q: Which of the following statements is true:**
- Refining synthetic images is not possible with unsupervised learning, self-regularization cannot be implemented in this problem
- Refining synthetic images is not possible with unsupervised learning, self-regularization can be implemented in this problem
- Refining synthetic images is possible with unsupervised learning, self-regularization cannot be implemented in this problem
- Refining synthetic images is possible with unsupervised learning, self-regularization can be implemented in this problem
* `[ D ]`


---

**Q: What is the aim of the introduction of a SimGAN (Simulated + Unsupervised learning) in the generation of synthetic images?**
- To decrease the time needed to create a dataset of synthetic images.
- To avoid the lack of realism in the images that may cause models to overfit to ‘unrealistic’ details in the synthetic images.
- To avoid the duplication of images that may cause models to overfit to details in the synthetic images.
- To learn a model that improves the realism of synthetic images from a simulator using labelled real data using a human interaction
* `[ B ]`


---

**Q: What is special about the loss function of the refiner network compared to the loss function of other generative networks?**
- It doesn’t use a log term.
- It is independent on how good it “fools” the discriminator network
- There is an extra regularization term
- None of the above
* `[ C ]`


---

**Q: Shrivastava et al. (2016) proposed Simulated+Unsupervised (S+U) learning by introducing a refiner network that reduces the realism gap between synthetic and real image distribution. Which of the following is a feature of the refiner network?**
- Adversarial loss
- Self-regularization loss
- Content modeling
- A and B are correct
* `[ D ]`


---

**Q: In the S+U learning network, the refiner is trained using a Local Adversarial Loss is used instead of a Global Discriminator Loss. Why?**
- To achieve the goal of S+U to use a set of unlabeled real images.
- Reduce over emphasise on certain image features rather than an emphasise on the complete image,  leading to artifacts.
- A method to improve the stability of adversarial training by updating the discriminator using a history of refined images, rather than only the ones in the current mini- batch.
- All of the above.
* `[ B ]`


---

**Q: Statement 1: The goal of Simulated+Unsupervised learning is to use a set of unlabeled real images to learn a refiner that refines a synthetic image. 
Statement 2: A key requirement for the refiner networks is that is should learn to model the generated image characteristics without introducing any artifacts. **
- Both statements are true
- Statement 1 is true
- Statement 2 is true
- None of the statements is true
* `[ B ]`


---

**Q: A problem of adversarial training is that the discriminator network only focuses on the latest refined images. This lack of memory may cause:**
- divergence of the adversarial training.
- the refiner network re-introducing the artifacts that the discriminator has forgotten about.
- Both A and B.
- None of the above.
* `[ C ]`


---

**Q: What is false about S+U learning in the SimGAN model?**
- The goal of S+U learning is to improve the realism of synthetic images
- S+U learning should preserve annotation information for training of machine learning models
- S+U learning should not generate images with artefacts
- All of the above are true
* `[ D ]`


---

**Q: In the paper on learning from simulated and unsupervised images, by Shrivastava et a., the authors:**
- Show that relying on real-world unsupervised data is a superior strategy to a hybrid approach.
- Use a combination of a refiner and a discriminator to integrate simulated and unsupervised data.
- Show the advantages that smooth simulated data has over unsupervised data, in terms of generalization performance.
- Make use of self-regularization concepts to improve performance on the training set.
* `[ B ]`


---

**Q: Why are needed the data created by a Generator Network?**
- to remove noise from images
- more data allow better classification
- to fool the Discraminator Network
- all of the above
* `[ B ]`


---

**Q: In the paper 'Learning from Simulated and Unsupervised Images through adversarial Training' a synthetic refined image is created using a GAN. To improve training performance self-regularization is introduced. Which of the following statement is true?**
- Self-regularization is added to make sure that the refined images do not look too much like their synthetic counterpart.
- Self-regularization is added to the loss function of the discriminator.
- Self-regularization minimizes the per-pixel difference between a feature transform of the synthetic and the real images.
- None of the above
* `[ D ]`


---

**Q: Which of the following about SimGAN is WRONG?**
- The refiner network minimized an adversarial loss to add realism and fool the discriminator.
- With an additional self-regularization loss, the refiner network is able to preserve the annotation information.
- To avoid artifact in the synthetic image, it limits the discriminator's reception field to local regions instead of the whole image.
- One of its goal is to train a very good simulator such that the generated synthetic  images are able to fool the discriminator.
* `[ D ]`


---

**Q: Which of the following techniques is used in S+U learning?**
- A 'self-regularization' term to preserve the annotation information of the simulator.
- A local adversarial loss to model the real image characteristics without introducing any artifacts.
- Updating the discriminator using a history of refined images to improve the stability of adversarial training.
- All of above.
* `[ D ]`


---

**Q: Which one of the following choices of the synthetic and real images is not true?**
-  Learning from synthetic images may not achieve the desired performance due to a gap between synthetic and real image distributions. 
- Training models with real images are expensive, since real images need to be annotated by human labor.  While training models on synthetic images can solve this problem. The annotations of synthetic images can be generated automatically.
- To establish the simulated and unsupervised learning model with the use of a set of unlabeled real images, we need to learn the model by minimizing a combination of two losses. The first loss is the discrimination between the real images and the synthetic images. The second loss is the difference between a feature transform of the synthetic and real images. 
- The simulated and unsupervised learning adds realism to the simulator while preserving the annotations of the synthetic images. 
* `[ C ]`


---

**Q: Which item cannot help GAN to preserve annotations, avoid artifacts and stabilize training?**
- a self-regularization loss
- a global adversarial loss
- using a history of refined images
- a local adversarial loss
* `[ B ]`


---

**Q: We can stabilize the GANs training by ?**
- Adding a self regularization term
- Using a local adversarial loss
- Updating the discriminator with a series of refined images
- All of the above
* `[ D ]`


---

**Q: Which of the following is correct about Simulated+Unsupervised (S+U) learning through adversarial networks?**
- Severe artifacts are preserved when training is done using history of refined images.
- Severe artifacts are preserved when training is done without using history of refined images.
- Gaze estimation error is high when using history of refined images than when not using the history.
- None of the above
* `[ A ]`


---

**Q: What is true about acversarial loss? **
- A global adversarial loss uses a fully
connected layer in the discriminator network, classifying
the whole image as real vs refined. The local adversarial
loss removes the artifacts and makes the generated
image significantly more realistic.
- Improve the stability of adversarial training
by updating the discriminator using the ones in the current minibatch.
- Another problem of adversarial training is that the
discriminator network only focuses on the history of refined
images and forgets about the order.
- All of the above.
* `[ A ]`


---

**Q: Which of the following is NOT the innovation point of this paper?**
- self-regularization loss
- sythetic images
- local adversarial loss
- using a history of refined images
* `[ B ]`


---

**Q: What is the use of a refiner network?**
- Improve quality of generated synthetic image
- Improve quality of actual, real image
- Improve architecture of generative adversarial network
- Reduce losses on misclassified examples
* `[ A ]`


---

**Q: What is the main difference between GAN’s and Simulated + Unsupervised learning / SimGAN?**
- S+U learning does not use a discriminator network
- S+U learning uses a refiner to refine synthetic images coming from a simulator
- GAN’s don’t use unsupervised learning
- S+U learning doesn’t use unsupervised learning
* `[ B ]`


---

**Q: In Simulated+Unsupervised learning (e.g. enhancing the realism of a synthetic image with certain defined parameters such as pose or gaze) it is useful to have 2 loss functions working together. These are the adversarial loss and the self-regularization loss. Which is the purpose of each one of this?  **
- The former is used to add realism and the latter to avoid diverging too much from the original input. 
- The former is used to add realism and the latter to avoid introducing artifacts. 
- The former is used avoid diverging too much from the original input and the latter to add realism.
- The former is used to avoid introducing artifacts and the latter to add realism. 
* `[ A ]`


---

**Q: Why is it good to train models on synthetic images and what is its problem?**
- Potentially avoiding the need for cheap annotations; May not achieve the desired performance due to a gap between synthetic and real image distributions
- Potentially avoiding the need for expensive annotations; May not achieve the desired performance due to a gap between synthetic and real image distributions
- Potentially avoiding the need for expensive annotations; May not achieve the desired performance but not due to a gap between synthetic and real image distributions
- Potentially avoiding the need for expensive annotations; May achieve the desired performance but with high cost
* `[ B ]`


---

**Q: Which of the following is not a contribution made by the authors of the paper 'learning from simulated and unsupervised images through adversarial training'?**
- They propose S+U learning that uses unlabeled real data to refine the synthetic images.
- They introduce the GAN framework to generate visually realistic images and a Structured GAN to learn surface normals.
- They train a refiner network to add realism to synthetic images using a combination of an adversarial loss and a self-regularization loss.
- They present qualitative, quantitative, and user study experiments showing that the proposed framework significantly improves the realism of the simulator output.
* `[ B ]`


---

**Q: What is an issue with simulated images**
- Making sure they have the same features as the real images
- None
- Very expensive to make
- Requires a lot of computing power.
* `[ A ]`


---

**Q: Which of the following is true about Simulated + Unsupervised (S+U) learning?**
- Learn a model to improve the realism of a simulators output using unlabeled real data
- Annotation information from the simulator is preserved
- Generate images without artifacts
- All of the above
* `[ D ]`


---

**Q: Which of the following strategy is not used for stableize the training process?**
- S+U learning
- a 'self-regularization' term
- a local adversarial loss
- updating the discriminator using a history of refined images
* `[ A ]`


---

**Q: Which is the problem of generating synthetic, automatically labeled images using unsupervised learning?**
- These images can contain artifacts used to fool the discriminator net
- The generated images are nowadays all very similar one to the other
- The labeling procedure is very expensive and time consuming
- Non of the above
* `[ A ]`


---

**Q: Which of the following options, according to the background and empirical results in this paper, is false?**
- Simulated+Unsupervised learning uses unlabelled real data to refine the synthetic images.
- In Simulated+Unsupervised learning the refined images are indistinguishable from real ones using a discriminative network.
- Simulated+Unsupervised learning requires user intervention at inference time.
- Simulated+Unsupervised learning add realism to synthetic images, using a combination of an adversarial loss and a self-regularization loss.
* `[ C ]`


---

**Q: Which is not an addition to standard GANS:**
- Changing the input of the generator
- Using historically refined images to train the discriminator
- Adding a regularization parameter to the loss function
- Applying morphological operations to images for data augmentation
* `[ D ]`


---

**Q: A GAN can be used to refine synthetic datasets. Among helpful network features for this task are self-regularization terms, a local adversarial loss and updating the discriminator using a history of refined images. Why is a self-regularization term useful for this task?**
- To stabilize training by avoiding drifting and avoiding introducing artifacts
- To preserve the annotation of the image by penalizing large changes between the synthetic and refined image
- Because that allows for vector arithmetics when the GAN is trained
- Because otherwise the input real dataset is simply copied.
* `[ B ]`


---

**Q: When humans were asked to determine whether an image was real or synthetic, what was approximately the human classification accuracy?**
- 30%
- 50%
- 70%
- 90%
* `[ B ]`


---

**Q: A problem that might occur when training the GANS neural network discriminator is that, due to the lack of memory, it only focuses on the latest refined images. In order to improve the stability of adversarial training, a solution can be:**
- Update the discriminator using a history of refined images;
- Update the discriminator using a history of the images in the current mini-batch;
- Update the discriminator with noisy images;
- Update the discriminator using generated images only.
* `[ A ]`


---

**Q: How can annotation information be preserved when training an refiner?**
- Introduce self regularization in the loss function
- Introduce regularization in the loss function
- Normalize input of the refiner
- Normalize output of the refiner
* `[ A ]`


---

**Q: Which of the following statements about S+U learning is incorrect? **
- With S+U learning the goal is to use unlabeled real images to learn a refiner that refines syntetic images
- A refined image should look like a real image in appearance while preserving the annotation information from the simulator when using S+U learning
- S+U learning tries to optimize a objective function that needs annotation information to create a good simulated image
- The loss function of S+U learning consists of two parts: one that adds realism to the images and one that preserves the annotation information
* `[ C ]`


---

**Q: On which neural network architecture is S+U learning based?**
- Recurrent neural network
- Auto-encoder
- LSTM
- GAN
* `[ D ]`


---

**Q: Why is fake data regnerators needed?**
- More data better classification
-  This is not the case it will make a deep-learning worser.
- Will remove noise from images
- None of the above
* `[ A ]`


---

**Q: What do the authors do to avoid having the generator over-emphasize image features in order to fool the discriminator, which leads to artifacts?**
- Alternatively trained the discriminator and the generator.
- A regularization term was used to correct the refiner from exaggerating.
- Applied a blur as a last step of refinement.
- Split the image into patches and have discriminator evaluate each.
* `[ D ]`


---

**Q: Which of these statements concerning Generative Adversarial Networks (GANs) is incorrect?**
- A self-regularizing term improves the ability of a GAN to generate realistically looking images
- Improving the model of the noise distribution in a GAN can increase its performance
- A refiner network combines unlabeled real images and synthetic images into refined images
- The local adversarial loss uses a fully connected layer in the discriminator network, classifying the whole image as real vs refined
* `[ B ]`


---

**Q: Choose the correct statement regarding the Simulated + Unsupervised learning method.**
- Unlabeled data is used by a refiner to enhance synthetic images.
- Unlabeled data is used to train the simulator to generate more realistic images.
- The refined image does not contain annotation information.
- At the end of an optimal training process, the discriminator must be able to classify the real and the refined images with high confidence.
* `[ A ]`


---

**Q: Training on synthetic images instead of on real images has become appealing but can be problematic due to:**
- Synthetic images are too perfect, creating only small opportunity for deviations
- Synthetic images are usually of larger size (memory), leading to slower computation
- Synthetic data often not being realistic enough, leading to a network failing to generalize well on real images
- Synthetic images are hard to obtain
* `[ C ]`


---

**Q: Why learning from synthetic images can be problematic?**
- We dont'n have enought synthetic images.
- Synthetic images are to big for effective learning.
- Synthetic images are not realistic enough.
- All of the above.
* `[ C ]`


---

**Q: 1) The method of local adversarial loss is equivalent to cross-entropy with two parameters.
2) A drawback of simulated images is that the simulations still need to be annotated by hand**
- Only 1 correct
- Only 2 correct
- Both correct
- Both false
* `[ A ]`


---

**Q: Which the following is the correct approach for improving the stability of training in GAN framework?**
- updating the discriminator using a history of refined images rather than only the ones from the current refiner network.
- limit the discriminator’s receptive field to local regions instead of the whole image
- introducing spurious artefacts while attempting to fool a single stronger discriminator
- None of the above
* `[ A ]`


---

**Q: A problem of adversarial training is that the discriminator network only focuses on the latest refined images. This lack of memory may cause:

1. Divergence of adversarial training
2. The refiner network re-introducing the artifacts that the discriminator has forgotten about**
- Statements 1 is correct, statements 2 is incorrect
- Statement 1 is incorrect, statement 2 is correct
- Both statements are correct 
- Both statements are incorrect
* `[ C ]`


---

**Q: Which statement is false about the refiner network in simulated + unsupervised learning?**
- The purpose is to filter out unrealistic data from the simulated data
- The refiner network can be trained similarly as a Generative Adversarial Network
- History can be used to improve stability
- It requires a relatively small amount of labelled real data
* `[ D ]`


---

**Q: Why would you use a GAN when trying to refine synthetic images?**
- Due to the nature of GANs, this will result in synthetic images which are very hard to distinguish from real images
- GANs are very stable during training, and thus are an excellent fit for this problem
- Because GANs are one of the only network structures which can properly interpret synthetic data
- You should not use a GAN for this problem, since refining is simply a matter of post-processing the image
* `[ A ]`


---

**Q: Which of the following is not part of the architecture of the simGAN described in the paper: “Learning from Simulated and Unsupervised Images through Adversarial Training”?**
- Refiner
- Discriminator
- Simulator
- Connector
* `[ D ]`


---

**Q: What is the function of Refiner in the S+U learning?**
- Add realism to synthetic images to reduce the difference between synthetic and read images
- Add realism to synthetic images to increase the difference between synthetic and read images
- Add realism to synthetic images to reduce the difference between synthetic images
- Add realism to synthetic images to reduce the difference between read images
* `[ A ]`


---

**Q: can synthetic examples be used to train a network?**
- Yes, always
- Yes, if specific precautions are taken
- No, never
- None of the above
* `[ B ]`


---

**Q: What is the goal of the generator network in the GAN framework?**
- Map a random vector to a realistic image
- Distinguish the generated images from the real images
- Generate a visually realistic image
- Generate a random image vector
* `[ A ]`


---

**Q: Which statement below is wrong?**
- The idea of training on synthetic instead of real images is appealing because the annotations are automatically available
- Learning from synthetic images can be problematic due to a gap between synthetic and real image distributions
- The GAN framework requires training two neural networks with competing goals, which is known to be unstable and tends to introduce artifacts
- The key requirement for S+U learning is that the refined image should look like a real image in appearance while abandoning the annotation information from the simulator
* `[ D ]`


---

**Q: When training images of eyes, what is the main motivation for using a history of refined images?**
- To increase noise in the data
- To decrease unrealistic artifacts in the image
- To ease the loss calculation
- None of the above
* `[ B ]`


---

**Q: In the “Learning from simulated and unsupervised images through adversarial training” paper the authors proposed SimGAN for refining synthetic images from a simulator. When it comes to the architecture of their method, the authors specify that the GAN framework, which requires training two neural networks with competing goals, is unstable and tends to introduce artifacts. What is their proposed solution for avoiding drifting and introducing artifacts when attempting to fool the discriminator?**
- The authors complement the adversarial loss with a self-regularization loss that penalizes large changes between the synthetic and refined images.
- The authors use a fully convolutional neural network that operates on a pixel level and preserves the global structure.
- The authors limit the discriminator’s receptive field to local regions instead of the whole image, resulting in multiple local adversarial losses per image.
- None of the above help combating the specified issue of GANs.
* `[ C ]`


---

**Q: What is the goal of Simulated+Unsupervised (S+U)?**
- Improve the realism of synthetic image
- Increase the learning speed of processing
- Reconstruct the learning structure
- None above all
* `[ A ]`


---

**Q: Which statement is not the shortage of adversarial training (while Updating Discriminator using a History of Refined Images)? **
-  The discriminator network only focuses on the latest refined images.
- This lack of memory may cause divergence of the adversarial training. 
- The refiner network re-introducing the artifacts that the discriminator has forgotten about. 
- Non of them. 
* `[ D ]`


---

**Q: Which of the following is an important consideration when training the author's SimGAN?**
- A regularization term for the Refiner itself must be introduced to prevent it from  changing the input image too much.
- To prevent drift and the production of artifacts in the refiner, a discriminator  should classify local image patches seperately instead of the whole image.
- To prevent re-introducing artifacts that the discriminator simply has forgotten about, the discriminator is shown random past images, as well as new images during each iteration.
- All of the above.
* `[ D ]`


---

**Q: What does the paper proposes the improvement of learning from synthetic images?**
- Sample more labeled images and compute combinations of them using GANs
- Blur the actual images using convolutions and filters so the algorithm learns more difficult data distributions
- Use a simulated+unsupervised learning to improve the realism of simulated images
- None of the above
* `[ C ]`


---

**Q: How does simulating data reinforce unsupervised learning algorithm.**
- It enables for obtaining labelled data from the simulator, so no human work needed
- It enables for better control over the labels
- The simulator can be improved to match more realistic data
- All of the above
* `[ D ]`


---

**Q: What is a typical problem encountered when training GANs**
- The adversary over-emphasizes on certain image features, causing artifacts
- The discriminator network only focusses on the latest generated adversary images, causing artifacts
- Discrete domains such as NLP cause GAN weight updates to be poorly defined
- All are possible problems encountered in training GANs
* `[ D ]`


---

**Q: Which of the following can be expected by using artificially generated synthetic images from Generative Adversarial Networks for training classifier model?**
- Improved generalisation due to increased availability labeled training data
- New artefacts developed in the synthetic images induce bias in the classifier model
- Significant improvement in learning duration without loss of accuracy in the classifier model
- All of above
* `[ B ]`


---

**Q: What is/are the TRUE statements about the main contribution and/or ideas of the paper?**
- the paper proposes a model with state-of-the-art results for gaze and hand-pose estimation without using any labeled read data
- the main problem with the synthetic hand pose images is not having the noise of the real images such as non-smooth depth boundaries
- to avoid introducing image artifacts while training a GAN architecture that tends to over-emphasize certain image features, the paper proposes limiting the discriminator’s receptive field to local regions of the image
- all of the above
* `[ D ]`


---

**Q: Which of the following statements is not correct?**
- DNNs may not be able to generalize to out-of-distribution inputs.
- The common pose transformation can be expressed as $v_i^{R,T}=T+v_i^R$, where R means rotate and T means translation
- Using human taking pictures will cause biase.
- Phenomenon in C can  not be solved by data augmentation
* `[ D ]`


---

**Q: which of these statements is/are correct?
1: S+U learning aims to use unlabeled real images to learn a refiner that can refine synthetic images, so they are free from artifacts
2: GAN's are known for their high stability and low artifact introduction.**
- only statement 1 is true
- only statement 2 is true
- both statements are true
- both statements are false
* `[ A ]`


---

**Q: Which of the following hold true?**
- Learning from synthetic data is better because network can now generalise details only present in this type of data.
- A refiner network should learn to model the real image characteristics without introducing any artificial artifacts
- Both the options (a) and (b) are correct
- Both the options (a) and (b) are wrong
* `[ B ]`


---

**Q: What does Simulated+Unsupervised learning offer?**
- Trying to learn a simulation algorithm.
- Training without having a labeled dataset.
- Labeling datasets.
- Generating training data.
* `[ B ]`


---

**Q: Which of the following statements is not correct: **
- Training supervised GANs require a lot of annotated data
- By adding a refiner to a GAN networks one can produce realistic images from synthetic images
- To preserve the annotation of the refined images from realistic images, adversarial loss was used only.
- In general GAN networks require training two networks with competing errors
* `[ C ]`


---

**Q: What is the goal of discriminator in GAN framework?**
- Map a random vector to a realistic image
-  Distinguish the generated from the real images
- Visual sequence prediction
- None of the above
* `[ B ]`


---

**Q: What is a problem of adversarial training, that was solved in SimGAN by using a buffer of refined images generated from previously networks? **
- The loss during training of the discriminator only focuses on the latest generated images causing divergence of the network 
- The loss during training of the refiner only focuses on the latest generated images causing divergence of the network
- The refiner tends to over emphasize certain features to fool the discriminator which introduces visual artefacts 
- The discriminator tends to over emphasize certain features to fool the refiner which introduces visual artefacts
* `[ A ]`


---

**Q: In GAN framework, what is the goal of a discriminator network?**
- Map a random vector to a realistic one
- Distinguish generated image from real images
- Minimize the local adversarial loss
- None of the above
* `[ B ]`


---

**Q: In the Simulated+Unsupervised learning settin of GANs, which statement is correct?**
- The refiner network should preserve annotation information without generating artifacts.
- The refiner network should modify annotation information without generating artifacts.
- Statistics of local patches sampled from the refined image should reflect the change in annotation information.
- The change in annotation information generated by the refiner network results in different statistics of local patches sampled from the refined image.
* `[ A ]`


---

**Q: Which of the following statements is FALSE?**
- By classifying local image patches, a discriminator can be improved
- By using a history of refined images, a discriminator can be improved
- Self-regularization in Feature-Space only works on grayscale images
- The refiner does not make use of labelled images
* `[ C ]`


---

**Q: What modifications did the authors of this paper did in GAN algorithm?
I. self-regularization term
II. local adversarial loss
III.  updating the discriminator using historical data**
- Only I
- II-III
- I-II-III
- I-III
* `[ C ]`


---

**Q: How can unlabeled samples be used to augment synthetic training data?**
- It is possible to train a refiner network together with a discriminator network, GAN style, to let the refiner network learn a transformation from the synthetic data, of which the labels are known, to look more like the unlabeled samples. This way you can get labeled training data which looks just like the real thing.
- You can train a classifier on the synthetic data, and use it to label the unlabeled samples, which can then be used during training.
- This is not recommended, because the synthetic training data will be out of distribution.
- By using domain adaptation methods.
* `[ A ]`


---

**Q: Which of the following is NOT a change introduced by Simulated+Unsupervised learning compared to generative adversarial networks**
- Introducing a 'self-regularization' term
- Introducing a local adversarial loss
- Updating the discriminator using a history of refined images
- Training a network that consists of both a Generator and a Discriminator.
* `[ D ]`


---

**Q: The contribution that the authors present is :**
- S+U learning that uses labelled real data to refine the synthetic images. 
- Train a refiner network to add realism to synthetic images using a combination of an adversarial loss and a self-regularization loss. 
- Use the GAN training framework to stabilize training without making any modifications.
- Both b and c
* `[ C ]`


---

**Q: Which of these components has a loss function by optimizing which it becomes harder to distinguish between real and not real data?**
- Simulator
- Refiner
- Discriminator
- None of them
* `[ B ]`


---

**Q: The SimGAN algorithm makes synthetic images go through a model (named "refiner") that is supposed to give them a more realistic look. When explaining the algorithm, the authors mentioned that extra caution had to be taken, in order to deal with the refiner and the changes it could induce in the annotations of the synthetically generated images. They were referring to the fact that**
- the refiner cannot change the type of information the images convey; for example if the synthetic image shows a hand in a "thumbs-up" position, then the refined image should still display that.
- the refiner should change the type of information the images convey; for example if the synthetic image shows a hand in a "thumbs-up" position, then it could change the pose of the hand, but it should look more realistic.
- the refiner cannot create impossible positions, like a hand that is completely twisted in the wrong direction.
- the refiner needs a more complicated objective function to capture the intent of generating more realistic versions of the images.
* `[ A ]`


---

**Q: Which of the following statements about the Simulated+Unsupervised (S+U) learning method the authors provide is false:**
- A synthetic image is generated with a black box simulator and it is refined using the refiner network.
- In order to preserve the annotations of synthetic images, they complemented the adversarial loss with a self-regularization loss that penalizes large changes between the synthetic and refined images.
- They used a fully convolutional neural network that operates on a pixel level and preserves the global structure.
- The used the whole image as the discriminator’s receptive field.
* `[ D ]`


---

**Q: For the paper: "Learning from Simulated and Unsupervised Images through Adversarial Training" what can be said about the qualitative method (asking human subjects to tel the pictures apart) that was used for evaluation?**
- This is a bad way of evaluating the model as humans are not good at this task.
- This is a bad way of evaluating the model as making the model mimic the training data as much as possible would yield great results.
- This is a good way of evaluating the model, as, according to the paper, humans had an accuracy of 50% and therefore, were randomly guessing.
- This is a good way of evaluating the model, as, according to the paper, humans were much better at telling apart the synthetic images and the adjusted images.
* `[ B ]`


---

**Q: In SimGAN, adding realism is done:**
- Using a refiner
- Using a black bock synthetic image generator
- Using adversarial loss with a discriminative network
- By generating images with artifacts
* `[ C ]`


---

**Q: Which one is wrong?**
- One of the interesting components to the SimGAN is that the generator starts from a simulated image rather than a random vector.
- Generator Loss for the SimGAN model, adversarial loss with respect to the discriminator.
- Generator Loss for the SimGAN model, self-regularization term which ensures the generator does not dramatically alter the original image.
- None
* `[ D ]`


---

**Q: In the work on (S+U) Training using GANs, what is responsible for ensuring that the refiner doesn’t learn how to generate artifacts that the discriminator had forgotten to identify?**
- Local Adversarial Loss
- Updating Discriminator using a History of Reﬁned Images 
- Neither A nor B can affect the presence of artifacts
- Both A and B
* `[ B ]`


---

**Q: What does a refiner network do?**
- Refine weights in a network
- Refine image aspects to make synthetic images look more realistic.
- Refine real images to make them appear more natural.
- None of the above.
* `[ B ]`


---

**Q: Which of the following statements is TRUE?**
- Simulated + Unsupervised (S+U) learning can be used to improve the realism of synthetic images from a simulator using unlabeled real data
- Synthetic data that is not realistic enough, leads the network to learn details only present in synthetic images and fails to generalize well on real images
- The GAN framework requires training two neural networks with competing goals 
- All of the above
* `[ D ]`


---

**Q: Why would you choose unsupervised training?**
- If you have no database with labels
- If you want to achieve a task with less coding time
- If you have less computer power
- If you want a solution that is more accurate
* `[ A ]`


---

**Q: The goal of Simulated+Unsupervised learning, as discussed in the paper by Shrivastava et al, is to use unlabelled real images to learn a refiner to refine a synthetic image to look like a real image. Which of the following two statements is correct regarding the key requirement that S+U learning poses?
1)	The refined image should look like a synthetic image in appearance
2)	The refined image should preserve the annotation information from the simulator**
- a) Both are correct
- b) Statement 1 is correct, Statement 2 is false
- c) Statement 1 is false, Statement 2 is correct
- d) Both are false
* `[ C ]`


---

**Q: Consider the following two statements about Simulated+Unsupervised learning:
1.The goal of Simulated+Unsupervised learning is to improve the realism of synthetic images from a simulator using unlabeled real data. 
2.To add realism the refiner network is trained using an adversarial loss, similar to GAN’s, such that the refined images are indistinguishable from real ones using a discriminative network. 
Which of these statements is true?**
- Both are true
- Both are false
- 1 is true and 2 is false
- 1 is false and 2 is true
* `[ A ]`


---

**Q: What does S+U learning do?**
- Adds realism to synthetic data 
- Preserves annotation information
- Generates images without artifacts
- All
* `[ D ]`


---

**Q: Why is it important to enhance the realism of generated images?**
- It is a important field for the video game industry
- Enables us to make training data for networks
- Makes it easier to design graphics
- None of the above
* `[ B ]`


---

**Q: Which statement about a GAN network that tries to make synthetic images look real (using adversarial training) is false?**
- The self-regularisation term ensures that the difference between the synthetic and real image is not too big
- The adversarial loss term makes sure that the discriminator cannot discern between real and refined images
- The self regularisation term changes the labels
- Training both networks is unstable since they have competing tasks
* `[ C ]`


---

**Q: GANs characteristic mechanism involves.**
- 2 adversarial network attempting to compute to get a lower global loss than their  counterpart.
- 2 network where 1 attempts to generate output to increase the loss of a secondary detector network.
- Global adversarial network, that always converges to a global optimum
- Refining images to generate photo-realistic outputs from synthetic images.
* `[ B ]`


---

**Q: What is the concept of adversarial loss**
- discriminator network made to focus also at earlier refined images, rather then the latest
- definition of a discriminator network that classifies all local image patches separately
- definition of a global discriminator network
- none of the above
* `[ B ]`


---

**Q: The adversarial loss used in training the refiner network is responsible for which of the following?**
- Bridge the gap between the distributions of real and synthetic images.
- Misleading the network into classifying the refined images as real.
- Preserve annotation information of the simulator
- All of the above
* `[ D ]`


---

**Q: In the paper, the authors propose a network to improve the realism of synthetic images from a simulator using unlabeled real data. They make several key changes to the standard GAN algorithm, one of which is adding a 'self-regularization' term. Why?**
- To preserve the annotations of synthetic images
- To add realism to the refined image
-  To model the real image characteristics without introducing any artifacts
- To improve the stability of adversarial training
* `[ A ]`


---

**Q: What is the advantage of using Simulated+Unsupervised (S+U) learning?**
- Using a history of refined images to improve the stability of the discriminator.
- Using a local adversarial loss that classifies all local image patches separately.
- Using a self-regularization term to decrease the gap between synthetic and real images.
- All of above
* `[ D ]`


---

**Q: The focus of the authors of paper 13 was...**
- ... to make it possible to train NNs with synthetic images with a combination of unsupervised and simulated learning.
- ... to propose an other way to train GANs.
- ... to publish a paper.
- ... to play with NNs.
* `[ A ]`


---

**Q: In which way you may improve performance of image generative network(based on GAN)**
- Use of adversarial Loss with Self-Regularization
- Use of local Adversarial Loss
- Updating Discriminator using a History of Refined Images
- All of the three
* `[ D ]`


---

**Q: What happens in adversarial training when memory is less?**
- Divergence of the adversarial training
- Discrimanator ovepowers the Refiner network
- Refiner network re-introducing the artifacts that the discriminator has forgotten
- Both A) and C)
* `[ D ]`


---

