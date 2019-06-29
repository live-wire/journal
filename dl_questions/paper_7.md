# Questions from `paper_7` :robot: 

**Q: Certain feature learning architectures can perform very well on object recognition tasks, without training. Why do random weights sometimes do
so well? **
- Certain convolutional pooling architectures can be inherently frequency selective and translation invariant, even with random weights.
- Certain convolutional pooling architectures can be inherently frequency selective and translation invariant, only with random weights.
- Certain convolutional pooling architectures can be inherently frequency selective and translation invariant, not with random weights.
- No convolutional pooling architectures can be inherently frequency selective and translation invariant, only with random weights.
* `[ A ]`


---

**Q: Which of the following w.r.t. circular and valid convolution is FALSE?**
- Valid convolution applies a filter f only at locations where f lies entirely within a given subregion of the original input image.
- Circular and valid convolution differ only near the edge of the input region.
- The optimal input for circular convolution is near optimal for valid convolution.
- Object recognition systems typically utilize circular convolution.
* `[ D ]`


---

**Q: Why the network architecture chosen by random initialization is very likely to well-performing?**
- Random-weight performance is not truly random but should correlate with the corresponding trained-weight performance, as both are linked to intrinsic properties of the architecture.
- The selection time is largely reduced since computationally expensive pretraining and finetuning are not needed in this procedure.
- Pretraining and finetuning is still necessary to attain near-top classification performance on CIFAR-10.
- Convolution is just one among many different architectural features that play a part in providing good classification performance.
* `[ A ]`


---

**Q: Comparing randomly weighted and pretrained/finetuned neural networks leads to which of the following conclusions?**
- Randomly weighted networks can substitute pretrained/finetuned ones since they achieve similar performance.
- The performance between the two is negatively correlated.
- Pretrained/finetuned networks are significantly faster compared to the random-weights approach.
- Random weights can be used to narrow down the network architecture options.
* `[ D ]`


---

**Q: What is FALSE about networks?**
- Randomly initialising networks can be very useful as a proxy for pre-training for fast architectural selection
- Performance of random-weight networks is significantly correlated with the performance of such architectures after pre-training and finetuning
- Convolutional pooling architectures enable even random-weight networks to be frequency selective
- Circular convolution, square-pooling architectures are not translation invariant.
* `[ D ]`


---

**Q: If the filter f is diffuse what is the optimal input close to?**
- a sinusoid at the maximum frequency in f
- Diffuse
- a sinusoid at the minimum frequency in f
- Random
* `[ A ]`


---

**Q: How can the convolutional square-pooling architecture be described?**
- As a single layer neural network. In the convolution layer, a bank of filters is applied at each position in the input image followed by a squaring and summing operation.
- As a two layer neural network. In the first convolution layer, a bank of filters is applied at each position in the input image, in the second pooling layer, neighboring filter responses are combined together by squaring and then summing them.
- As a three layer neural network. In the first convolution layer, a bank of filters is applied at each position in the input image, in the second interpolation layer, intermediate values are calculated to obtain a higher accuracy, in the third pooling layer, neighboring filter responses are combined together by squaring and then summing them.
- As an architecture that uses a predefined filter to obtain resulting values
* `[ B ]`


---

**Q: Review the following two statements about the circular convolution square pooling architecture:
\begin{enumerate}
    \item The architecture is inherently "frequency selective" due to the frequency of the optimal input being the frequency of the maximum magnitude in a filter.
    \item The architecture is "translation invariant" due to the arbitrary phase being unspecified.
\end{enumerate}
Which of the statements are true?**
- Statement 1 and 2 are true
- Statement 1 is true statement 2 is false
- Statement 2 is false statement 1 is true
- Statement 1 and 2 are false
* `[ A ]`


---

**Q: What does it mean in case random weights have the same or slightly worse results when compared to pretrained and finetuned weights?**
- That the random weights are by coincidence chosen very good
- That can mean that the architecture of the network alone can derive the most for the performance
- That choosing random weights is better for a network than pretrained and finetuned weights
- None of the above answers is correct
* `[ B ]`


---

**Q: For the most part, the key to a network’s performance lies in improving the learning algorithms. What is another key to a network’s performance?**
- a) The type of network architecture that is used
- b) The type of learning algorithm that is used
- c) The type of random weights that are used
- d) There is no other key to a network’s performance
* `[ A ]`


---

**Q: What does pre-training and fine-tuning do?**
- Increase the performance of the architecture
- Decrease the performance of the architecture
- Increase the time of parameter learning
- None of the above
* `[ A ]`


---

**Q: What is a benefit of the trend that architectures which perform well with random
weights also tend to perform well with pretrained and finetuned weights, and vice versa?**
- We do not need to train our networks anymore
- We can evaluate the mean performance of such architectures over several random initializations a lot faster and train them later
- We do not need convolutional layers anymore
- Circular convolution and valid convolution differ only near the edge of the input region
* `[ B ]`


---

**Q: Which of the following are properties of circular convolution, square-pooling architecture?
I - scale agnostic
II - translation invariant
III - frequency selective

Hint: As described in the paper, circular convolution is the behavior of convolution on fabricated periodic boundary conditions**
- I and II
- II and III
- I and III
- I, II, and III
* `[ B ]`


---

**Q: Statement 1: A feature of the circular convolution, square pooling architecture is: 
The frequency of the optimal input is the frequency of maximum magnitude in the filter f. Hence the architecture is frequency inselective
Statement 2: A feature of the circular convolution, square pooling architecture is: The phase φ is unspecified, and hence the architecture is translation variant.**
- Both statements are true
- Statement 1 is true
- Statement 2 is true
- None of the statements are true
* `[ D ]`


---

**Q: What is a circular convolution?**
- a convolution that permits wrap-around
- a convolution which is shaped circular
- a convolution which has no inputs
- a convolution which has no outputs
* `[ A ]`


---

**Q: Based on the research in the paper, what is something that you might want to do?**
- Perform a search over possible architectures before pretraining.
- Stop pretraining and finetuning the filters altogether.
- Both of the above
- None of the above
* `[ A ]`


---

**Q: What is key in unsupervised feature learning?**
- Training of the model
- Choosing a suitable model architecture
- Making models as deep as possible
- Use a suitable stochastic process to randomise the weights
* `[ B ]`


---

**Q: In the paper "On Random Weights and Unsupervised Feature Learning", a heuristic is proposed which should greatly reduce the amount of time taken to search the space of all available architectures in order to find the most suitable. What is this heuristic?**
- Pre-train and fine-tune each network, and evaluate performance
- Initialize weights in each network randomly, and evaluate performance
- First initialize weights in each network randomly, evaluate performance, then evaluate again the top performing classifiers after pre-training and fine-tuning
- First pre-train and fine-tune each network, evaluate performance, then evaluate again the top performing classifiers after random initialization of weights
* `[ C ]`


---

**Q: The random weight networks reflect intrinsic properties of their architecture. What can they be used for?**
- To avoid fine tuning
- To avoid pertaining
- For fast architectural selection
- To converge faster as a big learning rate can be used
* `[ C ]`


---

**Q: Which of following statements is NOT the reason why searching for the most suitable architectures is very important to achieve a good performance in object recognition tasks?**
- Particular architectures can naturally compute features well-suited to object recognition tasks even without trained weights.
- There's a surprising amount of performance just from their architecture alone.
- Many learnt networks lose out in terms of performance to random-weight networks with different architectural parameters
- In particular architectures, optimal inputs are often not sinusoid and sensitive to translations.
* `[ D ]`


---

**Q: Statement 1: Some architectures perform well without training on specific recognition tasks even with no training, because the architecture itself contains sufficient information about the task.
Statement 2: With carefully prepared models the learning phase may be omitted while still reaching state of art results.
Choose which of these statements if true or false:**
- 1: true, 2: true
- 1: true, 2: false
- 1: false, 2: true
- 1: false, 2: false
* `[ B ]`


---

**Q: What do the authors of ”On Random Weights and Unsupervised Feature Learn-ing” say is true?**
- It  is  not  important  to  distinguish  the  contributions  of  architectures  from those of learning systems by reporting the performance of random weights
- To get good performance, one cannot solely focus on the learning algorithm and neglect a search over a range of architectural parameters
- Random-weight networks do not reflect intrinsic properties of their architecture
- Random weights can be a substitute for pretraining and finetuning
* `[ B ]`


---

**Q: Which statement is NOT true?**
- Large-scale searches of the space of possible network architectures can be carried out by evaluating the mean performance of such architectures over several random initializations.
- Convolution is an architectural feature that plays a major role in providing good classification performance.
- The optimal input for the case of circular convolution is near-optimal for the case of valid convolution.
- Valid convolutional square-pooling architectures will respond near-optimally to sinusoids at the maximum frequency present in the filter.
* `[ B ]`


---

**Q: Which statement is FALSE?**
- the performance of random-weight networks is not truly random but correlates with the trained-weight performance. 
- convolutional pooling architectures enable even random-weight networks to be frequency selective
- The performance of random networks is much worse than the one of pretrained networks. However, there is a correlation between the random and pretrained networks.
- It is a good idea to distinguish between the contribution of architectures and of learning systems by reporting the performance of random weights 
* `[ C ]`


---

**Q: Starting from a speculation about the surprising results of random untrained weights, the authors of "On Random Weights and Unsupervised Feature Learning" then discover that:**
- The initialization of the weights is not affecting the speed of convergence
- The convolutional layer's performances are not dependent on the type of filter used
- The achitecture's choice is heavily affecting the performance in object recognition's tasks
- None of the above
* `[ C ]`


---

**Q: With the current state of research, what is a good application of unsupervised learning with random weights?**
- Pretraining.
- Finetuning.
- Architectural selection.
- Input selection.
* `[ C ]`


---

**Q: In the paper two different types of convolution are used: “valid” and “circular”. Which of the following statements is correct?**
- The two types of convolutions are exactly the same
- The results of the two convolutions differ only close to the center of the image
- The results of the two convolutions differ only close to the borders of the image
- Valid convolution is more flexible than circular convolution
* `[ C ]`


---

**Q: What is not true (as mentioned in the paper: “On Random Weights and Unsupervised Feature Learning”)?**
- A key feature of the circular convolution, square-pooling architecture is that the frequency of the optimal input is the frequency of maximum magnitude in the filter f. Hence the architecture is frequency selective.
- A key feature of the circular convolution, square-pooling architecture is that the phase is unspecified, and hence the architecture is translation invariant.
- Convolution is just one among many different architectural features that play a part in providing good classification performance. 
- An architecture with trained weights always outperforms an architecture with untrained weights because the trained architecture network is fully optimized.
* `[ D ]`


---

**Q: What is a fast, but efficient, way for weight initialization**
- Using random weights for n runs and selecting the most efficient
- Using random weights
- Applying Topographic independent components analysis (TICA)
- There is a strict trade off between strict and efficient for initializing weights
* `[ A ]`


---

**Q: What has more influence on the performance of a neural network?**
- The optimised weights
- The architecture
- A and B both have just as much influence
- The answer differs from case to case
* `[ B ]`


---

**Q: Which of these is TRUE about Pretrained vs Random-weight ConvNet?**
- Pretrained ConvNet are always better than random-weight ConvNet
- Random-weight ConvNet are always better than pretrained ConvNet
- Pretrained ConvNet are usually better than random-weight ConvNet
- Random-weight ConvNet are usually better than pretrained ConvNet
* `[ C ]`


---

**Q: The answers to this question are according to the paper ‘On Random Weights and Unsupervised Feature Learning’.
For object classification using convolutional neural networks, which of the following statements is true:**
- Pretraining and fine tuning the filters always increases the accuracy significantly. 
- Pretraining and fine tuning the filters can reduce the performance of convnets depending on the structure of the input data.
- Determining the performance of a convnet with random filter weights is a good heuristic for estimating the quality of the architecture.
- All convolutional pooling architectures are inherently frequency selective and translation invariant.
* `[ C ]`


---

**Q: Given the following two statements, determine their veracity: 1) "Randomly weighted unsupervised training methods can significantly facilitate the search for a good unsupervised learning architecture", and 2) "Randomly weighted convolutional networks can be frequency selective even if the filters used are randomly sampled."**
- 1) true 2) true
- 1) true 2) false
- 1) false 2) true
- 1) false 2) false
* `[ A ]`


---

**Q: What is the benefit of circular convolution campared to valid convolution?**
- cirvular convolution is computationally faster
- exact computation of optimal input 
- frequency selective 
- translation invariant
* `[ B ]`


---

**Q: Which of the following statements is true?
I:  Random weight networks reflect intrinsic properties of their architecture
II: Pretraining and finetuning can be substituted by using random weights**
- Only I is true
- Only II is true
- Both I and II are true
- Both I and II are false
* `[ A ]`


---

**Q: Which empirical result is not true?**
- Pretraining and Finetuning is necessary for top performance results
- Random-weights performance does not correlate with trained-weights performance
- Classification is faster than training weights
- Frequency selectivity and translation invariance are ingredients well-known to produce high-performing object recognition systems
* `[ B ]`


---

**Q: Which of the following statements are true?

1. The performance of random-weighted networks is significantly correlated with the performance of such architecture after preparing and fine-tuning. 
2. Random weights are substitute for pre-training and fine-tuning and thereby sidestepping the time-consuming learning process.**
- Statements 1 is correct, statements 2 is incorrect
- Statement 1 is incorrect, statement 2 is correct
- Both statements are correct 
- Both statements are incorrect
* `[ A ]`


---

**Q: Why are convolutional neural networks with random weights able to extract frequency information from images?**
- Randomly assigned filter weights have a maximal input image frequency which maximises the convolved output.
- The pooling layers makes the network translation invariant and the high input from certain frequency content in the images are preserved.
- Both A and B together.
- Neither A nor B.
* `[ C ]`


---

**Q: Which would be the preferred approach for carrying out a large-scale search of the space of possible architectures for a given task?**
- Compare the reported performance metrics of different architectures in the literature, select the top performing ones and evaluate them on the target data.
- Select a set of target architectures and evaluate their performance on the target data with random initialization (no training). A subset of the best ones is selected for pre-training and fine tuning.
- Select a set of target architectures, fine tune and pre-train them, and evaluate their performance on the target data. Then select the top performing one.
- There is no need for a large-scale search. There is a fixed set of architectures available for each task and there is no need to consider all the options.
* `[ B ]`


---

**Q: Which statement about ideas behind the Fast Architecture Selection is True?**
- On the same architecture, random-weight performance is correlated with trained-weight performance.
- Classification is significantly faster than pretraining and finetuning of networks.
- Both A and B are true.
- None of the above.
* `[ C ]`


---

**Q: Why can untrained networks that have randomly initialized weights be helpful?**
- Random initialization has better performance than starting with weights set to 0.
- Trained weights sometimes have lower performance.
- Unsupervised pretraining can improve model performance.
- Random weight network performance can correlate with trained network performance.
* `[ D ]`


---

**Q: What is a practical result from this study?**
- No more need for finetuning
- No more need for pretraining
- Fast architectural selection
- Faster computation
* `[ C ]`


---

**Q: Which of the following arguments is correct?**
- convolutional pooling architectures can be inherently frequency selective and translation invariant even with random weights
- Pre-trained weights always have a higher performance than 
- performance of random-weight networks is significantly correlated with the architecture selection
- All above
* `[ D ]`


---

**Q: In the random weights paper, what is the given explanation for the surprisingly good performance of random weight networks?**
- There is a chance the random weights were already close to the optimal weights
- The weights have no influence on the performance of the network
- The performance depends less on the weights, and more on the architecture of the network itself
- The architecture of the network is less important than the weights
* `[ C ]`


---

**Q: What was the main result from paper 7 on random weights?**
- Training a network can help at most 10%
- Circular convolution works better than valid convolution
- Proof that sinoidal filters perform best
- The performance of a trained network is strongly correlated with that of a random network
* `[ D ]`


---

**Q: Consider weights and hyperparameters of feature learning architectures. Which statement is true?**
- Certain feature learning architectures with random, untrained weights can do very well on object recognition tasks
- Pre training and hyperparameter tuning is essential part of deep learning architectures, thus it cannot be dismissed
- Pre training and hyperparameter tuning is not necessary but it will always give a better classification performance
- None of the previous answers are correct
* `[ A ]`


---

**Q: Random weights in unsupervised learning sometimes perform better. The statement is **
- FALSE
- TRUE
- Cannot be generalised 
- none of them
* `[ B ]`


---

**Q: What is the importance of network architecture?**
- It allows for random weight initialization
- It is not domain specific
- A and B
- None of the above
* `[ A ]`


---

**Q: Which of the following claims on Random Weights and Unsupervised Feature Learning is FALSE?**
- When using random weights classification can be significantly faster than pretraining and finetuning
- Current state-of-the-art feature detection systems derive a surprising amount of performance just from their architecture alone
- Random weights is in general a substitute for pretraining and finetuning
- Random-weight networks reflect intrinsic properties of their architecture (e.g. convolutional pooling architectures enable even random-weight networks to be frequency selective)
* `[ C ]`


---

**Q: Which of the following is not true about random weights?**
- Certain convolutional pooling architectures can be inherently frequency selective and translation invariant even with random weights.
- Random weights are substitute for pretraining and finetuning.
-  Unsupervised pretraining and discriminative finetuning improve classification performance relative to untrained random weights.
- A sizeable component of a system's performance can come from the intrinsic properties of the architecture, and not from the learning system.
* `[ B ]`


---

**Q: Why do random weights sometimes do so well?**
- Some convolutional pooling architectures can be frequency selective and translation invariant anyway.
- Because random weights come along with nice characteristics.
- Because using random weights makes pretraining and finetuning redundant.
- Because random weights can be used as a starting point.
* `[ A ]`


---

**Q: Which of the following two statements about the circular convolution, square-pooling architecture is false:  

\begin{enumerate}
\item The frequency of the optimal input is the frequency of maximum magnitude in the filter $f$. Hence the architecture is \emph{frequency invariant}.
\item The phase $\phi$ is unspecified, and hence the architecture is \emph{translation invariant}.
\end{enumerate}**
- Only 1 is false
- Only 2 is false
- Both are false
- None of the above are false
* `[ A ]`


---

**Q: What is true about 'valid' convolution?**
- Applies the filter at every position in x
- Applies the filter f only at locations where f lies entirely within x
- Allows the filter to “wrap around” in cases where it does not lie entirely within x
- None of the above
* `[ B ]`


---

**Q: What would you use a CNN with random weights for?**
- Fast architecture selection.
- Target applications with low-accuracy requirements without training.
- Recognizing noisy images.
- Training with higher learning rate.
* `[ A ]`


---

**Q: Using feature learning networks with random weights can speed up architecture selection. Why is this possible?**
- Because random weights are just as good as trained weights in feature learning.
- Because untrained architectures react to sinusoidal inputs
- Because the performance of untrained networks is correlated with the performance of the same network but trained
- because the architecture has no effect on the performance of a network.
* `[ C ]`


---

**Q: Which statement is false?**
- No feature learning architectures with random, untrained weights could ever outperform the same architectures with pretrained weights.
- Convolutional pooling architectures can be inherently frequency selective.
- Convolutional pooling architectures can be inherently translation invariant.
- B and C are true.
* `[ A ]`


---

**Q: Which of the following statements about random weights and unsupervised feature learning is false?**
- Random-weight networks reflect intrinsic properties of their architecture; for instance, convolutional pooling architectures enable even random-weight networks to be frequency selective
- For a fixed architecture, unsupervised pretraining and discriminative finetuning deteriorates classification performance relative to untrained random weights.
- Architectures which perform well with random weights also tend to perform well with pretrained and finetuned weights, and vice versa.
- Convolutional pooling architectures can be inherently frequency selective and translation invariant, even when initialized with random weights.
* `[ B ]`


---

**Q: Why do random weights sometimes do so well?**
- Certain convolutional pooling architectures can be inherently frequency selective and translation invariant.
- Certain convolutional mini-batch architectures can be inherently frequency selective and translation invariant.
- Certain convolutional pooling architectures can be inherently volume selective and translation invariant.
- Certain convolutional mini-batch architectures can be inherently volume selective and translation invariant.
* `[ A ]`


---

**Q: What is an efficient way to perform a search over a range of architectural parameters of possible network architectures?**
- Plot the classiﬁcation performance of random-weight architectures against trained-weight architectures
- Evaluate the mean performance of the architectures over several random initializations
- Pretrain and ﬁnetune each network, then evaluate the results in order to find the best parameters
- None of the above
* `[ B ]`


---

**Q: What trends can be observed when the classification performance of random-weight architectures against trained-weight architectures is ploted?**
- Perform well with random weights
- Tend to perform well with pre-trained and fine-tuned weights
- Random-weight performance is not truly random
- All of the above
* `[ D ]`


---

**Q: What are the characteristics of optimal input to a neuron for object recognition tasks ?**
- sinusoid and insensitive to translation
- sinusoid and sensitive to translation
- exponential and sensitive to translation
- exponential and insensitive to translation
* `[ A ]`


---

**Q: Which of the following is not true for this paper?**
- Certain feature learning architectures can perform very well on object recognition tasks, with training.
- Certain convolutional pooling architectures can be inherently
Frequency selective and translation invariant, even with random weights.
- The viability of extremely fast architecture search by using random weights to evaluate candidate architectures.
- Surprising fraction of the performance of certain state-of-the-art methods can be attributed to the architecture alone.
* `[ A ]`


---

**Q: What is the meaning of Valid convolution?**
-  It means that f applied only at vertical position inside x such that f lies entirely within x.
- It means that f applied only at each position inside x such that f lies entirely within x.
- It means that f applied only at horizontal position inside x such that f lies entirely within x.
- It means that f applied only at only vertical or horizontal position inside x such that f lies entirely within x.
* `[ B ]`


---

**Q: In the paper: "On Random Weights and Unsupervised Feature Learning", the authors showed that certain neural network architectures performed well on object recognition with random weight initialization. Which of the following is given as a reason for this**
- The distribution from which the random weights are selected removes the need to perform any further learning or parameter tuning hence the improved results
- The depth and complexity of the reviewed architectures allowed them to correct any negative effects due to the selection of random weights
- The architecture of the network might inherently posses characteristics such as frequency selectivity and translation invariance that by default make it suitable for recognition tasks even with random weights
- None of the above
* `[ C ]`


---

**Q: Which of the statement regarding to the circular convolution, square-pooling architecture is false? **
- The frequency of the optimal input is the frequency of maximum manitude in the filter $f$. Hence the architure is \textit{frequency selective}.
- The phase $\phi$ is unspecified, and hence the architecture is \textit{translation invariant}.
- Object recognition systems typically use circular convolution.
- Circular convolution and valid convolution differ only near the edge of the input region.
* `[ C ]`


---

**Q: According to this paper, what is the main practical use for random weight networks?**
- Fast classification
- fast architectural selection
- Usable in large ensembles
- Replace pretrained networks
* `[ B ]`


---

**Q: Which statement about the use of random weights in deep learning is not correct?**
- The architecture of a network plays a very important role in feature representation for object recognition.
- Convolutional pooling architectures enable random-weight networks to be frequency selective.
- Random-weights can substitute for pretraining and finetuning in a network.
- The performance on random-weight networks is significantly correlated with the performance of such architectures after pretraining and finetuning.
* `[ C ]`


---

**Q: Which statements about the “On random weights and unsupervised learning” paper are true:

Statement1: In the architecture Conv + Pool: If filter (kernel) f is highly frequency selective, we might expect that the optimal input will be diffuse, random or sinusoidal at frequency f if the filter itself is diffuse and random, or sinusoidal.

Statement2: In the architecture Conv + Pool: Regardless of the filter f (kernel) the optimal input will be near a sinusoid at the maximal frequency of frequencies present in a filter.

Statement3: Both circular and “valid” convolutions produce identical responses at the edges of the image, however in the center of the image results are different.

Statement4: The performance of random-weighted networks is significantly correlated with the performance of same architectures after pretraining and finetuning**
- 2 and 4 
- 1 and 2
- 1 2 and 3 
- 2 3 and 4
* `[ A ]`


---

**Q: The input that activates the filter the most is equal to**
- Sinusoid at the max frequency of the filter
- Random
- Sinusoid at high frequency
- Gaussian white noise
* `[ A ]`


---

**Q: Which of these statements is false?**
- Convolutional pooling architectures enable random-weight networks to be frequency selective
- Random weights are no substitute for pretraining and finetuning
- After convolution the image is always smaller than before the convolution
- A sizeable component of a system’s performance can come from the intrinsic
properties of the architecture
* `[ C ]`


---

**Q: Does architectures which perform well with random weights also tend to perform well with pretrained and finetuned weights?**
- Yes
- No
- There is no correlation
- No, but it works vice versa
* `[ A ]`


---

**Q: Which statement about the paper is FALSE?**
- Random weight network with convolutional pooling demonstrate importance if architecture for feature representation
- Various choice of distribution the weights were drawn did not affect results, as long as they had mean equal to 0
- Heuristic architecture search is slower than pre-training and fine-tuning
- Randomly initialized networks take much more time to classify compared to the ones using fine-tuning and pre-training
* `[ C ]`


---

**Q: Select the FALSE statement regarding valid and circular convolution:**
- Valid convolution applied on a grayscale image of [N,M] pixels  with a filter of size [K,K] that slides with stride 1 generates a feature map of [N-K + 1, M-K + 1]
- The optimal input for valid convolution is significantly different from the optimal one for circular convolution only near the border
- Both circular and valid convolution generate activation maps of a size at least equal with the input 
- Circular convolution is less natural, but it is used for computing the optimal input exactly
* `[ C ]`


---

**Q: Which of the following parts of a system to contributes strongly in the performance of that system?**
- Pretraining
- Finetuning
- Architecture
- All of the above
* `[ D ]`


---

**Q: Which of the following options, according to the empirical results in this paper, is false?**
- Convolutional pooling architectures can enable random-weight networks to be frequency selective.
- The architecture of the network has a relevant role in feature representation for object recognition.
- The performance of random-weight networks is correlated with the performance of architectures after pretraining and finetuning.
- Random weights can always be a substitute for pretraining and finetuning.
* `[ D ]`


---

**Q: Why do we still need pretraining and finetuning since the random parameter can achieve a surprisingly good performance?**
- It is necessary to attain near-top classification performance
- The performance of radom parameter is extremly unstable
- It is hard to generate a set of good ramdom numbers
- The architecture has to be more complex if we use random number
* `[ A ]`


---

**Q: Regarding the convolutional square-pooling architectures, what are the properties that allow the good performance of this network?**
- translation invariance and frequency selectivity
- translation invariance and frequency diversity gain
- translational symmetry and frequency diversity gain
- None of above
* `[ A ]`


---

**Q: It has been recently noticed that for object recognition tasks some neural networks (mainly convolutional ones) work well even if...**
- we use stochastic gradient descent just training some layers
- we use as much layers as possible
- we set randomly the weights in them
- we use feed forward architecture for them
* `[ C ]`


---

**Q: Which statement about Optimal input that activates pooling is false?**
- The optimal Input is one which matches the convolution filter best
- Optimal input for circular convolution is also near optimal for normal convolution
- Optimal input is sinusoidal with a frequency that of the max magnitude in the filter
- Phase of the input is irrelevant (translation invariance) 
* `[ A ]`


---

**Q: How do convolutions and max pooling help in object recognition**
- Convolutions help with feature selectivity and pooling with translational invariance
- Convolutions help with edge selectivity and pooling with translational invariance
- Convolutions help with edge selectivity and pooling with reducing the issues related to curse of dimensionality
- None
* `[ A ]`


---

**Q: What is a good way to envisage square pooling?**
- As a two layer Neural Network.
- As a single convolutional layer.
- As a randomly weighted hidden layer.
- As optimization using squared loss.
* `[ A ]`


---

**Q: Following are some statements related to the paper ' On random weights and unsupervised feature learning'
(a) To get a good performance, one had to give time to search over a range of architectural parameters while focusing on a learning algorithm
(b) Since pre-training and fine tuning do not consume relatively much time as compared to classification, it is always better to include them for accurate results
(c) Convolution pool architectures enable random-weight networks to be frequency selective**
- Statements (a) and (c) are correct; (b) is wrong
- Statements (b) and (c) are correct; (a) is wrong
- All statements are correct
- Statements (a) and (b) are correct; (c) is wrong
* `[ A ]`


---

**Q: What does Sakse et al suggest as mathematical reasoning for a one-layer convolutional pooling architecture to have some ability to classify object using only random weights with no training?**
- The optimal norm-one input can have an arbitrary phase, which makes the filter translationally invariant, enabling recognition of objects in spite of position differences.
- The optimal input can be recognized by the structure of the architecture invariantly of perturbations in the input frequencies. 
- For n-dimensional objects, the frequency filter will extract the n-k+1 optimal frequencies to classify your object with likelihood 1. 
-  None of the above
* `[ A ]`


---

**Q: Which of the following statements about the conclusions of this paper is correct?**
- A method for fast architectural selection is to evaluate its performance with random-weights as this is significantly correlated with
its performance after pretraining and finetuning
- Convolutional pooling architectures need to be pretrained and finetuned in order to be frequency selective and translation invariant
- For a fixed architecture, unsupervised pretraining and discriminative finetuning significantly improves  classification performance relative to untrained
random weights
- The biggest component of a system’s performance comes from the learning system, and not from the intrinsic
properties of the architecture
* `[ A ]`


---

**Q: The fact that some networks initialized with random weights were shown to perform surprisingly well in some tasks**
- showed that the accuracy of some models is highly due to the architecture of the network.
- means that the researchers got eally lucky, as random weight initialization should only give an accuracy comparable to that of guessing.
- means that professor David was wrong in his regularization lecture, and initializing weights with samples from the uniform distribution in (0, 1) is ok.
- is conjectured to be explained by the action of God, dark magic and unicorns. Just kidding, no one knows.
* `[ A ]`


---

**Q: Why are self-attention layers used in the proposed architecture?**
- It is less computationally complex
- Allows more parallelization of computation
- Allows learning long-range dependencies in the network
- All of the above
* `[ D ]`


---

**Q: What intrinsic properties of the architecture discussed in the paper have?**
- Frequency selective
- Translation invariant
- Weights require no pretraining and fine-tuning 
- All of the above
* `[ D ]`


---

**Q: Whats the is true about a neural network that is based on good architectures but with random weights and a neural network that is trained?**
- The first neural network will never outperform the pertained neural network
- The neural network which is based on good architectures doesn’t perform well on object recognition tasks
- The convolutional square-pooling architecture will generate many pooling unit outputs
- There is not any difference between the two models
* `[ C ]`


---

**Q: How can contributions of neural network architecture be distinguished form contributions by the learning algorithm**
- by comparing the performance of the neural network before and after training
- By comparing the performance of the pretrained neural network to the performance when using random weights 
- The contribution of the neural network architecture to performance is neglible
-  by trying multiple architectures using the same learning algorithm
* `[ B ]`


---

**Q: The circular convolution:**
- Applies filter at every position of the sub-region and allows filters to „wrap around” in cases where it does not lie entirely within sub-region
- Applies filter only at the location in which it lies entirely within the sub-region
- Applies Gaussian filters
- Applies filters where the frequency is higher
* `[ A ]`


---

**Q: What property(s) of neural network architecture are good for object recognition?**
- Translation invariance
- Frequency selectivity
- A & B
- None
* `[ C ]`


---

**Q: What can be an extremely fast way to perform architecture selection and evaluation for a NN?**
- Train different architectures and evaluate their performance.
- pretrain the networks through local receptive field Topographic Independent Components Analysis (TICA)
- Train a network, make small changes in the architecture and use the same weights as the previous network.
- Evaluate the performance of the network using random weights.
* `[ D ]`


---

**Q: Which of the following is true?**
- The learning of weights is always much more important than the architecture in deep neural networks
- Empirically testing network configurations based on the learning algorithm is much faster than testing based on the architecture of the network
- Convolutional networks with random weight values can sometimes give performance close to the weight-trained version of the same network
- Convolution is the only architectural feature that plays a part in providing good classification performance
* `[ C ]`


---

**Q: Which of the following statements on CNNs is true?**
- Random weight initialization results in random network performance, before training.
- There is no correlation between the network performance before training (right after initialization) and after training.
- In general; when the mean network performance of a random initialized architecture is higher than that of another architecture, then so will be the performance after training.
- Unsupervised pretraining takes on marginally more time than purely random initialization and is always required, before comparing architectural performances.
* `[ C ]`


---

**Q: The performance of random-weight networks is heavily dependent on: **
- Architecture
- Training set
- Validation set
- Random-weight networks are always unreliable
* `[ A ]`


---

**Q: What can we conclude with the usage of random weights when compared to pre-trained weight in a Convolutional Neural Network for image classification?**
- A combination of random weight along with pre-trained weights, should be considered for the convolutional network.
- Random weights perform better most of the times, in terms of accuracy when compared with the equivalent pre-trained weights.
- Random weights never perform better, in terms of accuracy when compared with the equivalent pre-trained weights.
- Many pre-trained networks lose out in terms of performance to random weights networks with different architectural parameters, hence its important to select the network architecture carefully.
* `[ D ]`


---

**Q: which of these statements is true?

I: a network with random weights and a superior architecture can outperform a network with trained weights and a lesser architecture
II: if we combine the top-performing architecture from a random weight architectural search with pretraining and finetuning, we are likely to find the overall top-performing architecture**
- only statement I is true
- only statement II is true
- both statements are true
- both statements are false
* `[ C ]`


---

**Q: What suggestion does the paper make in regards to initializing a network**
- One should always try to find optimal initialization weights for a network
- The initialization weight have more impact than the architechture of the network
- One should first optimize the architechture of the network before optimizing the initialization weights
- Optimizing initialization weight has almost no effect on later performance of the network
* `[ C ]`


---

**Q: What is the suggested reason (according to the paper’s author) why random weights sometimes do so well for convolutional networks?**
- Convolution is not dependent on location
- The architecture is frequency selective
- Both
- None of the above
* `[ C ]`


---

**Q: Does the performance of a neural network almost only depend on the learning system?**
- Yes, only the learning system has effect on the performance of a neural network
- Yes, but the intrinsic properties of the underlying architectures can also have some effect on the performance of a neural network
- No, the learning system does barely have any effect on the performance of a neural network
- No, the intrinsic properties of the underlying architecture can also have a lot of effect on the performance of a neural network
* `[ D ]`


---

**Q: Which of the following is true:**
- Any architecture shows good performance with random weights
- The optimal input for a filter used in convolution is similar to the type of filter
- Valid and circular convolutions produce identical responses in the interior of the input region
- Circular convolution is not applied at locations where the filter does not fit entirely
* `[ C ]`


---

**Q: Which of the following algorithms obtained highest accuracy on NORB? **
- Random weights
- Tiled convolutional neural nets
- Convolutional neural nets
- SMVs
* `[ B ]`


---

**Q: Which of the following characteristics of convolutional pooling object recognition systems have been observed to improve the performance of the system?**
- These systems can be inherently frequency selective
- These systems are translation invariant
- Both of above
- Neither of above
* `[ C ]`


---

**Q: Which of the following statements is true?**
- CNN architectures with a single convolution layer and a square pooling layer are proven to be invariant to affine transformations of the input
- Finite random filters whose weights are drawn from an arbitrary distribution exhibit a single maximal frequency with absolute certainty
- The stability of a neural network with with arbitrary weights relies heavily to the randomization range of its hidden parameters
- Random weight performance is indicative of the generalization properties of any neural netwrork
* `[ C ]`


---

**Q: Which of the following is/are correct about the use of random weights in feature learning?

1. Many learnt networks perform worse than networks that use random weights.
2. Architecture of a network plays at least as much of an important role as the learning process and should therefore not be underestimated.**
- Only 1 is correct
- Only 2 is correct
- Both are correct
- Both are incorrect
* `[ C ]`


---

**Q: Why random weight perform well in CNN **
- Some architectures yielded good performance with random weights
- Random weight convergence all the time in short iteration
- random-weight networks is signiﬁcantly correlated with the performance of such architectures after pre-training and ﬁne tuning
- a and c
* `[ D ]`


---

**Q: It has been shown that object recognition tasks can perform very well without any training. If any, what is the reason for this?**
- Mainly because of the architecture.
- Mainly because of the wrong datasets being used.
- Mainly because of the stochastic gradient descent.
- None of the above.
* `[ A ]`


---

**Q: Why random weights sometimes works well?**
- Because of applying good filters sometimes
- Because of applying good pooling sometimes
- Because of applying good padding sometimes
- Because of applying good non-linearity function sometimes
* `[ A ]`


---

**Q: The performance of a classifier depends on**
- the learning algorithm.
- Architecture of the classifier.
- Both A and B
- None of the above.
* `[ C ]`


---

**Q: What is the main point of this paper?**
- To show that random weights perform better than tuned weights
- To show that architecture has a big influence on performance
- To show that unsupervised learning always performs better
- None of the above 
* `[ B ]`


---

**Q: Which of the following is not true regarding the experiments in the paper on random weights and unsupervised learning?**
- The distribution that the random weights are drawn from does not effect the classification performance, as long as the distribution is centered at 0.
- Pretraining and fine tuning increases the performance of an architecture. 
- Random weights can be substituted for pretraining and tuning
- A good performance can be attained by both improving learning algorithms and developing the structure
* `[ C ]`


---

**Q: Which of these statements is incorrect?**
- The performance of a neural network largely depend on its architecture
- Convolution is just one among many different architectural features that play a part in providing good classification performance
- Regardless of the filter, the optimal input will be near a sinusoidal at the maximal frequencies or frequencies present in the filter
- A sizeable component of a system's performance comes from the learning system
* `[ D ]`


---

**Q: What is the difference between "valid" and "circular" convolution**
- Valid convolution does not wrap around the input, circular does.
- Valid convolution mirrors the edges of the input, circular wraps around.
- Circular convolution uses circular filters instead of square ones.
- Circular convolution works with polar coordinates instead of a regular grid of pixels.
* `[ A ]`


---

**Q: Which of the following statements are correct:
1.Architectures which perform well with random weights also tend to perform well with pretrained and fine-tuned weights.
2.Convolutional pooling architectures enable even random-weight networks to be frequency selective.**
- 1) and 2) both are correct
- Only 1) is correct
- Only 2) is correct
- Both are incorrect
* `[ A ]`


---

**Q: Which cannot explain the good performance of the learning with random weights?**
- Architecture
- Frequency selective
- Translation invariant
- None of the above
* `[ D ]`


---

**Q: Which of the following statements can NOT been said about a network with one convolution layer and a squared pooling layer with random weights?**
- His performance is correlated with the same pretrained and finetuned network
- His performance depends on the architecture of the network
- The filters of the network are frequency selective and translation invariant
- The filters of the network are frequency invariant and translation selective
* `[ D ]`


---

**Q: Which of the following methods would be the best method to determine initial weights in an architecture?**
- Try out different weights and determine the best (possibly by interpolation)
- Use random weights
- Pre training and finetuning
- Try out different weights pick the best then pretrain and finetune.
* `[ D ]`


---

**Q: Which of the following about random-weight network is correct?**
- Random weights help attain high performance of networks.
- Random weights doesn't help to improve the performance of networks at all.
- After pre-training and fine-tuning, random-weight networks will achieve better performance.
- Random-weights can be substituted by pre-training and fine-tuning.
* `[ C ]`


---

**Q: What will random weights at the start do?**
-  Saving time in computing 
- Have a negative effect on the network performance.
- Be slower, but more preciese network
- Add an extra layer of complexity that saves time of pretraining and finetuning
* `[ A ]`


---

**Q: Why can convolutional networks have good performance even with randomly assigned (untrained weights)**
- The architectural decisions can be inherently frequency selective, which aids classification
- The convolution operator in itself is not frequency selective, which helps with feature detection
- The depth of the network is almost as important as the weights of the network
- The weights do not contribute to the classification results, thus training has no effect
* `[ A ]`


---

**Q: Which of the following statements is TRUE about random-weighted networks?**
- random-weight are enabled by Convolutional pooling architectures of frequency selection by square pooling
-  random-weight networks is significantly correlated with the performance of lack of fine tuning and more pre-training
- random-weight networks are replaceable to pre-training and/or fine tuning's performance. 
- None of the above
* `[ A ]`


---

**Q: Which one is wrong?**
- The performance of random-weight networks is significantly correlated with the performance of such architectures after pretraining and finetuning, and random weights are substitute for pretraining and finetuning.
- The (classification) performance delivered by random CNN features is correlated with the results of their end-to-end trained counterparts 
- The top-performing networks had trained weights, and pretraining and finetuning invariably increased the performance of a given architecture.
- “Deep Learning” and “a Deep Neural Network (DNN) with unsupervised pre-training” are not totally the same things. But they have an overlap, since the specified model is a common type of DL model.
* `[ A ]`


---

**Q: Which of the following statements are true about convolution square-pooling architecture?

A in the convolution layer a bank of filters is applied at each position in the input image and in the pooling layer, neighboring filter responses are combined together by squaring and then summing them up.  
B Selectivity for a specific feature of the input is due to the convolution stage. 
C Robustness to small translations of the input is due to pooling stage.**
- A
- B
- C
- All of the above
* `[ D ]`


---

**Q: What does the performance of random weight networks tell us about CNNs?**
- Training CNNs only gives a marginal improvement so untrained networks can be used to save time.
-  Convolutional pooling architectures in themselves are already frequency selective.
- Random weights are only good at predicting randomly generated test images.
- Using random weights is a good method of prefiltering noise out of images.
* `[ B ]`


---

**Q: From paper 7, what does the relatively high performance of random-weight networks imply?**
- Pretraining and/or finetuning is in some cases unnecessary
- Convolutional networks perform particularly well without pre-training and/or finetuning
- Some learning algorithms are sensitive to overfitting, only decreasing the performance of the architecture
- Architectures that perform well with random weights typically also perform well when trained
* `[ D ]`


---

**Q: Which statement is NOT true about how random initialization of CNN parameters can help in finding good performing models?**
- With random initialization, one can quickly determine a set of suitable parameter values without the need for pretraining and finetuning
- One does not need pretraining and finetuning, as randomly initialized models perform as well as finetuned models
- Optimization using random weights take far less time than traditional optimization methods
- Large scale searches of the space of the model’s possible hyperparameter values can be done using the average performance of such architectures over several random initializations
* `[ B ]`


---

**Q: Which one is true ? Random weights perform so well,**
- particular architectures can naturally compute features well-suited to object recognition tasks.
- Because they are random and therefore not biased.
- This only works with squared max pooling.
- architectures which perform well with random weights tend to under preform with pretrained and finetuned weights
* `[ C ]`


---

**Q: The two key features for a circular convolution, square pooling architecture is:**
- (1) The optimal input should ensure frequency invariance of the neutal activation function & (2) The architecture follows an finite sum of periodic functions, hench the filter \textit{f} having several maximal frequencies.
- (1) The architecture distribution of node lenghts follows a joint normal distribution & (2) The architecture locking phase are fixed on the input frequency, thus defined.
- (1) The frequency amplitude of the moving average filter \textit{f} is centered around the frequency of the optimal input & (2) The phase $\phi$ can be expressed by a Laplace distribution
- (1) The frequency of the optimal input is the frequency of maximum magnitude in the filter \textit{f} & (2) The phase $\phi$ is unspecified
* `[ D ]`


---

**Q: Why would you use random weights and not train them?**
- To decrease computation time
- To get better models in general
- To test the network performance
- None of the above
* `[ A ]`


---

**Q: Why does a RESnet act as an ensemble of shallow neural networks.**
- Removing a layer in a RESnet has the same effect as removing a layer in a NNet
- Because most effective connections of a RESnet are relatively shallow compared to the total amount of layers of the RESnet
- Because multiple shallow NNet give the same variance en mean as a RESnet doe, given the same training and test data.
- A RESnet uses the values of a NNet to feed a new NNet
* `[ B ]`


---

**Q: Read the following statements.\newline
A. For good performance, we should not only focus on the learning algorithm but should also perform search over a range of architectural parameters.\newline
B. Architectures which perform well with random weights also tend to perform well with pretrained and finetuned weights.\newline
C. Architectures which perform well with pretrained and finetuned weights also tend to perform well with random weights.\newline
Select the correct options.\newline**
- B, C are correct.
- A, B are correct.
- A, C are correct.
- All are correct.
* `[ D ]`


---

**Q: What should be considered in order to get a model with good performance?**
- Learning algorithm
- Selection of a range of architectural parameters
- Both options
- None of the options
* `[ C ]`


---

**Q: In the paper “On Random Weights and Unsupervised Feature Learning” the authors make a modification to the convolutional square-pooling architecture in order to permit an analytical solution. More specifically, instead of treating the case of valid convolution, the consider circular convolution. What is the idea behind circular convolution?**
- Applying the filter f only at locations where f lies entirely within x.
- Applying the filter f only in the center of x.
- Applying the filter f at every position in x, allowing the filter to “wrap around” in cases where it does not lie entirely within x.
- Applying the filter f only in the border areas of x.
* `[ C ]`


---

**Q: which statement is wrong? **
- Convolutional pooling architectures enable even random-weight networks to be frequency selective
-  The performance of random-weight networks is significantly correlated with the performance of such architectures after pre training and fine tuning.
- Top-performing networks had trained weights, and pre training and fine tuning invariably increased the performance of a given architecture.
- Non of them.
* `[ D ]`


---

**Q: Which of the following is true about impact of random weights in unsupervised feature learning?**
- Fixed architecture, unsupervised pretraining and discriminative finetuning improve classification performance relative to untrained random weights.
- Segmented changing architecture, unsupervised pretraining and discriminative finetuning improve classification performance relative to untrained random weights.
- Fixed architecture, unsupervised pretraining and non-discriminative finetuning improve classification performance relative to untrained random weights. 
- Segmented changing architecture, unsupervised pretraining and non-discriminative finetuning improve classification performance relative to untrained random weights. 
* `[ A ]`


---

**Q: What are the two main characteristics that prove the fact that random weights are indeed doing well for some convolutional pooling architectures?**
- Frequency selectivity & Translation invariation
- Translation invariation & Color invariation
- Color invariation & Frequency selectivity
- Color invariation & Edge radius
* `[ A ]`


---

**Q: In section 2, the authors analyse the optimal input for activating certain filters of CNN architectures. They offer an expression for finding the optimal input. They find the phase $\phi$ can be left unspecified. Why?**
- Because they are using circular convolutions.
- Because the tested architecture is sensitive to translations.
- Because the tested architecture is frequency selective.
- Because CNN architectures are inherently translation invariant. 
* `[ D ]`


---

**Q: Why is using random weights in feature learning architecture a viable solution?**
- In some instances, the architecture has a much higher influence on the performance than the weights.
- Training the weights instead can sometimes reduce performance.
- Using random weights skips the possibility of getting stuck when training.
- Random weights can give a broader solution space than trained weights.
* `[ A ]`


---

**Q: Which of the following statements is TRUE?**
- Good performance solely depends on the choice of learning algorithm 
- Good performance mainly depends on the amount of training epochs
- Good performance mainly depends on the choice of network architecture and learning algorithm 
- Good performance mainly depends on choosing the right learning rate
* `[ C ]`


---

**Q: After paper 7, random weights perform well because of...**
- ... convolutional pooling architecture, which can be selective.
- ... because god plays with his dices.
- ... random weights represent a natural state.
- ... training only overfits.
* `[ A ]`


---

**Q: 0**
- 0
- 0
- 0
- 0
* `[ D ]`


---

**Q: Given Theorem 2.1 of paper 7, in case of multiple maximum frequencies in f, with the coefficients drawn independently from a continuous probability distribution How many maximal frequencies will M have?**
- 1 with M={(v,h)}
- 2 with M={(v,h)}
- 1 with M={(v,sh)}
- 2 with M={(v,sh)}
* `[ A ]`


---

**Q: Assuming that you are training your deep convolution network using random weights, by which method below you cannot get better performance generally?**
- Change the distribution of generating random weights(all centered 0)
- Data augmentation in training dataset
- Use pretraining and finetuning to train this model
- All of the three
* `[ A ]`


---

**Q: Which of the following sentences is true?**
- An architecture should exhibit translational variance due to a pooling operation
- An architecture should exhibit selectivity for a specific feature of the input due to the convolution stage.
- Both circular and valid convolution produce identical responses at the edges
- If the filter f used for convolution is diffuse or random, the optimal input will be diffuse or random 
* `[ B ]`


---

**Q: What should be taken into consideration to get a good performance for object recognition?**
- Learning algorithm
- A range of architectural parameters
- A and B
- A or B
* `[ C ]`


---

**Q: Architecture selection with random weights is compared to classic architecture selection with pretraining and finetuning:**
- Faster and still fairly reliable in finding the best architectures.
- Slower but more reliable in finding the best architectures.
- Faster but unreliable in finding the best architecture.
- Slower and less reliable in finding the best architectures.
* `[ A ]`


---

**Q: What is the goal for pre-training and fine-tuning instead of random initialization?**
- It is difficult to generate a good random samples.
- It helps to achieve near-top classification performance.
- The performance of random parameter is extremely unstable.
- The architecture has to be really fancy if we use random initialization.
* `[ B ]`


---

