# Questions from `paper_8` :robot: 

**Q: Most of the gradient in a residual network with 110
layers comes from paths that are only 10-34 layers deep. What partly explains this?**
- Residual networks avoid the vanishing gradient problem by introducing long paths which can carry gradient throughout the extent of very deep networks.
- General networks avoid the vanishing gradient problem by introducing short paths which can carry gradient throughout the extent of very deep networks.
- Residual networks avoid the vanishing gradient problem by introducing short paths which can carry gradient throughout the extent of shallow networks.
- Residual networks avoid the vanishing gradient problem by introducing short paths which can carry gradient throughout the extent of very deep networks.
* `[ D ]`


---

**Q: Which of the following regarding residual networks is FALSE?**
- Residual networks have 2^n paths connecting input to output layers.
- The paths in a residual network vary in length.
- Entire layers of a residual network can be removed without impacting performance.
- Highway networks are a special case of residual networks.
* `[ D ]`


---

**Q: Which statement below is wrong?**
- Residual network performance improves with adding more and more layers
- Residual networks can be seen as collections of many paths and the only effective paths are relatively shallow
- Removing a layer from a traditional architecture such as VGG leads to a tiny loss in performance
- Residual networks can be seen as a collection of many paths of differing length
* `[ C ]`


---

**Q: Which are the components that make up each layer of a residual neural network?**
- Residual module and ReLu unit.
- Residual module and skip connection.
- Residual block and skip connection.
- A sequence of convolution and ReLu unit.
* `[ B ]`


---

**Q: What is FALSE about residual networks?**
- Residual networks can be viewed as a collection of many paths
- The individual paths in residual networks do not strongly depend on each other, despite the fact that they are trained jointly
- The paths that contribute the most gradient during training are the long paths
- The individual paths in residual networks exhibit an ensemble-like behaviour, since their performance smoothly correlates with the number of valid paths
* `[ C ]`


---

**Q: As what can a residual network viewed?**
- a collection of deep paths
- a single ultra deep network
- a convolutional deep network
- a shallow network
* `[ A ]`


---

**Q: What is a residual network?**
- A neural network in which each layer consists of  a residual module that is bypassed by a skip connection
- A neural network that uses the difference between the predicted value  and actual value as an input
- A neural network that calculates the difference between the predicted value  and actual value
- A neural network that has a feature pipeline which cannot be bypassed
* `[ A ]`


---

**Q: Review the following two statements about the results from Veit et al (2016) in "Residual Networks Behave Like Ensembles of
Relatively Shallow Networks" on residual networks:
\begin{enumerate}
    \item The results show that residual networks can't be reconfigured at runtime
    \item Paths in a residual network do not strongly depend on each other
    although they are trained jointly.
\end{enumerate}
Which of the statements are true?**
- Statement 1 and 2 are true
- Statement 1 is true and statement 2 is false
- Statement 1 is false and statement 2 is true
- Statement 1 and 2 are false
* `[ C ]`


---

**Q: What is residual network?**
- Residual networks are neural networks in which each layer is as large as the previous layer
- Residual networks are neural networks in which each layer consists of residual module $f_i$  and a skip connection bypassing $f_i$
- Residual networks are neural networks in which each layer consists of residual module $f_i$ that cannot be skipped. 
- None of the above answers is correct.
* `[ B ]`


---

**Q: What distinguishes a Residual Neural Network (RNN) from a conventional feedforward network?**
- a) The output of one layer is subtracted from the output of another layer
- b) The output of one layer is multiplied with the output of another layer
- c) The output of one layer is added to the output of another layer
- d) The output of one layer is divided by the output of another layer
* `[ C ]`


---

**Q: What will happen when we keep on increasing the ensemble size of neural nets?**
- The performance keeps on increasing proportionally
- The performance keeps on increasing exponentially
- The rate of performance increase, reduces
- None of the above
* `[ C ]`


---

**Q: What does the following sentence implicate: "This result suggests that paths in a residual network do not strongly depend on each other although they are trained jointly."?**
- We do not need a lot of training samples to train this network.
- As expected, deleting any layer in residual networks reduces performance to
chance levels.
- This shows that to some extent, the structure of a residual network can be changed at runtime without affecting performance.
- The gradient vanishes
* `[ C ]`


---

**Q: Suppose a given Residual Neural Network has 10 blocks. If we remove 2 blocks from the network, how many paths will our modified network contain?**
- 1024
- 8
- 256
- 1
* `[ C ]`


---

**Q: Statement 1: For paths through residual networks there is precisely one path that goes through all modules and n paths that go only through a single module. From this reasoning, the distribution of all possible path lengths through a residual network follows a Binomial distribution. 
Statement 2: For paths through residual networks generally, data flows along all paths in residual networks. However, not all paths carry the same amount of gradient**
- Both statements are true
- Statement 1 is true
- Statement 2 is true
- None of the statements are true
* `[ A ]`


---

**Q: What is one of the problems residual networks might have?**
-  Vanishing gradients
- Outdated residuals
- Polluted convolutions
- Bad activation functions
* `[ A ]`


---

**Q: What is meant with a residual network?**
- A network  in which certain weights have vanished to 0
- A network that has skip connections.
- A network that explains the difference between test and training performance
- None of the above
* `[ B ]`


---

**Q: Ensembling is the process of:**
- Reducing the number of weights 
- Arranging a committee of neural networks in a simple voting scheme, where final output predictions are averaged
- Leaving out hidden layers
- Going from a sparse network to a fully connected network
* `[ B ]`


---

**Q: Which of the following statements regarding Residual Neural Networks is incorrect?**
- Residual Networks can be viewed as a collection of many paths, instead of an ultra-deep network
- Residual Networks paths exhibit ensemble-like behavior in terms of performance
- Only the longest paths contribute gradient during training
- Residual Network paths do not depend on each other
* `[ C ]`


---

**Q: Residual networks don’t use longer paths while training because :**
- Longer paths are difficult to train
- Paths are not strongly dependent of each other
- Residual networks can be written as a collection of shortest paths
- Longer paths don’t contribute to gradient
* `[ D ]`


---

**Q: Which of following statements is NOT true regarding residual neural networks?**
- Residual networks can be viewed as a collection of many paths.
- Residual networks resolve the vanishing gradient problem by preserving gradient flow throughout the entire depth of the network.
- Although the paths are trained jointly, they do not strongly depend on each other.
- Paths through the network that contribute gradient during training are shorter than expected.
* `[ C ]`


---

**Q: Choose the correct statement about residual networks:**
- The average path length of the gradients is usually significantly lower than the depth of the network.
- Removing a single layer from the network usually harms performance badly.
- In a residual network data flow is possible from any layer to any other layer.
- Deep paths are equally important as short paths during training.
* `[ A ]`


---

**Q: Which of the following is true?**
- Deleting a layer in residual networks at test time is equivalent to zeroing a quarter of the paths.
- For residual networks, deleting any layer reduces performance to chance levels
- The distribution of all possible path lengths through a residual network follows a Binomial distribution
- Residual networks paths are trained jointly, and thus strongly depend on each other
* `[ C ]`


---

**Q: Which statement is NOT true?**
- Removing paths from residual networks by deleting layers has a modest impact on performance.
- Residual networks can be seen as a collection of many paths of differing length.
- Paths through residual networks strongly depend on each other.
- Only the short paths are needed during training, as longer paths do not contribute any gradient.
* `[ C ]`


---

**Q: How can one explain that a residual network shows only minor changes in performance, even if several modules are removed at test time?**
- Residual networks can be seen as a collection of many paths. Thus, if one module is removed, still half of the number of paths stay intact, making information flow still possible
- Residual networks can be seen as a collection of many paths. If one module is removed, also only one path is removed, thus makin information flow still possible
- Residual networks can be seen as a collection of modules, which process information in a similar manner. Thus, removing one block, still allows the network to process information
- The statement in the question is not true. Performance in a ResNet significantly drops if modules are removed.
* `[ A ]`


---

**Q: What is the vanishing gradients problem in resiudal networks?**
- The gradients become so small that is not possible to proceed with SGD after a certain number of epoques
- Not all paths carry the same amount of gradient
- The loss function is so difficult to compute in the network that the gradients are estimated only on two samples at a time
- None of the above
* `[ B ]`


---

**Q: Why does the performance of convolutional neutral networks drop after a high amount of layers?**
- Vanishing gradients.
- Overfitting.
- Underfitting.
- None of the above.
* `[ A ]`


---

**Q: What is a skip connection?**
- a neuron which is not connected to anything
- a layer which is not fully connected
- a connection that leads to a layer that is deeper than the subsequent layer
- a layer in which the weights are all equal
* `[ C ]`


---

**Q: What is true about residual networks?**
- When a single layer is removed, the only viable path is corrupted. This result suggests that paths in a residual network do not strongly depend on each other although they are trained jointly.
- Residual networks behave like ensembles (performance smoothly correlate with the number of valid paths).
- The gradient magnitude of a path decreases exponentially with the number of modules it went through in the backward pass.
- All of the above
* `[ D ]`


---

**Q: How does removing a single layer at test time from a residual network behave versus a traditional neural network?**
- the residual network significantly reduces in accuracy, the traditional network does not
- Both networks significantly reduce their accuracy
- the traditional network significantly reduces in accuracy, the residual network does not
- Both networks are not significantly influenced
* `[ C ]`


---

**Q: Deleting a layer in a residual neural network at runtime:**
- Disables the network from doing anything
- Still allows the network to work, but at a greatly reduced performance
- Still allows the network to work, without significantly affecting performance
- Still allows the network to work, even increasing its performance
* `[ C ]`


---

**Q: Which of these is TRUE about Residual Network?**
- all paths in a residual neural networks are of the same length
- removing a residual block from a residual network has a modest or even null impact in the performance
- paths in a residual networks depend strongly on each other
- the distribution of all possible path lengths through a residual network follows a Bernoulli distribution
* `[ B ]`


---

**Q: Related to residual networks, which of the following statements is false:**
- Residual networks have O(2^n) implicit paths.
-  Residual networks are insensitive to having one layer removed.
- In residual networks path lengths are uniformly distributed.
- Paths do not strongly depend on each other, even when jointly trained.
* `[ C ]`


---

**Q: Consider the perspective given by Veit et al. on residual networks being collections of many paths with different lengths. Which of the following statements does not hold?**
- These paths exhibit ensemble-like behavior, as their performance smoothly correlates with the valid path count.
- Removing individual layers of a trained residual network typically has a much lower impact on performance than removing individual layers from trained conventional convolutional neural networks.
- The deepest paths are typically not required during training, as they do not contribute to the gradient.
- The paths depend strongly on each other, as they are jointly trained.
* `[ D ]`


---

**Q: In which way(s) do residual networks challenge the conventional computer vision view of a familiar architecture; processing inputs from lower level features up to task specific high-level features?**
- identity skip-connections that bypass residual layers
- two orders of magnitude deeper
- removing a single layer does not affect the performance noticably
- all of the above
* `[ D ]`


---

**Q: What is not a feature of a residual network?**
- Deleting a random layer has does not reduce the performance significantly
- The effective paths in the network are of equal length
- It behaves like ensembles of relatively shallow networks
- Data in a layer can flow to any subsequent layer
* `[ B ]`


---

**Q: What is not true about a Residual Network?**
- It consists of paths with different lengths
- The output of a layer is y = f(x) + x, with f(x) being the denotation for each layer
- Highway Networks are a special case of ResNets.
- Empirically it is shown that not all paths equally contribute gradient during learning
* `[ C ]`


---

**Q: Which of the following statements are true?

1. One key key characteristic of ensembles is that the performance smoothly correlates with the number of valid paths (Members).
2. Unraveled view reveals that residual networks can be viewed as a collection of many paths, instead of a single ultra deep network.**
- Statements 1 is correct, statements 2 is incorrect
- Statement 1 is incorrect, statement 2 is correct
- Both statements are correct 
- Both statements are incorrect
* `[ C ]`


---

**Q: What is the most significant insight behind the “unravelled veiw” of ResNets?**
- All the long paths are made up of smaller paths.
- Longer paths learn more than information than shorter paths due to more gradient drops.
- ResNets can be considered as an ensemble of networks.
- All of the above are true and are important as per the authors.
* `[ C ]`


---

**Q: What is not true about residual networks**
- Contrary to sequential networks, they can resist the removal of a layer because after this happens, there are still paths connecting the input with the output.
- The error of residual networks smoothly increases with the removal of residual modules.
- Reorganizing of the modules of residual networks is not possible as performance sharply decreases.
- Paths in residual networks do not depend heavily on each other.
* `[ C ]`


---

**Q: What's a possible explanation for Recurrent networks' robustness to deleting a layer, in comparison to the strictly sequential network VGG?**
- When a layer is removed, this will always impact the path from input to output in VGG.
- When a layer is removed, this will usually impact the path from input to output in VGG.
- The number of paths from input to output is only reduced by a factor of $\frac{1}{4}$ in Recurrent Networks.
- Recurrent networks bypass the deleted layer by their design.
* `[ A ]`


---

**Q: How do the authors conclude that residual networks behave like ensembles of shallow networks?**
- They compare the classification performance of the residual network to an ensemble of shallow networks.
- They remove layers from the network and observe the effect on classification performance.
- They show the performance of the residual network without skip connections.
- They apply dropout in order to regularize the network.
* `[ B ]`


---

**Q: What reason is given for the increased performance of residual networks?**
- they are deeper
- they behave like an ensemble
- they have short paths that contribute most to the gradient
- all of the above
* `[ B ]`


---

**Q: Which of the following arguments is wrong about the residual networks:**
- They can be seen as a collection of shorter paths instead of a one deep network
- They paths are trained jointly and strongly depend on each other
- Their performance correlates with the number of valid paths
- Deep paths do not contribute much to the gradiant and can be ignored during training
* `[ B ]`


---

**Q: Why does the removal of long paths have barely any impact on residual network performance?**
- Long paths generally do not contribute noticeable gradient during training
- Short paths are the most common paths in the network
- Long paths consist of short paths, thus the removal is compensated by the ensemble of short paths
- Any subset of paths can be removed without impacting performance significantly
* `[ A ]`


---

**Q: Residual networks**
- can solve the problem of vanishing gradient through a network
- depend strongly on each other
- have ensemble behaviour in the valid paths
- have most gradient in the longest paths
* `[ C ]`


---

**Q: How a gradient vanishing problem may be avoided?**
- By trying to shorten the effective paths of learning, thus trying a residual network architectures
- By increasing test set
- By removing a whole module from network architecture
- By increasing network depth
* `[ A ]`


---

**Q: Residual Networks models are always implemented with single-layer skips**
- This statement is always true
- This statement is always false
- Nothing can said in  generalised way
- None of them are correct
* `[ C ]`


---

**Q: How does deleting k residual modules from a network of original length n influence the number of valid paths?**
- The number of valid paths decrease to O(2^(n-k))
- The number of valid paths increase to O(2^(n-k))
- The number of valid paths stays the same
- None of the above
* `[ A ]`


---

**Q: Which of the following claims on Residual Networks  is FALSE?**
- Residual networks avoid the vanishing gradient problem by introducing short paths which can carry gradient throughout the extent of very deep networks
- Identity skip-connections bypass residual layers, allowing data to flow from any layers directly to any subsequent layers 
- Skip connections give rise to networks that are two orders of magnitude deeper than previous models 
- Removing single layers from residual networks at test time significantly affects performance 
* `[ D ]`


---

**Q: Which of the following is not true about the residual networks?**
- A. residual networks do not resolve the vanishing gradient problem by preserving gradient flow throughout the entire depth of the network.
- Residual networks enable very deep networks by shortening the effective paths.
- The residual networks can be viewed as a collection of many paths, instead of a single ultra-deep network.
- Not only short paths contribute gradient during training, but also deep path are required during training.
* `[ D ]`


---

**Q: What explanation is carried out by the paper for explaining the increased performance of residual networks, which is contrary to the hypothesis ``We obtain these results via a simple but essential concept - going deeper." proposed by another paper.**
- A residual network can be seen as a collection of many paths.
- A residual network can be seen as a single ultra deep network.
- A residual network can be seen as as single simple path.
- A residual network can be seen as a collection of many ultra deep paths.
* `[ A ]`


---

**Q: Which of the following is false:**
- Paths in a residual network strongly depend on each other when trained jointly.
- Paths in a residual network of varying length impact the network differently.
- Paths in a residual network behave like an ensemble
- In a residual networks path length follows a binomial distribution.
* `[ A ]`


---

**Q: Select the statement that is not true about residual networks.**
- They introduce identity skip-connections that bypass residual layers
- Skip connections give rise to networks that are two orders of magnitude deeper than previous models
- Removing single layers from residual networks at test time does not noticeably affect their performance
- None of the above
* `[ D ]`


---

**Q: Which answer is FALSE according to the article?**
- Residual networks can be viewed asa collection of many paths, instead of a single ultra-deep network
- Paths do not strongly depend on each other,even though they are trained jointly.
- Residual networks are highly sensitive to dropping layers.
- Only the short paths contribute gradient during training. Deep paths are not required during training.
* `[ C ]`


---

**Q: Residual networks increase in performance can best be described with the following explanation:**
- Deeper networks are always better
- The paths in a residual network act like an ensemble
- The shallow paths in a residual network do not contribute 
- The paths in a residual network are highly dependent on each other
* `[ B ]`


---

**Q: Having a residual network with 100 layers, what path lengths would you expect to determine the gradient?**
- Paths shorter than 30 layers.
- Paths between 30 and 50 layers.
- Paths longer than 50 layers.
- Impossible to answer.
* `[ A ]`


---

**Q: Consider the following two statements about residual networks:
1.	Residual network performance improves with adding more and more layers. 
2.	Residual networks can be seen as collections of many paths and the only effective paths are relatively shallow.**
- Both are true
- None are true
- 1 is true and 2 is false
- 1 is false and 2 is true
* `[ A ]`


---

**Q: What types of paths are leveraged in residual networks to enable very deep networks?**
- Only the short paths.
- Both long and short paths.
- Paths with many forks.
- None of the above.
* `[ A ]`


---

**Q: Which of the following statements about paths in residual networks is true?**
- The length of the paths through the network does not affect the gradient magnitude during backpropagation
- Effective paths in residual networks are not shallow
- Deleting a module from a residual network removes the short paths through the network
- The path lengths are closely centered around the mean of $n/2$
* `[ D ]`


---

**Q: What are the characteristics of paths through residual networks?**
- Distribution of path lengths
- Vanishing gradients in residual networks
- The effective paths in residual networks are relatively shallow
- All of the above
* `[ D ]`


---

**Q: What should be the function of gating function for highway networks to act like residual networks**
- 0
- 0.25
- 0.5
- 1
* `[ C ]`


---

**Q: Which of the following work is not related to this ResNet paper?**
- Highway networks
- Ensembling
- Dropout
- Recurrent network
* `[ D ]`


---

**Q: Which one is NOT the importance of short paths in residual networks?**
- Distribution of path length
- deleting any layer in residual networks reduces performance to chance levels
- Vanishing gradients in residual networks
- The effective paths in residual networks are relatively shallow
* `[ B ]`


---

**Q: In the paper, "Residual networks behave like Ensemble of Relatively Shallow Networks", the authors compared a full residual network with a network with a shorter (effective) paths and found no significant statistical difference between the two, which of the following is given as a possible explanation**
- The reduced network suffers less from the gradient vanishing problem than the full network and this compensates for a loss of accuracy due to a smaller network
- The authors were unable to proffer an explanation for this anomaly, they did suggest however that skip connections allowed for the effective path to have a similar performance with the full network
- In a residual network, the effective paths contribute the most to the gradient updates, hence the reason for the similarity in performance to the full network
- None of the above
* `[ C ]`


---

**Q: Which of the statements regarding to residual networks is false?**
- Residual network are networks in which each layer consist of a residual module $f_{i}$ and skip a connection layer bypassing $f_i$.
- In a residual network, data flows among many paths from input to output. 
- In a residual network, paths are trained jointly, therefore they strongly depend on each other. 
- Residual networks do no resolve the vanishing gradient problem by preserving gradient flow throughout the entire depth of the network.
* `[ C ]`


---

**Q: What is not a key difference between Residual networks and 'normal' convolutional networks**
- The use of Identity skip connections
- The depth that can be achieved
- removing
The effect of removing single layers at test time
- The use of convolutional layers
* `[ D ]`


---

**Q: Which statement about short paths in residual networks is correct?**
- The distribution of all possible path lengths through a residual network follows a gaussian distribution.
- The length of the paths through the network affects the gradient magnitude during backpropagation.
- The gradient magnitude of a path decreases logaritmicly with the number of layers it went through in the backwards pass.
- Using only the effective paths through a network is not enough since they constitute to a very small part of all the paths.
* `[ B ]`


---

**Q: Which statements about the “Residual Networks behave Like Ensembles of Relatively Shallow Networks“ are true?

Statement1: One key characteristic of ensembles is their smooth performance with respect to the number of members. In particular the performance increase from additional ensemble members gets smaller with increasing ensemble size.

Statement2: Entire layers can be removed from plain residual networks without impacting performance, indication that they strongly depend on each other

Statement3: The paths in a residual network do not strongly depend on each other although they are trained jointly.

Statement4: If the collection of paths were to behave like an ensemble, we would expect test-time performance of residual networks to smoothly correlate with the number of valid paths. And this is indeed what observed for ResNets.**
- 1 and 4
- 2 and 3 
- All answers are true
- 1 3 and 4
* `[ D ]`


---

**Q: Two distinguishable components of a residual network are**
- residual module and skip connection bypassing
- Residual module and memory gate
- Residual pooling layer and bypass
- None of the above
* `[ A ]`


---

**Q: In a deep network with 100 layers that possesses the 'skip connection' property, what paths are more likely to contain most of the gradient? **
- Paths that are 10 to 34 layers long
- Paths that are 35 to 64 layers long
- paths that are 65 to 84 layers long
- paths that are 85 to 100 layers long
* `[ A ]`


---

**Q: Is every path in residual network equally important during training?**
- Yes
- No, shorter path are more important
- No, longer path are more important
- No, only very short and very long path are important
* `[ B ]`


---

**Q: What are Residual networks?**
- Simply different name for Conv Nets
- Neural networks in which each layer consists of a residual module and skip connection bypassing residual module
- Feed forward networks with RELU layers
- Multilayer perceptron with average pooling layers
* `[ B ]`


---

**Q: Choose the TRUE statement:**
- The paths in residual networks have the same length
- The paths in a feedforward network have varying length
- Adding a residual block into a residual network increases the number of paths by 2
- Deleting a layer from a residual network is equivalent to a “Dropout” of half of the paths 
* `[ D ]`


---

**Q: What is the main reason that residual neural networks enable very deep networks?**
- It is possible to jump over some of the layers
- Only the short paths are leveraged during the training
- Residual neural networks are very efficient
- Both forward and backward propagation are used
* `[ B ]`


---

**Q: Which of the following options, according to the empirical results in this paper, is false?**
- Residual networks can be seen as a collection of many paths, instead of a single ultra-deep network.
- The paths, even though they are trained jointly, do not strongly depend on each other.
- The paths exhibit ensemble-like behavior in the sense that their performance smoothly correlates with the number of valid paths
- Longer paths contribute most of the gradient during training.
* `[ D ]`


---

**Q: What is the difference of networks compared with normal network?**
- Some layers are disabled randomly
- Some neurons are disabled randomly
- Some weights are changed randomly
- None of above
* `[ A ]`


---

**Q: What are the paths through the network that contribute gradient more during training in residual networks?**
- Deep paths
- Short paths
- All short and deep paths
- None of above
* `[ B ]`


---

**Q: In residual networks, which are the most important paths during training (the ones that contribute to the gradient the most)?**
- The longer paths, that arre able to skip to farther layers in the net
- The shortest ones
- The ones that are closer to the output layer
- The ones that are closer to the input layer
* `[ C ]`


---

**Q: Which characteristic about a residual network is false?**
- It behaves like an ensemble of neural networks, in that the deletion of a module (layer) smoothly increases the error of the net
- The effective paths are relatively shallow, as the shortest paths have a steeper gradient, and the longest paths are more common (lengths follows a binomial distribution)
- In the unraveled view, the paths strongly depend on each other
- Deleting a module (layer) most removes the long paths.
* `[ C ]`


---

**Q: Why are Residual Networks good at learning features? **
- They can go many layers deep and hence can learn multiple features from input data?
- They are essentially an ensemble of multiple networks and hence can extract various features
- They solve the vanishing gradient problem and can hence fine tune the kernels quite well to the training data
- None
* `[ B ]`


---

**Q: How do residual networks enable very deep networks?**
- By shortening their effective paths.
- By removing layers from the network.
- By only selecting networks with high weights
- None of the above
* `[ A ]`


---

**Q: Following are some statements related to the paper ' Residual Networks Behave Like Ensembles of Realtively Shallow Networks'
(a) Residual Networks can be viewed as a special case to highway networks
(b) Unraveled view reveals that residual networks can be views as a collection of many paths
(c) Longer paths in Residual networks contribute more gradient because of the higher information flow**
- Statements (a) and (c) are correct; (b) is wrong
- Statements (b) and (c) are correct; (a) is wrong
- Statements (a) and (b) are correct; (c) is wrong
- All statements are correct
* `[ C ]`


---

**Q: Residual networks are defined by each layer being defined by the previous layer as: y_i = f_i (y_{i -1}) +y_{i-1}. The unravelled view presented in Veit et al. reformulates y_i as a function of the input y_0 rather than the previous layer y_i, which of the following statement is correct:**
- Each near residual layer will from the unravelled point of view duplicate the existing network, resulting in O(2^n) implicit paths connecting input and output.
- The path lengths follow a binomial distribution resulting in the residual network having O(n over p) == O(n!/(p!(n-p)!) implicit paths connecting input and output.
- The shared structure of the residual network becomes apparent by the unrolling the recursion into linearly nested terms, showing that the residual network has O(n) implicit paths connecting input and output.
- None of the above
* `[ A ]`


---

**Q: Which of the following statements about residual networks is not a conclusion drawn from this paper?**
- Residual networks resolve the vanishing gradient problem by preserving gradient flow throughout the entire depth of the network
- The unraveled view revealed that residual networks can be viewed as a collection of many paths, which do not strongly depend on each other, although these paths are trained jointly
- The unraveled view revealed that residual networks can be viewed as a collection of many paths, which exhibit ensemble-like behavior in the sense that their performance smoothly correlates with the number of valid paths
- The unraveled view revealed that residual networks can be viewed as a collection of many paths. The paths through the network that contribute gradient during training are shorter than expected, and more specifically deep paths are not required during training as they do not contribute any gradient
* `[ A ]`


---

**Q: A residual neural network is**
- a sparse network, i.e. a network where the majority of the weights are set to 0, meaning only some portions of the full architecture are used.
- a network that computes residues, instead of the usual label predictors.
- the network you obtain by taking the difference of two accurate networks with the same architecture.
- a network trained to predict the error residues of a second network applied to a classification task.
* `[ A ]`


---

**Q: Why residual networks are resilient to dropping layers?**
- Residual networks can be seen as a collection of paths, and dropping one layer, leaves half of the paths valid connecting from the beginning to the end of the network.
- Residual networks have only certain layers that can be dropped, and if by chance, one drops one of those, they still work
- Because the initial layer is connected to the last one directly, so one can drop any layer in the middle.
- None of the above
* `[ A ]`


---

**Q: What is the difference between the hierarchy of residual networks and the classical visual hierarchy?**
- The classical visual hierarchy follows a strictly sequential pattern
- Residual networks depends each layer of processing only on the output of the previous layer
- The classical visual hierarchy has multiple paths of data flows.
- Each node in the classical visual hierarchy is fed with data of distributions generated from all previous nodes
* `[ A ]`


---

**Q: The residual networks can be viewed as a collection of many paths instead of a single deep network, that means**
- The paths are trained separately but they are depend on each other
- The paths through the network that contribute gradient during training are usually long
- Residual networks resolve the vanishing gradient problem by keeping the gradient flow stay to the entire depth of the network
- They exhibit ensemble-like behavior something that means that they perform well if the number of valid paths is high
* `[ D ]`


---

**Q: What is the benefit of training a model with stochastic depth?**
-  it encourages paths to produce good results independently
- it has no benefits
- It allows the model to be trained in less time by only using a subset of the data
- both A and C
* `[ A ]`


---

**Q: Deleting a module from a residual network:**
- Removes the long paths through the network
- Removes the short paths through the network
- Removes the broken paths through the network
- Removes noisy paths through the network
* `[ A ]`


---

**Q: What do residual connections add?**
- Multiple paths through a network
- Redundancy in a network
- Gradient updates for training deep networks
- All of the above
* `[ D ]`


---

**Q: Which of the following is false with regard to residual network?**
-  It allows you to create much deeper networks without degradation in performance. 
- It allows the use of a linear activation function and still can approximate functions in a good way.
- In a residual network, as the number of layers increase, the training error keeps decreasing.
- Information from one layer can get directly sent forward to deeper layers, skipping the layers in-between.  
* `[ B ]`


---

**Q: Which of the following contributes to the increased performance of residual networks?
i. These networks are deeper than other approaches, and can be viewed as a collection of many paths
ii. The paths in residual networks do not strongly depend on each other
iii. The paths in a residual network exhibit ensemble-like behaviour, with the performance correlating with the number of valid paths
iv. The paths that contribute gradient during training are rather short**
- Only i, ii, iii
- Only i, iii, iv
- None of i, ii, iii and iv
- All of i, ii, iii and iv
* `[ D ]`


---

**Q: "The error in a deep Residual network is almost invariant to removing a single layers." This is...**
- False; because removing any layer out of a deep network destroys the predefined structure on which it has been trained and all trained weights become essentially random again.
- True; because the the entire network has $2^n$  (semi-independent) paths and by removing 1 layers still half off all, and most of all effective, layers remain.
- Impossible; as no layer can be removed due to the mismatch in dimensionality with the next layer.
- True; because it's the same as bypassing it, which is already 50% of the output of that layer, due to the inherent structure of the residual network. 
* `[ B ]`


---

**Q: Residual networks can be seen as:**
- A collection of many paths
- A single ultra deep network
- A single shallow, ultra wide network
- A square network
* `[ A ]`


---

**Q: What is the reason behind residual networks’ increased performance?**
- They tend to behave efficiently, by preserving paths contributing the most to the gradient (Shallow Paths) and neglect paths contributing least to the gradient (Deeper Paths).
- They tend to behave efficiently, by preserving paths contributing the most to the gradient (Deeper Paths) and neglect paths contributing least to the gradient (Shallow Paths).
- All of the above.
- None of the above.
* `[ A ]`


---

**Q: Which of these are valid explanations for the increased performance of residual networks?**
- a residual network can be viewed as a collection of many paths instead of a single deep network, thereby offering more flexibility.
- although all paths in a residual network are trained together, they do not strongly depend on eachother, so they each contain more unique information
- residual networks can contain paths that are much longer than the depth of the original deep net, these paths can store more gradient and therefore perform better
- the paths that contribute the most gradient are short, so there is no need to preserve gradient flow across the entire depth of the network, thereby improving performance
* `[ C ]`


---

**Q: Which statement are supported by the Paper about Residual Networks
1. A resiudal Network can be seen a series of paths
2. Removing layers in a Residual Network has the same effect as in a normal network**
- Both statements are supported
- Only statement 1 is supported
- Only statement 2 is supported
- Both statements are NOT supported
* `[ B ]`


---

**Q: What is true about residual paths.**
- The effective paths are shallow compared to the total network
- Most residual paths are long paths
- The longer the residual path the lower the gradient
-  All of the above
* `[ D ]`


---

**Q: During the training of residual networks, which paths contribute the most to the gradient?**
- The shorter paths have barely any effect on the gradient while longer paths have much effect on the gradient
- The longer paths have barely any effect on the gradient while shorter paths have much effect on the gradient
- The shorter paths have just as much effect on the gradient as the longer paths
- The longer paths do not have any effect on the gradient in comparison with the shorter paths.
* `[ D ]`


---

**Q: The residual networks:**
- Tend to keep the magnitude of the gradient through all the possible paths
- Present a high interdependency between their paths
- Have paths of varying lengths
- Drastically lose on performance when paths are removed
* `[ C ]`


---

**Q: What is not true about residual networks? **
- Residual network paths through residual networks vary in length
- Residual network paths show ensemble like behavior
- Residual network paths are longer than expected
- Residual networks avoid vanishing gradient problem
* `[ C ]`


---

**Q: In a residual network, most of the paths used in training are**
- Short paths
- Long Paths
- Paths of length around half of the total number of layers
- Cannot be generalised
* `[ A ]`


---

**Q: Which of the following statements does NOT hold true?**
- Very deep neural networks exhibit decreased accuracy compared to its shallow counterparts if they don't come with a layer-bypassing strategy 
- As the layers of a neural network go deep, gradients have smaller magnitudes, a fact which can cause overfitting
- Individual paths of a residual neural network most probably bypass 50% of the modules of the network 
- Identification of effective paths of a residual network can permit a massive compression of its overall architecture without compromising its generalization properties 
* `[ B ]`


---

**Q: Which of the following are correct about residual networks ?

1. Residual networks do not resolve the vanishing gradient problem by preserving gradient flow throughout the entire depth of the network
2. The paths through the network that contribute gradient during training are shorter than expected.**
- Only 1 is correct
- Only 2 is correct
- Both are correct
- Both are incorrect
* `[ C ]`


---

**Q: why residual network increases performance**
- it is a collection of many paths 
- it goes deeper than the other 
- shows ensemble like behaviour
- all
* `[ D ]`


---

**Q: Complete following statement: "When one would rewrite a residual network as an explicit collection of paths, paths through residual networks ... ".**
- are the same in length. 
- vary in length.
- are the same in width.
- vary in width.
* `[ B ]`


---

**Q: What is the correct description for residual network?**
- Paths are rather independent to each other; longer paths do not contribute any gradient
- Paths are rather independent to each other; shorter paths do not contribute any gradient
- Paths are rather dependent to each other; longer paths do not contribute any gradient
- Paths are rather dependent to each other; shorter paths do not contribute any gradien
* `[ A ]`


---

**Q: Which of the statements below are true regarding residual networks?**
- Layers in residual networks can comprise of multiple convolutional layers.
- Layers in residual networks can contain a single convolutional layer.
- Both A and B.
- None of the above.
* `[ A ]`


---

**Q: What is a residual network?**
- A one-layer CNN
- A network containing paths to feed data into layers a few hops away
- A fully-connected network
- A network using residue to compute the stochastic gradient descent
* `[ B ]`


---

**Q: Which of the following is/are true regarding the residual networks?
I. There are 2^n different paths connecting the input to the output.
II. Residual networks do not show the properties of ensemblies.
III. Subtracting one or more modules from the residual network do not generally effect the performance rate drastically.
IV. It is not possible for a residual network with swapped modules to run efficiently**
- I-II
- I-II-III
- I-III
- II-IV
* `[ C ]`


---

**Q: Which of these statements is incorrect?**
- Neurons in early visual areas have small receptive fields and are sensitive to basic visual features, e.g., edges and bars
- Removing single layers from residual networks does not noticeably affect their performance
- Dropping out individual neurons during training leads to a network that is equivalent to averaging over an ensemble of exponentially many networks
- Paths in a shallow network strongly depend on each other
* `[ D ]`


---

**Q: Do residual networks really solve the vanishing gradient problem?**
- Yes.
- No, not really. The gradient flows through the shorter paths and still vanishes in the longer ones.
- Yes, but only for extremely deep (1000+ layers) networks.
- Yes, but only once you delete the modules representing the shorter paths during training time.
* `[ B ]`


---

**Q: Which of the following statements are correct:
1)Paths in residual network strongly depend upon each other.
2)Residual networks can be viewed as a collection of many paths.
3)Residual Network resolve the vanishing gradient problem.**
- All of the statements are correct
- Only 2) and 3) is correct
- Only 2) is correct
- Only 3) is correct
* `[ C ]`


---

**Q: Which experiment cannot support that the effective paths are shallow in Residual Networks?**
- Deleting individual layer
- Deleting many modules
- Reordering modules
- None of the above
* `[ D ]`


---

**Q: Which of the following statements is correct about a residual network?**
- A highway network has the same behaviour if the learned gates in the highway network are t(.)=0.4
- The distribution of possible path lengths is exponential
- Removing residual layers will mostly effect short paths
- The short paths will contribute the most to the gradient during backpropagation.
* `[ D ]`


---

**Q: Which of the following statements is true?
I Residual networks can be seen as a collection of many paths
II The paths in residual networks are trained jointly, however they are not strongly dependent on each other.**
- Only I is true
- Only II is true
- Both I and II are true
- None of the statements are true
* `[ C ]`


---

**Q: Which of the following about Residual Network is WRONG?**
- Removing single layers from residual network does not noticeably affect their performance.
- Shorter paths are of more importance as the longer paths don't contribute any gradient.
- The performance of residual networks increases with adding more layers.
- Residual networks preserve gradient flow throughout the entire depth of the network and thus resolve the vanishing gradient problem.
* `[ D ]`


---

**Q: Removing residual modules mostly removes long paths?**
- yes, so there will be better logistical connections 
- yes, the network will have a better dividing of robustness weight distrubution
-  no, shorter paths are better so change for overfitting
- no, it will more likely have a change to overfit 
* `[ B ]`


---

**Q: In what sense does the behaviour of residual networks resemble that of model ensembles?**
- Removing nodes gradually decreases the performance
- Training is much slower in both cases
- Both are methods are solutions to the vanishing gradient problem
- Both are combinations of multiple models, but combined in different ways
* `[ A ]`


---

**Q: Which of the following is true about characteristics of short paths in residual networks?**
- Distributed connected path lengths through the residual networks enable short paths
- Vanishing gradient magnitude during backpropagation
- Relatively shallow path lenghts being less effective path lengths to the residual network
- None of the above
* `[ B ]`


---

**Q: Which on is not conclusion of this paper**
- The unraveled view is equivalent and showcases the many paths in Resnet.
- Residual Networks consist of many paths , although trained jointly, they do not strongly depend on each other.
- Most paths through a resnetare relatively short.
- During training, gradients only flow through short paths
* `[ A ]`


---

**Q: Which of the following statements are true?

A A key characteristic of ensembles is their smooth performance with respect to a number of members.
B In dropping entire layers can be removed from plain residual networks without impacting performance, indicating they are not dependent on each other. **
- A
- B
- A and B
- None of the above
* `[ C ]`


---

**Q: What are beneficial properties of residual networks?**
- They improve execution time as they train on the residual of the network, which is all nodes that aren't connected.
- There are multiple paths that can be taken through the network which solves problems such as the vanishing gradient problem.
- Residual networks can speed up execution by skipping non vital layers.
- Residual networks are more popular for the industry as the computational load can be changed during runtime by selecting which path to take.  
* `[ B ]`


---

**Q: According to paper 8, what impact of BatchNorm on the training process is most key to its success?**
- prevention of exploding or vanishing gradients
- robustness to different settings of hyperparameters such as learning rate and initialization scheme
- keeping most of the activations away from saturation regions of non-linearities
- It reparameterizes the underlying optimization problem to make its landscape significantly more smooth
* `[ D ]`


---

**Q: Which statements are true regarding network ensembles and residual networks:
1.	The performance of ensembles depends smoothly on the number of members
2.	Residual models and ensembles behave in the same fashion regarding removal  of their **
- Only 1 is true
- Only 2 is true
- Both are true
- None are true
* `[ C ]`


---

**Q: What is not true about residual networks ?**
- Residual networks can be viewed as a collection of many paths, instead of a single ultra deep network.
- There are many paths through residual networks.
- The effective paths in residual networks are relatively shallow.
- All paths through residual networks are of the same length.
* `[ D ]`


---

**Q: Which distribution does the possible path lengths in a residual network follows?**
- Dirichlet distribution
- Poisson distribution
- student-T bivariate distribution
- Binomal distribution
* `[ D ]`


---

**Q: What is the difference between a residual network and a regular deep network?**
- The batch normalization procedure
- The possible path between the layers
- The layers itself are different
- The activation function must be different for a residual network
* `[ B ]`


---

**Q: 1) Random weights are a good substitue for pretraining and finetuning
2) The structure of CNN is more important than the initialisation of the weights.**
- Only 1 is correct
- Only 2 is correct
- 1 and 2 are correct
- none are correct
* `[ B ]`


---

**Q: Read the following statements.\newline
A. The paths through the network that contribute gradient during training are shorter than expected.\newline
B. During training, deep paths are not required as they do not contribute any gradient.\newline
C. Residual networks can be viewed as a single ultra deep network.\newline
Select the correct options.\newline**
- B, C are correct.
- A, B are correct.
- A, C are correct.
- All are correct.
* `[ B ]`


---

**Q: What are the benefits of the residual networks?**
- They can be seen as a collection of many paths, instead of a single ultra deep network
- These paths do not strongly depend
on each other
- The paths exhibit ensemble-like behavior
- All mentioned
* `[ D ]`


---

**Q: In the paper “Residual Networks Behave Like Ensembles of Relatively Shallow Networks” one of the authors’ conclusion is focused on the properties of effective paths in residual networks. According to the paper, which types of paths contribute most of the gradient during training?**
- Longer paths
- Relatively shallow paths
- All the paths contribute equally to the total gradient magnitude
- The paths do not contribute in any way when computing the total gradient magnitude
* `[ B ]`


---

**Q: which statement is wrong? **
- Residual networks can be viewed as a collection of many paths
-  Ensemble-like behavior in the sense that their performance smoothly correlates with the number of valid paths
-  The paths through the network that contribute gradient during training are shorter than expected
- Non of them.
* `[ D ]`


---

**Q: Which of the following is/are suggestive approaches to residual networks to behave like ensembles in shallow networks?**
- Removing residual modules mostly removes long paths
- Enablement of connection to highway networks
- Effect of stochastic depth training procedure
- All of the above
* `[ D ]`


---

**Q: In residual networks, paths can have multiple sizes. The contribution towards the gradient is dependent on these sizes. In general what size is usually contributing in larger portions towards this gradient?**
- Short (10-34 layers)
- Long (100-110 layers)
- Medium (50-60 layers)
- all of the above
* `[ A ]`


---

**Q: The authors find that removing residual modules from residual networks...**
- ... increases the error rate smoothly with the number of modules removed.
- ... increases the error rate exponentially with the number of modules removed.
- ... does not affect the error rate below a rather high threshold.
- ... makes the network as good as a random chance classifier.
* `[ A ]`


---

**Q: How do residual networks avoid the vanishing gradient problem?**
- By keeping the weights bounded.
- Because of the increased dependence of paths in the network.
- Skips introduced by residual networks allows the gradient to loop.
- Short paths are created throughout the network which allows the gradient to stay signifcant.
* `[ D ]`


---

**Q: Which of the following statements is TRUE?**
- Paths in a residual network strongly depend on each other
- Paths in a residual network do not necessarily depend on each other
- The longest paths in a residual network contribute most to the gradient
- The effective paths in a residual network are the shortest paths
* `[ B ]`


---

**Q: After the authors of paper 8, residual networks can help train very deep networks...**
- ... by avoiding vanishing gradients.
- ... by avoiding too high computational cost
- ... by avoiding varying path lengths.
- ... by avoiding your mother.
* `[ A ]`


---

**Q: 0**
- 0
- 0
- 0
- 0
* `[ A ]`


---

**Q: To show not all transformations within a residual network are necessary, one can delete individual modules. During whih phase should one do this?**
- Before training remove the residual model from a single building block by changing y_i to y_{i-1}+f_i(y_i-1) to y_i'=y_{i-1} 
- after training remove the residual model from a single building block by changing y_i to y_{i-1}+f_i(y_i-1) to y_i'=y_{i-1}
- after training remove the residual model from 2 building blocks by changing y_i to y_{i-1}+f_i(y_i-1) to y_i'=y_{i+2}
- Before training remove the residual model from 2 building blocks by changing y_i to y_{i-1}+f_i(y_i-1) to y_i'=y_{i+2} 
* `[ A ]`


---

**Q: A conventional residual network have 8 implicit paths connecting input and output, how many blocks does this network have. If a block is added, how many path would the new network have?**
- 3, 16
- 8, 64
- 3, 16
- 8, 64
* `[ A ]`


---

**Q: Which of the following is true?**
- The distribution of all possible paths through a residual network follows a Bayesian distribution.
- It is essential that all paths in a residual network carry the same amount of gradient
- A residual network should be have at least 3 blocks to justify its architecture
- The performance increase from additional ensemble members does not scale linearly with ensemble size.
* `[ D ]`


---

**Q: How does the paper show that residual networks behave similar to relatively shallow network ensembles?**
- network ensembles?
Introduces unravelled view which illustrates residual networks as a collection of many paths
Performs lesion study to show these paths are not strongly dependent on each other
Studies gradient flow through residual networks to show only short paths contribute gradient during training, not deep paths.
All the above
- network ensembles?
Introduces unravelled view which illustrates residual networks as a collection of many paths
Performs lesion study to show these paths are not strongly dependent on each other
Studies gradient flow through residual networks to show only short paths contribute gradient during training, not deep paths.
All the above
- network ensembles?
Introduces unravelled view which illustrates residual networks as a collection of many paths
Performs lesion study to show these paths are not strongly dependent on each other
Studies gradient flow through residual networks to show only short paths contribute gradient during training, not deep paths.
All the above
- network ensembles?
Introduces unravelled view which illustrates residual networks as a collection of many paths
Performs lesion study to show these paths are not strongly dependent on each other
Studies gradient flow through residual networks to show only short paths contribute gradient during training, not deep paths.
All the above
* `[ D ]`


---

**Q: What is a residual block?**
- A single fully connected layer.
- A layer comprising multiple fully connected layers.
- A single convolutional layer.
-  A layer comprising multiple convolutional layers.
* `[ D ]`


---

**Q: What is the difference of networks compared with normal network?**
- Some weights are changed randomly
- Some neurons are disabled randomly
- Some layers are disabled randomly
- None of above
* `[ A ]`


---

