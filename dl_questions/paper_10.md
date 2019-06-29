# Questions from `paper_10` :robot: 

**Q: When is Group Normalization a better choice than Batch Normalization?**
- When the system's parameters have been designed for Batch Normalization.
- When a very large batch size is available (>50  per worker).
- When a small batch size is available (<4 per worker).
- When the mean of the input batch is close to 1.
* `[ C ]`


---

**Q: The main advantage of group normalization (GN) over batch normalization (BN) is:**
- GN has a better performance in terms of error rate.
- GN has a better performance in terms of generalization
- GN can reliably work on smaller batch sizes, thus reducing memory constraints
- BN requires you to define the size of batches beforehand, thus placing restrictions on the model design.
* `[ C ]`


---

**Q: GroupNorm starts to perform better than BatchNorm when:**
- The batch size is larger.
- The batch size is smaller.
- The group axis is larger.
- The group axis is smaller.
* `[ B ]`


---

**Q: Group Norm will become Layer Norm if we:**
- set the group number as G = 1
- set the group number as G = 2
- set the group number as G = 3
- set the group number as channel number: G = C
* `[ A ]`


---

**Q: Which statement is true? Statement 1: Group normalization divides the channels into groups and computes within each group the mean and variance for normalization. Statement 2: Batch normalization’s error becomes considerably higher with small batch sizes. Group normalization‘s behavior is more stable and insensitive to the batch size.**
- Statement 1 is correct & Statement 2 is correct
- Statement 1 is incorrect & Statement 2 is correct
- Statement 1 is correct & Statement 2 is incorrect
- Statement 1 is incorrect & Statement 2 is incorrect
* `[ A ]`


---

**Q: Which of the following statements about group normalization is incorrect?**
- Group normalization can be a good alternative to batch normalization due to many classical-features being group-wise features
- Group normalization becomes the same as linear normalization if the group number is limited to 1
- Group normalization becomes the same as instance normalization if the group number is set to G = 2C, which means there are two channels per group 
- Group normalization does not exploit the batch dimensions, making its computations independent of batch sizes
* `[ C ]`


---

**Q: Which of the following about normalization methods is FALSE?**
- In batch normalization, the pixels in a layer that share the same channel index are normalized together.
- Group normalization becomes equal to instance normalization when the group number is equal to 1. 
- It is required for batch normalization to work with a sufficiently large batch size.
- Batch normalization’s mean and variance computation introduces uncertainty, which helps regularization.
* `[ B ]`


---

**Q: How is Group Normalization (GN) related to Layer Normalization (LN) and Instance Normalization (IN)?**
- GN is equivalent to LN when the number of groups is equal to one.
- GN calculates the mean and variance along a batch of samples.
- GN is more restricted compared to LN.
- GN is equivalent to IN when using more than one channels per group.
* `[ A ]`


---

**Q: What is the best technique for Normalization with large batch size?**
- Group Normalization
- Batch Normalization 
- Both Batch Normalization and Group Normalization are good with large batch size
- Instance Normalization
* `[ C ]`


---

**Q: Why is group normalization performing better than batch normalization for small batch sizes?**
- This statement is false: Batch normalization always performs better than group normalization 
- For small batch sizes, batch mean and variance can be overly stochastic and inaccurate, which makes batch normalization perform worse. Group Normalization also depends on the batch size, but has better normalization methods
- For small batch sizes, batch mean and variance can be overly stochastic and inaccurate, which makes batch normalization perform worse. Group Normalization on the other hand is independent of the batch size
- Group normalization groups together several batches if the batch size is too small
* `[ C ]`


---

**Q: What problems does group normalization aim to fix?**
- with big batch sizes, batch normalization is not accurate
- with small batch sizes, batch normalization is not accurate
- batch normalization is not effective most of the time
- none of the above
* `[ B ]`


---

**Q: Group normalization (GN) is an alternative to Batch Normalization (BN) when the problem at hands requires smaller batch sizes (which is when BN tends to perform poorly). The main difference between the two is that**
- GN normalizes features over a group of channels but for each data point, while BN normalizes features over the whole batch, for a specific channel.
- GN normalizes features over a subset of the batch, while BN normalizes features over the whole batch.
- GN normalizes features over the whole batch, but only for a group of coordinate indices of the input, while BN normalizes features over the whole batch and the whole input.
- GN normalizes features over a group of layers, while BN is used after every layer.
* `[ A ]`


---

**Q: Group normalization**
- Computes the mean and standard deviation along the N,H,W axes.
- Computes the mean and standard deviation along the C,H,W axes.
- Computes the mean and standard deviation along the H,W axes.
- Computes the mean and standard deviation along the H,W axes and along a group of channels
* `[ D ]`


---

**Q: Which statement is wrong about BN, GN, IN, LN?**
- GN, IN, LN are independent in the batch size.
- Classifications with GN always perform better than Classifications with BN.
- IN, LN are just extreme case of GN.
- BN has limitation on batch size. 
* `[ B ]`


---

**Q: Which normalization method is more likely to perform the best in a small batch sizes?**
- Batch normalization
- Group normalization
- Layer normalization
- Instance normalization
* `[ B ]`


---

**Q: What would be a reason to use Group normalization rather than the more commonly used Batch normalization**
- Because Group normalization converges faster
- Because the images are grouped in the original dataset
- Because Group normalization generally has a better accuracy
- Because Batch normalization might take up to much memory
* `[ D ]`


---

**Q: What is the difference between Group normalization and batch normalization?**
- Group Normalization only works with large size batches otherwise it will have a decrease in accuracy. Meanwhile Batch Normalization works for single size batches just as good as large size batches. 
- For Group normalization you can have small batches without having a decrease in its accuracy. Meanwhile with Batch Normalization this is not possible.
-  Group Normalization is a technique that combines multiple layers of Batch Normalization
- Group Normalization is the same as batch normalization but then only uses batches of size 1 before grouping them together. Meanwhile batch normalization takes entire batches.
* `[ B ]`


---

**Q: Which of the following regarding networks trained with the respective normalisation is true?**
- General Norm might be implying some regularisation for Batch Norm.
- Networks trained with Batch norm decrease in error when the batch size is reduced
- General norms behaviour is less stable than batch norm wiht very small samples sizes
- Batch norm removes the batch size constraint in general norm.
* `[ A ]`


---

**Q: What is the relation between GN and its counterparts?**
- When the number of groups equals 1, it becomes IN.
- When the channels per group equals 1, it becomes LN.
- When the channels per group equals 1, it becomes BN.
- None of above
* `[ D ]`


---

**Q: What best describes group normalisation?**
- The idea of Group Normalization is processing feature by group-wise normalization over channels
- It normalises different groups of the classes in the dataset
- It is the same as batch normalisation but only with smaller groups
- None of the above
* `[ A ]`


---

**Q: Group normalization (GN) is a means to train deep neural networks that is thought to substitute batch normalization (BN), having very similar results both on the training and test set. In which cases anyway GN works significantly better than BN?**
- When the batch size used in the net is very big
- When the batch size used in the net is very small
- If we are dealing with deep convolutional neural networks
- If we are dealing with VGG neural networks
* `[ B ]`


---

**Q: Statement 1:  Group Normalization becomes LayerNorm if we set the group number as G = 2. 
Statement 2: Group Normalization behaves very stable over a wide range of batch sizes. **
- Both statements are correct
- Statement 1 is correct
- Statement 2 is correct 
- None of the statements are correct
* `[ C ]`


---

**Q: Which of the following is FALSE about Group Normalization ?**
- Group Normalization computes mean & variance for normalization from the set of pixels of same batch index.
- Group Normalization computes mean & variance for normalization from the set of pixels sharing the same channel index.
- Group Normalization does not overcome the shortcoming of BatchN's error increase in cases of small batch size. 
- All of the above.
* `[ D ]`


---

**Q: What is NOT true about Group Normalization (GN)?**
- GN divides channels into groups and normalizes the features within each group.
- GN does not normalize along the batch dimension, and its computation is independent of batch sizes.
- GN can naturally transfer from pre-training to fine-tuning.
- GN’s accuracy is stable in a small range of batch sizes.
* `[ D ]`


---

**Q: Consider the situation where you are training a deep network and have a memory constraints that limits your batch size. What strategy of data normalization would you choose in order to maintain a relatively good classification performance?**
- Group normalization
- Batch normalization
- Data normalization
- Do not normalize at all
* `[ A ]`


---

**Q: What is one of the main problems when normalizing over the batch dimension?**
- Batch normalization slows down exponentially when batch dimension grows
- Batch normalization's error increases rapidly when batch size decreases
- Batch normalization only works with a minimum batch size
- Batch normalization cannot calculate variance over very small batches
* `[ B ]`


---

**Q: Recall that the following variables refer to the given dimensions of the input space:
H,W - spatial axes
C - channel axis
N - batch axis

Match the given input normalization techniques to their corresponding axes of normalization:
1: normalize on  H, W
2: normalize on H, W, N
3: normalize on H, W, C
4: normalize on H, W, and portions of C
I: Layer Norm
II: Group Norm
III: Instance Norm
IV: Batch Norm**
- 1 -> I
2 -> II
3 -> III
4 -> IV
- 1 -> III
2 -> IV
3 -> I
4 -> II
- 1 -> III
2 -> II
3 -> I
4 -> IV
- 1 -> IV
2 -> III
3 -> II
4 -> I
* `[ B ]`


---

**Q: What main problem of Batch Normalization (BN) does Group Normalization (GN)  tackle?**
- Batch Normalization does not work when the internal covariate shift in the network is large, Group normalization minimizes this internal covariate shift
- Batch Normalization does not work on all data types, Group Normalization is able to generalize better and provide improved performance on video and audio data
- The performance of Batch Normalization is dependent on the batch size, Group Normalization gives better performance when batch sizes are relatively small
- Batch Normalization does not work on very deep networks, due to the vanishing gradient problem. Group Normalization helps prevent this vanishing gradient in deeper networks
* `[ C ]`


---

**Q: What limitation of Batch Normalization is solved by Group Normalization?**
- dependence on batch size
- the need for a large training set
- restricted to relatively simple models
- all of the above
* `[ A ]`


---

**Q: What is TRUE about Group Normalization?**
- it exploits batch dimensions
-  its computation depends by the batch size
- it seems to be more stable (in term of accuracy) in changes of mini-batch size than BatchNormalization
- it's even called Batch Renormalization
* `[ C ]`


---

**Q: What is a possible advantage of Group normalization when compared to batch normalization**
- Group normalization improves performance of training with large batch sizes
- Group normalization improves performance of training with small batch sizes
- None, they are actually equivalent, according to the article.
- None of the above.
* `[ B ]`


---

**Q: Which statement is true about group and batch normalisation (GN/BN)?**
- GN's computation is independent of batch sizes and its accuracy is stable in a wide range of batches
- GN's computation is dependent of batch size and its accuracy is stable in a small range of batches
- In general, normalising the input data elongates the training process
- GN's computation favors a large batch size
* `[ A ]`


---

**Q: In what way group normalization improves upon batch normalization?**
- Group is better in large batch sizes 
- Group is better in small batch sizes 
- Batch size is irrelevant to performance
- The error of batch does not increase as we decrease the size of the batches 
* `[ B ]`


---

**Q: What is the batch norm **
- All channels in groups
- One image per batch
- One channel per group
- One channel
* `[ B ]`


---

**Q: Group normalization (GN) solves the fact that batch normalization (BN) has an increased error rate for smaller batch sizes. What is the main reason that BN has this problem?**
- The statistics estimation of the batch is inaccurate for small batch sizes
- The additional layers of BN can not handle smaller batch sizes
- The covariances cannot be calculated properly for smaller batch sizes
- We do not know the exact reason
* `[ A ]`


---

**Q: When does group normalization outperform Batch Normalization?**
- When facing a computer vision problem
- When facing a sound recognition problem
- When there is a lot of data present
- When there is little data present.
* `[ D ]`


---

**Q: What happens to the error of the Batch Norm and Group Norm respectively  when the batch size gets smaller?**
- It stays the same / It stays the same
- It goes down / it goes up
- It goes down / it stays the same
- It goes up / it goes up
* `[ C ]`


---

**Q: Which of the following statements is true?

1. Group Normalization does not exploit the batch dimensions, and its computation is independent of batch sizes.
2. Group Normalization can outperform its Batch Normalization based counterpart, especially in visual recognition type of tasks.**
- Statements 1 is correct, statements 2 is incorrect
- Statement 1 is incorrect, statement 2 is correct
- Both statements are correct 
- Both statements are incorrect
* `[ C ]`


---

**Q: What statement is FALSE?**
- Batch Normalization suffers from inaccurate batch statistics estimation when the batch size is small
- Group Normalization is a simple alternative to Batch Normalization
- Group Normalization divides the channels into groups and computes within each group the mean and variance for normalization
- Group Normalization performs worse than Batch Normalization on very small batch sizes
* `[ D ]`


---

**Q: In "Group Normalization" Wu and He introduce group normalization, a method to apply normalization to small batches. Which method is most similar to this method in its approach to normalization e.g. (normalization over channels and batch, or only over channels, etc)**
- Batch normalization
- Layer normalization
- Instance normalization
- L2 norm
* `[ B ]`


---

**Q: Which of the following statements about Normalization is wrong?**
- Batch Normalization’s error increases rapidly when the batch size becomes smaller
- Group Normalization is batch-independent
- Batch Normalization is not appropriate for training larger models and transferring features to computer vision tasks due to constrained memory consumption
- Group Norm becomes a Batch Norm when all channels are put in one group.
* `[ D ]`


---

**Q: In 2018, Wu and He have set forth the Group Normalization (GN) technique. Which statement holds about this technique?**
- The GN method normalizes along both the batch and the group dimension, for most optimal performance.
- The GN method can be described as a layer that divides channels into groups and normalizes the feature values within each group.
- Even though GN can generalize well, it is outperformed by BatchNorm (BN) in scenarios with very small sample sizes.
- For larger batch sizes, BN is outperformed by GN.
* `[ B ]`


---

**Q: What do both group normalisation and batch normalisation have in common?**
- large batches (memory intensive to normalise).
- Low model accuracy for small (mini)-batch size.
- ease optimisation and enable very deep networks to converge. 
- Exploits layer dimensions and computation is independent of batch size.
* `[ C ]`


---

**Q: What is the main advantage of group normalisation compared to batch normalisation?**
- GN is computionally less expensive
- independent of batch size
- GN can be naturally transferred from pre-training to fine-tuning
- none of the above
* `[ B ]`


---

**Q: Which statement about Group Normalization is False?**
- Group Normalization was inspired by group-wise representations by design, such as SIFT, HOG and GIST.
- Group Normalization is equal to Layer Normalization and Instance Normalization in extreme cases.
- Group Normalization normalizes the features (or pixels) along (parts of) the Batch, Height and Width axes.
- None of the above
* `[ C ]`


---

**Q: Why would one choose Group Normalization over regular Batch Normalization?**
- When the batch size becomes smaller, the batch statistics estimation become inaccurate 
- A model for high quality segmentation needs to be trained
- The computer for training has unlimited memory
- There is not enough data available to train a deep network
* `[ A ]`


---

**Q: To which method is group normalizaiton not related?
In other words, which method can group normalization not mimic with specific parameter tuning?**
- Batch normalization
- Layer normalization
- Instance normalization
- It is related to all above methods
* `[ A ]`


---

**Q: What is the correct description for Batch Normalisation?**
- Making landscape smooth
- Reducing internal covariate shift
- Reducing the executing time
- Ensuring the generalisation among all data
* `[ A ]`


---

**Q: What is the main benefit of Group Normalization (GN) as opposed to Batch Normalization (BN)?**
- GN's computation is independent of batch sizes, and its accuracy is stable in a wide range of batch sizes.
- GN achieves a significantly lower error rate while using a lot less memory than BN. This is done by varying the considered group sizes.
- GN achieves a significantly lower error rate while using a lot less memory than BN. This is done by internally grouping together a more varied set of training data.
- None of the above.
* `[ A ]`


---

**Q: Which of the statements about Group Normalization (GN) is false?**
- Group normalization's computation is dependent of batch sizes and its accurancy is unstable in a wide range of batch sizes.
- Group normalization divides the channels into groups and computes within each group the mean and variance for normalization.
- Group normalization is the same as layer normalization if the group number is equal to 1. 
- Group normalization is the same as instance normalization if the group number is set as G = C (i.e. one channel per group).
* `[ A ]`


---

**Q: What is true?**
- Batch normalization is a special case of layer normalization
- Group normalization is a special case of batch normalization
- Instance normalization is a special case of group normalization
- Layer normalization is a special case of instance normalization
* `[ C ]`


---

**Q: When is Group normalization(GN) considered better than Batch normalization(BN)?**
- When the model is large (batch size becomes small)
- GN is always worse than BN
- When batch size remains same
- GN is always better than BN
* `[ A ]`


---

**Q: Which of the following statements about Normalization Methods is FALSE?**
- Layer norm and instance norm are batch independant.
- Layer norm and instance norm can be seen as the extremes along the channel axis
- Batch norm behaves stably over a wide range of batch sizes
- GN does not exploit the batch dimension
* `[ C ]`


---

**Q: What is the primary conceptual flaw with BatchNorm in real-world implementations?
A - The mean and covariance are computed for a single channel in a particular hidden layers across a batch during training
B - The mean and covariance computed during training are not applied during inference time due to changing batch sizes
C - The mean and covariance computed during training are copied over during inference
D - None**
- A and B
- B and C
- C and A
- D
* `[ B ]`


---

**Q: Which of the following statements is/are true?

A In batch normalization, reducing the batch size increases the model error dramatically. 
B Group normalization is proposed as a layer that divides channels into groups and normalizes the features with each group. Moreover, it does not exploit batch dimension and it’s computation is independent of bath sizes. **
- A
- B
- A and B
- None of the above
* `[ C ]`


---

**Q: The introduced method, Group Normalization, is not:**
- Independent of batch sizes.
- Stable in terms of accuracy.
- Useable for both pre-training and fine-tuning.
- Better than Batch Normalization.
* `[ D ]`


---

**Q: Consider that for an particular mini-batch for training on images we have 3 dimensions, N being the batch size, C being the channels and H,W representing the spatial position of the pixel. Group Normalization works by splitting which dimension into groups for normalization?**
- H,W
- N
- C
- Both N & C
* `[ C ]`


---

**Q: Which of the statement is NOT TRUE about batch normalizations**
- It normalizes the features by the mean and variance computed within a mini batch
- The stochastic uncertainty of the batch statistics also acts as a regularizer
- Accurate estimations can be made even with small batch sizes
- None of the above
* `[ C ]`


---

**Q: What does Group Normalization try to solve**
- The problems with understanding how batch normalization works
- In case the batches become to small counter the increasing error. Allow for faster and more parallel training.
- Allow for faster training
- Allow slower training but a removes all overfitting
* `[ B ]`


---

**Q: which of the following about Batch Normalization (BN) and Group Normalization is not true?**
- BN's error increases rapidly when the batch size becomes smaller, caused by inaccurate batch statistics estimation.
- Normalizing the input data can make training faster.
- The pre-computer statistics may also change when the target data distribution changes.
-  Batch renomalization can solve the BN's issue involving small batches, when the batch size decreases its accuracy will not be effected. 
* `[ D ]`


---

**Q: Which of the following reasons led researchers to propose Group Normalization (GN) instead of Batch Normalization (BN) in high-level computer vision tasks?**
- GN exhibits consistently better classification capabilities compared to BN in object recognition tasks
- Batch normalization impedes the identification of spatial/temporal features in video processing 
- BN is a more greedy approach considering both informational and computational resources
- GN yields lower training errors which are indicative of a more efficient optimization
* `[ C ]`


---

**Q: 1. Batch Norm work well if the batch size is small
2. Group Norm is independent of the batch size to a large extent
Select the correct statement/statements**
- 1
- 2
- 1 and 2
- None
* `[ B ]`


---

**Q: Group normalization is**
-  A technique that helps in training deep neural networks with small batch sizes.
- An alternative to Batch normalization when batch size is very large.
- Normalizing strategy with large batches, which is memory intensive.
-  A technique having higher error rates than Batch normalization.
* `[ A ]`


---

**Q: Group normalization is particularly useful when training on domains that prohibitively require small batch sizes. Which of the following is a reason why Batch normalization is not effective in such cases **
- Batch normalization is a computationally intensive task and there are little gains when it is run on small batch sizes.
- Normalizing the mean and variance for small batch sizes typically proves difficult especially for complex models
- Smaller batch sizes reduce the accuracy of estimation of batch statistics which in turn increases the model error when using batch normalization.
- None of the above
* `[ C ]`


---

**Q: Why Group Normalization can perform better than Batch Nomalization on image classification and image segmentation tasks?**
- The feature that used on classical computer vision stemming from group-wise representations (SIFT, HOG, GIST) and a deep neural network will probably extract features by using a group-wise-like representation.
- Group Normalization can reduce the internal covariance shift and this can be extended too much better image classification performance.
- Group Normalization introduces two extra parameters that con- train the estimated mean and variance of BN within a certain range, reducing their drift when the batch size is small. 
- None of the above.
* `[ A ]`


---

**Q: What is NOT a sort of normalization named in the paper?**
- Batch normalization
- Class normalization
- Layer Normalization
- Instance normalization
* `[ A ]`


---

**Q: Group normalization**
- does not exploit batch size
- becomes instance normalization if we set group number to 1
- s use is limited to shallow networks
- normalizes values within a whole channel
* `[ A ]`


---

**Q: What differs Group Normalization (GN) from Batch Normalization (BN)?**
- GN performs BN-like computation but along the channel dimension
- GN introduces two extra parameters that constrain the estimated mean and variance of BN within a certain range
- GN’s computation is independent of batch sizes
- GN proposes to normalize the ﬁlter weights
* `[ C ]`


---

**Q: What is main benefit of Group normalization versus Batch normalization.**
- Can use bigger batch size
- Can run on GPUs
- Can by use to computer vision task
- It's worse in every way
* `[ A ]`


---

**Q: What is the effect of BN when changing from large batch sizes and small batch sizes and how does GN overcome this? **
- Decrease with big gaps; Dependence of the batch axis
- Decrease with big gaps; Independence of the batch axis
- Decrease with small gaps; Independence of the batch axis
- Increase with big gaps; Independence of the batch axis
* `[ B ]`


---

**Q: Which statement is correct about group normalization?  **
- A small batch size always helps
- A large batch size always helps
- Batch size has no effect on normalization 
- None of the above are correct
* `[ D ]`


---

**Q: What is the proposed application area of group normalization?**
- Shallow networks.
- Reducing overfitting.
- Training using small batches.
- Help regularization.
* `[ C ]`


---

**Q: What is NOT an advantage of Group Normalization over Batch Normalization?**
- Computation is independent of batch sizes.
- It’s accuracy is stable in a wide range of batch sizes.
- It can be naturally transferred from pre-training to fine-tuning.
- It can always be applied to neural networks, unlike Batch Normalization.
* `[ D ]`


---

**Q: Which of the statements below are true about Group Normalisation(GN)?**
- GN's computation is independent of batch sizes.
- GN's computation is dependent of batch sizes.
- GN's accuracy is stable in a wide range of batch sizes.
- Both A and C.
* `[ D ]`


---

**Q: In what scenarios does Group Norm ourperforms Batch Norm?**
- When the batch sizes are very small
- When the input data are images
- When the dataset is too big
- None of the above
* `[ A ]`


---

**Q: The error rate of batch normalisation rapidly increases as batch size decreases. Why is this an issue?**
- A smaller batch-size leads to much more memory usage, which increases the chance of errors being made.
- It is not true that the error rate increases as batch size decreases.
- In tasks like computer vision, small batches are required in order to prevent huge memory consumption.
- None of the above mentioned statements are true.
* `[ C ]`


---

**Q: What is the problem of small group batch size?**
- The result becomes unstable
- The features become more
- The features become better
- Some features may be lost
* `[ A ]`


---

**Q: Which of the following normalization methods is preferred for training larger models and transferring features to computer vision tasks including detection, segmentation, and video, which require small batches?**
- Batch Normalization
- Group Normalization
- Layer Normalization
- Instance Normalization
* `[ B ]`


---

**Q: Why does batch normalization performance decrease when the batch size decreases**
- Because as the batch gets smaller the estimated mean and variance become noisier
- because of the internal covariate shift
- The reason is not understood
- wrong answer
* `[ A ]`


---

**Q: The equation for the set used by Group Normalization to calculate $\mu$ and $sigma$ is $S_i = { k | k_N , \lfloor \frac{k_C}{C/G} \rfloor = \lfloor \frac{i_C}{C/G} \rfloor }$,where G is the number of groups, C is the number of channels, N is the batch size. How can Group Normalization be made equivalent to Layer Normalization?**
- If the number of groups G is set to 1.
- If the number of groups G is set to C.
- If the number of groups is set to N.
- None of the above, as it is not possible.
* `[ A ]`


---

**Q: Which of the following about BatchNorm (BN) and Group Normalization (GN) is NOT correct?**
- BN performs badly with small batch size.
- BN achieves higher accuracy with sufficiently large batch size which is extremely memory consuming for large training dataset.
- GN is independent of batch sizes and thus outperforms BN.
- Unlike how BN does, GN divides channels into groups and performs normalization on them.
* `[ C ]`


---

**Q: What is the problem of Batch Normalization that Group Normalization is aiming to fix?**
- The error of Batch Normalization increases rapidly when the batch size becomes smaller, because of the inaccurate batch statistic estimation.
- The error of Batch Normalization increases dramatically when the batch size becomes larger.
- Batch Normalization is significantly more computationally expensive than Group Normalization.
- None of the above
* `[ A ]`


---

**Q: What statement about Group normalization is FALSE?**
- Group normalization is performing better than batch normalization when batch size is smaller
- It is other name for Local Response Normalization
- It targets the issue of limited memory while using Batch Normalization
- Uses floor function for definition of set, which is later used to define mean and standard deviation
* `[ B ]`


---

**Q: Which of the following statements can be said about Group Normalization (GN) and Batch Normalization (BN) ?**
- If the number of groups in GN is equal to the amount of channels, then GN does the same as BN
- Using GN leads to a regularization effect, leading to a better test error performance and generalization
- GN leads to a more stable convergence of the network than BN if a small batch size is used
- GN and BN will have approximately the same performance when the group size is equal to the batch size
* `[ C ]`


---

**Q: as the size of the batches, how is the error rate affected when using batch normalisation?**
-  The error rate goes up the smaller the batch size.
-  the error rate goes down the smaller the batch size.
-  The error rate goes down until it reaches a minimum then shoots to infinity.
-  The error rate goes up until it reaches a maximum then tends towards zero.
* `[ A ]`


---

**Q: Which of the following techniques works best when optimizing with normalization when we are dealing with small batch sizes?**
- Batch Normalization
- Group Normalization
- Layer Normalization
- Instance Normalization
* `[ B ]`


---

**Q: Why does group normalization according to the author of the paper: "Group Normalization" perform better than batch normalization when the batch size is rather small?**
- Because, when using batch normalization, the batch mean and variance estimation become more inaccurate when they are computed over a small amount of batches, while group normalization does not depend on the batch size.
- Because, when using batch normalization, the batch mean and variance estimation become more inaccurate when they are computed over a small amount of batches, while group normalization performs much better with a small amount of batches.
- Because, when using batch normalization, the batch mean and variance estimations become more unstable and unpredictable with a small amount of batches, while group normalization becomes more stable when working with small amount of batches.
- Because batch normalization heavenly depends on large batch sizes, while group normalization heavenly depends on small batch sizes.
* `[ A ]`


---

**Q: What is not an advantage of Group Normalization over Batch Normalization?**
- It performs better on smaller batch sizes
- Its error is more or less independent on the amount of images per worker
- It can naturally transfer from pre-training to fine-tuning
- It is faster than batch normalization
* `[ D ]`


---

**Q: What is Group Normalisation’s main advantage over Batch Normalisation?**
- GN’s computation is independent of the batch sizes, making its accuracy stable in a wide range of batch sizes.
- GN is computationally cheaper than BN.
- GN has been developed before BN and is therefore more widespread than BN.
- There are no distinct advantages of GN over BN.
* `[ A ]`


---

**Q: What is NOT a motivation for using Group Normalization?**
- Group Normalization can be applied with small batch-sizes
- Using group-wise features such as HOG in your model
- Group Normalization can easily transfer from pretraining to finetuning
- All are valid motivations
* `[ D ]`


---

**Q: Which of the following normalization methods does not perform computations along the batch axis?**
- BatchNorm
- LayerNorm
- InstanceNorm
- GroupNorm
* `[ A ]`


---

**Q: What is the smallest batch size in order for Group Normalization to still work?**
- 32
- 16
- 8
- it does not matter how small the batch size is
* `[ D ]`


---

**Q: Computing the mean and standard deviation over a set of channels is done by**
- Batch normalization
- Group normalization
- Layer normalization
- Instance normalization
* `[ B ]`


---

**Q: Which of the following options, according to the empirical results in this paper, is false?**
- The accuracy of Group Normalization is stable in a wide range of batch sizes.
- Group normalization divides the channels into groups and computes within each group the mean and variance for normalization.
- In Batch Normalization reducing the batch size decreases the model error.
- Group normalization doesn’t exploit the batch dimension, so its computation is independent of batch sizes.
* `[ C ]`


---

**Q: Which of these statements concerning batch normalization is incorrect?**
- Batch normalization benefits from the stochasticity under some situations and its error therefore decreases when the batch size becomes smaller
- Normalizing the input data makes training faster
- Group normalization does not need to exploit batch dimensions in order for it to be an effective normalization layer
- Batch normalization may perform better than group normalization in some systems due to hyperparameter optimalizations
* `[ A ]`


---

**Q: What computation do feature normalization methods perform?**
- Substracting the mean and dividing by the standard deviation for all features computed by layers
- Substracting the mean and dividing by the standard deviation for all features computed by layers, differing in the set that computes the mean and standard deviation
- Divide all features by the norm of the feature vector
- None of the above
* `[ B ]`


---

**Q: How does Group normalization differ from BatchNorm?**
- It normalizes within a batch of channels
- It normalizes within a single channel
- It normalizes across a single layer
- It normalizes across a batch of training examples
* `[ A ]`


---

**Q: What is the main drawback of Batch Normalization?**
- It has a large overhead when dividing data into batches
- The batch size needs to be high, else the probability of the mean or variance being wrong is too high
- The model will underfit as it is only training on a small batch of data
- It is only effective if the batch is at least as big as half the available training data.
* `[ B ]`


---

**Q: Batch normalization is a technique which allowes various networks to train. The choice of the batch dimension introduces problems in deep learning. Which of the following statements is false when it comes to the batch normalization size:**
- A small batch would lead to inaccurate estimation of the batch statistic
- A small batch would lead to the increasing of the model error
- A small batch would lead to the decreasing of the model error
- The batch size should be sufficiently large for obtaining good results
* `[ C ]`


---

**Q: What is not an advantage from group normalization over batch normalization?**
- GN's computation is independent of batch sizes
- GN's accuracy is stable over batch sizes
- GN can be transferred from pre-training to fine tuning
- GN has a lower memory usage 
* `[ D ]`


---

**Q: Which following normalization method is relevant with batch size?**
- Batch normalization
- Group normalization
- Instance normalization
- Layer normalization
* `[ A ]`


---

**Q: What is the main advantage of group normalization over batch normalization?**
- GN works better with small batch sizes, whereas BN starts to fail as the batch size gets smaller.
- GN is more memory efficient.
- GN gets a higher training accuracy faster.
- GN outperforms BN in all cases.
* `[ A ]`


---

**Q: What problem does Group Normalization solve?**
- The high computational cost of Batch Normalization.
- The low computational cost of Batch Normalization.
- The inaccurate values achieved by Batch Normalization for large batch sizes.
- The inaccurate values achieves by Batch Normalization for small batch sizes.
* `[ D ]`


---

**Q: What can be said about batch normalization and group normalization?**
- Group normalization generally perform better
- Batch normalization generally perform better
- Group normalization is more robust than batch normalization to changes in batch size
- Batch normalization is more robust than group normalization to changes in batch size
* `[ C ]`


---

**Q: Which is not correct about the Group Normalization?**
- Divide the channels into groups
- Small batch size
- Related to Layer and Instance Normalization
- None of the above
* `[ D ]`


---

**Q: Consider the group normalization hyperparameter G and the number of channels C. Layer normalization is an:**
- Example of a group normalization with G = C 
- Example of a group normalization with G = C/2
- Example of group normalization with G = 32
- Example of group normalization with G = 1 
* `[ D ]`


---

**Q: should you normalize per batch or per group for better accuracy?**
- Group
- Batch
- Doesn't matter
- Depends on other variables
* `[ A ]`


---

**Q: Which statement about group normalisation is false?**
- Group normalisation’s accuracy is stable in a wide range of batch sizes
- In group normalisation, normalisation is performed on a groups of channels
- Group normalisation outperforms batch normalisation with small batches
- Performance of group normalisation is dependent on the batch size
* `[ D ]`


---

**Q: What is the most widely accepted explanation of BatchNorm’s success?**
- ICS - internal covariance shift
- The use of very small batch sizes (size about 2)
- The induced randomness that contributes to data augmentation
- None of the above
* `[ A ]`


---

**Q: Which one is wrong ?**
- In this project, they investigated GN’s performance on learning representations for reinforcement learning (RL) tasks
- The use of batch has also become a source of drawbacks:  Sometimes you have to use small batch for example due to memory constraints; The used batch is varying which means the concept of batch can be varying in different scenarios.
- They proposed a new normalization method called group normalization whose key property is that is independent of the concept of batch.
- The result shows that some form of normalization matters as it help optimization of CNN, but normalization on batch may not always ideal
* `[ A ]`


---

**Q: What is the required batch size for batch normalization to work properly?**
- 2
- 8
- 16
- 32
* `[ D ]`


---

**Q: Which of the following statements are true? (Based on "Group Normalization")

stat1: Compared to the BatchNormalization Group Normalization will suffer from the reduced size of the mini-batch

stat2: It is required from GN to have sufficiently large batch size e.g. 32 per worker

stat3: Compared to Layer Normalization (LN) GN is more restricted because each group of channels (instead of all of them) are assumed subject to the shared mean and varicance.

stat4: Compared to Instance Normalization (IN) GN can only rely on the spatial dimension for computing the mean and variance and it misses the opportunity of exploiting the channel dependence.

stat5: BN(BatchNorm) technique still can outperform the GN for the task of object detection and segmentation for  COCO and video classification in Kinetics**
- 1 and 2 
- 3 and 5
- 2 and 5
- all false
* `[ D ]`


---

**Q: Group normalization:**
- Is a better version of Batch normalization
- It performs better than BatchNorm for small batches/groups
- For large enough batch sizes it always performs worse than BatchNorm
- It is a different terminology for BatchNorm
* `[ B ]`


---

**Q: Complete following statement: "The error of BatchNorm ... when the batch size becomes smaller, caused by inaccurate batch statistics estimation".**
- decreases slowly
- increases slowly
- decreases rapidly
- increases rapidly
* `[ D ]`


---

**Q: The authors of paper 10 introduce group normalization because ...**
- ... batch normalization has high error with a big batch.
- ... batch normalization has high error with a small batch.
- ... group normalization decreases in error with increasing batch size.
- ... group normalization decreases in error with decreasing batch size.
* `[ B ]`


---

**Q: What is the difference between Batch Normalization (BN) and Group Normalization (GN)?**
- BN normalizes over instances, GN normalizes over channels
- BN normalizes over channels, GN normalizes over instances
- BN normalizes over pixels, GN normalizes over channels
- BN normalizes over channels, GN normalizes over pixels
* `[ A ]`


---

**Q: Group normalisation**
- normalises over a batch over a single channel.
- normalises over a single instance over all channels.
- normalises over a single instance over a single channel.
- normalises over a single instance over a group of channels.
* `[ D ]`


---

**Q: Assume we use a neural network for two pattern detection tasks on high resolution pictures. The second task uses pictures with half the resolution of the first. Which task is more suited for Batch normalisation?**
- The first.
- The second.
- Both are equally suited.
- Batch normalization can't be used for pattern detection tasks.
* `[ B ]`


---

**Q: Which method provides the best results when trying to detect an object?**
- Batch Normalization (BN)
- Group Normalization (GN)
- Layer Normalization (LN)
- Instance Normalization(IN)
* `[ D ]`


---

**Q: Which of the following method is not a normalised method? **
- Batch normlization
- Layer normlization
- Group normlization
- Adaptive normlization
* `[ D ]`


---

**Q: Which of the following normalization methods often forces the user to compromise between the model design and the batch sizes?**
- IN
- BN
- GN
- LN
* `[ B ]`


---

**Q: which statement about Batch normalization and Group normalization is not true?**
- GN will outperform BN when the batch size gets very small
- BN uses the mean and variance to normalize features within a batch
- GN is independent of batch size
- Normalization of input data slows down the training process
* `[ D ]`


---

**Q: This paper introduces Group Normalization (GN) as a simple alternative to Batch Normalization (BN). Which of the following is not a problem of BN that has lead the authors to implement GN?**
- A small batch leads to inaccurate estimation of the batch statistics.
- Reducing BN’s batch size increases the model error dramatically.
- The restriction on batch sizes is very demanding in tasks like computer vision.
- BN do not enable very deep networks to converge.
* `[ D ]`


---

**Q: Which is not an advantage of Group Normalization compared to Batch Normalization?**
- Its computation is independent of batch sizes
- Its accuracy is stable in a wide range of batch sizes
- It can be easily implemented by a few lines of code in modern libraries
- Its computations costs are lower
* `[ D ]`


---

**Q: Which of the following problems occur in batch normalization and is addressed in group normalization?**
- Batch-wise normalization is not legitimate at inference time and the mean and variance are pre-computed.
- The pre-computed statistics may also change when the target data distribution changes.
- There is no normalization performed when testing.
- All of the above.
* `[ D ]`


---

**Q: Which of these drawbacks of Batch Norm has forced the researchers to develop Group Normalization?**
- BatchNorm increases internal covariance shift
- Accuracy is dependent on the batch size
- BatchNorm tends to modify the optimization landscape
- All the variants
* `[ B ]`


---

**Q: What are the advantages of Group Normalisation(GN) over Batch Normalisation(BN)**
- BN is sensitive to the bath size while GN is not.
- GN can minimise computational requirements for training networks.
- Both A and B
- None of the above because BN is better than GN.
* `[ C ]`


---

**Q: Which of the following statement is not correct:**
- Batch Normalization error increase rapidly if the batch size decreases cause by inaccurate batch statics estimation 
- Group Normalization is independent of the batch size
- Group Normalization benefits from stochasticity in small group sizes
- Group Normalization can be a strong alternative to Batch Normalization in segmentation and video classification
* `[ C ]`


---

**Q: Given neural network A and neural network B. A uses batch normalization and B uses group normalization. What can be said?**
- With a small batch size A will take longer to train than B
- With a small batch size B will take longer to train than A
- With a large batch size A will take longer to train than B
- None of the above are true
* `[ A ]`


---

**Q: Which of the following is an advantage of Group normalization over batch normalization?**
- It is independent of batch sizes and accuracy is stable over batch sizes
- It divides channels into groups and normalizes features within each group and therefore doesn't exploit batch dimension
- It is applicable to sequential and generative models
- All of the above
* `[ D ]`


---

**Q: Group normalization divides a certain axis of the feature map tensor into groups. Which axis is it?**
- The channel axis C
- The batch axis N
- The spatial axis H
- The spatial axis W
* `[ A ]`


---

**Q: In the group norm for visual representations the set representting the pixels in which the standard deviation and mean are computed, can be formulated as:

$S_i={k|k_N=i_N,└\frac{k_C}{C/G}┘=└\frac{i_C}{C/G}┘}$ where  └.┘ represents the floor operation. This equation can be interpreted as:**
- The pixels in the same group are normalized together by the same mean and sigma, and Group Norm learns per channel gamma and Beta.
- With i and k in different channels if each group of channels are stored in sequential order along the C axis.
- The hyperparameter G is independent of the number of channels per group and computed as a function of $i_C$ and $k_C$.
- Each channel can have a different number of groups and all the pixels of a single group are normalized individually.
* `[ A ]`


---

**Q: Consider the following statements and chose the relevant option.
a) Batch norm inherently has regularization capabilities
b) Group Normalization performs better than Batch Norm when the batch size is large
c) Group Normalization performs better than Batch Norm when the batch size is small**
- Statement (a) and (b) are correct; Statement (c) is wrong 
- Statement (b) and (c) are correct; Statement (a) is wrong 
- Statement (a) and (c) are correct; Statement (b) is wrong 
- All the statements are correct
* `[ C ]`


---

**Q: Which of these normalization techniques, is independent of batch size and results in the smallest error, for a fixed architecture and hyper-parameter set.**
- Batch Normalization
- Layer Normalization
- Group Normalization
- Instance Normalization
* `[ C ]`


---

**Q: Which of the following is not a drawback of batch normalization?**
- Reducing the batch size has a huge effect on the error
- The batch statistics become inaccurate when batch size is small
- Batch normalization usage requires a compromise between model design and batch sizes
- None of the above
* `[ D ]`


---

**Q: Which of the following statements is true?**
- Batch Normalisation (BN) performs better on bigger batch sizes than Group Normalisation (GN).
- Group Normalisation (GN) performs better on smaller batch sizes than Batch Normalisation (BN).
- Group Normalisation (GN) can not be used on recurrent neural networks (RNN).
- Group Normalisation (GN) is dependent on the batch size.
* `[ B ]`


---

**Q: With respect to video classification which conclusions hold good? 
1.GN is competitive with BN when BN works well for same batch size. 
2.For smaller batch size, BN's accuracy is reduces while GN's remains the same. **
- 1
- 2
- Both
- None
* `[ C ]`


---

**Q: Group Nomalization (GN) relates to Layer Normalization (LN) and Instance Normalization (IN) respectively:**
- GN normalizes over the same groups as LN and IN
- LN does statistics on a subsample of the input and IN links subsamples to a output space, GN assemble these two concepts
- LN and IN uses "frozen" mean and variance over all channels, which is aswell is adapted to GN
- GN becomes a LN when,  G=1 sharing mean and variance over all channels & GN becomes IN when, G=C e.i. sharing mean and variance over a selected channel.
* `[ D ]`


---

**Q: Regarding GN and it's properties, pick the statement that is true.
1.Despite the relation to these methods, GN does not
require group convolutions.
2.GN is a generic layer, as we evaluate in standard ResNets
3.GN also learns the per-channel gamma and beta, and just as LN and IN they compute independent computations al.**
- 1
- 2
- 3
- All of the above.
* `[ D ]`


---

**Q: What is the key difference between batch and group normalization?**
- [Batch] Less intensive and performs poorer with scarce amounts of data.
- [Group] Less intensive and performs better with scarce amounts of data.
- [Group] More intensive and performs poorer with scarce amounts of data.
- [Batch]  Less intensive and performs poorer with scarce amounts of data.
* `[ B ]`


---

**Q: which of the statements is not right**
- BN's error increase rapidly when batch size decreases
- GN can be naturally transferred from pre-training to fine-tuning
- GN is a normalization layer without the batch dimension
- GN is not related to LN and IN
* `[ D ]`


---

**Q: What statement about Group Normalization is true?**
- GN only exploits the layer dimensions, but is influenced by varying batch sizes
- GN exploits both the batch and the layer dimensions, and its computation is independent of batch sizes.
- GN only exploits the layer dimensions, and its computation is independent of batch sizes.
- GN exploits both the batch and the layer dimensions, and its computation is independent of batch sizes
* `[ C ]`


---

**Q: Relations of group normalization: What is true? Group normalization becomes Layer normalization if the group number is set to 1. Group normalization becomes Instance normalization if the group number is set to 1.**
- A is true, B is false
- A is true, B is true
- A is false, B is false
- A is false, B is false
* `[ A ]`


---

**Q: which of the following is true about group normalization (GN) and layer normalization (LN)**
- GN becomes LN if we set the group number as G = 1.
- GN is less restricted than LN
- GN has an improved representational power of over LN
- all
* `[ D ]`


---

**Q: What options below may improve the performance of the deep net work, inspired by research from batch normalization**
- Reducing internal covariate shift
- More smooth landscape of the corresponding optimization problem 
- More predictive and stable behavior of the gradients
- All of the three
* `[ D ]`


---

**Q: What problems of batch normalisation does group normalisation solve?**
- BN's error increases rapidly for small batch sizes
- BN cannot handle large batches
- BN can handle few channels
- GN is much faster
* `[ A ]`


---

**Q: For sample size of 2 per batch which normalization method should be selected?**
- Group Normalization
- Batch Normalization
- Instance Normalization
- Layer Normalization
* `[ A ]`


---

