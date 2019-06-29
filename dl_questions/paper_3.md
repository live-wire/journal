# Questions from `paper_3` :robot: 

**Q: CIFAR-10 is a dataset comprised of tiny pictures depicting objects of 10 different classes. For many years, it has been used as a performance measure for several algorithms. The fact that the same exact dataset has been used to rank dozens of algorithms for the past few years concerned some researchers, because**
- the higher accuracy of the "best" models may be explained by the fact that the iterative process of improving a model has led the models to overfit the test set.
- as the years go by, he images in the CIFAR-10 dataset may have become obsolete.
- the resolution of the images on the dataset is not on par with the resolution with which current cameras take pictures.
- the size of the dataset is not representative of the average size of the datasets available in modern applications.
* `[ A ]`


---

**Q: Which of the following statements about overfitting is true?**
- Adapting design choices using the test set does not cause overfitting.
- Decreasing accuracy when testing on a new test set can always be attributed to overfitting on the original test set.
- The test set can be used to select the best model without causing overfitting.
- The training set overfitting can be quantified by computing the difference between the training and test accuracy.
* `[ D ]`


---

**Q: Image classification models are inherently sensitive to:**
- Shifts in the underlying data distribution
- The choice of hyper parameters, such as learning rate.
-  Type of object that needs to be identified
- Small shifts in the color of images
* `[ A ]`


---

**Q: What is CIFAR-10?**
- An algorithm for image recognition
- A technique for updating weights of a neural network
- An established computer-vision dataset
- A technique to overcome the problem of overfitting at object recognition
* `[ C ]`


---

**Q: Do CIFAR-10 classifiers generalize to “CIFAR-10” (a new CIFAR-10 like test set)?**
- A.	Yes, there is no significant difference between the performance between the original CIFAR test and a new one.
- B.	They have difficulties with it, because there is a large difference in the performance, there is very likely overfitting happening due to a lot of hyperparameter tuning; therefore current classifiers have difficulty generalizing to natural variations in image data.
- C.	They have difficulties with it, cross-validation on the original cifar-10 dataset shows a large decrease in performance; therefore current classifiers have difficulty generalizing to natural variations in image data.
- D.	They have difficulties with it. It is likely that a small distribution shift in a test set caused already a large decrease in performance; therefore current classifiers have difficulty generalizing to natural variations in image data.
* `[ D ]`


---

**Q: Which of the following with respect to classifiers is TRUE?**
- Training set overfitting of a classifier describes the difference between the training accuracy and the accuracy on the underlying data distribution.
- Test set overfitting of a classifier denotes the gap between the test accuracy and the accuracy on the underlying data distribution.
- The statistical reproducibility of a classifier can be evaluated by running published code on the same test data.
- The computational reproducibility of a classifier can be investigated by evaluating the classifier on truly new data.
* `[ B ]`


---

**Q: Which one  of the following statements about the paper "Do CIFAR-10 Classifiers Generalize to CIFAR-10?" is wrong?**
- Current accuracy numbers are brittle and susceptible to even minute natural
variations in the data distribution.
- A broad range of old deep learning models show a large drop in accuracy on the truly new test set.
- The difference in drop degree between old models and relatively recent model is likely not due to underfitting based on adaptivity.
- More recent models with higher original accuracy show a smaller drop and better overall performance.
* `[ C ]`


---

**Q: What can the gap in generalizability of CIFAR-10 models, shown by Recht et al. in their reproducibility study, be attributed to?**
- Statistical error in the generation of the new test set.
- Insufficient hyperparameter tuning on the new test set.
- The images in the new test set being fundamentally harder to classify than in the original test set.
- None of the above.
* `[ D ]`


---

**Q: Why do the authors expect to find overfitting for CIFAR-10 classifiers?**
- most models have problems with overfitting
- test sets have been used to select these classifiers
- the classifiers have too many parameters
- the data set is relatively small
* `[ B ]`


---

**Q: What are the two types of overfitting?**
- Learning set overfitting and test set overfitting
- Learning set overfitting and experimental  set overfitting
- Training set overfitting and experimental set overfitting
- Training set overfitting and test set overfitting
* `[ D ]`


---

**Q: Why is the CIFAR-10 dataset a good choice for a reproducibility study? In other words, if you want to check the generalizability of some models, which characteristics should the dataset that all these models are benchmarked on have?**
- Its creation process is transparent.
- It has been the object of intense research for a long time.
- There are multiple classification models whose implementation is available.
- All of them are correct.
* `[ D ]`


---

**Q: How does the relative error change when testing CIFAR-10 classifiers on the new dataset provide?**
- It increases more for more sophisticated and precise algorithms
- It increases more for easier and less sophisticated algorithm
- It does not seem to have a correlation with the initial validation error
- This feature has not been discussed in the article
* `[ A ]`


---

**Q: Which statement (mentioned in the paper: “: Do CIFAR-10 Classifiers Generalize to CIFAR-10?”) is true?**
- When testing the models on a “new created” dataset, the simpler models have a lower relative error compared with more complex models (with higher original accuracy).
- When testing the models on a “new created” dataset, the simpler models have a higher relative error compared with more complex models (with higher original accuracy).
- The absolute gap is larger for models that perform the best on the original test set compared to models that did worse on the original test set.
- All of the above mentioned statements are true.
* `[ A ]`


---

**Q: What statements about the overfitting are true?

statement1: Adding a minor modification of standard machine learning competitions avoids the sort of overfitting that can be achieved with aggressive adaptivity. 

statement2: A split of data 66%/34% for training to test datasets is a good start. Using cross validation is better, and using multiple runs of cross validation is better again. 

statement3: You can increase the accuracy of your model by decreasing its complexity.

statement4:  There are at least two types of overfitting: training set overfitting and test set overfitting. Since the overall goal in machine learning is to generalize to unseen data, we argue that the second notion of overfitting through test set adaptivity is more important.**
- 2 3 and 4
- 3 and 4
- 4
- All statements are true
* `[ D ]`


---

**Q: Which of the following is not a hyperparameter?**
- Learning Rate
- Dropout Ratio
- Train/Test Ratio
- Weight Decay
* `[ C ]`


---

**Q: What are good examples of hyperparameters of a convolutional network?**
- Initial learning rate, dropout, weight decay.
- Parameters of Loss functions and ReLU functions
- parameters which live in 4 or more dimensions
- Parameters which remain consistent across multiple universes
* `[ A ]`


---

**Q: Which of the following reasons is NOT part of the choice for the CIFAR-10 dataset?**
- CIFAR-10 is one of the most used datasets in the field of machine learning
- The dataset creation proces for CIFAR-10 is transparant and documented
- CIFAR-10 poses a difficult enough problem such that the dataset is still subject of research
- CIFAR-10 is a fairly noisy dataset, so good performing models are less likely to overfit
* `[ D ]`


---

**Q: The main goal of the paper about CIFAR-10 dataset is to:**
- show that, even with a complex dataset, models can score high
- explain techniques to assess the validity of models
- investigate whether or not current ML and DL models overfit on CIFAR-10
- mathematically model the change of accuracy among the two datasets
* `[ C ]`


---

**Q: Which of the following is NOT a troubling trend as described by the paper 'Troubling Trends in Machine Learning Scholarship'**
- Using both mathematical and natural language statements without tightly linking them.
- Making statements about some results that are understood to the reader as explanations, but are merely sentences that convey the speculative thoughts of the author that might support the apparent results.
- The author proposes more than one change to improve results, leading to the reader believe that all changes are necessary, while in fact only one of the changes contributed to the improved results
- Using too much mathematical expressions in a paper that may scare away people that wished to enter the ML community.
* `[ D ]`


---

**Q: in "Do CIFAR-10 Classifiers Generalize to CIFAR-10?" the authors hypothesise what could cause the error gap between their data and the data used for the original classifiers. Which of the following seemed to have the biggest effect on accuracy?**
- Statistical error
- Near duplicate removal
- Hyper-parameter tuning
- Cross validation
* `[ B ]`


---

**Q: Why CIFAR 10 classifiers do not generalize to CIFAR 10?**
- Training set overfitting
- Test set overfitting
- Distribution shift 
- None of the above
* `[ C ]`


---

**Q: Which of the following expressions about the overfitting problem is true?**
- We can define two notions of overfitting (“test set overfitting” and “validation set overfitting”) and CIFAR-10 dataset suffers of the test set overfitting 
- We can define two notions of overfitting (“training set overfitting” and “test set overfitting”) and the test set overfitting is more important to generalize to unseen data
- We can define two notions of overfitting (“training set overfitting” and “test set overfitting”) and the training set overfitting is more important to generalize to unseen data
- None of the above
* `[ B ]`


---

**Q: What is the explanation given for the gap betweeen original and new accuracy?**
- statistical error
- differences in near-duplicate removal
- cross-validation
- creation of a set of 'harder'images
* `[ D ]`


---

**Q: In the paper “Do CIFAR-10 Classifiers Generalize to CIFAR-10?” a large drop in accuracy was found when testing the performance of CIFAR-10 classifiers on a similar, but different dataset. How do the authors explain this gap?**
- The classifiers are implicitly overfitted on the test set by adapting the design choices to the test set
- The classifiers perform worse due to a small distribution shift between the original CIFAR-10 dataset and the new dataset
- The new dataset contains ‘harder’ images, and thus is more difficult for all classifiers
- The hyperparameters of the model were incorrect, and should be retuned
* `[ B ]`


---

**Q: CIFAR -10 doesn’t contain images**
- which are almost same 
- which has duplicates
-  which are labeled 
- which are not labeled
* `[ D ]`


---

**Q: What is test set overfitting?**
- The gap between the test accuracy and the accuracy on the underlying data distribution.
- The difference between the training accuracy and test accuracy.
- The mean error of the training session.
- The difference between the size of the test set and the training set.
* `[ A ]`


---

**Q: While creating a new test CIFAR-10 test dataset, the best approach to the generalization capability of a classifier is to :-**
- induce some variety in the dataset by taking a DSLR camera, snapping images of the same classes and cropping to 32 x 32
- take more test images from the Tiny Images Dataset and ensure that there aren't any similarities between the the new test dataset and the old test and old train dataset
- A and B
- None
* `[ B ]`


---

**Q: The authors found a large gap in accuracy when using a different database. What was the main reason for this gap?**
- CIFAR-10 does not generalize very well
- Statistical error
- Differences in near duplicate removal
- Hyper parameter tuning
* `[ A ]`


---

**Q: What is "mathiness"?**
- The use of a mixture of symbols and words that leaves ample room for slippage between statements in natural language versus formal language
- When the mathematics involved in a paper is not related to the paper but refers to a different level of abstraction
- The ability to translate easily into mathematics every concept
- None of the above
* `[ A ]`


---

**Q: Which of the following explanation is not possible reason of the gap between the orignal and new accuracy in the paper?**
- statistical error
- differences in near-duplicate removal
- training on part of the new test data
- Few Changes in the relative Order
* `[ D ]`


---

**Q: Regarding the large gap between original and new accuracy, which of the following statements could NOT be a hypothesis behind the gap? Note that the hypothesis might not be the real reason.**
- It's statistical fluctuations that cause the large gap.
- A large number of removed near-duplicates lead to some accuracy drop.
- There are distribution shifts between the original dataset and the new one.
- The new data set is quite different from the original one because it is generated differently.
* `[ D ]`


---

**Q: What explains most the significant drop of accuracy for the new test set compared to the CIFAR-10 dataset?**
- Not re-tuning the hyperparameters of a classfication model
- Not applying cross-validation
- The CIFAR-10 dataset contains near-duplicates while the new test set does not
- A small distribution shift between the original CIFAR-10 dataset and the new test set
* `[ D ]`


---

**Q: If one wants to recreate a dataset with new images and test a classifier on both datasets to see if it performs equally well on both: what is NOT important for creating the new dataset?**
- The new dataset has to contain the same number of images as the original one
- There should no (near) duplicates in the two datasets
- The difficulty of the two datasets has to be similar
- The sub-class data distribution of the two datasets has to be as similar as possible.
* `[ A ]`


---

**Q: When comparing results of classifying acknowledged test data sets and new (manually) created test sets using predefined classifiers, what could explain a gap between accuracy?**
- The VC-dimension
- Wrong hyperparameter tuning
- Computing power
- Visualization
* `[ B ]`


---

**Q: In the paper "Do CIFAR-10 Classifiers Generalise to CIFAR-10?", the authors test CIFAR-10 trained classifiers on a new set of test data which has been curated to have a similar sub-class distribution to the original CIFAR-10 dataset. When building classifiers in general, is this good practice? Choose from the following statements.**
- Yes, as testing on a dataset of similar distribution to the original ensures that the data doesn't overfit to the training set or test set.
- Yes, as testing on more data will give us more accurate results with regards to classifier performance.
- No, because the design set distribution will never be a similar approximation to the true distribution of the data.
- No, as this will not let us determine if the classifier generalises beyond the original test set. We should gather data with a similar distribution to what we would expect in the environment the classifier will run in.
* `[ D ]`


---

**Q: Which one of the following is NOT the step to build the new test set?**
- Understanding the sub-class distribution
- Analyzing current CIFAR-10 dataset
- Collecting new images
- Final assembly
* `[ B ]`


---

**Q: When testing a classifier with a test set, the distribution of the test set should:**
- Be as wide as possible;
- Be as different from the original dataset set as possible;
- Be as close the original dataset set as possible;
- Be a shifted version of the original dataset.
* `[ C ]`


---

**Q: What is CIFAR-10 in machine learning?**
- CIFAR-10 is an activation function.
- CIFAR-10 is a collection of images that are commonly used to train machine learning and computer vision algorithms
- CIFAR-10 is an image classification model
- CIFAR- 10 is an algorithm to compute gradients
* `[ B ]`


---

**Q: Which of the following statements can not be an explaination for the difference in performance of classifiers for two different data sets?**
- The statistical confidence intervals are so low that there is a chance that the performances are the same
- The images in one of the sets are harder to classify
- The parameters were better tuned for one of the sets
- The distributions of the sets differ
* `[ A ]`


---

**Q: Why do the authors of the paper collect a new test set to generalize CIFAR-10 trained classifiers on?**
- Popularity of CIFAR-10
- Transparancy and documentation of CIFAR-10
- To compare against already established models
- All of the above
* `[ D ]`


---

**Q: What is one of the main reasons that classifier performance tests tend to produce over optimistic results?**
- test results are not overly optimistic, they always give a good estimate of the true classification performance.
- We usually don’t have access to new unseen data, therefore many tests tend to overfit to already seen data.
- The results are fake.
- The data that used for training is too simple compared to the real world data
* `[ B ]`


---

**Q: What happens to models' accuracy in the NEW test set when in the trained set of CIFAR-10 is added part of the NEW test set?**
- Accuracy grows significantly
- Accuracy grows but not significantly
- Accuracy drops significantly
- Accuracy decrease but not significantly
* `[ B ]`


---

**Q: In the paper: "Do CIFAR-10 classifiers generalize to CIFAR-10", why do the authors go through a detailed selection process for the new test set**
- They wanted to observe the folk-law of Machine learning
- They wanted to ensure that the distribution of the new test set matched the original test set as closely as possible.
- They wanted to be able to publish the new images as an extension to the CIFAR-10 dataset for future researchers
- None of the above
* `[ B ]`


---

**Q: What is CIFAR-10?**
- CIFAR-10 is a technique used to generate datasets using 10 classes of input.
- CIFAR-10 is a dataset commonly used to benchmark the performance of image classifiers.
- CIFAR-10 describes a type of convolutional neural network with 10 hidden layers.
- None of the above.
* `[ B ]`


---

**Q: Which of the following statement is incorrect regarding Ciphar-10 dataset:**
- adding truly unseen images to a dataset will decrease the accuracy by 4 to 10 percents
- In terms of relative error, the models with higher original accuracy tend to have a smaller increase.
- the current research methodology of “attacking” a test set for an extended period of time is surprisingly resilient to overfitting.
- ciphar-10 is a subset of Tiny Images dataset
* `[ B ]`


---

**Q: Which of the following statements are true?

1. Overfitting happens when a models learn the details and noise in the training data to the extent that it negatively impacts the performance of the model on new data.
2. The goal of machine a learning algorithm is to produce a model that generalises well to training data.**
- Statements 1 is correct, statements 2 is incorrect
- Statement 1 is incorrect, statement 2 is correct
- Both statements are correct 
- Both statements are incorrect 
* `[ A ]`


---

**Q: What is a key (realistic) assumption that enables for unbiased model estimate?**
- Classifiers must be independent of the test data set.
- Test data set is sufficiently large.
- The optimum of the loss function has been found.
- The neural network has enough hyperparameters.
* `[ A ]`


---

**Q: As the original accuracy is significantly higher than the new accuracy in most cases, does this mean the models are overfitting on CIFAR-10?**
- Yes, as overfitting can be easily explained by the gap of test accuracy and training accuracy
- No, overfitting is only the difference between training accuracy and test accuracy, the new dataset doesn't influence this at all
- Yes, as the model is not able to generalize to CIFAR-10, this is an obvious sign of overfitting
- No, the models could be tuned specifically for the original training and test set and therefore perform worse.
* `[ D ]`


---

**Q: If you have a ML model, which was trained and evaluated using the CIFAR-10 dataset, showing an accuracy of 95%, what would you expect if you input a new dataset of unknown images with a very similar data distribution?**
- The ML model would most likely perform the same- the accuracy would remain within 0.5% of the original test.
- The ML model would most likely perform better. The accuracy would improve by 0.6% or more.
- The ML model would certainly perform worse. An accuracy drop of around 5% or higher is expected.
- Insufficient information to make such a prediction.
* `[ C ]`


---

**Q: Which of the following statements is true?**
- The gap in results the original data and newly created data are for a large part accounted by statistical errors.
- Model tuning based on test error does not lead to overfitting on a standard data set.
- Constantly 'attacking' a test set will lead to overfitting.
- Randomly selecting parts of a large data set as test set showed large distribution shifts for these test sets.
* `[ B ]`


---

**Q: Imagine that somehow, new data fed to a classifier is truly unseen and there is still a drop in accuracy. Then the classifier trained on the old data is…**
- a) overfitted
- b) underfitted
- c) perfectly fitted
- d) has zero error on the test data
* `[ A ]`


---

**Q: Big general datasets like Tiny Images or CIFAR-10 are often used to train machine learning algorithms. Which statement is correct?**
- It is possible to achieve a perfect accuracy for image recognition applications.
- Datasets are used to train the algorithms.
- Datasets can only be used for image recognition.
- In order to create a image recognition algorithm, it is necessary to have pictures with a very high quality.
* `[ B ]`


---

**Q: Which statement about CIFAR-10 is false?**
- The purpose of CIFAR-10 was to create a cleanly labeled subset of Tiny Images
- CIFAR stands for Canadian Institute of Advanced Research
- It is based on 10 000 images split into 14000 classes (--> why it is false: impossible)
- Manually labeling (by means of students) was necessary
* `[ C ]`


---

**Q: Which statement is true?**
- If a deep neural network shows 100% accuracy on the training set we can conclude that it failed to generalize.
- Developing algorithms using the same test sets can cause overfitting even if we apply cross validation properly
- Using cross validation we can avoid all kinds of overfitting
- It is not a problem if we adapt our model design choices to the test set
* `[ B ]`


---

**Q: What is the effect of optimizing and comparing models on a single certain test set, like discussed in the CIFAR-10 Paper?**
- Gain of comparability between models, gain of generalization power
- Gain of comparability between models, loss of robustness
- Loss of comparability between models, loss of robustness
- None of the above
* `[ B ]`


---

**Q: In order to investigate how well current image classifiers generalize to truly unseen data, a new test set for the CIFAR-10 image is collected. What is a reason for creating this CIFAR-10 test set?**
- The dataset creation process for CIFAR-10 is transparant and well documented.
- CIFAR-10 is a dataset that is not often used in machine learning.
- CIFAR-10 poses a problem that is not too difficult.
- There is not much known about CIFAR-10 and therefore it is interesting to study. 
* `[ A ]`


---

**Q: Which of the following the reason for choosing the CIFAR-10 dataset for checking the "Generalisation" properties of the classifiers?**
- CIFAR-10 is currently one of the most widely used datasets in machine learning and serves as a test ground for many computer vision methods.
- The dataset creation process for CIFAR-10 is transparent and well documented.
- CIFAR-10 poses a difficult enough problem so that the dataset is still the subject of active research.
- All of the above.
* `[ D ]`


---

**Q: What is NOT a given reason why the authors tested the CIFAR-10 dataset?**
- CIFAR-10 still poses a relevant and difficult problem
- CIFAR-10 is one of the most used datasets
- CIFAR-10 holds properties that make it very suitable for testing neural networks
- CIFAR-10 is transparent and well documented
* `[ C ]`


---

**Q: Which of the following is no reason for the gap between the expected accuracy of ML methods and its accuracy on truly unseen images?**
- Statistical fluctuations
- Incorrectly configured hyperparameters
- Unseen images are "harder" to recognize than testset images
- Underfitting on the unseen images
* `[ D ]`


---

**Q: What can be inferred as the major reason for drop in accuracy of the classifying algorithm when tested on the new data set**
- Statistical Error
- Very stringent near duplicate image removal 
- Hyper-parameter tuning
- Large distribution shift from the original data set
* `[ B ]`


---

**Q: We adopt the standard classification setup and posit the existence of a “true” underlying data distribution D over labeled examples (x; y). Consider the case where we want to split dataset test set (Dtest) and training set (Dtrain)  for classifier training and testing. Which of the following holds true?**
- For a sufficiently large test set Dtest, standard concentration results show that Dtest is a good approximation of D as long as the classifier does not depend on Dtest  
- Even with small set of  Dtest, standard concentration results show that Dtest is a good approximation of D as long as the classifier does not depend on Dtest and samples are drawn randomly from D.
- No matter what training set size is standard concentration results show that it is impossible to approximate D with high degree of confidence.
- None of these propositions are true
* `[ A ]`


---

**Q: Which the following is WRONG?**
- Try not to optimize the model based on the test set.
- CIFAR-10 classifiers do not generalize to CIFAR-10.
- Ideally, the classifier wants to generalize to unseen data. However, we don't always have access to true data distribution and thus we use separate test set to  evaluate the classifier.
- In the above option C, the justification is that the classifier is independent of the test set.
* `[ D ]`


---

**Q: In terms of overfitting, the main contribution of the paper is that it shows that:**
- CIFAR-10 classifiers do not test set overfit.
- CIFAR-10 classifiers test set overfit.
- CIFAR-10 classifiers do not training set overfit.
- CIFAR-10 classifiers training set overfit.
* `[ A ]`


---

**Q: What is test set overfitting?**
- When we choose our test set in a way that we achieve high accuracy.
- A way to quantify overfitting as a difference between test set accuracy and underlying distribution.
- When our test set is too small to give relevant information of the real performance.
- A way to quantify overfitting as a difference between test set accuracy and training set accuracy.
* `[ B ]`


---

**Q: Which of the following could not be a reasonable explanation for the gap between the original accuracy in CIFAR-10 and the accuracy in the new dataset?**
- Differences in the removal of near-duplicate images for the two datasets
- The need of re-tuning of the algorithm's hyperparameters
- The accidental creation of a more difficult test set by including a set of "harder" images
- The new, truly unseen data which have the same distribution with the data in CIFAR-10 dataset
* `[ D ]`


---

**Q: Did the analysis show any overfitting on the analized network-dataset pairs?**
- Yes, just training set overfitting.
- Yes, just test set overfitting.
- Yes, both training and test set overfitting.
- No.
* `[ A ]`


---

**Q: According to the paper “CIFAR-10 Classifiers Generalize to CIFAR-10”, what is the FALSE statement?**
- When testing on truly unseen data, all the tested models accuracy dropped significantly
- The relative order of the classifiers remains almost the same on the new test data
- The gap in accuracy when tested on truly unseen data could be explain by the different test size used (2000 – the paper new test dataset VS 10000 – CIFAR-10 test dataset)
- Tuning hyperparameters of VGG model using the new test data by performing grid-search does not improve the accuracy significantly
* `[ C ]`


---

**Q: Different types of overfitting exists:

1. Training set overfitting
2. Test set overfitting

Which answer describes these two the best**
- 1. Quantify to overfitting as the performance difference of the training set with unseen data. One should do everything to avoid this type of overfitting.

2. By adapting the model to the underlying data patern it might start overfitting to the test data. This type of overfitting isn't a big issue.
- 1. Quantify to overfitting as the performance difference of the training set with test data. This overfitting tends to happen

2. By adapting the model to the underlying data patern it might start overfitting to the test data. This type of overfitting is big issue as your test set lose validaty
- 1. Quantify to overfitting as the performance difference of the training set with unseen data. This overfitting tends to happen

2. By adapting the model to the underlying data patern it might start overfitting to the test data. This type of overfitting isn't a big issue.
- 1. Quantify to overfitting as the performance difference of the training set with unseen data. This overfitting tends to happen

2. By adapting the model to the underlying data patern it might start overfitting to the test data. This type of overfitting is big issue as your test set lose validaty
* `[ B ]`


---

**Q: How do authors find the duplicate pictures?**
- L2-nearest-neighbor
- Manual
- Fourier transform
- Edge detection
* `[ A ]`


---

**Q: Which one of the following cannot be concluded from the research of Recht et al. (2018) on the generalization of CIFAR-10?**
- There are no signs of test set overﬁtting on CIFAR-10
- Model tuning based on the test error does not lead to overﬁtting on a standard data set
- More studies should collect insightful new data and evaluate existing algorithms on such data
- None of the above
* `[ D ]`


---

**Q: Would you suggest to keep using CIFAR-10? Why or why not? (Please pick the most correct one.)**
- Yes. Because it actually does not lead to overfitting through test-set adaptivity even though it may find it hard to generalise to natural variations in image data.
- No. Because competitive adaptivity on this dataset leads to overfitting.
- Yes. Because this is the standard and used everywhere.
- No. Because it shows a large drop in accuracy when tested on a slightly changed different data set.
* `[ A ]`


---

**Q: State-of-the-art work in the domain of Neural Networks in Image processing is validated by using a small number of test-set repositories (e.g. CIFAR-10, ImageNet etc). Which of the following can be described as a controversial aspect of this strategy?**
- Fine tuning of NN hyper-parameters is not facilitated
- The estimated accuracy of the NNs is sensitive to small distribution variations in the labeled datasets thus compromising the robustness of accuracy estimates
- Over-estimation of current NN techniques accuracy can stall the development of more complex and efficient algorithms
- The developed model design techniques are adapted to the available test set thus causing severe over-fitting
* `[ B ]`


---

**Q: What is CIFAR 10 dataset?**
- A set of pre-labelled images used for the purpose  of machine learning
- Every new data set used for identification
- A single class of images used for training
- A set of blurred images used for improving the efficiency of the learning algorithms
* `[ A ]`


---

**Q: Which of the method can be used to find whether there exits distribution shift in the new test set:**
-  The distribution shift can be found by hyperparameter tuning of the classifier.
- The distribution shift can be found by training on part of the new test set.
- The distribution shift can be found by cross validation.
- The distribution shift can be found by both of the three methods.
* `[ B ]`


---

**Q: By using new dataset created in the paper “Do CIFAR-10 Classifiers Generalize to CIFAR-10?” there is observed a gap. Which statement is TRUE?**
- Gap is caused by statistical error as it is not significant
- Gap for all NN is around 5 %
- On new dataset, we can observe that all networks maintain the same tendency and have the very same relative order
- Usage of new data set proves that Cutout data augmentation shows more significant improvement than on CIFAR-10
* `[ D ]`


---

**Q: Which of the following statements is true?**
- A classifier trained on the CIFAR 10 dataset can only be used for images from this dataset.
- A classifier trained on the CIFAR 10 dataset can be used to correctly classify the CIFAR 100 dataset to its 100 classes.
- A classifier trained on the CIFAR 10 dataset can be used on images of the same classes that are not part of the CIFAR 10 dataset.
- A classifier trained on the CIFAR 100 dataset can be used to correctly classify the CIFAR 10 dataset to its 10 classes.
* `[ C ]`


---

**Q: The authors define two notions of overfitting. Which type is being argued to be better and why?**
- Training set overfitting, because the difference between the training set accuracy and test set accuracy gives a good estimation of how the algorithm has overfitted on the training data
- Training set overfitting, because deep neural networks can achieve a very high accuracy on the training data which might signal possible overfitting
- Test set overfitting, because it alligns with the overall goal to generalize to unseen data.
- Test set overfitting, because using cross-validation asures the validity of the test accuracy for truly unseen data.
* `[ C ]`


---

**Q: The authors find that all the existing neural networks, that have a high test error on the CIFAR-10 data set, preform significantly worse on a newly created similar data set. What is the reason for this drop in accuracy?**
- The new dataset contains less data
- All the existing models are sensitive to small variations in the data distribution
- The new dataset contains more difficult images
- All the existing models are overfitted on the widely used dataset CIFAR-10
* `[ B ]`


---

**Q: What of the following is a core assumption of machine learning?**
- That testing and training sets come from the same distribution.
- That to get good results you need a lot of training data.
- Data must be labeled to be able to train on.
- All of the above.
* `[ A ]`


---

**Q: What is most true with regard to the “distribution shift” mentioned in the paper the use of CIFAR inspired novel dataset?**
- There is a stark difference in the number of the images included in each class.
- The CIFAR database is not representative of the ground-truth distribution of Image data.
- There is a bias in the CIFAR database that shifts the mean when training networks.
- The keywords per class are better modelled by a binomial distribution than by a Gaussian distribution. 
* `[ B ]`


---

**Q: Which of the following steps is not part of the CIFAR-10 creation process?**
- Create a subsection of the TinyImages database
- Smoothen the images
- Let students annotate the images
- Remove (near) duplicates
* `[ B ]`


---

**Q: Which of the following is not a correct notion of overfitting?**
- The gap between the test accuracy and the accuracy on the underlying data distribution
- Having a model with high bias
- The difference between the training accuracy and the test accuracy
- A model that contains more parameters than can be justified by the data
* `[ B ]`


---

**Q: What are possible explanations of the gap in accuracy between the generated test data and the test set data that is always used for CIFAR10?**
- Statistical Error
- Hyperparameter tuning 
- Inspecting hard images
- All of the above
* `[ D ]`


---

**Q: What is CIFAR-10?**
- Classifier
- Dataset of images
- Type of neural network
- Deep Learning conference
* `[ B ]`


---

**Q: Choose the correct option**
- Using the test set to further tune the hyperparameters of a model enhances it's generalisation capability
- Cross-validation is a good method to check for the models generalisation capability
- To improve a classifiers generalisation capability, it's better to increase it's accuracy over multiple experiments over the same data rather than focus on testing on truly new data
- All of the above are correct
* `[ B ]`


---

**Q: Mark the false sentence.**
- The data in the test set used to evaluate the model’s performance must be from the same distribution as the original data.
- Cross-validation represents a more reliable form to measure the ability of a model to generalize to unseen data.
- The model’s accuracy from the original data set is always identical to the accuracy from the new generated test set.
- None of the above is false.
* `[ C ]`


---

**Q: Which of the following is correct use of cross validation?**
- Selecting variables to include in a model
- Comparing predictors
- Selecting parameters in prediction function
- All of the Mentioned
* `[ D ]`


---

**Q: Differences in accuracy between the CIFAR-10 test set and the new test set can be best explained by:**
- The time it took the construct both test sets
- The subclass distribution of the new set is matched to the original CIFAR-10 data set
- Differences in near duplicate removal
- Important changes in relative order
* `[ C ]`


---

**Q: What cause do the authors give to explain the biggest part of the accuracy gap?**
- Statistical error
- Cross-validation
- Distribution shift
- Overfitting
* `[ C ]`


---

**Q: What is the issue with existing classifier models, according to the authors based on their findings?**
- the classifier models do not handle even benign variations in image data very well
- by trying to improve performance on the same dataset (such as CIFAR-10), newer models would overfit to it
- there is a negative correlation between accuracy rates for CIFAR-10 and for unseen datasets
- using regularisation terms on newer models risks underfitting
* `[ A ]`


---

**Q: Which fact causes the drop in the accuracy of CIFAR-10 classifiers?**
- relative ordering change
- distribution shift
- training set overfitting
- test set overfitting
* `[ B ]`


---

**Q: Which of the following statements are correct: 1. The results of the paper did not support the hypothesis of adaptivity-based overtraining.  2. There was no significant gap between original and new accuracy scores.**
- Both 1 and 2 are correct
- Only 1 is correct
- Only 2 is correct
- Neither are correct
* `[ B ]`


---

**Q: Which of the below statements are true regarding the performance of the classifiers on the new CIFER-10 test set?**
- All models show a large drop in accuracy from the original to the new test set.
- VGG showed the best results without any significant accuracy drop.
- This experiment proved that the classifiers are stable and perform optimally even for a new test set.
- None of the above.
* `[ A ]`


---

**Q: One way to quantify overfitting is by using:**
- Training set overfitting: as the difference between the training accuracy and the test accuracy.
- Training set overfitting: as the accuracy of the loss function
- Test set overfitting: as the gap between the test accuracy and the accuracy of the underlying data distribution 
- Test set overfitting: as the degree of distribution shift between the test data set and training data set
* `[ A ]`


---

**Q: Statement 1: The goal for the CIFAR-10 dataset was to create a cleanly labelled subset of Tiny Images. 
Statement 2: A simple nearest mean search sufficed for determining the Tiny Image keyword for every image in the CIFAR-10 dataset since every image in CIFAR-10 had an exact duplicate. **
- Both statements are true
- Statement 1 is true
- Statement 2 is true
- Both statements are true
* `[ B ]`


---

**Q: Which of the following reasons is most likely to explain the observed gap between original CIFAR-10 data and the create data by the authors of the paper “Do CIFAR-10 Classifiers Generalize to CIFAR-10?”?**
- The created dataset has a distribution shift.
- The data is not correctly created following the CIFAR-10 described method.
- Statistical error fluctuation.
- Hyperparameter tuning.
* `[ A ]`


---

**Q: What is a possible reason for the drop in accuracy on the new test set compared to CIFAR-10?**
- Statistical error
- Change in dataset distribution
- New dataset is "harder" than CIFAR-10 due to duplicate removal
- Successive generations of models overfitting on CIFAR-10
* `[ C ]`


---

**Q: Which of the following is false according to the paper about CIFAR-10**
- The absolute gap is larger for models that perform worse on the original test set and smaller for models with better published CIFAR-10 accuracy
- The reason that the gap between original and new accuracy is high could be due to the fact that the test images created were accidentally harder than the initial data.
- The results showed overfitting in CIFAR-10
- The CIFAR data set is a subset of Tiny Images
* `[ C ]`


---

**Q: Which of the following statements true?
I: One way to quantify overfitting is as the difference between the training accuracy and the test accuracy.
II: A notion of overfitting is the gap between the test accuracy and the accuracy on the underlying data distribution.**
- I
- II
- I & II
- None
* `[ C ]`


---

**Q: The gap in accuracy (difference in accuracy from the original set to the new test set) is ...?**
- negative for all the models in consideration
- positive for all the models in consideration
- mostly negative
- mostly positive
* `[ B ]`


---

**Q: Which of the following is not a main contributor to the the gap between original and new accuracy?**
- Statistical fluctuations/errors
- Not well tuned hyperparameters of the model
- Differences in near-duplicate removal
- Training with too many test sets together with the original test sets
* `[ D ]`


---

**Q: Which of these statements concerning the generalisability of classifiers is not true?**
- The ground truth of a data distribution is usually not accessible
- The test set should be used to select a model
- Overfitting can be quantified as the difference between the training and test accuracy for a model
- Overfitting can be quantified as the gap between the test accuracy and the accuracy on the underlying distribution
* `[ B ]`


---

**Q: In order to avoid bias in deep learning, one can:**
- Evaluate 2 different deep learning architectures
- Evaluate the data-set on the same architecture only once
- Evaluate the data-set on the same architecture only twice
- Bias cannot be avoided in deep learning
* `[ A ]`


---

**Q: If a classifier gives 95% training accuracy and 94% test accuracy, then what can we interpret about the classifier?**
- The classifier generalizes to the given dataset and so can generalize to unseen data as well
- The classifier overfits the data and needs generalization
- The classifier generalises only to the given dataset but can has a possibility of performing badly on truly unseen dataset
- The classifier under fits the data.
* `[ C ]`


---

**Q: What does 'overfitting' mean according to the article?**
- The gap between the training accuracy and the test accuracy
- The gap between the test accuracy and the accuracy on the underlying data distribution
- Both, but A) is more important
- Both, but B) is more important
* `[ D ]`


---

**Q: Consider a classifier A that has a certain test accuracy on a benchmark test set. We want to test the hypothesis that A suffers of adaptivity-based overfitting, so we create a new test set of truly unseen data, that is as close as possible to being drawn from the same distribution as the original test set. However, the accuracy on this new test set drops and has a large gap from the accuracy on the original test set. What are methods to explain and recover some of this gap?**
- Statistical error
- Hyperparameter tuning
- Training on part of the new test set
- All of the above
* `[ D ]`


---

**Q: What is the difference between training set and test set overfitting?**
- Training set overfitting depends on the difference between training and test set accuracy, where test set overfitting depends on the difference between the test set and the underlying data distribution accuracy.
- Training set overfitting depends on the difference between the apparent error and the bayes error accuracy, where test set overfitting depends on the difference between the test set and bayes error accuracy. 
- Training set overfitting depends on the difference of the accuracy between the training set and the underlying data distribution of the training set, where test set overfitting depends on the difference of the accuracy between the test set and the underlying data distribution of the test set.
- There is no difference between these types of overfitting.
* `[ A ]`


---

**Q: Which of the following is not a reason for the performance difference of classifiers on test set of truly unseen images having distibution close to their original test (as outlined by Recht et al. for CIFAR10 classifiers)**
- The hyperparameters are always tuned on test set, making the classifer performance custom to the test set.
- Even small distribution shift in data results in significant difference in performance
- The test set is often used to adapt the model and this makes the test set an inaccurate measure of true performance
- The classifiers can capture small differences in the distribution that are trivial to humans ans this affects their performance on the unseen test set 
* `[ A ]`


---

**Q: Is it possible to over-fit on a test set?**
- No, the test set is hidden from the model so it can't use it
- Yes, if all models are compared using the same test set, it is no longer a representation of how they behave on unseen data but on how well they do with one test set
- No, over-fitting happens on training data
- Yes, only if you train on your test set
* `[ B ]`


---

**Q: You want to assess the overfitting of your classifier. Within the 95% confidence interval, 1% of the error can be found. You assess overfitting best by:**
- Hand
- Looking at the statistics
- Tuning hyperparameters
- This is situationally different
* `[ D ]`


---

**Q: Does the experiments on CIFAR-10 show overfitting ?**
- It shows training set overfitting and this is the most important kind.
- It shows no training or test set over fitting, but the accuracy for new data does drop, this is explained with "distribution shift"
- It shows training and test set overfitting, simply due to the fact there is new unseen data.
- It does not show training set overfitting, cause the training set only achieves 75% accuracy. 
* `[ B ]`


---

**Q: Which statement is wrong?**
- The data collection for CIFAR-10.1 was designed to minimize distribution shift relative to the original dataset.
- This paper (Do CIFAR-10 Classifiers Generalize to CIFAR-10?)evaluates the performance of CNN on a test dataset as the CIFAR-10. It shows that  performance drops on the new dataset.
- Adaptive data analysis  to guarantee the generalization, for example  a small gap between accuracy of true data distribution (what we want to optimize) and training accuracy (what we optimize in practice). 
- The results show that models that were better on the original test-set were not always better on the new test-set.
* `[ D ]`


---

**Q: Consider the following statements:

1. Using standardised datasets leads to test set adaptivity overfitting.
2. Generalization experiments focus on statistical reproducibility by evaluating classifiers on truly new data.

Which ones are correct?**
- 1
- 2
- 1 and 2
- neither
* `[ B ]`


---

**Q: Which of the following statements about the reliability of our current measures of progress in machine learning is false?**
- While there is a natural desire to compare new models to previous results, it is evident that the current research methodology undermines the key assumption that the classifiers are independent of the test set. This mismatch presents a clear danger because the research community could easily be designing models that only work well on the specific test set but actually fail to generalize to new data. 
- Due to adapting the CIFAR-10 test set for several years, there has been stagnation. There is as expected no progress in contemporary machine learning. The current research methodology of “attacking” a test set for an extended period of time is not resilient to overfitting. 
- The test cast doubt on the robustness of current classifiers. The classification accuracy of widely used models drops significantly. 
- Their study contained of three steps: Curating a new test, matching the sub-class distribution of the new set to the original CIFAR-10 dataset. Evaluating the performance of 30 image classification models on the new test set. Finally, the third step investigates multiple hypotheses for the discrepancies between the original and new accuracies.
* `[ B ]`


---

**Q: A model will generalize well if:**
- The training data set subclass distribution is random, to represent the natural world as it is.
- If cross-validation of the testing data set is possible
- The test data set is independently drawn from the same distribution as the training data set.
- The training data set is not stringent on near duplicates.
* `[ C ]`


---

**Q: What statement is incorrect. **
- An online standard database. like CIFAR-10 can be used to evaulate and compare deep networks.
- An online standard database, like CIFAR-10 is useful to train your model on, which can then be used for its true purpose.
- It is important to create new standards to accomodate new deep network models.
- Current deep networks perform better on standard databases than older generation deep networks.
* `[ B ]`


---

**Q: what is a problem in using a static dataset for comparing accuracy results**
- It allows for a competitive setting to improve results
- It increases the complexity of comparing results
-  It may reduce the accuracy of the true underlying distribution for the dataset
- None of the above
* `[ C ]`


---

**Q: What is the goal of a reproducibility study on CIFAR-10?**
- Tuning parameters and seeing which one performs the best
- Evaluating the performance of classifiers on CIFAR-10 against overfitting
- Finding a model that could minimize loss function
- ALL of above
* `[ D ]`


---

**Q: Which of the following is not an effect of "suitcase words"**
- Conflation can lead to overestimating the abilities of current systems
- It necessitaties extra prose unpacking the actual meaning of the wording used
- It clarifies technical papers
- These words function as aspirational prose (for example in marketing)
* `[ C ]`


---

**Q: Which of the following is/are true about the CIFAR-10 dataset?**
- Most widely used dataset
- dataset creation process is transparent and well documented
- Still a subject of active research
- All of the above 
* `[ D ]`


---

**Q: What are good ways to estimate the generalizability performance on unseen data of a classifier? 1. Determine error rate on a training set 2. Determine error rate on an independent testing set 3. Use cross validation to estimate error rate 4. Train the classifier on the test set and determine error rate on the test set**
- 1,2
- 3,4
- 1,3
- 2,3
* `[ D ]`


---

**Q: The gap between the accuracy over test data and unseen data is due to:**
- Training set overfitting
- Test set overfitting
- Distribution shift between original and new test sets
- Statistical fluctuations
* `[ C ]`


---

**Q: A small distribution shift on the test set has **
- No effect on the generalization results
- A small generalization problem
- Impacts the generalization greatly
- None of the above
* `[ C ]`


---

**Q: When creating a test set for your classifier, what is an important property of the test set to focus on?**
- That it has the same distribution as your unknown dataset in which the classifier will be used
- That the number of samples of your test set has does not exceed the maximum amount
- That the orientation of the test set is the same as the training set
- That it has a different distribution than your training dataset
* `[ A ]`


---

**Q: What is training se overfitting?**
- Diffrence between the training accuracy and test accuracy.
- Diffrence between the training accuaracy and real accuaracy.
- Diffrence between the test accuaracy and real accuaracy.
- Diffrence between the rea accuaracy and predicted accuaracy.
* `[ A ]`


---

**Q: Which of the following contribute regard accuracy in  the image classification in machine learning **
- Hyperparameter tuning
- Inspecting image
- A and B
- none 
* `[ C ]`


---

**Q: Most of the deep neural networks or evaluated on standard datasets such as CIFAR-10 and 100 but a new datet was created to check the generalising capability of CIFAR 10 classifiers.What can mainly explain the gap in the test set performances of deep neural network on CIFAR-10 and the new test set?**
- Differences in near-duplicate removal
- Lack of Hyper parameter tuning  while testing on new test set
- Over-fitting on test set
- Statistical fluctuations
* `[ A ]`


---

**Q: What does CIFAR-10 not contain?**
- The complete Tiny Images dataset
- airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and
truck
- Images with 32 x 32 resolution
- RGB color images
* `[ A ]`


---

**Q: What risks are present when model selection is performed based on the performance on a test dataset?**
- The performance of the selected model on the test data is an underestimation of the true performance of the model.
- The performance of the selected model on the training data is an overestimation of the true performance of the model.
- The performance of the selected model on the test data is an overestimation of the true performance of the model.
- The performance of the selected model on the training data is an underestimation of the test performance of the model.
* `[ A ]`


---

**Q: which statement is wrong?**
- One way to check overfitting is looking at the difference of training accuracy and testing accuracy. 
- Another notion of overfitting is the gap between the test accuracy and the accuracy on the underlying data distribution. 
- Cross-validation is a principled way of measuring a model’s generalization ability, so we don't need to care about the accuracies on the original test set. 
- The second notion of overfitting is more important in machine learning in general.
* `[ C ]`


---

**Q: Logistic regression assumes which of the following?**
- Linear relationship between continuous predictor variables and the outcome variable.
- Linear relationship between continuous predictor variables and the logit of the outcome variable.
- Linear relationship between continuous predictor variables.
- Linear relationship between observations.
* `[ B ]`


---

**Q: CIFAR-10 is a:**
- A modern deep learning algorithm
- A accuracy measure used to evaluate different deep learning aproaches
- A database of test images
- A super computer in the USA used for deep learning purposes
* `[ C ]`


---

**Q: What is the (argued) main reason for all classifiers' worse performance on the authors' custom test data compared to the CIFAR-10 test data?**
- The classifiers overfit to the CIFAR-10 training data.
- The classifiers overfit to the CIFAR-10 test data.
- A Distribution Shift between the CIFAR-10 data and the custom data can occur.
- Less custom test data led to the curse of dimensionality in all of the classifiers.
* `[ C ]`


---

**Q: According to the authors do their experiments reveal overfitting, what claims where made? 
Choose the most complete statement.**
- Yes, training set overfitting occurs.
Overfitting being the accuracy difference between training and test set always happens on the trainings data when there is an 100% accuracy.
No , test set overfitting occurs.
The model was explicitly fit to the test set, therefor validity of the test is lost.
- Yes, aggressive adaptivity always leads to overfitting.
- No, adding minor modification of standard machine learning competitions avoids the sort of overfitting that can be achieved with aggressive adaptivity.
- Due to multiple years of competitive adaptivity on this dataset, there is stagnation on held out data. In fact, the best performing models on our new test set see an increased advantage over more established baselines. 
* `[ C ]`


---

**Q: What is the most likely explanation of deep learning algorithms that perform well on CIFAR-10 performing worse on a new data set?**
- A slight difference in distribution between CIFAR-10 and the new data set
- Overfitting to the CIFAR-10 training data
- Difference in near-duplicate removal between data sets.
- Overfitting by adapting the model to the CIFAR-10 test data.
* `[ A ]`


---

**Q: The measurement of the current reliability of a algorithm can be doubt because of:
I Too small data sets are used to test the algorithm
II The same test data is often re-used throughout the algorithm and model design
III Not enough “unseen” data is used to test the algortihm**
- Only III
- I and II
- II and III
- I, II and III
* `[ C ]`


---

**Q: According to the paper, what causes the gap in error rates between the unseen data and the cifar 10 test set?**
- Since the models have been tuned using the test error, they have overfit to that particular test set
- There must have been a small distribution shift between the new test set and the original cifar10 dataset. Thus, it can be concluded these classifiers have trouble generalizing to natural variations in image data.
- The models need new hyperparameter tuning to get back to their original error rates.
- The new set was accidentally made more difficult.
* `[ B ]`


---

**Q: On regard to experiments for testing classifier models for overfitting, which of the following scenarios is the most accurate method to reproduce the results reliably?**
- From the same dataset used in the original experiment, ignore the training dataset and re-run the classification experiments on the remainder of the data samples.
- From the same dataset used in the original experiment, train the model with the data samples that were used for testing in the original experiment and test the model with the data samples that were used as training set in the original experiment.
- Use completely new data samples with wide degree of difference with the original dataset.
- Use completely new data samples, which are similar to the original dataset.
* `[ D ]`


---

**Q: Why the authors chose the CIFAR-10 dataset to investigate how well image classifiers generalise to truly unseen data?**
- CIFAR-10 is the first most common dataset in NIPS 2017
- CIFAR-10 draws from the larger Tiny Images repository that has significantly more fine-grained labels
- CIFAR-10 can be processed easily because of the limited research on this area
- None of the above
* `[ B ]`


---

**Q: Possible sources of troubling trends in machine learning are :**
- A big reviewer pool 
- Slow expansion of the community 
- Complacency in the face of progress
- Being clear on which problems are open and which are solved
* `[ C ]`


---

**Q: How is the decreased accuracy explained over the new dataset?**
- The models have been adapted during the years to the CIFAR-10 dataset creating overfitting
- There's a small distribution shift between CIFAR-10 and the new dataset, pointing out the difficulty of creating a dataset that generalizes well.
- The high competitiveness of the field has made researchers misrepresent the results anounced
- The difference is so small that could be explained due to statistical error
* `[ B ]`


---

**Q: If the accuracy of an algorithm on dataset 1 is $a_0$, what is the accuracy on a subpopulation of this dataset that is $k$ times harder? **
- a_0 / k
- 1 - k*(1-a_0)
- (1-a_0) / k
- none of the above
* `[ B ]`


---

**Q: In the paper "Do CIGAR-10 Classifiers generalize to CIGAR-10" the authors created a new dataset that was similar to the original dataset. According to the authors what was the cause of the drop in performance on their dataset**
- Improper use of cross-validation
- Hyperparameter tuning
- A statistical error
- None of the above
* `[ D ]`


---

**Q: In the CIFAR-10 experiment, what is the main conclusion that can be drawn from the fact that CIFAR-10 classifiers were having decreased performance against the newly created dataset?**
- The dataset was considered significantly more difficult
- The models show overfitting on CIFAR-10: model design choices were adapted according to the test set results
- The newly created test set was from a significantly different data distribution than the original CIFAR-10 test set
- CIFAR-10 classifiers can't generalize well to small natural variations in image data
* `[ D ]`


---

**Q: The famous CIFAR-10 dataset, was widely used at NIPS 2017.
During the construction of the dataset the researchers used which similarity measure to eliminate "near-duplicates"?**
- Jaccard similarity coefficients
- Mahalanobi's distance
- Euclidean distance
- Hashing similarity (MinHash)
* `[ C ]`


---

**Q: What is the current problem of choosing dataset?**
- Some data sets are too new and unseen
- Some data sets are not representative
- Some data sets might work well only in some certain algorithms but are not generalised
- Some data sets size is too small
* `[ C ]`


---

**Q: What is the database from which most images are taken for image recognition algorithms?**
- CIFAR-10
- CIFAR-13
- CIFAR-69
- CIFAR-30000
* `[ A ]`


---

**Q: Concerning the evaluation of the performance of models, which sentence best describes the goal of a learning algorithm?**
- To produce a model that generalises well to unseen data.
- To reach a high level of test and training accuracy.
- To provide computationally efficient classifications or regressions.
- Best representation of the ground truth distribution.
* `[ A ]`


---

**Q:  Consider a dataset DS1 consisting of 80 million RGB color images of resolution of 32x32. The images are organized by roughly 75,000 keywords and 1,000 to 2,500 images per keyword. A dataset CIFAR10 is derived from the DS1 dataset by following same creation process steps. Another dataset DS2 is formed from the same DS1 dataset and following the same creation process as for CIFAR10, keeping in mind that it's distribution is as close as to the CIFAR10 dataset(not repeating the images from CIFAR10). The accuracy of the two dataset is calculated on different models, and compared with each other.

Which of the following statement for the two datasets, CIFAR10 and DS2 is correct?**
- Both datasets will have same accuracy on all the models as both are derived from the same parent dataset.
-  There will be some gap between the accuracies, as a result of a small distribution shift between the original CIFAR-10 dataset and DS2.
- There will be a huge gap between the two accuracies, as DS2 dataset is a truly unseen data.
- None of the above
* `[ B ]`


---

**Q: What can be a reasonable guess when your deep learning model behaves poor  on a new image dataset?**
- The distribution of the new dataset is very different from the training dataset.
- The images in the new dataset is much harder to recognize than those in the test dataset
- Statitical errors occur as the new dataset is not large enough
- all of the three
* `[ D ]`


---

**Q: How are the overall accuracy for the original test and the new test set related? (β and β’ relative frequencies of sub-populations of original test set and new test set respectively)**
- accnew = β'/β accorig
- accnew = β/β' accorig
- accnew = accorig
- not related
* `[ A ]`


---

**Q: The gap between original and new accuracy of CIFAR 10 is concerningly large, which hypothesis is not considered in the paper?**
- Bias offset
- Hyperparameter tuning
- Training on part of our new test set
- Cross-validation
* `[ A ]`


---

**Q: The authors of the third paper found a significant gap between the old and their newly reproduced accuracy. To defend their findings, the authors explore several potential causes. What is not a potential cause explored by the authors?**
- Statistical error
- Cross-validation
- Hyperparameter tuning
- Dataset size
* `[ D ]`


---

**Q: Which statement about the selection of CIFAR-10 is false?**
- It was selected as the dataset creation process is transparent and well documented
- CIFAR-10 is currently one of the most widely used datasets in ML.
- CIFAR-10 poses a difficult enough problem so that the dataset is still under active research. 
- CIFAR-10 is freely accessible, and no fees need to be paid to access it. 
* `[ D ]`


---

