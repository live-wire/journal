# Questions from `paper_5` :robot: 

**Q: Theoretically, one of the attractive properties of RANSAC is that it returns an optimal solution with a predefined, user-controllable probability. What is the main problem scientist run into applying RANSAC algorithm?**
- Assumption that all all-inlier (minimal) samples lead to the optimal solution in practice does not lead to an acceptable solution
- Assumption that selecting random set of samples and fitting a line to it will lead to an acceptable solution
- Fitting a line on selected samples and counting the inliers will only return local optimal solution, that cannot be applied on a whole image.
- By selecting the random samples it is impossible to calculate the  base transformation estimate.
* `[ Option A ]`


---

**Q: LO-RANSAC runs faster than RANSAC:**
- always
- for low inlier ratios
- for hight inlier ratios
- never
* `[ Option B ]`


---

**Q: Which of the following is not a step of RANSAC algorithm:**
- Select minimal random subset from the input data.
- Model parameters that fit the sample are computed.
- All inliers to the model are found and the quality of the model parameters evaluated.
- The result of the quality of the obtained fit is further improved by running a cost function on the outliers.
* `[ Option D ]`


---

**Q: What is true about the formula t_tot=C_R*K+C_LO*[log(K)], where t_tot is the total running time. (used in the paper: ΓÇ£Fixing the locally optimized RANSACΓÇ¥)**
- K is a function of the computed probability of finding always the maximum number of inliers.
- K is the average time of LO procedure.
- K is the time of standard RANSAC single hypothesis generation and verification round.
- K denotes the number of cycles.
* `[ Option D ]`


---

**Q: The LO-RANSAC method is introduced because:**
- The RANSAC method is too slow;
- The RANSAC method needs too much data;
- The RANSAC method doesnΓÇÖt always get the optimal solution
- None of the above
* `[ Option C ]`


---

**Q: What is valid about RANSAC?**
- RANSAC returns an optimal solution with a predefined and user-controllable probability
- The assumption that all all-inlier samples lead to the optimal solution is the theoretical guarantee of its correctness
- RANSAC as a randomized algorithm, returns different outputs each time it is executed.
- All of the above
* `[ Option D ]`


---

**Q: What are the salient features of the LO+-Ransac
A - limiting the fraction of inliers in the optimization operations
B - Finding an alternative to the top-hat, 0-1 loss function by using the quadratic function
C - Having a C++ implementation
D - None**
- A and B
- A and B and C
- A
- D
* `[ Option A ]`


---

**Q: The main difference in terms of performances between RANSAC and Locally Optimized RANSAC is time, but this is not very relevant unless they face a particular situation. Which one?**
- When a low number of inliers in considered at every iteration of the algorithm
- When a high number of inliers is considered at every iteration of the algorithm
- When we deal with homography estimation
- When the considered pictures ar made just by a few pixels
* `[ Option B ]`


---

**Q: Which of these statements is false for the LO-RANSAC procedure?**
- It is very stable
- It is very precise
- More sensitive to the choice of inlier-outlier threshold
- Offers a significantly better starting point for bundle adjustment than the Gold Standard method
* `[ Option C ]`


---

**Q: Local Optimization RANSAC model parameter estimation can be sped up by imposing a limit on the number of what?**
- a) outliers
- b) weights
- c) inliers
- d) LO-RANSAC model estimation cannot be sped up
* `[ Option C ]`


---

**Q: In the paper 'Fixing the Locally Optimized RANSAC, several technical improvements of the LO-RANSAC were proposed. Which of the following is NOT an improvement?**
- The LO$^{+}$-RANSAC offers a stable robust estimation despite its randomized nature.
- Limiting the number of inliers included in the (iterative) least squares significantly reduces the execution time and often even improves the precision.
- Lightweight LO' is much faster compared to plain RANSAC even for difficult problems. 
- LO$^{+}$-RANSAC offers a significantly better starting point for bundle adjustment than the Gold Standard.
* `[ Option C ]`


---

**Q: What could do when we find the light sources, window shape and the scene outside a picture used in computer graphics?**
- provide a better model of the light rays in the scene to render synthetic objects that will be inserted inside the picture.
- find pattern of intensities on the surfaces
- construct a model to produce the blur kernel
- find the difference between origin figure and the figure with shadows
* `[ Option A ]`


---

**Q: Why is LO-RANSAC considerably slower than RANSAC?**
- Because the full local optimization step uses the advanced interpolation method and therefore it dominates execution time
- Because the full local optimization step uses the iterated linear least squares method and therefore it dominates execution time
- Because the full local optimization step uses extensive interpolation and therefore it dominates execution time
- Because the full local optimization step uses iterative convolution and therefore it dominates execution time
* `[ Option B ]`


---

**Q: Which of the following options does not contribute to speed up the LO-RANSAC algorithm?**
- Limit the number of correspondences that participate in the estimation of the parameters of the Iterative Least Squares model.
- Use the truncated quadratic cost function instead of MSAC-like cost functions for the separation of inliers and outliers.
- Do not call the Local Optimization of the LO-RANSAC during the first $K_start$ iterations.
- None of the above options.
* `[ Option B ]`


---

**Q: Which of the statement for the novel LO-RANSAC is not true?**
- very stable
- very precise in a broad range of conditions
- sensitive to the choice of inlier-outlier threshold
- offers a significantly better starting point for bundle adjustment
* `[ Option C ]`


---

**Q: What is NOT a reported advantage of the proposed LO+-RANSAC method?**
- it decreases variability over the output
- significantly higher execution time compared with the LO-RANSAC method
- it achieves far better accuracy than the LO-RANSAC method
- it offers a better starting point for bundle adjustment than the gold standard
* `[ Option C ]`


---

**Q: The RANdom SAmpling Consensus algorithm is a random, iterative algorithm. At each iteration, a "minimal subset" from all the data points is selected. This minimal subset is minimal in the sense that**
- it has the smallest number of data points needed to create the mathematical object that we want to fit to the whole data
- it has the same mean and variance as the whole data set, but has less points
- it is just some subset of the whole data set that has strictly less elements
- its mean is $0$
* `[ Option A ]`


---

**Q: How does the light-weight version of LO-RANSAC achieve execution time improvements?**
- During an iteration, only a single iterative least squares is applied on each model that is better than the one that was previously best.
- During an iteration, no iterative least squares is applied.
- During an iteration, only a single iterative least squares is applied on each model $k^{th}$ model.
- During an iteration, a single iterative least squares is applied.
* `[ Option A ]`


---

**Q: When do LO-RANSAC (Locally Optimised RANSAC) adds optimisation step?**
- When a better (than current) model is found
- When each one LO-RANSAC step is finished
- When each ln(k) LO-RANSAC steps are finished, where k is the total number of LO-RANSAC steps
- When each lg(k) LO-RANSAC steps are finished, where k is the total number of LO-RANSAC steps
* `[ Option A ]`


---

**Q: 1. The original RANSAC algorithm always leads to the optimal solution
2. The biggest improvement of the locally optimised RANSAC (LO-RANSAC) algorithm is the decrease in computational time**
- Both statements are correct
- Statement 1 is correct, statement 2 is incorrect
- Statement 2 is correct, statement 1 is incorrect
- Statement 2 is correct, statement 1 is incorrect
* `[ Option D ]`


---

**Q: The most substantial achievement of the authors' modification to the Locally Optimized RANSAC was:**
- making it faster
- increasing its accuracy
- yielding better results in specific applications
- generalizing its use for a wider range of applications
* `[ Option A ]`


---

**Q: Which is NOT an improvement of RANSAC algorithm?**
- Stop the iteration when the result is good enough
- Get more data for each sample and generate more accurate model
- Apply bundle adjustment
- Maximize the deviation of samples
* `[ Option D ]`


---

**Q: Which of the following is false about the LO+ RANSAC algorithm?**
- The execution time is lower than the execution time of LO-RANSAC
- The speed is comparable to plain RANSAC
- It offers a better starting point for bundle adjustment than the gold standard
- The precision is lower than of LO-RANSAC
* `[ Option D ]`


---

**Q: Which of the following options is not a characteristic of the novel $LO^+-RANSAC$ algorithm, based on the paper's evaluation?**
- It improves the execution speed.
- It offers a stable estimation.
- It offers a significantly better starting point for bundle adjustment than the Gold standard.
- It is very random in nature.
* `[ Option D ]`


---

**Q: The main steps of the RANSAC algorithm are:
\begin{enumerate}
       \item random sampling of points required to fit the model,
       \item model fitting,
       \item find inliers based on the extracted model and 
       \item evaluate model.
\end{enumerate}

After which of the above steps, an optimization step is introduced regarding the LO-RANSAC algorithm?**
- 1
- 2
- 3
- 4
* `[ Option C ]`


---

**Q: When is local optimization most performed in LO-RANSAC?**
- In estimation problems with low outlier ratios
- In estimation problems with high outlier ratios
- In estimation problems with low inlier ratios
- In estimation problems with high inlier ratios
* `[ Option D ]`


---

**Q: LO plus RANSAC is made faster by **
- By selecting a random subset of point and using it as seed to compute the model
- By triggering the algorithm to stop when more than a certain number of inliers are found
- Limiting the number of inliers used for least squares calculation
- None of the above
* `[ Option C ]`


---

**Q: In the optimized algorithm, how does the author achieve the purpose to reduce computing time**
- By introducing a limit on the number of correspondences during the step of iterative LSQ
- By reducing error scale threshold
- By reducing user-defined probability of finding optimized solution
- By increasing error scale threshold
* `[ Option A ]`


---

**Q: Which of the following steps is exclusive to Locally Optimized RANSAC when comparing it with RANSAC?**
- Random Selection of the minimal subsets from the input data.
- Computing the model parameters fitting the samples.
- Optimization of the best found model.
- Finding all the inliers to the model parameter and evaluating the quality of the model parameters.
* `[ Option C ]`


---

**Q: Consider the following statements related to the discussion in the paper - "Fixing the Locally optimized RANSAC"
(a) Locally optimised RANSAC is improved by reducing the number of inliers in the iterated linear least squares which is used to improve the quality of the estimated model
(b) Since RANSAC is a randomisation algorithm, the variants LO, LO+ also share the trait of having approximately same amount of variance in the detected inliers and accuracy

Choose the correct option **
- (a) and (b) are both correct
- (a) is wrong and (b) is correct
- (a) is correct and (b) is wrong
- (a) and (b) are both wrong
* `[ Option B ]`


---

**Q: What statement is not true:**
- RANSAC sufferes from the "not all all-inlier samples are good" problem
- RANSAC does not always return the same result
- RANSAC LO is always as fast as RANSAC
- RANSAC LO' (improved RANSAC LO) is unstable
* `[ Option D ]`


---

**Q: LO-RANSAC is slower than plain RANSAC in which of the following scenarios?**
- Number of inliers is large
- Number of inliers is small
- LO-RANSAC is always slower than plain RANSAC
- None of the above
* `[ Option A ]`


---

**Q: Which of the following statements about Locally Optimized RANSAC is/are true?

1. LO-RANSAC is significantly slower than normal RANSAC due to the many executions of the local optimization step. 
2 The advantage LO-RANSAC gets over normal RANSAC is the lower variance in the number of outliers and the lower variance in the accuracy**
- Only 1 is correct
- Only 2 is correct
- Both are correct
- Both are incorrect
* `[ Option B ]`


---

**Q: Consider the following statements and determine if they are TURE or FALSE.
(a) For the inputs with large inliner-outliner ratios, LO-RANSAC can be very slow as iterated least square steps needs to run through the inliners.
(b) Hence, limiting the number of inliners considered in the local optimisation step can improve the execution speed however will considerably damage the output quality.**
- TURE, TURE.
- TURE, FALSE
- FALSE, TRUE.
- FALSE, FALSE.
* `[ Option B ]`


---

**Q: What are the main advantages of the algorithm proposed in the paper, over the standard LO-RANSAC?**
- The ouput is more stable, and the execution time is reduced
- The ouput is more stable, and it needs less parameters
- It uses less memory and it needs less parameters
- the execution time is reduced and it uses less memory
* `[ Option A ]`


---

**Q: What of the following one is the proper definition of SfM?**
- The process of reconstructing 3D structure from
its projections into a series of images taken from different
viewpoints.
- The process of reconstructing 2D structure from
its projections into a series of images taken from different
viewpoints.
- The process of reconstructing 2D structure from
its projections into a series of images taken from the same
viewpoint.
- The process of reconstructing 3D structure from
its projections into a series of images taken from the same
viewpoint.
* `[ Option A ]`


---

**Q: Which of the statements is true:**
- LO RANSAC does not use inliers.
- LO RANSAC is slow due to its iterations over every inlier
- LOΓÇÖ RANSAC is better than LO+ RANSAC
- LO+ RANSAC is better than LO RANSAC
* `[ Option B ]`


---

**Q: What is NOT true about local optimization (LO) RANSAC?**
- Assumes that some inliers could be noise
- LO optimizes the new best RANSAC's hypothesis by fitting a new model which uses more inliers.
- LO RANSAC terminates faster than ordinary RANSAC, therefore LO RANSAC is always faster.
- N.A.
* `[ Option C ]`


---

**Q: What part of the standard RANSAC method is changed by the authors, to make the algorithm more stable, more precise and faster? **
- a different number of random matches is drawn for calculating the transformation matrix
- a different cost function for evaluating the inliers and outliers 
- limiting the number of inliers in the least squared method
- GPU acceleration of the least squared method
* `[ Option C ]`


---

**Q: What is the difference between RANSAC and LO-RANSAC?**
- LO-RANSAC adds an optimization step after the verification phase.
- LO-RANSAC only uses a subset from the start of the algorithm.
- LO-RANSAC is a low-computation version of the original RANSAC.
- LO-RANSAC has more tunable parameters compared to the original RANSAC.
* `[ Option A ]`


---

**Q: Which of the following statements is correct? **
- The speed of the lightweight LOΓÇÖ is much better compared to plain RANSAC even for easy problems with very high inlier ratios
- The LO+ - RANSAC offers a stable robust estimation despite its randomized nature
- The Gold Standard offers a significantly better starting point for bundle adjustment than the LO+ - RANSAC
- RANSAC returns the same outputs each time it is executed
* `[ Option B ]`


---

**Q: In the LO-RANSAC paper, what does the "LO" stand for?**
- LOw
- LOcal
- LOss
- Least Occurrent
* `[ Option B ]`


---

**Q: Compared with the standard RANSAC algorithm, what is not an advantage of the LO-RANSAC algorithm?**
- The LO procedure is relatively simpler
- The execution time is closes to standard oneΓÇÖs
- It offers a stable robust estimation
- It is less sensitive to inlier-outlier threshold choices
* `[ Option A ]`


---

**Q: Take a look at the following two questions concerning the LO$^+$ RANSAC algorithm which of the following properties hold true?
\begin{enumerate}
    \item The algorithm provides a better starting point for bundle adjustment than for example the Gold Standard
    \item Does not provide a robust solution due to its randomized nature
\end{enumerate}
Which of the statements are true?**
- Statement 1 and 2 are true
- Statement 1 is true, statement 2 is false
- Statement 1 is false, statement 2 is true
- Statement 1 and 2 are false
* `[ Option B ]`


---

**Q: Which of the following is a problem in 2D image stitching:**
- It is impossible to estimate the stitching field accurately due to complex interactions between 3D scene and camera parameters
- A single global transformation (homography) is suitable only in special conditions
- A single global transformation leads to misalignment and ghosting effect
- All of the above
* `[ Option D ]`


---

**Q: What is meant by the 'not all all-inliers samples are good' problem? Considering the Ransac algorithm.**
- Often a significant data-dependent fraction of all-inlier samples does not lead to an acceptable solution.
- That the assumption that all all-inlier samples lead to the optimal solution is invalid.
- Both A and B
- Neither A or B
* `[ Option C ]`


---

**Q: Which of following statements is FALSE regarding the LO+ RANSAC?**
- It replaces 0-1 loss function with truncated quadratic loss function.
- It constraints the number of inliers to reduce the LO time consumption.
- It runs "iterative least squares" only once instead of repetitions.
- The number that LO part of the algorithm is executed is relatively small compared to the RANSAC part of the algorithm.
* `[ Option C ]`


---

**Q: LO-RANSAC improves the quality of the estimated model but slows down the procedure for a large number of inliers. What is the reason for this effect?**
- It uses a modified cost function.
- It adapts the threshold of the distances between points and the fitting line depending on the number of inliers.
- It uses (iterated) linear least squares.
- More outliers are present.
* `[ Option C ]`


---

**Q: What is FALSE about LO-RANSAC?**
- LO-RANSAC is slower than plain RANSAC
- LO-RANSAC is superior to RANSAC in terms of accuracy
- LO in LO-RANSAC stands for Local Optimiziation
- LO-RANSAC always obtains a correct solution
* `[ Option D ]`


---

**Q: Which are true?

I. LO-RANSAC uses local optimization
II. LO' is faster than RANSAC but performs close to LO-RANSAC**
- I
- II
- Both
- Neither
* `[ Option C ]`


---

**Q: What is one of the advantages of the LO+-RANSAC algorithm described in the paper?**
- The algorithm is much faster than the traditional RANSAC algorithm.
- The algorithm is more stable than RANSAC.
- LO+-RANSAC uses no randomness.
- LO+-RANSAC always performs better than LO-RANSAC.
* `[ Option B ]`


---

**Q: which of the following properties does the novel algorithm LO-RANSAC hold?**
- non-random or stable
- less sensitive to the choice of inlier-outlier threshold
- It offers a significantly better starting point for bundle adjustment
- All of above
* `[ Option D ]`


---

**Q: Which is not a reason for LO-RANSAC to be adjusted in the paper titled: "Fixing the Locally Optimized RANSAC"**
- The LO procedure of the LO-RANSAC significantly affects the running time of the algorithm
- The LO procedure is relatively complex with a high number of parameters
- The LO procedure does use an inlier limit
- The LO RANSAC adds an optimisation step after the verification phase
* `[ Option C ]`


---

**Q: what is a serious drawback of ransac**
- having to select inliers
- that it returns different results each time it is run
- finding sufficient inliers
- None of the above
* `[ Option D ]`


---

**Q: lo-ransac fixes the problem wiht finding **
- Local-maximum Local-maximum 
- local-minum 
- regressing calculation 
- all of the above
* `[ Option B ]`


---

**Q: What assumption of RANSAC is not always true and is improved in Local Optimized RANSAC?**
- The assumption that some of the outlier samples are noise and not used in the optimal solution
- The assumption that the set of all inlier samples lead to the optimal solution
- The assumption that all minimal sets of inlier inlier samples lead to the same optimal solution
- The assumption that all outlier samples are noise and not used in the optimal solution
* `[ Option C ]`


---

**Q: Which one of following contribution is not the LO+-RANSAC making?**
- Limiting the number of inliers in the least squares significantly
- The lightweight LOΓÇÖ for easy problem
- Local optimization step to random minimal samples
- A better starting point for bundle adjustment 
* `[ Option C ]`


---

**Q: In which case the LO-RANSAC might negatively impact the execution time**
- For estimation problems with low inlier ratios
- For image pairs with high fraction of inliers
- LO-RANSAC never negatively impacts the execution time
- LO-RANSAC is always slower than RANSAC
* `[ Option B ]`


---

**Q: which of the following is false about RANSAC?**
-  RANSAC returns different outputs each time it is run.
-  RANSAC assumes that the inliers lead to an optimal solution.
- LO-RANSAC is slower than regular RANSAC
- LO-RANSAC is very random in nature.
* `[ Option D ]`


---

**Q: What is the tradeoff that has to be made in RANSAC style algorithms?**
- The tradeoff between estimation quality and computational time
- The tradeoff between the number of variables that a model is fitted over and estimation quality
- The tradeoff between the number of input parameters and estimation quality
- The tradeoff between the number of input parameters and computational time
* `[ Option A ]`


---

**Q: In what way does the LO-RANSAC method differ from the standard RANSAC method?**
- A lower amount of samples is selected for each iteration
- An extra model parameter is added to describe the computational load of each outcome
- In the verification step, the performance is only determined based on the outliers to the model
- An optimization step is added after the verification step
* `[ Option D ]`


---

**Q: What is the benefit of RANSAC with local optimization?**
- Generating a good estimation from a sample of inliers when not all inliers give the same estimation
- Better able to reject outliers
- Faster on data with a lot of inliers
- Generates a better estimation than RANSAC asymptotically
* `[ Option A ]`


---

**Q: What is the main idea of the improved LO-RANSAC, LO+?**
- Speed up the local optimization step by restricting the number of used inliers for the estimation of the model parameters in the iterative least squares to a value dependent of the number of inliers
- Speed up the local optimization step by restricting the number of used inliers for the estimation of the model parameters in the iterative least squares to a value independent of the number of inliers
- Speed up the local optimization step by using iterative least squares
- Speed up the local optimization step by restricting the number of used outliers for the estimation of the model parameters in the iterative least square
* `[ Option B ]`


---

**Q: What can you do to reduce the execution time of LO-RANSAC especially for large number of inliers, while not affecting the precision?**
- Limit the number of inliers in the (iterated) linear squares
- Verify some instead of all inliers in verification
- Lower \eta user-required probability of finding the optimal solution
- Lower \Theta inlier-outlier error threshold (error scale)
* `[ Option A ]`


---

**Q: Which one is one of the optimizations made to Lo-Ransac**
- Optimization of the LO step 
- Take into account all the points in the algorithm
- All of the above
- None of the above
* `[ Option A ]`


---

**Q: The paper claims that the Boston image pair, LO+-RANSAC with MSAC-like gain returned the same resulting homography for all 10000 runs. Why is this important?**
- Homography is suceptible to noise which can lead to different results over the 10000 runs unless using an algorithm with noise-elimination.
- RANSAC is a randomized algorithm, so having consistent results shows the model has good stability.
- This indirectly proves the homography is 100% accurate.
- MSAC-like gain only works with stable results.
* `[ Option B ]`


---

**Q: Which statement is true about the various RANSAC methods: (Original) Ransac, Local Optimization (LO) Ransac and Lightweight Local Optimization (LO$^+$) Ransac?**
- LO Ransac uses  (iterated) Least Squares to improve the quality of the estimated model.
- Neither Ransac, LO Ransac nor LO$^+$ Ransac guarantee the same result for repeated experiments.
- LO$^+$ Ransac significantly improves the running time of LO Ransac by limiting the number of correspondences that participate in improving the current model's parameters.
- All of the above.
* `[ Option D ]`


---

**Q: Which of the following statements are true rrgarding local optimisation of RANSAC :
A)LO-RANSAC is faster than the RANSAC method
B)LO-RANSAC gives better accuracy than RANSAC**
- A) is true
- B) is true
- Both A) and B) are true
- Both A) and B) are false
* `[ Option B ]`


---

**Q: Local optimization of the RANSAC reduces the randomness of the outputs by:**
- By forcing an upper bound on the number of inliers
- By adjusting the sensitivity of cost functions to the choice of inlier error scale
- By minimizing the local optimization overhead
- Reducing the variance in the number and accuracy of detected inliers
* `[ Option D ]`


---

**Q: Consider the Lo-Ransac algorithm: If we have a large set of inliers for the local optimization step, what happens if we only use a random subset with a set size for model estimation?**
- Execution time decreases, accuracy decreases
- Execution time increases, accuracy remains the same
- Execution time increases, accuracy decreases
- Execution time decreases, accuracy remains the same
* `[ Option D ]`


---

**Q: We can reduce the overhead in RANSAC algorithm by**
- Averaging the inliers before calculation of least squares.
- Introducing a limit on the number of inliers used in the least squares computation.
- eliminating the inliers using thresholding operation
- Using MSAC-like kernel to increase tolerance
* `[ Option B ]`


---

**Q: In the local optimization  step the Least Squares method is used in order to improve the quality of the model. How can the execution time be decremented when the number of inliers is large?**
- By limiting the number of matches that take part in the estimation of the model's parameters. 
- By estimating the parameters with lower speed.
- By increasing the number of matches .
- None of the above.
* `[ Option A ]`


---

**Q: In the paper: "Fixing the Locally Optimized RANSAC (LO-RANSAC)",  the authors propose an improvement over the locally optimized RANSAC algorithm. Which of the following describes one of such improvements**
- The authors propose an improved LO-RANSAC algorithm that executes in the similar time as the original LO-RANSAC algorithm but is more accurate.
- The authors propose an improved LO-RANSAC algorithm that executes faster than RANSAC and is more accurate than the original LO-RANSAC implementation.
- The authors propose an improved LO-RANSAC algorithm that executes almost as fast as RANSAC but is almost as accurate as the original LO-RANSAC implementation.
- None of the above
* `[ Option C ]`


---

**Q: which one is wrong?**
- The benefits of using affine correspondences have the potential to boost LO-RANSAC approach, enabling rapid runtime, with significantly fewer RANSAC-iterations and local optimization steps. 
- The locally optimized RANSAC adds an optimization step before the verification phase, if a so-far-the-best model is found.
- To reduce the time consumption, introducing a limit on the number of inliers that participate in estimation of model M' parameters
- LO+-RANSAC add local optimization steps within RANSAC to improve accuracy.
* `[ Option B ]`


---

**Q: This paper visits the problem of local optimization for RANSAC. The LO-RANSAC algorithm tries to solve the problem of the RANSAC, the (incorrect) assumption that all  all all-inlier samples are good. However this LO-RANSAC method has some advantages which were not mentioned in the original paper, in terms of running time. For image pairs with a high fraction of inliers, where a small number of random samples is sufficient for finding the solution, the original LO procedure significantly effects the running time. How did the paper solve this?**
- by excluding a fraction of the inliers;
- by modifying the iterative least squares by removing the limit on the number of inliers used in the least squares computation;
- by increasing the sample size;
- by modifying the iterative least squares by introducing a limit on the number of inliers used in the least squares computation;
* `[ Option D ]`


---

**Q: Which statement is false about RANSAC, LO-RANSAC and LO+-RANSAC?**
- RANSAC returns different outputs each time it is executed.
- LO-RANSAC applies a local optimization step to promising hypotheses generated from random minimal samples.
- LO+-RANSAC limits the number of inliers used in the least squares computation.
- LO+-RANSAC has a shorter execution time than RANSAC
* `[ Option D ]`


---

