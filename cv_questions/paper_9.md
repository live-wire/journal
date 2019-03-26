# Questions from `paper_9` :robot: 

**Q: Why would you normalize the data points?**
- To get an uniform F matrix for every configuration of the cameras
- To speed up the calculations
- To make the F matrixes normalized
- To reduce the complexities of the calculations
* `[ Option A ]`


---

**Q: In the eight-point algorithm, the equation that should be minimized is A*f=0 subject to the constraint that f is a unit vector. Which is the solution to this system of equations?**
- The least eigenvector of A^{T}A.
- The largest eigenvector of A^{T}A.
- The second largest eigenvector of A^{T}A.
- The second largest eigenvector of A.
* `[ Option A ]`


---

**Q: Which statement is false?**
- An undesirable feature of the eight-point algorithm is that the method of transformation leads to a different solution for the fundamental matrix.
- It can be useful to first translate the coordinates so that the centroid of the points is at the origin. This reduces the ΓÇ£badΓÇ¥ effect of the offset on the corresponding matrix A^T A.
- It is important to carry out normalization (enforce the singularity constraint) of the image coordinates first to prevent that the most important entries in the fundamental matrix are precisely those that are subject to the largest relative perturbation.
- The normalized 8-point algorithm always gives lower errors compared with ZhangΓÇÖs iterative method.
* `[ Option D ]`


---

**Q: In a set of linear equations of the form Af $= 0$, where f has 9 entries and A is the equation matrix, the minimum number of equations required to obtain a trivial solution are**
- 7
- 8
- 9
- 10
* `[ Option B ]`


---

**Q: Review the following statements about the eight-points algorithm:
\begin{enumerate}
    \item The Eight-Point algorithm is known to be robust against noise
    \item Due to the complexity in it's implementation, the eight-point algorithm is not used often
\end{enumerate}**
- Statement 1 and statement 2 is true
- Statement 1 is true and statement 2 is false
- Statement 1 is false and statement 2 is true
- Statement 1 is false and statement 2 is false
* `[ Option D ]`


---

**Q: In the Eight-Point algorithm, what technique is used to enforce that the estimated fundamental matrix F has rank 2?**
- Singular value decomposition.
- Eigendecomposition.
- Lower-upper factorization.
- Normalization.
* `[ Option A ]`


---

**Q: Let $x'$ and $x$ denote matched points in two different images. The fundamental matrix $F$ is such that $x'^T F x = 0$. If now we transform the coordinates $x'$ and $x$ by $\hat{x}' = T'x$ and $\hat{x} = Tx$ and find the fundamental matrix $\hat{F}$ such that $\hat{x}'^T \hat{F} \hat{x} = 0$, then**
- $\hat{F} = T'^{-T} F T^{-1}$
- $\hat{F} = T'^T F T$
- $\hat{F} = T^{-T} F T'^{-1}$
- $\hat{F} = T^T F T'$
* `[ Option A ]`


---

**Q: Choose the correct statement about the non-isotropic scaling normalization transformation.**
- After this transformation the points form a cloud about the origin.
- After this transformation the centroid of the points is at the origin.
- After this transformation the principal moments are both equal to unity
- All of the above.
* `[ Option D ]`


---

**Q: Why the fundamental matrix is normally full-rank in practice?**
- Inaccuracies in the measurement
- The error in computer
- The wrong choice of images
- To speed up calculation
* `[ Option A ]`


---

**Q: Why is the proposed normalized 8-point-algorithm still relevant when an iterative algorithm is used?**
- This introduces easier to solve linearity into the iterative algorithm
- An iterative version of the 8-point algorithm is even more effective than the regular algorithm
- The normalized 8-point algorithm can provide a better starting point for the iterative algorithm
- The iterative algorithm is based on the standard 9-point algorithm
* `[ Option C ]`


---

**Q: When an accurate estimate of the fundamental matrix is desired, what is a big advantage of using the improved 8-point algorithm, prior to using a more complex method?**
- There is no advantage in doing this
- This creates a good comparison between the performance of the 8-point algorithm and a more complex method
- The proposed algorithm will provide a good initial estimate as a starting point for iteration
- The overall process will become much faster
* `[ Option C ]`


---

**Q: What is the problem with the original 8-point algorithm?**
- too slow
- too sensitive to noise
- too complicated
- all of the above
* `[ Option B ]`


---

**Q: What is true about the eight-point algorithm, according to the authors?**
- Although the normalized eight-point algorithm does not offer much better results than the unnormalized one, it provides a significant speed-up
- The normalized eight-point algorithm yields the best results, compared to all other methods
- Performing non-isotropic scaling rather than isotropic provides additional complexity without much benefit
- When using the eight-point algorithm to provide a starting point for iterative algorithms, it is better to use the unnormalized version
* `[ Option C ]`


---

**Q: Considering the following steps of the eight-point algorithm; select the right sequence of steps, in order to get best performance in terms of lowest average error.

\begin{enumerate}
\item Normalization: Transformation of the image coordinates using transforms $T$ and $T'$.
\item Solution: Finding matrix $F$ by solving a set of linear equations.
\item Constraint enforcement: Replacing $F$ by the closest singular matrix.
\item Denormalization: Replacing $F$ by $T'^TFT$. 
\end{enumerate}**
- 2-3
- 2-1-3-4
- 1-2-3-4
- 1-2-4-3
* `[ Option C ]`


---

**Q: What is the great advantage of the eight-point algorithm?**
- The system is needs 8 points to calculate the least squares minimization problem 
- The system is linear
- The eight points are enough to calculate the transformation
- The rank solutions of the Frobenius norm
* `[ Option B ]`


---

**Q: What are the benefits of the normalized eight-point algorithm when compared to the best iterative algorithms for the computation of the fundamental matrix?**
- The normalized 8-point algorithm performs almost as well as the best iterative algorithms.
- The normalized 8-point algorithm runs about 20 times faster than the best iterative algorithms.
- The normalized 8-point algorithm is far easier to implement than the best iterative algorithms.
- All of the above.
* `[ Option D ]`


---

**Q: Which statement about the traditional Eight Point Algorithm is true?**
- It is highly sensitive to noise in the matching points.
- Getting the number of required matching points needed to solve the Fundamental Matrix is usually unachievable.
- The solution for the Fundamental Matrix is insensitive to transformations in the input.
- None of the above.
* `[ Option A ]`


---

**Q: Why use normalized 8-point algorithm?**
- Unnormalized 8-point algorithm may be ill-conditioned
- It recovers more correspondences
- Normalized 8-point algorithm can solve larger systems 
- Normalized 8-point algorithm can solve systems with fewer observations
* `[ Option A ]`


---

**Q: What is the main advantage of the 8-point algorithm over other algorithms to calculate the fundamental matrix?**
- It is more robust to noise
- It requires only 8 point matches
- It is simpler, so it is much faster
- It performs better
* `[ Option C ]`


---

**Q: Which of the following is NOT the great advantage of the eight-point alogorithm?**
- High accuracy
- Linear
- Fast
- Easily implemented
* `[ Option A ]`


---

**Q: What is NOT true about the fundamental matrix?**
- The fundamental matrix conveniently encapsulates the epipolar geometry of the imaging configuration.
- The eight-point algorithm can be used to compute the fundamental matrix.
- The fundamental matrix may be used to reconstruct the scene from two uncalibrated views.
- The fundamental matrix is singular and of rank three.
* `[ Option D ]`


---

**Q: Which following statement about the normalized eight-point algorithm is false?**
- It performs almost as well as the best iterative algorithms
- It runs about 20 times faster than the best iterative algorithms
- Normalization of input is trivial
- The data normalization method can be widely used to other algorithms
* `[ Option C ]`


---

**Q: Consider the situation where you are using 8 points algorithm to recover 3D model from images. What you should consider in order to get good results from the algorithm?**
- Finding a transformation (translation and scaling) of the points in the image that maximizes the result of the algorithm
- Finding a transformation for each point which you use for solving linear equations
- Algorithm is insensitive to translation and scaling thus you do not need to worry about transforming images to match each other
- None of the above.
* `[ Option A ]`


---

**Q: Which normalization transformation strategy will likely be most beneficial when applying the eight-point algorithm?**
- Apply an isotropic scaling in the first stage.
- Apply a non-isotropic scaling in the first stage.
- Apply any of the proposed normalization transformations in both stages.
- Apply either of the proposed normalization transformations in the first stage.
* `[ Option C ]`


---

**Q: what is not true about the eight-point algorithm?**
- with normalization, it can perform almost as well as the best iterative algorithms
- it runs much faster than iterative algorithms with similar performance
- normalization usually offers minor improvements to the accuracy of the 8 point algorithm, in the order of 10^0
- at the time of writing of the paper, it was commonly thought that the eight-point algorithm was very susceptible to noise and therefore not very useful 
* `[ Option C ]`


---

**Q: What happens if we donΓÇÖt use normalization step in 8 point algorithm?**
-  The singular values found will have huge differences between them
- The singular values found will be zero
- The fundamental matrix could be computed faster
- None of the above
* `[ Option A ]`


---

**Q: Which of the following do not belong to the scaling steps before the eight-point algorithm proposed in the 'In Defense of the Eight-Point Algorithm' paper?**
- Points are translated so that their centroid is at the origin
- Points are scaled that the average distance from the origin is $\sqrt{2}$
- Points are translated and scaled for both images independently
- None of the above
* `[ Option D ]`


---

**Q: Which of the following is a major draw back of a direct implementation of the eight point algorithm**
- It is computationally expensive when compared with other iterative algorithms which solve for the fundamental matrix
- The eight point algorithm is typically very complex to implement.
- It is highly sensitive to noise when determining point correspondence
- None of the above
* `[ Option C ]`


---

**Q: Which is not one of the three steps of isentropic scaling, as mentioned in the paper**
- The points are translated so that their centroid is at
the origin
-  The points are then scaled so that the average distance
from the origin is equal to \sqrt{2}
-  The points are then scaled so that the average distance
from the origin is equal to 1
-  This transformation is applied to each of the two images independently.
* `[ Option C ]`


---

**Q: Which of the following steps is NOT part of the normalized 8-point algorithm?**
- Translating image coordinates to put the centroid at the origin
- Rescaling the image coordinates to set the average distance to the origin to sqrt(2)
- Projecting the second image to the orthographic frame of reference of the first image 
- All of the above are present in the 8-point algorithm 
* `[ Option C ]`


---

**Q: Biggest advantage of the classic 8 point algorithm is **
- insusceptible to Noise
- Non-Linear
- Computationally efficient
- Resilient to change of scale
* `[ Option C ]`


---

**Q: What is an issue with the 8-pt algorithm for estimating the fundamental matrix in uncalibrated cameras**
- Since the cameras are uncalibrated, the fundamental matrix derived from this algorithm is not accurate
- A change in the coordinate system of the camera pixels, leads to a change in the final fundamental matrix
- The final fundamental matrix cannot be used for other applications such as image rectification.
- None
* `[ Option B ]`


---

**Q: How can we enforce the singularity of the fundamental matrix F? Order the following steps:
1 ΓÇô Set $SΓÇÖ = diag(s_{11}, s_{22}, 0)$
2 ΓÇô Compute the singular value decomposition of the fundamental matrix
3 ΓÇô Compute $F = USΓÇÖV^T$
4 ΓÇô Compute the singular value decomposition of the diagonal matrix S**
- 2, 4, 1, 3
- 2, 1, 3
- 2, 4, 3
- 3, 1, 2
* `[ Option B ]`


---

**Q: Why do researchers think eight-point algorithm is useless?**
- It is computationally expensive.
- Eight points are not enough to get the accurate essential matrix.
- Without normalization of the inputs, the algorithm works poorly.
- None of above.
* `[ Option C ]`


---

**Q: What steps can you implement beforehand that would improve the eight point algorithm most?**
- Translate the centroid of the points to the origin.
- Non-isotropic scaling of the points.
- Isotropic scaling of the points.
- Combining answer A and C.
* `[ Option D ]`


---

**Q: The 8 Points Algorithm is  a very well known technique to built the essential matrix, that describes the relationship between two images of the same subject taken from uncalibrated cameras. It is very sensitive to noise in its base version though, how can we significantly enhance its performances?**
- Using more point correspondances between teh two images (more that 8)
- Normalizing the two input images
- Running it multiple times on different feature points
- Substituting it with an iterative procedure, that is usually faster
* `[ Option B ]`


---

**Q: What transformations are necessary to stabilize the 8-point algorithm**
- Rotation and projection
- Normalization and scaling
- Shrink and project
- None of the above
* `[ Option B ]`


---

**Q: Which of the following is false regarding the eight-point algorithm?**
- It is much faster than the best iterative algorithms
- It can performs just as well as the best iterative algorithms
- Without normalization of the inputs, the algorithm is virtually useless
- Performing an affine transformation on the input does not change the resulting fundamental matrix
* `[ Option D ]`


---

**Q: Which of the following statements is false?**
- The fundamental matrix is a basic tool in the analysis of scenes taken with two calibrated cameras.
- An important property of the fundamental matrix is that it is singular, in fact of rank two.
- The eight-point algorithm is frequently used for computing the fundamental matrix from a set of eight or more point matches. 
- Without normalization of the inputs, the eight-point algorithm performs quite badly.
* `[ Option A ]`


---

**Q: What is NOT an improvement for the 8 point algorithm that computes the fundamental matrix between uncalibrated cameras?**
- Bringing the centroid of all of the points to the origin
- Scaling the points so that the average distance from the origin is equal to $\sqrt{2}$
- Making the principal moments of the set of points to unity
- Replacing the fundamental matrix with a matrix that maximizes their Frobius Norm
* `[ Option D ]`


---

**Q: Which of these statements concerning the eight point algorithm is incorrect?**
- Normalizing the inputs of the eight point algorithm will make the performs worse
- The results of iterative algorithms depend on their initial estimate
- The results of iterative algorithms are prone to local minima
- The non-isotropic scaling scheme for the normalization transform performs just as well as the isotropic scaling in case of the eight point algorithm
* `[ Option A ]`


---

**Q: Why would you use normalization for the eight point algorithm?**
- It just happens to be a good trick
- It is not necessarily needed to obtain good results
- The solution vector for the fundamental matrix is better defined
- None of the above
* `[ Option C ]`


---

**Q: Why was the normalised 8 point algorithm developed?**
-  Because different image coordinates can lead to different solutions to the fundamental matrix F
-  if we do not normalise the image coordinates the resulting matrix F will over scale the transformed point
-   if we do not normalise the image coordinates the resulting matrix F will under scale the transformed point
-  It leads to faster computation of the F matrix
* `[ Option A ]`


---

**Q: What is FALSE about the Eight-Point Algorithm?**
- It performs almost as well as the best iterative algorithms
- It runs about 20 times faster than the best iterative algorithms
- It is far easier to code than the best iterative algorithms
-  It is more accurate than the best iterative algorithms without normalization of the input
* `[ Option D ]`


---

**Q: Which of the following methods will result in the best performing algorithm regarding (correctness/speed)?**
- Eight-point algorithm without normalization
- Eight-point algorithm with normalization
- An advanced iterative algorithm
- None of the above.
* `[ Option B ]`


---

**Q: How will normalization of the coordinates benefit the eight-points algorithm performance?**
- It reduces the sensitivity to noise
- It improves the speed
- It reduces the complexity
- It calibrated the cameras
* `[ Option A ]`


---

**Q: This papers shows that the performance of the 8-points algorithm can be drastically improved, to the extend that itΓÇÖs almost as good as the state of the art iterative algorithms. What step is necessary for this performance?**
- Normalization
- Converting to grayscale
- Define an extra ΓÇÿEΓÇÖ matrix
- Convert the matrix to rank(5), leading to an always non-singular matrix
* `[ Option A ]`


---

**Q: Which proposed solution resolves some of the susceptibility to noise issues for the eight-point algorithm?**
- Repeating the estimation procedure of the fundamental matrix multiple times.
- Finding the cross factor values that constitute the base pairs of the coordinate system.
- Rounding the values multiple times to converge to a global optimum.
- Normalizing of translation and scaling matrices.
* `[ Option D ]`


---

**Q: What can be improved regarding the Eight-Point Algorithm, to make it almost as good as the best iterative algorithms to estimate the fundamental matrix of a two view problem?**
- a) scale the coordinates of the interest points to have the same magnitude
- b) translate the coordinates of the interest points to the origin
- c) normalizing the coordinates of the interest points
- d) the Eight-Point Algorithm cannot be improved
* `[ Option C ]`


---

**Q: For eight-point algorithm, which method it needs for good performance, say make the result virtually useful?**
- Normalisation
- Centralisation
- Regularisation
- None of A, B and C, some other methods not mentioned here
* `[ Option A ]`


---

**Q:  what can be done to correct for a rank defficiency in a matrix, while reducing as little noise as possible.**
- remove the bottom rows
- singular value decomposition
- remove the rightmost columns
- transpose the matrix
* `[ Option B ]`


---

**Q: What's the biggest advantage of the eight-point algorithm**
- Ability to handle black and white images
- Linear complexity
- Logarithmic complexity
- Little sensitivity to noise
* `[ Option B ]`


---

**Q: Which of the following is not true about  Non-Isotopic scaling?**
- The points are transformed so that their centroid is at the origin.
- It is a non affine transformation.
- The principal moments are both equal to unity.
- The set of points will form an approximately symmetric circular cloud of points.
* `[ Option B ]`


---

**Q: How is the problem of heavy dependency from noise of the 8-points algorithm solved in the paper "In defense of the eight-points algorithm"?**
- The images from the different uncalibrated cameras are overlapped and pixel by pixel the mean value is taken into consideration
- From each image we get some augmented ones with a gaussian blur and we average all of them
- The fundamental matrix is calculated on the basis of more than 8-points
- Before running the actual algorithm, normalize the coordinates of the matching points (translation and scaling)
* `[ Option D ]`


---

**Q: Which of the following statements stand for the Singularity Constraint? 
i) The fundamental matrix always has rank 2
ii) Rank 2 of the fundamental matrix needs to be enforced (usually). 
iii) By letting  $F = UDV$ be
the SVD of $F$, where D is a diagonal matrix $D = diag(r, s, t)$
satisfying $r \geq s \geq t$. We let $F' = Udiag(r, s, 0)V^{T}$
iv) Most applications do not necessarily require rank 2.**
- ii) and iv) are correct
- i), ii), iii) are correct
- ii) and iii) are correct
- None of i), ii), iii) and iv) are correct
* `[ Option C ]`


---

**Q: Following are some of the statements related to the paper 'In defense of the eight point algorithm'.
A)Even if you transform the input, the solution for the fundamental matrix remains the same.
B)Lack of homogeneity in the image coordinates is one of the reasons to transform the images before solving for fundamental matrix**
- Both statements are correct
- statement B is correct ; statement A is incorrect
- statement A is correct ; statement B is incorrect
- Both the statements are incorrect
* `[ Option B ]`


---

**Q: which one is wrong?**
- In this paper, the analysis of the problem shows that the problem is caused by the poor distribution of the homogeneous image coordinates in their space, R3.
- In general, this estimate of the fundamental matrix is a better one than would have been obtained by estimating from the un-normalized coordinates.
- The basic eight-point algorithm is here described for the case of estimating the essential matrix E: Formulating a homogeneous linear equation;Solving the equation;Enforcing the internal constraint.
- The coordinate system of each of the two images should be transformed, independently, into a new coordinate system according to the following principle: 
 After the translation, the coordinates are uniformly scaled so that the mean distance from the origin to a point equals 2.
* `[ Option D ]`


---

**Q: What is NOT one of the main advantages of the eight point algorithm?**
- Performs better than  the best iterative algorithms.
-  It runs about 20 times faster.
- It is easy to code. 
- It is widely applicable to different problems. 
* `[ Option A ]`


---

**Q: Why is it needed to rescale and translate the points **
- To improve numerical stability and decrease effects of noise
- To make sure the matrix A (composed of pairs of feature points) is singular
- To make sure that the fundamental matrix is not singular
- To speed up the computation of the fundamental matrix F
* `[ Option A ]`


---

**Q: Which of the following is considered a disadvantage of Eight-Point algorithm?**
- It is extremely susceptible to noise
- It is slow
- It is more complicated than other iterative methods
- No transformations can be applied on input
* `[ Option A ]`


---

**Q: How can we mediate the 8 point algorithm's sensitivity to noise?**
- By applying it multiple times and taking the best result
- By normalisation of the matched points
- By combining it with other iterative algorithms
- By not setting a condition on the determinant of the fundamental matrix
* `[ Option B ]`


---

**Q: Following are some of the statements related to the paper 'In defense of the eight point algorithm'. 
a) Even if you transform the input, the solution for the fundamental matrix remains the same
b) Lack of homogeneity in the image coordinates is one of the reasons to transform the images before solving for fundamental matrix
Choose the correct option**
- Both statements are correct
- Statement A is wrong; Statement B is wrong
- Statement A is correct ; Statement B is wrong
- Both statements are wrong
* `[ Option B ]`


---

**Q: Which of the following statement for transformations in the paper 9 is not true?**
- For isotropic scaling, the points are translated so that their centroid is at the origin.
- For isotropic scaling, the points are then scaled so that the average distance from the origin is equal to sqrt(3).
- For isotropic scaling, this transformation is applied to each of the two images independently.
- For non-isotropic scaling, their centroid is at the origin.
* `[ Option B ]`


---

**Q: Performance of the eight-point algorithm can be improved by **
- applying translation and scaling operations before applying the algorithm
- passing the matrix through it multiple times
- Reducing the size of fundamental matrix by taking only a part of it
- Using a sharper image for reducing noise
* `[ Option A ]`


---

**Q: Which adjustments gives the best increase in performance of the eight point algorithm? Use: **
- Normalized coordinates with non isotropic scaling
- Unnormalized coordinates with isotropic scaling
- Unnormalized coordinates with Non isotropic scaling
- Normalized coordinates without scaling
* `[ Option A ]`


---

**Q: Regarding the scaling of the 8-point algorithm, which of the following statement is true?**
- For isotropic scaling, coordinates need to be scaled so that the average distance from the origin is equal to square root of 2.
- For non-isotropic scaling, coordinates need to be scaled so that the average distance from the origin is equal to square root 2.
- For isotropic scaling, the origin is defined by selecting an arbitrary point from the point set as the reference point.
- For non-isotropic scaling, the origin is defined by selecting an arbitrary point from the point set as the reference point.
* `[ Option A ]`


---

**Q: What is essential for the 8-point algorithm and what was the prevailing view of it that made it useless?**
- normalisation; susceptible to noise
- normalisation; not susceptible to noise
- number of points; susceptible to noise
- number of points; not susceptible to noise
* `[ Option A ]`


---

**Q: Select the TRUE statement:**
- The eight-point algorithm is approximately 10 times slower than the iterative algorithms
- The eight-point algorithm is approximately 20 times faster than the iterative algorithms
- The eight-point algorithm performs almost as well as the best iterative algorithm without the need of normalizing the inputs
- Even if it achieves better performances, the eight-point algorithm is much harder to implement
* `[ Option B ]`


---

**Q: What statement of eight-point algorithm is not right?**
- without using normalization, it performs badly 
- added complexity of the normalization algorithm is not significant
- After normalization, eight point's performance is not worse than state-of-the-art iterative algorithm
- initial estimate is not important to the quality of iteratively estimated result
* `[ Option D ]`


---

**Q: Which of the following is true:

I. There is a big difference in the error when using isotropic or non-isotropic scaling with the eight-point algorithm.
II. With normalization the eight-point algorithm performs much beter.**
- I
- II
- Both
- Neither
* `[ Option B ]`


---

**Q: 0**
- 0
- 0
- 0
- 0
* `[ Option A ]`


---

**Q: The eight-point algorithm for computing the essential matrix is a classic paper. In that paper,  the essential matrix is used to compute the structure of a scene from two views with calibrated cameras. However, the eight-point algorithm performs quite badly, often with errors as large as 10 pixels, which renders it completely useless. The 9th paper goes back to this algorithm, offering a solution to use the eight point algorithm together with a pre-processing step  to cover its susceptibility to noise. What's the advantage of this algorithm, compared to other iterative methods, even with its suscptibility to noise?**
- The essential matrix conveniently encapsulates the epipolar geometry of the
imaging configuration;
- The normalization of the coordinates improves the condition of the problem of computing the essential matrix;
- This algorithm is linear, therefore its implementation is simple and it is a fast method;
- It will use a linear least squares minimization solver for more than 8 points;
* `[ Option C ]`


---

**Q: What happens if we perform the eight point algorithm on normalized image coordinates?**
- We get the fundamental matrix.
- We get the essential matrix.
- None of the above.
- We can't do that.
* `[ Option B ]`


---

**Q: Which statement is true regarding the 8 point algorithm?**
- Transforming the coordinates of the points does not effect the algorithm's solution.
- Scaling the coordinates of the points does not effect the algorithm's solution.
- Normalizing the coordinates to some fixed canonical frame can make the algorithm independent of the choice of the coordinates.
- All of the above.
* `[ Option C ]`


---

