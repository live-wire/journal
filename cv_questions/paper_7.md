# Questions from `paper_7` :robot: 

**Q: What makes features appear and disappear in reality as the camera moves?**
- Occlussions
- Camera noise
- Different lighting
- Reflections
* `[ Option A ]`


---

**Q: Which of the following is correct about the factorization method that was introduced by the authors?**
- It can recover the shape and motion under orthography without computing depth as an intermediate step.
- It uses the singular-value decomposition technique to factor the measurement matrix into two matrices which represent object shape and camera rotation respectively.
- It can also handle and obtain a full solution from a partially filled-in measurement matrix (when some features are occluded from some frames) that may result either when some features are occluded from some frames or tracking failures of points of interest.
- All of the above
* `[ Option D ]`


---

**Q: What is the advantage of an algorithm based on singular-value decomposition?**
- It is numerically well behaved and stable.
- It is a very fast algorithm
- It always converges towards a solution
- It can handle noise very well
* `[ Option A ]`


---

**Q: Using the factorization method, we cannot recover the following from a sequence of images under orthographic projection**
- Material
- Shape
- Motion
- We can recover all of the above
* `[ Option A ]`


---

**Q: Review the following two statements about applying the factorization method in SFM by Tomasi and Kanade in "Shape and Motion from Image Streams under Orthography:
a Factorization Method"
\begin{enumerate}
    \item The factorization method, in comparison to often applied existing solutions, is more robust against image noise.
    \item Camera translation along the optical axis is not accounted for in orthographic projection.
\end{enumerate}
Which of the statements are true?**
- Statement 1 and 2 are true
- Statement 1 is true and statement 2 is false
- Statement 1 is false and statement 2 is true
- Statement 1 and 2 are false
* `[ Option A ]`


---

**Q: If we assume to use orthographic projection in a Structure from Motion procedure, which of the following movements of the camera we mus avoid?**
- Rotation along the horizontal axis
- Motion along the vertical axies
- Displacement along its optical axis
- Circular motion around the object
* `[ Option C ]`


---

**Q: The measurement matrix W can be factorized into a shape matrix S and a rotation matrix R through singular-value decomposition (SVD). Because of the rank theorem, the registered measurement matrix W-tilde has at most three nonzero singular values. Therefore, in the SVD of W, what do the three largest values represent?**
- a) The coordinates of the object centroid
- b) The singular values that exceed the noise level
- c) The coordinates of the mean of all feature points
- d) There is no physical meaning to these values
* `[ Option B ]`


---

**Q: In this paper, a factorization method is used to estimate object shape and camera rotation from an image stream. What is not successfully dealt with is:**
- noise
- identification of false features
- occlusions
- all of the above
* `[ Option B ]`


---

**Q: How are the missing feature points because of occlusions dealt ?**
- The feature points are excluded
- The feature points are reconstructed 
- The feature points are randomly chosen
- None of the above
* `[ Option B ]`


---

**Q: What is Structure-from-Motion (SfM)?**
- The algorithm used to calculate feature points in images where the subject is moving
- The process of reconstructing 3D structure from its projections into a series of images taken from different viewpoints
- The process of understanding which transformation has been used only by looking at two different images of the same subject
- None of the above
* `[ Option B ]`


---

**Q: Which reference model and projection model have researchers assumed to have the position of feature points be specified by their image coordinates depths?**
- Perspective projection and a camera-centered representation
- Perspective projection and a world-centered representation
- Orthographic projection and a world-centered representation
- Orthographic projection and a camera-centered representation
* `[ Option A ]`


---

**Q: Regarding sfm, the measurement matrix W should be rank 3, why is this most of the time, in practice, not the case?**
- Because of noise
- Because affine transformations have 4 degrees of freedom
- The statement is false, the matrix W always has rank 3
- None of the above
* `[ Option A ]`


---

**Q:  What is the main problem(s) in using a camera centered representation for shape from motion?**
- the shape estimation is difficult
- the shapes are unstable
- it is sensitive to noise
- All of the above
* `[ Option D ]`


---

**Q: Which statement is correct according to the paper ΓÇ£Shape and motion from image streams under orthography: a factorization methodΓÇ¥: 1: not require a smooth motion assumption. The identification of false features, that is, of features that do not move rigidly with respect of the environment, remains an open problem. 2: In the absence of noise, an unknown location of a feature can be reconstructed if the point is visible in at least three more frames and if there are three more points that are visible in all four frames.**
- Statement 1 is correct & Statement 2 is correct
- Statement 1 is incorrect & Statement 2 is correct
- Statement 1 is correct & Statement 2 is incorrect
- Statement 1 is incorrect & Statement 2 is incorrect
* `[ Option A ]`


---

**Q: What is not a reason that outdoor images are harder to process than those produced in the controlled environment of a laboratory?**
- Lighting changes less predictable
- The motion of the camera is more difficult to control, resulting in images blurred by motion
- Because of image disparities that are caused by camera's jumps and jerks
- The resolution is a limiting factor
* `[ Option D ]`


---

**Q: WhatΓÇÖs an essential constraint for recovering the shape and motion from an image stream, when using the articleΓÇÖs proposed ΓÇÿfactorizationΓÇÖ method?**
- Perfect image (no noise)
- Orthographic projection
- Use of RANSAC
- No occlusions
* `[ Option B ]`


---

**Q: What is the rank of the registered measurement matrix $\tilde{W}$?**
- at most 2
- at most 3
- at most 4
- can't really say, as it also depends on noise conditions
* `[ Option D ]`


---

**Q: Which statement is true about the decomposition of the Measurement Matrix $\hat{W}$? **
- If $Q$ is any invertible 2x2 matrix, the matrices $\hat{R}Q$ and $Q^{-1}\hat{S}$ is a valid decomposition of $\hat{W}$.
- We can find the true rotation matrix $R$ and shape matrix $S$ by imposing metric constraints on $Q$ to find its values.
- Both A and B are correct.
- None of the above.
* `[ Option B ]`


---

**Q: The rank theorem states that the correlation among measurements made in image stream under orthography has a simple expression no matter:**
- the camera motion
- the shape of the object
- both the camera motion and the shape of the object
- none of the above
* `[ Option C ]`


---

**Q: Why can a good estimate of motion be reconstructed regardless of some noise in the input data?**
- Because noise does not affect the motion of the camera.
- The data can be cleaned easily.
- The many points available provide redudendant measures which can be exploited.
- Noise can be elimanted in a lab settings, outside of this the results are not very robust.
* `[ Option C ]`


---

**Q: Which of the following is false regarding occlusions in the structure-from-motion problem?**
- The measurement matrix is partially filled in.
- The factorization method cannot be directly applied.
- The measurement matrix cannot be built and an alternative data structure must be used.
- A feature should appear in enough, but not all, frames in order to be recoverable.
* `[ Option D ]`


---

**Q: The rank theorem states that the correlation among measurements made in an image stream under orthography has a simple expxression:**
- depending on the camera motion.
- depending on the shape of an object.
- depending on the camera motion and the shape of an object.
- independent of the camera motion nor the shape of an object.
* `[ Option D ]`


---

**Q: How does the proposed model deal with partial occlusion in the series of images?**
- Occluded points are ignored
- Frames with too many occluded/missing features are ignored
- An occluded point can be reconstructed using frame estimation
- An occluded point can be reconstructed using previous occurences in other frames and the constructed camera model
* `[ Option D ]`


---

**Q: What is NOT true about structure from motion (SfM)?**
- SfM recovers scene geometry and camera motion from a sequence of images.
- Finding structure from stereo vision is a similar problem as SfM.
- SfM is a strategy for 3D reconstruction from unordered image collections.
- In SfM camera poses are solved for and always added one by one to the collection.
* `[ Option D ]`


---

**Q: What does the rank theorem say, considering structure from motion?**
- Without noise, the registered measurement matrix is at most of rank three.
- Without noise, the measurement matrix is at most of rank three.
- Even with noise, the registered measurement matrix is at most of rank three.
- Even with noise, the measurement matrix is at most of rank three.
* `[ Option A ]`


---

**Q:  The factorization method can robustly recover shape and motion from a sequence of images under orthographic projection. What problem the factorization method does not solve?**
- The identification of false features
- Small movement problem (when camera moves just a bit)
- Noise in the image problem
- Offers solution the occlusion problem (when certain features disappears from images)
* `[ Option A ]`


---

**Q: What is wrong about the factorization method of shape and motion:**
- No matter what the camera motion is, the rank theorem make the image under orthography have a simple expression
- No matter what the shape of object is, the rank theorem makes the image under orthography have a simple expression
- The factorization method uses a long interframe camera motion to simplify feature tracking
- The rank theorem leads to a systematic procedure to solve the occlusion problem
* `[ Option C ]`


---

**Q: Which of the following statements is true?

I: The Factorization Method is an iterative process that can complete a previously partially empty measurement matrix
II: Application of The Rank Theorem reduces the effect of noise**
- Only statement I is correct
- Only statement II is correct
- Both statements are correct
- Neither statement is correct
* `[ Option C ]`


---

**Q: In absence of noise, a missing measurement can be hallucinated if there are atleast ____  more frames and ___________ points present in all 4 frames. **
- 3, 3
- 3,4
- 4,3
- 4,4
* `[ Option A ]`


---

**Q: Which of the following statement of the paper is not true?**
- An image stream can be represented by the 2FxP measurement matrix of the image coordinates of P points tracked through F frames.
- Under orthographic projection this matrix is of rank 4.
- Factorization method uses the singular-value decomposition technique to factor the measurement matrix into two matrices.
- The method gives accurate results, and does not introduce smoothing in either shape or motion.
* `[ Option B ]`


---

**Q: What is NOT the key problems towards building a truly general-purpose pipeline?**
- Applicable
- Robustness
- Accuracy
- Completeness
* `[ Option A ]`


---

**Q: What is the major reason that the camera-centred representation usually leads to unstable results?**
- The reconstruction is very sensitive to non-line perspective responses.
- Data points are not sufficient if only one camera is used.
- Reconstruction is made by examine subtle differences between samples, which is corrupted by noise.
- None of A, B or C
* `[ Option C ]`


---

**Q: Which of the following statements is false?**
- Features can appear and disappear from the image because of occlusions.
- The factorization method can robustly recover shape and motion from a sequence of images under orthographic projection. 
- The rank theorem states that under orthography the measurement matrix is 5. 
- The rank theorem is the basis of the factorization method.
* `[ Option C ]`


---

**Q: How can one determine the amount of noise in the measurement matrix?**
- Ratio of maximum and minimum eigenvalues obtained via SVD of measurement matrix
- Ratio of 3rd and 4th eigenvalues obtained via SVD of measurement matrix
- Norm of all eigenvalues values of measurement matrix
- None of the above
* `[ Option B ]`


---

**Q: What does the Condition for Reconstruction state? An unknown image point in one frame can be reconstructed ifΓÇª**
- the image point is visible in at least three other frames
- there are at least three other points that are visible in the same and three other frames
- noise is absent or neglectable 
- all of the above statements are true
* `[ Option D ]`


---

**Q: What is, according to the 'outline of the complete algorithm', not a step of the algorithm?**
- Singular value decomposition
- Compute rotation matrix R & shape matrix S
- Compute matrix Q, by imposing metric constraints
- Match features
* `[ Option D ]`


---

**Q: Two statements about the Rank Theorem. Statement 1: The registered measurement matrix $\tilde{W}$ can be expressed in matrix form: $\tilde{W} = RS$, where $R$ represents camera rotation and $S$ is the shape matrix. Statement 2: The Rank Theorem says that, in absence of noise, the registered measurement matrix $\tilde{W}$ is at most of rank three. **
- Statement 1 is true, statement 2 is false.
- Statement 1 is false, statement 2 is true.
- Both statements are true.
- Both statements are false.
* `[ Option B ]`


---

**Q: What surprising result do we obtain from the rank theorem?**
- The correlation among the measurements made in an image stream under orthography has a simple expression, no matter what the shape of the object and camera motion is
- It states that all transformation in image reconstruction are affine transformations
- We can obtain a surprisingly simple expression for image transformations in general
- The rank theorem allows is to efficiently exclude false matches between various points
* `[ Option A ]`


---

**Q: How can  the full motion and shape solution can
be found? **
- Column-wise extension: factor $W_{6x4}$ to find a partial
motion and full shape solution, and propagate it to
include motion for the remaining frame
- Row-wise extension: factor $W_{8x3}$ to find a full
motion and partial shape solution, and propagate it
to include the remaining feature point. 
- Column-wise extension: factor $W_{8x3}$ to find a full
motion and partial shape solution, and propagate it
to include the remaining feature point. 
- None of the above
* `[ Option C ]`


---

**Q: In SFM, if we have four 3D points and one of them is occluded in the one of the frames, how many measurements and frames do we need? **
- 3 more frames and 3 more corresponding points 
- 4 more frames and 4 more corresponding points 
- 3 more frames and 4 more corresponding points 
- None
* `[ Option A ]`


---

**Q: What is true about the distance between the scene and the camera?**
- The importance of this distance can be ignored since we work in world-coordinates, meaning the the average position is subtracted from all the coordinates. 
- For a correct reconstruction, this distance must be large compared to the camera movement.
- The importance of this distance can be ignored, as long as both camera vectors are orthogonal to the vector pointing from the camera to the scene.
- This distance must be larger than the width of the object that is imaged.
* `[ Option B ]`


---

**Q: What is the maximum rank of the measurement matrix W without noise?**
- 2
- 3
- 4
- Depending on the shape and size. 
* `[ Option B ]`


---

**Q: What is the assumption made by the Rank Theorem?**
- The origin of the world reference system is positioned at infinity.
- The origin of the world reference system is positioned at the centroid of the P points.
- The origin of the world reference system is positioned at a randomly selected point within the set of P points.
- The origin of the world reference system is positioned at the center of each frame.
* `[ Option B ]`


---

**Q: In order to retrieve the 3D coordinates of $P$ feature points detected in a set of $F$ images, a factorization method can be used. The measurement matrix $W$ (with shape $2F\times P$) holds the coordinates of all feature points in all images. To recover both the $3 \times P$ matrix $S$ with the 3D coordinates of the points and the $2F \times 3$ matrix $R$ with the camera rotations, $W$ needs to be factorized**
- using SVD, which will give $W = O_1DO_2$, where $D$ is a diagonal matrix with the ordered singular values of $W$. Then we extract the $3$ larger singular values and use those to build $S$ and $R$.
- into three matrices $L, D$ and $U$, respectively lower triangular, diagonal and upper triangular. Then $R = LD$, $S = U$.
- into $W = LW_- + RW_+$, where $W_-$ only has negative entries and $W_+$ only has positive entries. Then $R$ and $S$ are a transformation of $W_-$ and $W_+$.
- using Stochastic Gaussian Elimination, to find three random singular values of $W$ and then build $S$ and $R$ from those.
* `[ Option A ]`


---

**Q: In structure from motion problems,  when there are some points that are occluded in only some frames what can be done to solve the shape and motion recovery problem?**
- Run the interest point detection algorithm again in order to detect new interest points. 
- The shape and motion recovery is not possible when there are occluded points in the images.
- Apply an affine transformation to the occluded points before running the same and motion recovery algorithm.
- Exclude the occluded points in the calculations, as long as there are enough other points it will still work. 
* `[ Option D ]`


---

**Q: In the paper, "Shape and Motion from Image Streams under Orthography: a Factorization method", the authors state reasons why  shape estimation from a camera centered representation can be difficult and noise sensitive. Which of the following is NOT a possible reason**
- For small camera motions, it can be difficult to distinguish between effects of translation and rotation which makes shape reconstruction difficult and noise sensitive
- For images very far from the centre of projection, computation of shape estimation as relative depth can be very noise sensitive since the depth difference between top and bottom of an object is small relative to the values
- Determining the structure from a stream of images requires the elimination of ambiguity which is itself sensitive to noise
- None of the above
* `[ Option C ]`


---

**Q: Which of the following statements is NOT true regarding the factorization method?**
- Registered measurement matrix W^~ is highly rank-deficient under orthography.
- Registered measurement matrix W^~ would also be of rank 3 even with noise using approximate rank.
- Factorization method could be used to solve occlusion problem in shape and motion from Image streams.
- Factorization method does not show robustness in the out-door experiments.
* `[ Option D ]`


---

**Q: What is TRUE about the ranking theorem:**
- With noise the measurement matrix W is at most of rank 3
- Without noise the measurement matrix W is at least of rank 3
- If the measurement matrix contains approximated values (based on at least three frames) of points not visible in a certain frame, the rank is at most 3
- Without noise the registered measurement matrix $\~W$ is at most of rank 3
* `[ Option D ]`


---

**Q: What the main reason for singular value decomposition in affine structure from motion?**
- Because of noisy measurements
- To speed up calculation
- To eliminate affine ambiguity
- To find a subset of inlier data points
* `[ Option A ]`


---

**Q: What are the outcomes of the factorisation method and what are its benefits?**
- Shape and motion; Introduce no smoothing in shape and motion.
- Missing features; Recovered images
- Missing features; Model tracking to recover missing features
- Shape and motion; Obtain a partial replica of the original image 
* `[ Option A ]`


---

**Q: What is NOT a reason why using camera-centered representation for shape estimation difficult, unstable and noise sensitive?**
- When camera motion is small, effects of camera rotation and translation can be confused with each other
- Because the scale of an object is unknown
- Because the computation of shape as relative to depth is noise sensitive as it is a small difference between large values
- None of the above
* `[ Option B ]`


---

**Q: In the absence of noise, an unknown image measurement pair \[ (u_fp, v_fp) \] in frame f can be reconstructed if...**
- The point p is visible in at least three more frames and three more points are visible in all those frames as well as in the original frame.
- The point p is visible in at least two more frames and three more points are visible in all those frames as well as in the original frame.
- The point p is visible in at least one more frames and three more points are visible in all those frames as well as in the original frame.
- The point p is visible in at least three more frames and two more points are visible in all those frames as well as in the original frame.
* `[ Option A ]`


---

**Q: How does the factorization method approach noise in the measures?**
- Taking the only the 3 greatest singular values in the factorization (ignoring the rest) since the rank of the measurement matrix is 3
- When noise is present, one needs to get more measures for the factorization method to work
- Noise is not approached yet
- None of the above
* `[ Option A ]`


---

**Q: Which of the following description is correct?**
- Camera rotation and translation do not always make confusion when the motion is small
- The computation of shape as relative depth is always not very sensitive to noise
- Factorization method which cannot robustly recover shape and motion from a sequence of images under orthographic projection
- None of above is correct
* `[ Option D ]`


---

**Q: Suppose there is a feature disappeared due to the occlusion, if we want to reconstruct the object, what conditions should we meet?**
- this point should be visible at least 3 more other frames, with 3 more other feature points
- this point should be visible at least 2 more other frames, with 3 more other feature points
- this point should be visible at least 3 more other frames, with 2 more other feature points
- this point should be visible at least 2 more other frames, with 2 more other feature points
* `[ Option A ]`


---

**Q: Why some of the originally detected features might be abandoned**
- They are in dark areas of the image
- They are in bright areas of the image
- They are on the corners
- Their appearance changes two much between frames
* `[ Option D ]`


---

**Q: A matrix W is created from measurements, that contains no noise, under an orthography projection. What is the rank of matrix W? **
- 1
- 2
- 3
- 4
* `[ Option C ]`


---

**Q: What happens to the measurement matrix M when features appear and disappear due to occlusion or tracking failures?**
- M is only partially filled
- The measurement matrix rank will not be exactly 3
- It does not affect the application of factorization method
- None of the above
* `[ Option A ]`


---

**Q: What is the rank of the measurement matrix when one assumes orthogonality?**
- 3
- 5
- 7
- 1
* `[ Option A ]`


---

**Q: What will be improved with the rank theorem?**
-  robust and fast calculations
- less errors is calculation
- Do filtering on 3D image points
- Non of the above 
* `[ Option A ]`


---

**Q: What is a sufficient condition for reconstruction in the absence of
noise when given an unknown image measurement pair?**
- A frame f can be reconstructed if point p is visible in at least three more frames and if there are at least three more points that are visible in all four frames. 
- A frame f can be reconstructed if point p is visible in at least two more frames and if there are at least three more points that are visible in all three frames. 
- A frame f can be reconstructed if point p is visible in at least three more frames and if there are at least two more points that are visible in all four frames. 
- A frame f can be reconstructed if point p is visible in at least one more frame and if there is at least one more points that is visible in both frames. 
* `[ Option A ]`


---

**Q: Which camera effects are ignored by orthographic projection?**
- Camera translation along the optical axis
- Camera translation along one of the image plane axes
- Camera rotation along the optical axis
- Camera rotation along one of the image plane axes
* `[ Option A ]`


---

**Q: Which of the following is/are correct about the rank theorem?

1. The rank theorem states that without noise, the registered
measurement matrix is at most of rank three. 
2. The only difference between the rank theorem and the rank theorem for noisy measurements, is the difference in rank.**
- Only 1 is correct
- Only 2 is correct
- Both are correct
- Both are incorrect
* `[ Option A ]`


---

**Q: Why the measurement matrix W needs to be approximated to have rank 3 when noise is present?**
- Because in this case the estimate for the shape and rotation matrices is the best one.
- Because noise is eliminated.
- Because the first 3 eigenvalues contain the most important information.
- All of the above.
* `[ Option D ]`


---

**Q: Which of these statements is incorrect?**
- Rank theorem leads to factorization of the measurement matrix into shape and motion in a well-behaved and stable manner
- The factorization method exploits the redundancy of the measurement matrix to counter noise sensitivity of structure-from-motion
- The factorization method allows using very short interframe camera motion to simplify feature tracking
- The rank theorem is hard to use because it states that the correlation among measurements made in an image stream under orthography does not have a simple expression
* `[ Option D ]`


---

**Q: How does the author solve the presence of noise?**
- A possibly large, full subblock of W is first decomposed by factorization
- Gaussian blur
- Use more trainning data
- Change to a more complex architecture
* `[ Option A ]`


---

**Q: As the camera moves, features can appear and disappear from the image because of occlusions. Suppose that a feature point is not visible in the current
frame, to recover its position in space we need**
- 3 keypoints which are visible in the current frame, as well as other frames.
- Corresponding camera position
- Motion sequence of camera
- Any of the above
* `[ Option D ]`


---

**Q: According to rank theorem what is the rank of measurement matrix represented  by 2F x P (P points tracked through F frames)?**
- P
- F
- 2F
- 3
* `[ Option D ]`


---

**Q: What is NOT true about the registered measurement matrix?**
- The entries are normalized by their row means
- A row represent one frame
- A column represent one vertical point
- The rank is 3 under no noise
* `[ Option C ]`


---

**Q: Tomasi and Kanada's Shape and Motion from Image streams under orthography uses:

I. Factorization Method
II. Rank Theorem**
- I
- II
- Both
- Neither
* `[ Option C ]`


---

**Q: Which one is wrong?**
- The factorization method uses the singular-value decomposition technique to factor the measurement matrix into two matrices which represent object shape and camera rotation respectively.
- The method gives accurate results, and introduce smoothing in either shape or motion. 
- The paper prove that the rank theorem, which is the basis of the factorization method, is both surprising and powerful.
- The theorem proposed in this paper is powerful because the rank theorem leads to factorization of the measurement matrix into shape and motion in a well behaved and stable manner. 
* `[ Option B ]`


---

**Q: 0**
- 0
- 0
- 0
- 0
* `[ Option A ]`


---

