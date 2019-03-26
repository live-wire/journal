# Questions from `lecture_4` :robot: 

**Q: How many bins are needed to represent each pixel of a 4 x 4 subwindow?**
- 8
- 12
- 24
- 36
* `[ Option A ]`


---

**Q: What size does W need to be to factorize matrix D?**
- It depends on the amount of cameras
- 3x1
- 3x2
- 3x3
* `[ Option D ]`


---

**Q: What is true about the projective 15dof type of motion ambiguity which is described by the matrix $\begin{bmatrix}
A & t \\
v^T & v
\end{bmatrix}$ ?**
- It preserves intersection and tangency.
- It preserves parallelism and volume ratios
- It preserves angles and ratios of length
- It preserves angles and lengths
* `[ Option A ]`


---

**Q: What is true about the affine 12dof type of motion ambiguity which is described by the matrix $\begin{bmatrix}
A & t \\
0^T & 1
\end{bmatrix}$ ?**
- It preserves intersection and tangency
- It preserves parallelism and volume ratios
- It preserves angles and ratios of length
- It preserves angles and lengths
* `[ Option B ]`


---

**Q: What is the name of the optical center?**
- Center of projection
- Ray point
- Center of equation
- Pinhole
* `[ Option A ]`


---

**Q: Which of the four types of ambiguity does not need constraints on the camera matrix or scene?**
- Similarity
- Projective
- Euclidean
- Affine
* `[ Option B ]`


---

**Q: Which of the following statements is true?**
- We cannot approximate the Laplacian of Gaussian with a difference of Gaussian (DoG)
- Each octave has 8 images
- An ideal SIFT detector is not invariant to illumination
- The Laplacian is used for interest point detection
* `[ Option D ]`


---

**Q: Bundle adjustment is a _____ method that is used to ____ error in reprojection?**
- Linear, minimize
- Non-linear, minimize
- Linear, maximize
- Non-linear, maximize
* `[ Option B ]`


---

**Q: Which of the following is not deemed to be a desirable property of a SIFT image descriptor?**
- scale invariant
- illumination invariant 
- viewpoint invariant 
- resize invariant
* `[ Option D ]`


---

**Q: Review the following two statements about structure from motion ambiguity:
\begin{enumerate}
    \item Projective ambiguity preserves the intersections and tangency
    \item Similarity ambiguity preserves angles and lengths of sides.
\end{enumerate}
Which of the statements are true?**
- Statement 1 and 2 are true
- Statement 1 is true and statement 2 is false
- Statement 1 is false and statement 2 is true
- Statement 1 and 2 are false
* `[ Option B ]`


---

**Q: Running the Structure from Motion algorithm the "measurement matrix"  D is introduced. It is proven that with negligible noise its rank is 3, so singular value decomposition is used to rewrite it as D=U*W*V' . What is matrix W?**
- It contains the left eigenvectors of matrix D
- on its rows we can find the eigenvalues of D, sorted in decreasing order
- On its diagonal we can find the singuar values of D
- It is a matrix having the same dimensions of D
* `[ Option C ]`


---

**Q: Using the SIFT procedure we refer to a scale space for highlighting keypoints in the image.  What is it?**
- It is a collection images, that are the results of blurring the same starting image with gaussian filters having different sigmas
- It is the algorithm by which we can detect the positions of the kepoints in the image grid
- It is the result of covolving the initial image with different normalized laplacians of gausssians
- It is a bunch of images containing the x and y derivatives of the initial image
* `[ Option A ]`


---

**Q: What is the dimensionality of the SIFT descriptor?**
- 8
- 16
- 128
- 256
* `[ Option C ]`


---

**Q: If a transformation Q is necessary on the image scene for some algorithm, what should be applied to the camera matrices in order for the image to stay the same?**
- a) the transpose of transformation Q
- b) the inverse of transformation Q
- c) the square of transformation Q
- d) the square root of transformation Q
* `[ Option B ]`


---

**Q: Which of the following transformations has the largest ambiguity in structure from motion?**
- Euclidean
- Projective
- Affine
- Similarity
* `[ Option B ]`


---

**Q: Which of the following claims is false about structure from motion?**
- The singular value decomposition (SVD) is not unique
- In orthographic projection, we enforce the constraints of image axes being perpendicular and scale being 1
- to remove translation, we need to perform singular value decomposition (SVD)
- Not all points are visible from all views
* `[ Option C ]`


---

**Q: Which of the following is true about the window size of 16x16 window for SIFT descriptors?**
- The window size is always constant
- The window size depends on the corresponding scale of the feature point
- The window size depends on the resolution of the image
- None of the above
* `[ Option B ]`


---

**Q: Which of the following matrix represents parallel projection**
- [u v 1]ΓÇÖ = [1 0 0 0; 0 1 0 0; 0 0 0 0; 0 0 0 1] * [x y z 1]ΓÇÖ
- [u v 1]ΓÇÖ = [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1] * [x y z 1]ΓÇÖ
- [u v 1]ΓÇÖ = [1 0 0 0; 0 1 0 0; 0 0 0 0; 0 0 1/f 0] * [x y z 1]ΓÇÖ
- [u v 1]ΓÇÖ = [1 0 0 0; 0 1 0 0; 0 0 1/f 0; 0 0 0 1] * [x y z 1]ΓÇÖ
* `[ Option A ]`


---

**Q: In the SIFT descriptor we have a 128-dimensional descriptor because we consider**
- 16 subwindows, each one with a 8-bins histogram
- 4 subwindows, each one with a 32-bins histogram
- 8 subwindows, each one with a 16-bins histogram
- Half of a full RGB color scale (0-255)
* `[ Option A ]`


---

**Q: The orhographic projection is a special case of perspective projection in which the distance between COP and image plane is**
- 0
- Twice the focal distance
- Infinite
- None of the above
* `[ Option C ]`


---

**Q: How many degrees of freedom does an Euclidean transformation matrix have?**
- 6
- 5
- 4
- 8
* `[ Option A ]`


---

**Q: Which of the below transformations is not affine?**
- Identity.
- Translation.
- Rotation.
- Perspective.
* `[ Option D ]`


---

**Q: Regarding Structure from Motion, how what should be the rank of the measurement matrix D?**
- The same as the number of measurement points
- The same as the number of cameras
- Twice the number of cameras
- 3
* `[ Option D ]`


---

**Q: Regarding Structure from Motion, how can you enforce the correct rank of the D matrix?**
- Only regard the number of points equal to the desired rank
- Only regard the number of cameras equal to the desired rank
- Take the singular value decomposition of D and take the first 3 rows/columns, and set the rest to 0
- You cannot simply enforce a matrix to have a certain rank
* `[ Option C ]`


---

**Q: what is a SIFT descriptor?**
- A property of a corner point
- A rotationally invariant property of a corner point
- A 128 dimensional vector
- All of the above
* `[ Option D ]`


---

**Q: Why can a difference of two gaussians be used to substitute a laplacian**
- Because there is only a constant difference
- Because there is only a (non-constant) linear difference
- Because there is only a (non-constant, non-linear) polynomial difference
- None of the above
* `[ Option B ]`


---

**Q: Why would you scale-normalize a Laplacian response in the first place?**
- To make the matching of different points scale invariant
- To make the matching of different points viewpoint invariant
- To make the matching of different points illumination invariant.
- None of the above
* `[ Option A ]`


---

**Q: What should be the rank of the measurement matrix D and how achieve this rank?**
- The rank of measurement matrix D should be 3 and this is achieved through setting all the singular values to 0 except for the first 3.
- The rank of measurement matrix D should be 3 and this is achieved through reducing the side of D to a 3x3 matrix.
- The rank of measurement matrix D should be 2 and this is achieved through setting all the singular values to 0 except for the first 2.
- The rank of measurement matrix D should be 2 and this is achieved through reducing the side of D to a 2x2 matrix.
* `[ Option A ]`


---

**Q: What is the rank that the measurement matrix D = MS must have?**
- Rank 1
- Rank 2
- Rank 3
- Rank 4
* `[ Option C ]`


---

**Q: What is FALSE about orthographic projection?**
- It is also called 'parallel projection'
- The focal length f and the object depth coordinate z are infinite
- Orthographic projection can be used for model drawings, because it does not suffer from projection warpings
- The projection matrix has all 1's on the diagonal starting from the top left
* `[ Option D ]`


---

**Q: Whats the dimension of the resulting vector containing the directions of the SIFT descriptor?**
- 8
- 16
- 128
- 256
* `[ Option C ]`


---

**Q: When performing an affine structure from motion, the D matrix in the D=MS equation has 2*m rows and n columns (for m cameraΓÇÖs and n points). Why is that?**
- For every point, there is an x and y coordinate (or u and v) in the cameraΓÇÖs coordinates
- Because thereΓÇÖs 2 cameras needed for SfM
- Because of homogeneous coordinates
- For the Singular Value Decomposition
* `[ Option A ]`


---

**Q: The first two rows of matrix $D$ are $[1,2,3,4,5; 6,7,8,9,10]$. Estimate the centered feature coordinates for the given block of $D$. **
- $[-2, -1, 0, 1, 2; 2, 1, 0, -1, -2]$
- $[-2, -1, 0, 1, 2; -2, -1, 0, 1, 2]$
- $[-4.5, -3.5, -2.5, -1.5, -0.5; 0.5, 1.5, 2.5, 3.5, 4.5]$
- $-2.5, -2.5, -2.5, -2.5, -2.5; 2.5, 2.5, 2.5, 2.5, 2.5$
* `[ Option B ]`


---

**Q: Given 10 images of 15 fixed 3D points, what will be the size of the motion matrix $M$?**
- $10\times 3$
- $3 \times 10$
- $20 \times 3$
- $20 \times $
* `[ Option C ]`


---

**Q: Which of the following is part of the SIFT algorithm?**
- In each octave, images are blurred using an Average Kernel, which is proven to not introduce new details under certain assumptions.
- For each blurred image per octave, the Laplacian is computed. Interest points lie at local maxima and minima of the resulting images.
- After computing a histogram of orientations of the keypoints, the maximum bin is selected and the other bins are discarded, no matter how big they were.
- None of the above.
* `[ Option D ]`


---

**Q: Why is it impossible to recover the absolute scale of the scene using the Structure from Motion as described in the lecture?**
- Because the Projection Plane is located infinitely far away from the Center of Projection.
- Because of the ambiguity of the absolute scale; scaling of the scene can be undone by scaling the camera by $\frac{1}{scale}$.
- We need matching points from multiple camera's, which is unrealistic under the Structure from Motion model.
- None of the above.
* `[ Option B ]`


---

**Q: What is the window size around each keypoint for SIFT?**
- 4x4
- 10x10
- 16x16
- 100x100
* `[ Option C ]`


---

**Q: What is the dimensionality of a SIFT descriptor?**
- 16
- 32
- 64
- 128
* `[ Option D ]`


---

**Q: Which assumptions can be used to resolve affine ambiguity?**
- a1 ┬╖ a2 = 0, |a1|^2 = |a2|^2 = 1
- a1 ┬╖ a2 = 1, |a1|^2 = |a2|^2 = 0
- a1 ┬╖ a2 = 1, |a1|^2 = |a2|^2 = 1
- a1 ┬╖ a2 = 0, |a1|^2 = |a2|^2 = 0
* `[ Option A ]`


---

**Q: How are SIFT descriptors used in image point matching? **
- The image statistics can be used to directly compute a transformation from one image to the other.
- SIFT computes image features that can be used to compare points from the images.
- SIFT descriptors estimate the rotational angle between two points.
- The SIFT algorithm dynamically matches points best on the 'best-for-fit' criterion.
* `[ Option B ]`


---

**Q: To achieve scale-invariance in a corner detector a possible method is:**
- Use Difference of Gaussians to compare 2 images with different scales.
- Work on the ΓÇÿscaleΓÇÖ dimension by computing responses with different sigmas and the selecting the maximum of these. 
- Optimize a transformation of the image such that scale-related outliers are removed.
- Blur the image with a gaussian filter with a large sigma so all the finer details are coarser.
* `[ Option B ]`


---

**Q: In a structure-from-motion exercise there are 15 cameras and 40 matching points were found between them. Taking this into account which is the dimension of the measurements matrix D: **
- 30 by 40
- 15 by 40
- 15 by 80
- 30 by 80
* `[ Option A ]`


---

**Q: In the context of Descriptors Matching, if $d_2^*$ is the best match and $d_2^{**}$ is the second best match from image 2, for $d_1^i$ from image 1, which formula describes best what matches we want to keep? ($\theta$ is a threshold between 0 and 1)**
- $\frac{dist(d_1^i, d_2^*)}{dist(d_1^i, d_2^{**})} < \theta$
- $\frac{dist(d_1^i, d_2^**)}{dist(d_1^i, d_2^{*})} < \theta$
- $dist(d_1^i, d_2^*) - dist(d_1^i, d_2^{**}) < \theta$
- $dist(d_1^i, d_2^{**}) - dist(d_1^i, d_2^{*}) < \theta$
* `[ Option A ]`


---

**Q: How do we reduce the rank of the matrices that are decomposed from factorizing a measurement matrix to 3?**
- Just set all the singular values to zero except the last 3.
- Just set all the singular values to zero except the first 3.
- By doing a factorization trick.
- Just make the measurement matrix smaller.
* `[ Option B ]`


---

**Q: How are edges filtered out from keypoint descriptors?**
- By thresholding the average energy of the keypoints
- By comparing it with its matched pair
- By applying the harris detector
- By looking at the pixel gradients
* `[ Option C ]`


---

**Q: Why is it allowed to reduce the rank of the SVD of the measurement matrix to 3?**
- The rest of the values represent the noise in the measurements
- The removed section is preserved in the other parts of the decomposition
- The decomposition matrices are diagonal
- Is does not matter since SVD already gives the fitting output
* `[ Option A ]`


---

**Q: What is NOT a property of corner points?**
- Orientation
- Magnitude
- Scale
- Position
* `[ Option B ]`


---

**Q: Which kernel constructs a scale space without spurious resolution?**
- Radial basis kernel
- Gaussian kernel
- Laplacian kernel
- Box kernel
* `[ Option B ]`


---

**Q: What is the size of the sift descriptor?**
- 32
- 64
- 128
- 256
* `[ Option C ]`


---

**Q: Is it possible to recover scale from a reconstruction with the structure from motion technique?**
- Yes
- No
- Only if a reference dimension is known.
- Only if 3 reference dimensions are known.
* `[ Option C ]`


---

**Q: Consider Orthographic and Perspective projections. Which statement is true?**
- Orthographic projection in contrary to perspective projection does not take perspective into account
- Perspective projection in contrary to Othographic projection does not take perspective into account.
- Both projections take perspective into account, but perspective distortion is different in each of them.
- None of the projections take perspective into account.
* `[ Option A ]`


---

**Q: Consider the situation where we would like to rotate the image. Lets asume we can apply transofrmation for both - camera and image. Which statement is false?**
- In order to correctly rotate the image we should apply trasformation T to camera and T inverse for image
- In order to correctly rotate the image we should apply transformation T to camera only
- In order to correctly rotate the image we should apply transformation T to picture only
- Normalize data before passing it through the network and normalize only on the output layer
* `[ Option A ]`


---

**Q: which rank does the measurement matrix D must have**
- 2
- 3
- 4
- 5
* `[ Option B ]`


---

**Q: Which of following is false for the dealing with measurement matrix missing data?**
- decompose matrix into dense sub-blocks
- factorize each sub-block
- fuse the factorized results
- none is false
* `[ Option D ]`


---

**Q: How do SIFT descriptors realise rotation invariance?**
- the coordinates of the descriptor and the gradient orientations are rotated relative to the keypoint orientation
- the orientation of both the descriptor and the keypoint are made equal by aligning them with the direction of the principal axis of orientation of the gradient 
- During the matching phase, each of the descriptor vector of the second picture is compared to the descriptor vector in the first picture in each of the possible 128 ways
- All descriptor vectors start with their biggest value, in that way they are always lined up correctly
* `[ Option A ]`


---

**Q: Which of the following statements is true?

I: With structure from motion techniques, the absolute scale of a scene can be recovered
II:  With similarity ambiguity, there are 7 degrees of freedom**
- Only statement I is correct
- Only statement II is correct
- Both statements are correct
- Neither statement is correct
* `[ Option B ]`


---

**Q: For key-point orientation if the 36 binned histogram has 2 values over the threshold what would be the correct choice ?**
- Assign the key-point with orientation corresponding to the higher bin 
- Assign the key-point with orientation corresponding to the lower bin
- Split the key-point into 2 key-point assigning one orientation to each of the 2 key-point 
- Reject the key-point
* `[ Option C ]`


---

**Q: Which of the following defines affine projection in the best way ?**
- linear mapping
- non-linear mapping
- linear mapping with translation
- non linear mapping with translation
* `[ Option D ]`


---

**Q: Which of the follow degree of freedom is not true?**
- Projective 14dof
- Affine 12dof
- Similarity 7dof
- Euclidean 6dof
* `[ Option A ]`


---

**Q: Which of the following precedure is not included in the structure from motion algorithm?**
- Construct a 2m*n measurment matrix D
- Factorize D
- Create only the shape matrices
- Eliminate affine ambiguity
* `[ Option C ]`


---

**Q: Which of the following is not the desirable characteristic according to SIFT?**
- Scale invariance
- Similarity invariance
- Rotation invariance
- Illumination invariance
* `[ Option B ]`


---

**Q: Which function relationship does transformation (x,y,z) -> (xf/z, yf/z)belong to?**
- linear
- linear after division by z
- nonlinear
- non-linear before division by z
* `[ Option C ]`


---

**Q: For orthographic projection, if the object has a distance L away from the image plane, then what is the distance from the centre of projection to the image plane?**
- 2L
- L
- L/2
- Infinite
* `[ Option D ]`


---

**Q: Imaging you would like to reconstruct the relative positions of 10 points in 3D space in terms of affine structure from motion with 3 cameras, what is the dimension of the measurement matrix?**
- 3x10
- 3x20
- 6x10
- 6x20
* `[ Option C ]`


---

**Q: Which of the statements regarding to the types of ambiguity is false?**
- Projective ambiguity preserves intersection and tangency.
- Affine ambiguity preserves parallelism and volume ratios.
- Similarity ambiguity preserves ratios of volume and length.
- Eucledian ambiguity preserves angles and length.
* `[ Option C ]`


---

**Q: Which of the following statements is false?**
- A possible solution to deal with missing data is to decompose the matrix in sub-blocks, factorize each sub-block and fuse the results.
- Bundle adjustment is a linear method for refining structure and motion.
- A general affine camera combines the effects of an affine transformation of the 3D space, orthographic projection and an affine transformation of the image.
- Affine projection is a linear mapping and translation in inhomogeneous coordinates.
* `[ Option B ]`


---

**Q: Which type of ambiguity do we have to deal while reconstructing a scene without any constraints ?**
- Projective
- Affine
- Similarity
- Euclidean
* `[ Option A ]`


---

**Q: Why are edges among uninformative keypoints for matching images?**
- They are not distinctive enough for matching
- They are computationally expensive to matching
- Both of the above reasons are valid
- They are actually very informative for matching
* `[ Option A ]`


---

**Q: How can we find matching points using descriptors?**
- By comparing the values of the descriptors of the first image to the values of the descriptors of the second image
- By computing the Euclidian distance of the descriptors of the first image to all descriptors of the second image and comparing the ratio of the two best matches to a threshold
- By taking the ratio between the maximum value of the descriptors of the first image and the second largest value of the descriptors of the second image and comparing it to a threshold
- By computing the Euclidian distance of the descriptors of the first image to all descriptors of the second image and keeping only those points which best match is larger than the second-best match
* `[ Option B ]`


---

**Q: The last step of the proposed Structure from Motion (SfM) algorithm is the elimination of the affine ambiguity. How could this be achieved? ($a_{1}$ and $a_{2}$ are the image axes)**
- $a_{1} \cdot a_{2} = 0$ and $|a_{1}|^2 = |a_{2}|^2 = 1$
- $a_{1} \cdot a_{2} \neq 0$ and $|a_{1}|^2 = |a_{2}|^2 = 1$
- $a_{1} \cdot a_{2} = 0$ and $|a_{1}|^2 \neq |a_{2}|^2 \neq 1$
- $a_{1} \cdot a_{2} \neq 0$ and $|a_{1}|^2 \neq |a_{2}|^2 \neq 1$
* `[ Option A ]`


---

**Q: How are SIFT descriptors matched?**
- By looking at the euclidean distance between the keypoints
- By looking at descriptors with the same scale
- By looking at the magnitude of the gradient
- By looking at the euclidean distance to all descriptors in the other image
* `[ Option D ]`


---

**Q: What is the difference between the 'octaves'? **
- The amount of pixels in each image
- The amount of images
- The color-band looked at
- The kernel used
* `[ Option A ]`


---

**Q: Which statement about SIFT is false?**
- SIFT computes the Laplacian of Gaussian.
- SIFT detects points that are local extrema with respect to both space and scale.
- SIFT matches features using Euclidean distance.
- In SIFT, each octave consists of 5 progressively blurred images.
* `[ Option A ]`


---

**Q: Which statement about orthographic projection is false?**
- Orthographic projection is a special case of perspective projection.
- Orthographic projection is a form of parallel projection.
- Orthographic projection is a commonly used in technical drawings.
- In orthographic projection, the distance from the center of projection to the image plane is finite.
* `[ Option D ]`


---

**Q: We use descriptors to match different points between images. Now, if we consider a 16x16 window divided in subwindows of 4x4 and 8 bins, what is now the dimension of the descriptor vector?**
- 128
- 256
- 64
- 32
* `[ Option A ]`


---

**Q: Which properties of each pixel do we need to construct the descriptor?**
- Magnitude and orientation
- Magnitude and color
- Color and orientation
- None of the above
* `[ Option A ]`


---

**Q: Which of the following steps is wrong in the SIFT Scale Space? **
- Take the original image and progressively blur it $5x$ with increasing $\sigma$.
- Resize the image to its half and repeat the process
- Each "octave" has 5 image.
- Each image in an octave has the same ΓÇ£scale" (amount of blur)
* `[ Option D ]`


---

**Q: Why is it important that the measurement matrix have rank 3?**
- Rank 3 provides redundancy, in case of occlusions the features could be refactored.   
- Permits large number of points and frames to be processed in a conceptually simple and computationally efficient way to reduce the effects of noise
- An iterative procedure can find a complete solution in case the features appear and disappear from successive frames  
- All of the above
* `[ Option D ]`


---

**Q: In Structure from Motion,  we create a dataset matrix. What are its dimensions and why assuming we have two points in 3D and we have 3 cameras with corresponding points for both the points**
- D = [4,3]
- D = [3,4]
- D = [2,3]
- None
* `[ Option A ]`


---

**Q: Once SIFT feature coordinates are found, we need to orient them as well. If the resulting histogram of pixel gradients has more than 1 value over the pre-determined threshold, what should we do? **
- Give that coordinate the orientation corresponding to the bin with maximum magnitude
- Give that coordinate the orientation corresponding to the bin with minimum magnitude
- Make multiple keypoints with the same coordinate and assign each of them the magnitude of the histogram bin above the predetermined threshold as its orientation
- None of the above
* `[ Option C ]`


---

**Q: Why do we reduce the decomposed matrices in SfM to rank 3?**
- Otherwise the matrices are too large and sparse, so computations become lengthy and inefficient
- Otherwise they are not SPD (semi-positive definite), which would make the Cholesky decomposition impossible
- Othewise we do not end up with 3D coordinates for all the points in the 3D-scene. 
- To reduce the noise present in the images from camera movement
* `[ Option D ]`


---

**Q: Two statements on why we can assume an orthographic projection for large distance: 
	1) 1/z is low, meaning that for large distance the image is small
	2) for a long distance from the object, the projective perspective is neglectible**
-  1 = false, 2 = true
- 1 = true, 2 = false
- both false
- both true
* `[ Option D ]`


---

**Q: From all the types of ambiguities, what is the name of the  one who preserves the angles and lengths?**
- euclidian 
- similarity
- affine
- projective
* `[ Option A ]`


---

**Q: How is it calculated the measurement matrix D?**
- D=MxS where M is the motion and S is the scale
- D=MxS where M is the number of Measurements and S is the shape
- D=MxS where M is the number of camera orientations and S is the scale
- D=MxS where M is the motion and S is the shape
* `[ Option D ]`


---

**Q: The SIFT descriptor guarantees that the found key points are specified in stable 2D coordinates, meaning that they can be found in two distinct images independent of:**
- Their location within each of the images (x and y coordinates).
- Scale.
- Orientation.
- All of the above.
* `[ Option D ]`


---

**Q: The goal of structure from motion technique is to obtain:**
- The motion and shape matrices.
- The measurements and shape matrices.
- The shape matrix and the 3D location of the points.
- The measurements and the motion matrices.
* `[ Option A ]`


---

**Q: If $X$ refers to a set of 3D points and $x$ refers to the same set of points, but captured by a camera, then the camera matrix $P$ is such that $x = PX$. When we have the set $x$ and want to recover the 3D positions of the points, there is some ambiguity in finding $P$ and $X$, because**
- if $Q$ is some invertible transformation and $P, X$ is a solution of $x = PX$, then setting $P' = PQ^{-1}$ and $X' = QX$ gives that $x = P'X'$ is also a solution.
- if $Q$ is some invertible transformation and $P, X$ is a solution of $x = PX$, then setting $P' = QP$ and $X' = XQ^{-1}$ gives that $x = P'X'$ is also a solution.
- the problem is numerically unstable as the number of cameras increases.
- the algorithm to find $P$ and $X$ assumes the measurements and matches are perfect and have no noise.
* `[ Option A ]`


---

**Q: To eliminate the ambiguity that is present when recovering 3D positions of feature points in several images, further restrictions on the setting of the algorithm are imposed,**
- such as making the axes orthonormal.
- such as preventing a point to be used in more than 3 images.
- such as assuming the pictures were taken at a fixed distance $D$ from the object.
- such as setting the norm of each 3D output point to 1.
* `[ Option A ]`


---

**Q: Which of the following is not part of the SIFT feature detection method?**
-  downsampling of original image
-  discarding low contrast keypoints
-  discarding key points located on the edges
-  linearly transforming the image and recalculating the keypoints
* `[ Option D ]`


---

**Q: Given the 3D point (x=1, y=2, z=3) what will be the coordinates of the point projected onto a projection plane with distance f=2 from the COP?**
-  x=0.5, y=0.6 z=2
-  x=0.4, y=0.8 z=2
-  x=0.8, y=0.4 z=2
-  x=-0.4, y=0.8 z=-2
* `[ Option B ]`


---

**Q: In the SIFT algorithm which of the following is NOT a valid feature descriptor**
- First Order Gradient magnitude of the image
- First Order Gradient orientation of the image
- Pixel neighbour RGB values
- None of the above
* `[ Option C ]`


---

**Q: A general affine camera combines all the following effects in image generation EXCEPT**
- Affine transformation of the 3D space
- Orthographic projection
- Affine transformation of the image
- None of the above
* `[ Option D ]`


---

**Q: On which situation a projection could be think of a linear transformation?**
- The object is close to the camera.
- The object is still.
- The object is relatively large.
- The object is far from the camera.
* `[ Option D ]`


---

**Q: Which of following statements is NOT a type of structure from motion ambiguity?**
- Projective ambiguity.
- Affine ambiguity.
- Similarity ambiguity.
- None of above.
* `[ Option D ]`


---

**Q: Select the TRUE statement:**
- Images in the same octave have the same amount of blur
- Images in the same octave have progressively increasing sizes
- The number of octaves depends on the size of the original image 
- The increasing scale is independent of the original image size
* `[ Option C ]`


---

**Q: What type(s) of ambiguity preserves angles?**
- Projective
- Affine
- Similarity
- All of the above
* `[ Option C ]`


---

**Q: In an orthographic projection, which input dimension is ignored?**
- X
- Y
- Z
- Homogeneous coordinate
* `[ Option C ]`


---

**Q: What is not a type of ambiguity?**
- Projective ambiguity
- Affine ambiguity
- Similarity ambiguity
- Distortional ambiguity
* `[ Option D ]`


---

**Q: When the image plane is far away from the COP, what effects will you observe?**
- You see the size of projected images changes almost linearly as you move
- The projected image is almost a parallel projection of the object
- You see the size of projected images changes exponentially as you move
- Both A and B
* `[ Option D ]`


---

**Q: What can be reconstructed of the original object from Motion?**
- Scale
- Parallelism
- Volume ratio
- All of the above
* `[ Option D ]`


---

**Q: How do we normalize the descriptor in SIFT?**
- By taking the average
- By dividing by the largest value
- By dividing by the mean of all descriptors
- by dividing by its mean
* `[ Option D ]`


---

**Q: Why can we only recover a projective reconstruction when we do not constrict the camera matrix or the scene?**
- Because of the ambiguity of transformations; when we change the camera and scene with the same transformation, we get the same result.
- Because we have no way of filtering the information
- Because camera restriction makes a projective reconstruction impossible
- Because of the ambiguity of transformations a projective reconstruction is the best we can do regardless of further constrictions.
* `[ Option A ]`


---

**Q: Which of the following sentences describes SIFT?**
- For each feature, we compute an eight bin histogram of gradient magnitudes for 16 subwindows around the feature, resulting in an 128 dimensional descriptor.
- For each feature, we compute a two bin histogram of gradient directions for 6 subwindows around the feature, resulting in an 12 dimensional descriptor.
- For each feature, we compute an eight bin histogram of gradient magnitudes resulting in an 8 dimensional descriptor.
- For each feature, we compute an eight bin histogram of gradient directions for 16 subwindows around the feature, resulting in an 128 dimensional descriptor.
* `[ Option D ]`


---

**Q: What is the assumption for orthographic projection?**
- Distance between COP and image plane is "infinite".
- Distance between COP and image plane is small.
- Distance between object and image plane is "infinite".
- Distance between object and image plane is small.
* `[ Option A ]`


---

**Q: Why is the measurement matrix rank 3?**
- Because we only need 3 measures to compute the motion an shape matrices
- Because the nature of the measurements is 3D
- All of the above
- None of the above
* `[ Option B ]`


---

**Q: What are desirable characteristics of descriptors?**
- Scale invariance
- Rotation invariance
- Viewpoint invariance
- All of the above
* `[ Option D ]`


---

**Q: As introduced in coordinates transformation, which of the following dimension description is wrong?**
- 3D point (3*1)
- Camera to pixel coordinate transformation matrix (3*3)
- Perspective projection matrix (3*4)
- World to camera coordinate transformation matrix (4*4)
* `[ Option A ]`


---

**Q: Which of the following method is not a part of multiple view geometry?**
- Correspondence (stereo matching)
- Scene geometry (structure)
- Camera geometry (motion)
- Image geometry (detector)
* `[ Option D ]`


---

**Q: In factorization method we prove the rank theorem, what is the rank of measurement matrix are required under orthography?**
- 3
- 2
- 1
- more than 3
* `[ Option A ]`


---

**Q: When we factorize a measurement matrix, how can we deal with singular value decomposition?**
- Set all singular value to 0 except first 3 
- Set all singular value to 1 except first 3
- Set all singular value to 0 
- Set all singular value to 1
* `[ Option A ]`


---

**Q: What are the other types of descriptors **
- HoG
- CSV
- DoG
- SVM
* `[ Option A ]`


---

**Q: How can be efficiently approximate the Laplacian of Gaussian**
- DoG
- HoG
- SVM
- CSV
* `[ Option A ]`


---

**Q: What is the effect of the rank constrain in the singular value decomposition of a measurement matrix?**
- Noise in the decomposition solution is reduced
- Motion ambiguity is avoided
- Less camera positions are needed
- Less picture features are needed
* `[ Option A ]`


---

**Q: Given is the projection matrix P, the 3D points matrix X, the correspondences x and a random transformation Q. Which of the following formulas is correct about motion ambiguity?**
- $x = (PQ)(Q^{-1}X)$
- $x = (PQ^{-1})(QX)$
- $x = (QP)(XQ^{-1})$
- $x = (Q^{-1}P)(XQ)$
* `[ Option B ]`


---

**Q: Which of the following is incorrect. Normalizing SIFT features in the direction of the dominant gradient allows descriptors to:**
- Become invariant to image distance
- Become invariant to rotation
- Become invariant to translation
- Become invariant to all affine transformations
* `[ Option D ]`


---

**Q: Which of the following about SFT is true:**
- If we transform both camera matrices and scene by same transformation, the images do not change
- If we transform camera matrices by A and scene by 1/A, the images do not change
- If we transform camera matrices by 1/A, the images get transformed by a factor of A
- If we transform scene and camera matrices by same transformation, it is possible to recover the absolute scale
* `[ Option B ]`


---

**Q: How many subwindows are normally used when using SIFT?**
- 4
- 16
- 32
- 256
* `[ Option D ]`


---

**Q: What does affine ambiguity mean?**
- The scaling stays the same however parallel lines donΓÇÖt stay parallel
- The scaling differs however parallel lines do stay parallel
- The scaling differs and parallel lines donΓÇÖt stay parallel
- The scaling stays the same and parallel lines do stay parallel
* `[ Option B ]`


---

**Q: What should the propertie(s) of features has/have?**
- Repeatibility
- Saliency & Locality
- Compactness and efficients
- All true
* `[ Option D ]`


---

**Q: When with sift become the features scale invariant?**
-  When the 16x16 to 4x4 you normalize it
- The historgram of the rotations you average 
- When you compute the 4x4 gradients
- None
* `[ Option C ]`


---

**Q: Which of the following is not a part of the keypoint orientation process?**
- Create a histogram with 36 bins.
- Weigh each orientation by its magnitude.
- If there are multiple peaks above 80%, reduce the scale.
- Assign to each keypoint a dominant orientation, ╬╕i.
* `[ Option C ]`


---

**Q: When removing uninformative keypoints, what is the way to identify a corner through a moving kernel?**
- The kernel shows no change in all directions
- The kernel shows significant change in all directions
- The kernel shows no change along one direction
- All of the above
* `[ Option C ]`


---

**Q: Why do we need to rotate the gradient orientations by the keypoint orientation?**
- To be able to match an image that is rotated relative to the first
- There is no reason to rotate
- For numerical stability
- To account for changes in lighting conditions
* `[ Option A ]`


---

**Q: Which of the following is a good approximation for the laplacian?**
- Gaussian of size kf - Gaussian of size f
- Gaussian of size f - Gaussian of size kf
- Gaussian of size kf - Gradient of gaussian of size f
- Gaussian of size f - Gradient of gaussian of size f
* `[ Option A ]`


---

**Q: Which of the following correspond to the structure question?**
- Given a point in just one image, how does it constrain the position of the corresponding point in other images?
- Given 2D point matches in two or more images, where would the corresponding points be given another camera position?
- Given a set of corresponding points in two or more images, what are the camera matrices for these views?
- None of the above
* `[ Option D ]`


---

**Q: What rank must the measurement matrix $M$ be in structure from motion?**
- 2
- 3
- 4
- 5
* `[ Option B ]`


---

**Q: In the affine structure from motion, if there are 4 points and 3 cameras , what are the dimensions of the camera and 3D points matrix?**
- Camera matrix: 6x3
3D points matrix: 3x4
- Camera matrix: 8x3
3D points matrix: 3x2
- Camera matrix: 3x3
3D points matrix: 3x4
- Camera matrix: 3x4
3D points matrix: 4x3
* `[ Option A ]`


---

**Q: Which of the followind statements is true regarding perspective?**
- Its a linear transformation.
- When z goes to infinity, the perspective translation can be approximated to be a linear function.
- Perspective changes a lot when the camera is far.
- The function of perspective is exponential.
* `[ Option B ]`


---

**Q: Which of these is not a desirable characteristic for image matching?**
- Scale invariance
- Rotation invariance
- Illumination invariance
- Colour invariance
* `[ Option D ]`


---

**Q: Which of these statements is incorrect?**
- To approximate a Laplacian of Gaussian for a sigma, we need two gaussians with the sigma
- The low contrast keypoints can be removed using a threshold for the magnitude of the intensity of a keypoint location
- Images are usually matched in SIFT using the Euclidean distance between each descriptor in the first image and all of the descriptors in the other image
- Each octave used by SIFT has 8 images
* `[ Option D ]`


---

**Q: In SIFT algorithm, what do we do to reduce the influence of different illumination?**
- Normalize the descriptor
- Rotate the image
- Change RGB image into grayscale image
- Double the intensity of all pixels
* `[ Option A ]`


---

**Q: How do we deal with missing data when affine images?**
- Decompose matrix into dense sub-block, factorize each block, combine them
- Use avarage value of the surrounding pixels
- Use the smallest value of the surrounding pixels
- Use the largest value of the surrounding pixels
* `[ Option A ]`


---

**Q: We can approximate the Laplacian of Gaussian by**
- Multiplication of octaves
- Difference of octaves
- Summation of octaves
- Convolution of Octaves
* `[ Option B ]`


---

**Q: To find keypoints at subpixel locations we use**
- Taylor expansion
- Fourier transformation
- Convolution
- Ray tracing
* `[ Option A ]`


---

**Q: What does a 16 x 16 window of SIFT descriptor contain?**
- Laplacian
- Gradient Magnitude
- Orientation
- Both B) and C)
* `[ Option D ]`


---

**Q: Which of the following statements are correct
1)Scaling the image in different octave in the SIFT algorithm has the effect of blurring
2)SIFT uses both edges and corners as key point**
- Only 1) is correct
- Only 2) is correct
- Both are correct
- Both are incorrect
* `[ Option A ]`


---

**Q: What does motion refer to in structure of motion?**
- Motion of the camera
- Motion of the scene
- Motion of the feature points
- None of the above
* `[ Option A ]`


---

**Q: What is not needed for the factorization method?**
- Center each frame around the origin at 0
- Compute the SVD for the matrix
- Create the motion and structure matrix
- All of the above are needed
* `[ Option D ]`


---

**Q: A Laplacian of Gaussian can be approximated with**
- First derivative of a Gaussian
- Second derivative of a Gaussian
- Difference of Gaussians
- It is not possible
* `[ Option C ]`


---

**Q: Structure from motion**
- only works when a camera rotates
- only works when a camera moves horizontally
- works with rotation and horizontal movements, but there are different algorithms needed
- only works with a movie as motion is needed
* `[ Option C ]`


---

**Q: Which one is wrong**
- There are three stages to the SIFT algorithm: Scale-space extrema detection;   Orientation assignment; Keypoint descriptor.
- A SIFT descriptor for a 16 ├ù 16 image window W is a vector of 128 numbers, which can be thought of as being grouped in 16 consecutive groups of 8 numbers each. In this example.  Each group of 8 numbers is a histogram of orientations of the image gradient in one of 16 sections, each orientation weighted by the corresponding gradient magnitude.
- One can make descriptors rotationally invariant by assigning orientations to the key points and then rotating the patch to a canonical orientation.
- Orientation based descriptors are very powerful because robust to changes in brightness
* `[ Option A ]`


---

**Q: Which one is wrong**
- There are two main types of HOG descriptor: a rectangular shaped R-HOG descriptor, and a circular shaped C-HOG descriptor.
- In SIFT, make a patch descriptor rotationally invariant by constructing Histograms of Gradients in a neighborhood around the feature point. Assign the largest bin as the corresponding direction of the keypoint. Rotate all detected features to make the corresponding orientations vertically aligned
- First step pf is SIFT to estimate a scale space extrema using the Difference of
Gaussian (DoG).  Then, a key point orientation
assignment based on local image gradient, and lastly a
descriptor generator to compute the local image descriptor for
each key point based on gradient magnitude and
orientation.
- Gaussian pyramid is used in many computer vision algorithms such as SIFT.
* `[ Option C ]`


---

**Q: SIFT is invariant to which of the following features?**
- Illumination.
- Rotation.
- Viewpoint.
- All of the above. 
* `[ Option D ]`


---

**Q: For SIFT descriptors, around each keypoint a 16x16 window is taken and each window is split further into 16 subwindows. This proces finally leads to how many dimesional descriptors?**
- 64
- 128
- 256
- 512
* `[ Option B ]`


---

