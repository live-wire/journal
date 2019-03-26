# Questions from `lecture_5` :robot: 

**Q: How many degrees of freedom does the fundamental matrix F have?**
- 7
- 2
- 5
- 9
* `[ Option A ]`


---

**Q: What kind of points do the matrixes E and F use respectively?**
- Real world points and Image points
- Real word points and Real word points
- Image points and Real word points
- Image points and Image points
* `[ Option A ]`


---

**Q: When performing image matching, the difference between using the same camera with a different angle and two cameras in different spatial locations is that:**
- In the first configuration the fundamental matrix needs to be estimated and in the second is the homography.
- The first makes use of the Epipolar constraint, but the second does not.
- In the first configuration the homography needs to be estimated and in the second is the fundamental matrix.
- Both matching problems can be approached the same way. Only the intrinsic camera parameters must be corrected.
* `[ Option C ]`


---

**Q: What is the Epipolar constraint? **
- A point in one camera corresponds exactly to a point in the other camera.
- A point in one camera corresponds to a line in the other camera. 
- A line in a camera matches a line in the other camera.
- None of the above describe the Epipolar constraint.
* `[ Option B ]`


---

**Q: Epipolar geometry**
- An epipolar line.
- An epipolar point.
- An epipolar plane.
- An Epipole.
* `[ Option A ]`


---

**Q: Which statement is True regarding the homography and the fundamental matrix?**
- Homography does not include camera translations while the Fundamental matrix does include camera translations.
- Homography needs 4 points correspondences (3 for an affine homography) while the Fundamental matrix needs 8 points correspondences.
- They are both normalized with the formula: \hat{x}-Tx and \hat{xΓÇÖ}-TΓÇÖxΓÇÖ
- All of the above statements are true.
* `[ Option D ]`


---

**Q: Which one of the following is false?**
- All epipolar lines pass through the epipole
- Baseline is the line joining the camera center to the image center
- Epipolar line is the intersection of epipolar plane and image plane
- Epipolar plane contains the baseline and the world point
* `[ Option B ]`


---

**Q: In epipolar geometry, the dot product of 2 perpendicular vectors with magnitudes v1 and v2 gives**
- Zero
- A vector perpendicular to both the vectors
- A vector in the same plane 
- A scalar with magnitude v1.v2
* `[ Option A ]`


---

**Q: Review the following two statements about epipolar geometry:
\begin{enumerate}
    \item The epipole is the point of intersection of the baseline with the image plane
    \item The epipolar line is the intersection of the epipolar plane with the image plane
\end{enumerate}
Which of the statements are true?**
- Statement 1 and 2 are true
- Statement 1 is true and statement 2 is false
- Statement 1 is false and statement 2 is true
- Statement 1 and statement 2 are false
* `[ Option A ]`


---

**Q: Consider the following statement about homography vs fundamental estimation, which ones are incorrect?**
- Optimization with a homography goes according to $x' = Hx$ and in the case of a fundamental matrix we have $x'Fx=0$
- Normalization has a similair format in a homography and a fundamental matrix
- They both make use of RANSAC, but in the case of the fundamental matrix the determinant of F needs to be kept equal to 0
- A homography and a fundamental matrix have the same amount of points correspondences
* `[ Option D ]`


---

**Q: Statement 1: The essential matrix is used for computing the epipolar lines associated with x and xΓÇÖ, the points in the first and second image. Statement 2: A property of the essential matrix is that its matrix product with the epipoles equate to zero. **
- Statement 1 is true, statement 2 is false.
- Statement 1 is false, statement 2 is true.
- Both statements are true.
- Both statements are false.
* `[ Option C ]`


---

**Q: Which of the following factors does not affect the intrinsic parameters of a camera model?**
- Image resolution
- Exposure
- Image center point
- Focal length
* `[ Option B ]`


---

**Q: In epipolar geometry, the baseline is the line that connects two camera centers $O_1$ and $O_2$. Suppose now that in image 1 you have a point $P_1$ that should have a correspondent, call it $P_2$, in image 2. The plane defined by the baseline and the line $O_1P_1$ is the ___________ , that will intersect image 2 and create the ______________ . It is on that line that $P_2$ must be.**
- epipolar plane, epipolar line
- projective plane, epipolar line
- epipolar plane, perspective line
- projective plane, perspective line
* `[ Option A ]`


---

**Q: In the setting of epipolar geometry, the fundamental matrix $F$ is the $3\times 3$ matrix that satisfies $x'^T F x = 0$, if $x'$ is a set of points in image $1$ and $x$ the set of corresponding points in image $2$. In order to find $F$ one usually uses the $8$-point algorithm, that uses _____ pairs of points, that results in ______ equations to find the _______ unknowns of $F$.**
- $8$, $8$, $9$
- $4$, $8$, $8$
- $4$, $8$, $9$
- $9$, $9$, $9$
* `[ Option A ]`


---

**Q: Which of the following is(are) intrinsic parameter(s)?**
- The camera's focal length
- The image center point
- Radial distortion parameters
- All of the above
* `[ Option D ]`


---

**Q: Choose the correct statement about the epipolar constraint?**
- Potential matches of point p from one view of an object to another point p' in a different  view should lie in the epipolar line.
- Potential matches of point p from one view of an object to another point p' in a different  view should lie in the epipolar circle.
- Potential matches of point p from one view of an object to another point p' in a different  view should be just a point.
- None of the above
* `[ Option A ]`


---

**Q: What is epipolar plane?**
- Plane containing baseline and world point
- Plane containing baseline and focus point
- plane containing baseline and true point
- plane containing baseline and gold point
* `[ Option A ]`


---

**Q: What is the function of the fundamental matrix?**
- Find the relation between points in two images
- Find the relation between moving speed in two images
- Match points in two images
- Stitching two images togethor
* `[ Option A ]`


---

**Q: Why is it much easier to recover depth from a stereoscopic view than a panoriamic (rotation only) view?**
- A stereoscopic view contains more information
- A panoramic view becomes near orthographic at large distances
- Stereoscopic views allow for triangulation between image points
- Panoramic images do not contain any depth
* `[ Option C ]`


---

**Q: What is the golden standard method?**
- A method to find the epipolar line in a stereoscopic projection
- A method to remove noise from the images and feature points
- Normalize the features to have a mean of 0 and variance of 1 for lower error on the 8-point algorithm
- Minimize the error between actual points and the reprojection of the calculated 3D points
* `[ Option D ]`


---

**Q: Which of the following statements is true about homography estimation VS fundamental matrix estimation?
1. With homography estimation, there needs to be a translation of the camera
2. When estimating the fundamental matrix, one needs (at least) 8 corresponding points**
- Both statements are true
- Statement 1 is true, statement 2 is false
- Statement 2 is true, statement 1 is false
- Both statements are false
* `[ Option B ]`


---

**Q: How many DoF does the essential matrix E have (E = [t]x*R)**
- 2, for translation
- 3, for rotation
- 5, for rotation and translation
- 6, for affine transformations
* `[ Option C ]`


---

**Q: The 8-point algorithm estimates the:**
- essential matrix
- fundamental matrix
- epipoles
- epipolar constraint
* `[ Option B ]`


---

**Q: Disparity is ΓÇª to depth**
- directly proportional
- quadratically proportional
- inversely proportional
- quadratic inversely proportional
* `[ Option C ]`


---

**Q: The benefit of the epipolar constraint is that:**
- it imposes a one-to-one correspondence between points
- it narrows down the correspondence problem to a 1D search
- it reduces the number of equations (i.e matches) needed to obtain an accurate homography
- it adds a third dimension to the final image which is vital for 3D reconstruction
* `[ Option B ]`


---

**Q: The camera calibration:**
- refers to extrinsic parameters
- cannot be extracted from two images taken with different cameras if not already known
- involves the focal length of the camera
- is not related to the fundamental matrix
* `[ Option C ]`


---

**Q: What is the size of the equation matrix $A$?**
- $3\times 3$
- $n\times 3$
- $2n\times 9$
- $n\times 9$
* `[ Option D ]`


---

**Q: Which of the following statements is wrong?**
- all epipolar lines intersect at the epipole
- epipole is the point of intersection of baseline with the epipolar plane
- baseline is the line joining the camera centers
- an epipolar plane intersects left and right image planes in epipolar lines
* `[ Option B ]`


---

**Q: What is the fundamental matrix**
- The matrix that describes the diagonal matrix 
- The mathematical formulation that enforces this epipolar constraint
- Describes the baseline of the image
- The homogeneous linear ordinary differential equations
* `[ Option D ]`


---

**Q: What is the epipolar line **
- Line between images centers
-  Line in another images diagonal
- Line that describes where the point in second image would be
- Line that is circular in the other images
* `[ Option C ]`


---

**Q: Which of the following statements is False?**
- The baseline always intersects both images.
- All epipolar lines intersect at the epipole.
- An epipolar plane intersects the left and right image planes in epipolar lines.
- None of the above.
* `[ Option A ]`


---

**Q: What is the Rank (R) and the number of Degrees of Freedom (DoF) of the fundamental matrix?**
- R = 3, DoF = 9
- R = 3, DoF = 8
- R = 2, DoF = 8
- R = 2, DoF = 7
* `[ Option D ]`


---

**Q: What is the fundamental difference between mosaic and epipolar, regarding the cameras?**
- Mosaic assumes one camera was used to generate the images, whereas epipolar assumes multiple cameras can be used.
- Mosaic assumes the used camera to be stationary, whereas epipolar assumes, if a single camera was used, that it has moved.
- Both A and B are True
- None of the above.
* `[ Option C ]`


---

**Q: How does RANSAC for mosaic differ from RANSAC for epipolar (reconstruction)?**
- For epipolar, we need more correspondence points than for mosaic.
- In epipolar RANSAC, we compute the inliers by looking at perpendicular errors between the points and the epipolar lines in each image.
- Both A and B are True.
- None of the Above.
* `[ Option C ]`


---

**Q: What is difference between essential and fundamental matrix?**
- No difference, different name for same thing
- Essential matrix corrects for intrinsic parameters of camera only
- Fundamental matrix corrects for intrinsic parameters of camera only
- Fundamental matrix is essential matrix with correction of intrinsic parameters
* `[ Option D ]`


---

**Q: Why is fundamental matrix rank deficient?**
- It has non-trivial nullspace which is the epipole.
- Because of homogeneous coordinates
- It has 9 unknowns and 8 degrees of freedom
- It is not rank deficient.
* `[ Option A ]`


---

**Q: What does the fundamental matrix describe?**
- The camera parameters (focal length etc)
- The transformation between two images
- The epipolar constraint
- None of the above
* `[ Option B ]`


---

**Q: How many degrees of freedom does the essential matrix have?**
- 2
- 3
- 4
- 5
* `[ Option D ]`


---

**Q: Which of following is not TRUE about EPIPOLAR GEOMETRY∩╝ƒ**
- Epipolar plane: plane containing baseline and world point
- Epipole: point of intersection of baseline with the epipolar plane
- Epipolar line: intersection of epipolar plane with the image plane
- Baseline: line joining the camera centers
* `[ Option B ]`


---

**Q: How many points can define the fundamental matrix of two projective cameras?**
- 9
- 8
- 3
- 2
* `[ Option B ]`


---

**Q: How many point correspondences are needed for homography estimation and fundamental matrix estimation respectively?**
- 3 and 7
- 4 and 7
- 3 and 8
- 4 and 8
* `[ Option D ]`


---

**Q: What is NOT an intrinsic camera parameter?**
- Pixel sizes
- Image center point
- Radial distortion
- Rotation matrix
* `[ Option D ]`


---

**Q: Which following statement about fundamental matrix F is false?**
- F is related to the camera parameters
- different coordinates can lead to different solutions for F in the 8-point algorithm
- F would not be influenced by the noise 
- F is singular, in fact of rank 2
* `[ Option C ]`


---

**Q: Which following statement about homography matrix H is false?**
- It is not a camera translation
- The SVD solution is equivalent to Least Squares
- It has 8 point correspondences
- 3 of the point correspondences are for affine homography
* `[ Option C ]`


---

**Q:  Consider the situation where you are comparing the RANSAC algorithm for image matching and fundamental matrix estimation. What difference in algorithms is NOT true?**
- RANSAC for image matching usually gives better results because it operates with 3 points instead of 8
- RANSAC for image matching operates with 3 points and RANSAC for fundamental matrix estimation operates with 8 points
- In both cases of RANSAC algorithm some cost function is being minimized
- In both cases we compute inliers and discard the outliers
* `[ Option A ]`


---

**Q: Consider essential and fundamental matrices. What is the key difference?**
- the essential matrix is a metric object pertaining to calibrated cameras while the fundamental matrix describes the correspondence in more general and fundamental terms of projective geometry.
- the fundamental matrix is a metric object pertaining to calibrated cameras while the essential matrix describes the correspondence in more general and fundamental terms of projective geometry.
- We can infer fundamental matrix only from essential matrix and not the other way around
- There are no differences between these two matrices.
* `[ Option A ]`


---

**Q: With respect to the computation of the fundamental matrix, we can get different solutions in $F$. How can we solve this?**
- Normalize all coordinates to have 0-mean, solve the optimization for $\hat{F}$, and de-normalize to get $F$.
- Normalize all coordinates to have 0-mean, scale the points s.t. the average distance from the origin is $sqrt(2)$, solve the optimization for $\hat{F}$, and de-normalize to get $F$.
- Solve the optimization for $\hat{F}$
- Scale the points s.t. the average distance from the origin is $sqrt(2)$, solve the optimization for $\hat{F}$, and revert the scaling to get $F$.
* `[ Option B ]`


---

**Q: With respect to the computation of the fundamental matrix, what do we do if we do not know the intrinsic camera parameters $K$ and $K'$? How can we solve this?**
- Set $x'Fx= 0$ with $F = K'^{-T}EK^{-1}$, where E is the essential matrix.
- Assume that $F = K'^{-T}FK^{-1}$.
- We can not solve this.
- By applying a translation.
* `[ Option A ]`


---

**Q: What is the most accurate description of an epipolar line?**
- An epipolar line is a line that can be drawn to directly connect two corresponding points in different images. 
- An epipolar line is the line in one image that represents all pixel locations on which a particular point from another image of the same scene can lie
- An epipolar line represents the axis along which two cameras are rotated relative to eachother
- An epipolar line represents the translation of one camera relative to another camera
* `[ Option B ]`


---

**Q: Which of these statements is true?
I: the Essential matrix describes the intrinsic values of a stereo camera setup
II: with normalization different solutions for the Fundamental matrix can be found**
- only statement I is true
- only statement II is true
- both statements are true
- both statements are false
* `[ Option D ]`


---

**Q: What is the minimum number of points needed for affine transformation?**
- 1
- 2
- 3
- 4
* `[ Option C ]`


---

**Q: For a point in the left image, how many points in right image satisfy the epipolar constraint?**
- 10
- 50
- 1
- A series of points that form a line
* `[ Option D ]`


---

**Q: Which of the following is/are correct?

1. One of the differences between homography and the fundamental matrix is that in the first there is no camera translation, while there is in the second.
2. The fundamental matrix also requires more point correspondences than homography. The fundamental matrix requires 6 to be exact.**
- Only 1 is correct
- Only 2 is correct
- Both are correct
- Both are incorrect
* `[ Option A ]`


---

**Q: Which of the following is not an intrinsic parameter?**
- Focal length
- Rotation matrix
- Image center point
- Radial distortion parameters
* `[ Option B ]`


---

**Q: For which of the following would you NOT use the "Eight Point" algorithm**
- Constructing a mosaic from images taken by a fixed camera rotating about a its centre of projection
- Reconstructing a scene from images taken by a moving camera
- Reconstructing a scene from images taken by multiple (and possibly un-calibrated) cameras
- None of the above
* `[ Option A ]`


---

**Q: Which of the following is NOT associated with the Epipolar constraint**
- It reduces the problem of finding corresponding matches between two images taken by different cameras (or a single camera) to a one-dimensional problem
- It constrains the corresponding location of a point (or feature) in one view to a line carved out by the plane connecting the world view point of the feature and optical centers of the different views.
- Enforcing the Epipolar constraint allows for determining a matrix which relates location of points of an image in one view to their location in a second view of the same image.
- None of the above
* `[ Option D ]`


---

**Q: Where do all epipolar lines cross,?**
- In the camera origin
- In the epipole
- In the bottom left corner
- In the center if the image
* `[ Option B ]`


---

**Q: What is not a step in the 'RANSAC for solving the Epipolar constraint' algorithm shown in the lecture**
- Compute point correspondences between the 2 images
-  Use the [renormalized] 8-point algorithm to solve F from the selected correspondences.
- Randomly select 8 pairs of points x_k, x_k'.
- Draw the epipolar lines
* `[ Option D ]`


---

**Q: What rank has the fundamental matrix F?**
- 1
- 2
- 3
- 8
* `[ Option B ]`


---

**Q: The epipolar geometry is NOT favored of tomography when?**
- The camera is translated
- The camera is rotated
- Two similar cameras are used (on different locations)
- Two different cameras are used (on different locations) 
* `[ Option B ]`


---

**Q: Benefit of image rectification**
- Epipoles are moved to the center of the image.
- Image contents are uniformly scaled to a desirable size.
- All epipolar lines intersect at the vanishing point.
- All epipolar lines are perfectly horizontal.
* `[ Option D ]`


---

**Q: Which is incorrect about Essential Matrix**
- Size 3x3
- Singular 
- Rank 2
- 3 DoF
* `[ Option D ]`


---

**Q: In a stereo camera setup, if the image planes are co-planar, where are the epipoles located?
A - On the intersection between baseline and image planes
B - The left epipole is at -inf and the right epipole is at +inf
C - The left epipole is at +inf and the right epipole is at -inf
D - None**
- A and B
- A and C
- A and B and C
- D
* `[ Option C ]`


---

**Q: If a camera setup involves forward translation between two images, where are the epipoles located?**
- Epipoles are located on two parallel lines
- Epipoles are located at z=infinity with x,y as h/2 and w/2
- Epipoles are location at z=infinity and for convenience purpose we assume them both to be at one point
- None
* `[ Option C ]`


---

**Q: Which statement is true if the cameras are co-planar?**
- All epipolar lines intersect at one point, the epipole.
- Epipolar lines are parallel and epipoles lie at infinity.
- Epipolar lines are vertical and epipoles are located outside the image.
- Epipolar lines are parallel and epipoles lie in the middle of the camera view.
* `[ Option B ]`


---

**Q: Which three points are connected by the epipolar plane?**
- World point ΓÇô optical center of right image ΓÇô optical center of left image
- World point ΓÇô epipole of right image ΓÇô epipole of left image
- World point ΓÇô epipolar line of right image ΓÇô epipolar line of left image
- None of the above.
* `[ Option A ]`


---

**Q: Which of the following statements is NOT a correct definition of the corresponding term?**
- Epipolar plane: plane containing baseline and world point.
- Epipole: point of intersection of baseline with the image plane.
- Epipolar line: intersection of epipolar plane with the image plane.
- Baseline: line joining the scene points.
* `[ Option D ]`


---

**Q: How many degrees of freedom does the fundamental matrix have?**
- 5
- 6
- 7
- 8
* `[ Option C ]`


---

**Q: Do epipolar lines always converge to a single point?**
- Yes, this point would correspond to where the other camera would be.
- Yes, this point is called the epipole.
- No, the epipolar lines can also be parallel
- No, the epipolar lines never converge.
* `[ Option C ]`


---

**Q: Why are coordinates normalized before the 8-point algorithm?**
- Because image coordinates can have different ranges depending on the camera parameters.
- Because the image coordinates have to be reduced to 8 dimensions.
- Because the 8-point algorithm only works with zero-mean.
- Because the maximum distance between image coordinates can be maximum sqrt(2).
* `[ Option A ]`


---

**Q: Which of the following sentences is not true about the fundamental matrix F in epipolar geometry?**
- F is a 3x3 matrix
- F is rank 2
- F can be determined only up to scale
- F is symmetric
* `[ Option D ]`


---

**Q: How can we define inliers in RANSAC procedure for solving the epipolar constrain?**
- They are the points lying on their respective epipolar line
- They are the points standing within a given threshold from their epipolar line
- They are the number of feature points present on the epipolar lines
- They are the number of epipolar lines which are actually passing through the image epipole
* `[ Option B ]`


---

**Q: What does the epipolar constraint say?**
- The points in the images lie in a same depth
- The point in the first view, occurs in a line on the second view 
- The points in the views have a constant change of intensity
- None of the above
* `[ Option B ]`


---

**Q: Why is the epipolar constraint useful?**
- Because it helps to find better matches in the images
- Because reduces the correspondence problem to a 1D search
- Because its solves the Fundamental Matrix equation
- None of the above
* `[ Option C ]`


---

**Q: Which of the following is not needed to solve that different coordinates can lead to different solutions for F in the Eight-Point Algorithm?**
- Re-normalizing the mean of all the coordinates in each image
- Solving x'Fx = 0
- De-normalizing F to obtain the true F
- Re-normalizing the variance of all the coordinates in each image
* `[ Option D ]`


---

**Q: What is an extrinsic parameter?**
- Focal length
- Pixel sizes
- Rotation matrix
- Image center point
* `[ Option C ]`


---

**Q: Which of the following statements about epipolar is false?**
- The epipolar constraint states that the image point in the first view, must occur in the second view on the line carved out by a plane connecting the world point and optical centers.
- The epipolar constraint is useful because it reduces the correspondence problem to a 1D search along an epipolar line.
- RANSAC can be used to solve the epipolar constraint.
- The epipole is the point of intersection of the epipolar plane with the image plane.
* `[ Option D ]`


---

**Q: Assume we have matches with outliers between two images: x and x'. Which of the following statements about homography estimation and fundamental matrix estimation is false?**
- Homography estimation has no camera translation and fundamental matrix estimation has camera translation.
- For homography estimation 3 point correspondances are needed, for fundamental matrix estimation 8 point correspondances are needed.
- Homography optimization: $x' = Hx$ and fundamental matrix optimization: $x'^TFx = 0$.
- Normalization of homography estimation is equal to fundamental matrix estimation.
* `[ Option B ]`


---

**Q: What is the difference between homography $H$ and fundamental matrix $F$ estimation?**
- Normalization of the input is only needed for estimating $F$
- The amount of needed point correspondences
- The determinant of $F$ is always higher than $H$
- There are no differences between estimating $H$ and $F$
* `[ Option B ]`


---

**Q: Why is the fundamental matrix $F$ rank deficient and is a rank of two enough for a estimation?**
- Because it is constrained in the Singular Value Decomposition when estimating $F$
- Because the epipoles always lie on the epipolar lines
- Because images are always in 2D
- Because the $F$ matrix is denormalized in his estimation
* `[ Option B ]`


---

**Q: Which of these definitions is not correct**
- An epipolar plane is a plane consisting of the baseline and a world point
- An epipole is a point of intersection between the baseline and the world point
- An epipolar line is an intersection between the epipolar plane with the image plane
- The baseline is a line joining the camera centers
* `[ Option B ]`


---

**Q: Which of these statements is incorrect?**
- All epipolar lines intersect at the epipole
- An epipolar plane intersects the left and right image planes in an epipolar lines
- The epipolar constraint reduces the correspondence problem to a 2D search along an epipolar plane
- Stereo is the shape from motion between two views
* `[ Option C ]`


---

**Q: Which are NOT intrinsic camera parameters?**
- Focal length
- Center point
- Radial distortion 
- Rotations
* `[ Option D ]`


---

**Q: How many random samples does Ransac for each iteration need for the fundamental matrix problem?**
- 3
- 4
- 8
- 9
* `[ Option C ]`


---

**Q: Which of the following is false with respect to the mosaic and epipolar structure from motion problem? **
- Mosaic uses a single camera fixed in the same point
- Epipoilar uses two cameras rotated and translated from each other.
- Depending on the movement of the cameras, an epipolar problem can be considered the same as a mosaic problem. 
- Depending on the movement in the photographed scene, an epipolar problem can be considered the same as a mosaic problem. 
* `[ Option D ]`


---

**Q: You are given two pictures taken from two different cameras that are translated and rotated from each other. What is the minimum number of points that you need, in order to estimate the transformation matrix from a point in one image to the same point in the other image?**
- 2
- 4
- 6
- 8
* `[ Option D ]`


---

**Q: Which definition is NOT correct?**
- Epipolar plane: Plane containing baseline and world point
- Epipole: Point of intersection of baseline with the image plane
- Epipolar line: Intersection of epipolar plane with the image plane
- Baseline: Line joining the epipole centers
* `[ Option D ]`


---

**Q: What is NOT a step of using RANSAC for solving the Epipolar constraint?**
- Compute point correspondences between the 2 images
- Randomly select 8 pairs of points x_k, x_k'
- Use the 8-point algorithm to solve F from the selected correspondences
- Compute Euclidian distances between the points and the epipolar lines in each image
* `[ Option D ]`


---

**Q: What is not a difference between homography and the fundamental matrix?**
- Camera translation
- The amount of points needed
- The method of normalization
- None of the above
* `[ Option C ]`


---

**Q: Which of the following parameters is extrinsic instead of intrinsic?**
- Focal length
- Image center point
- Pixel size
- Rotation matrix
* `[ Option D ]`


---

**Q: Which statement about the epipolar geometry is False?**
- Epipolar plane only contains baseline but no world points
- Epipole is the point of intersection of baseline with the image plane
- All epipolar lines intersect at the epipoles
- An epipolar plane intersects the left and right image planes in epipolar lines
* `[ Option A ]`


---

**Q: Which statement about the mathematical formulation that enforces epipolar constraint is False?**
- The fundamental matrix has a rank of 3
- F is a 3*3 matrix called the Fundamental matrix.
- We can determine F up to scale.
- F has 9 unknowns and 7 degrees of freedom
* `[ Option A ]`


---

**Q: How many equations do you need to solve the epipolar constraints?**
- 8
- 6
- 3
- 5
* `[ Option A ]`


---

**Q: What is the fundamental difference in the real world between homography and the fundamental matrix?**
- The fundamental matrix is more efficient, less computations
- The fundamental matrix can only be used for grayscale images
- The fundamental matrix allows for camera translation
- The fundamental matrix does not have to be normalized
* `[ Option C ]`


---

**Q: Which of the following is not an intrinsic parameter?**
- Focal length
- Pixel sizes (mm)
- Image central point
- Rotation matrix
* `[ Option D ]`


---

**Q: Which formula optimizes the fundamental matrix?**
- x'^TFx = 0
- Fx = 0
- Fx = b
- x'^TFx = Fx'
* `[ Option A ]`


---

**Q: What is the definition of the Epipolar Line?**
- a) point of intersection of baseline with the image plane
- b) intersection of epipolar plane with the image plane
- c) line containing baseline and world point
- d) line joining the camera centers
* `[ Option B ]`


---

**Q: According to the 8-Point Algorithm, how many feature points between two images are necessary to solve the set of linear equations to estimate the fundamental matrix?**
- a) any multiple of eight
- b) at most eight
- c) at least eight
- d) any amount of points
* `[ Option C ]`


---

**Q: Epipolar is useful because it reduces the correspondence problem to what?**
-       
1D search 
- 2D search
- 3D search
- All above
* `[ Option A ]`


---

**Q: The Fundamental matrix is estimated over image point pairs in () ?**
- 1D
- 2D
- 3D
- All above
* `[ Option B ]`


---

**Q: what is the mathematical formulation that enforces the epipolar constraint**
- F=x^t0x
- xFx^t=0
- xx^t=F0
- x^TFx=0
* `[ Option B ]`


---

**Q: How many points are required at minimum to estimate the fundamental matrix**
- 7
- 8
- 9
- 10
* `[ Option B ]`


---

**Q: Why is Epipolar Constraint helpful**
- Because it allows to use only 3 images to find the corresponding point
- Because it enables to create a 3D representation of an object using a infrared camera
- Because it reduces the problem of searching for the corresponding point to just 1 dimension
- Epipolar Constrains is just a theory that is not used in practice
* `[ Option C ]`


---

**Q: If the camera is only moving forward, the epipole will**
- move along the y axis
- move along the x axis
- not be a part of all of the images
- have the same coordinates in all of the images
* `[ Option D ]`


---

**Q: What is the definition of the Epipolar line?**
- Line joinining the camera centers.
- Intersection of epipolar plane with the image plane.
- Point of intersection of epipolar plane with the image plane.
- Plane containing baseline and world point.
* `[ Option B ]`


---

**Q: Which of the following is part of the RANSAC procedure for solving the Epipolar constraint?**
- Compute the correspondences between the two images.
- Compute perpendicular errors between the points and theepipolar lines in each image, for all k.
- Identify inliers over all correspondences, over all k.
- All of the above.
* `[ Option D ]`


---

**Q: Why is the epipolar constraint useful? **
- It reduces the correspondence problem to a 1D search along an epipolar line
- It reduces the number of points needed to find the solution to 6
- It is quadratic, so it is less complex than finding the calibration parameters of the two cameras
- None of the above
* `[ Option A ]`


---

**Q: Given the homography (1) and the fundamental matrix (2), which of the following statments is FALSE?**
- (1) is with no camera translation, (2) allows camera translastion
- (1) need 4 point correspondences to be estimated, (2) needs 8 point correspondences
- (1) supports RANSAC, (2) does not support RANSAC
- det(H) = det (F) = 0
* `[ Option D ]`


---

**Q: What is the main difference between Epipolar and Mosaic? **
- Epipolar assumes different cameras (or moving cameras) while Mosaic is just a camera turning around its axis. 
- Epipolar uses the same transformations as Mosaic to obtain stitched images. 
- They both use RANSAC the same way. 
- None of the above. 
* `[ Option A ]`


---

**Q: What is the main difference in the estimation algorithms of Homography vs. Fundamental matrix?  **
- Same number of point matches are required for both
- The errors in the case of the Fundamental matrix estimation are computed as perpendicular errors between the points and the epipolar lines in each image. 
- The both deal with camera translation
- None of the above
* `[ Option B ]`


---

**Q: Which of the following statements are correct?**
- LEFT epipole is the projection of optical center of RIGHT camera on LEFT image plane
- LEFT epipole is the projection of optical center of LEFT camera on LEFT image plane
- LEFT epipole is the projection of optical center of LEFT camera on RIGHT image plane
- None of the above
* `[ Option A ]`


---

**Q: If the cameras in a stereo setup are diverging then which of the following about the horizontal coordinates of the epipoles is correct?**
- left epipole has +ve coordinates; right epipole has -ve coordinates
- left epipole has -ve coordinates; right epipole has -ve coordinates
- left epipole has +ve coordinates; right epipole has +ve coordinates
- None of the above
* `[ Option D ]`


---

**Q: Consider conditions under which an epipolar constraint used in stereo matching holds between images from two cameras. Which one is true?**
- The two cameras must have coplanar projection planes
- The two cameras must face in the same direction (i.e., have parallel optical axes).
- The two images must be rectified.
- There are no restrictions on camera locations or orientations, an epipolar constraint always applies.
* `[ Option D ]`


---

**Q: What is the biggest benefit of image rectification for stereo matching?**
- All epipolar lines intersect at the vanishing point.
- All epipolar lines are perfectly vertical.
- All epipolar lines are perfectly horizontal.
- Epipoles are moved to the center of the image.
* `[ Option C ]`


---

**Q: What is it needed to consider for calculating the image depth with stereo?**
- Info of camera pose
- Image point correspondences
- Calibration
- All of the previous answer should be considered. 
* `[ Option D ]`


---

**Q: What of the following characteristics does NOT correspond to the fundamental matrix estimation?**
- 8 point correspondences. 
- Camera traslation.
- Optimization : x'^(T)Fx=0
- De-Normalization : F= (T')^(-1)*F*T
* `[ Option D ]`


---

**Q: What is the epipolar constraint in image alignment (given matched points)?**
- A point from image 1 must lie on a line in image 2 carved by the plane connecting the optical centers to the world point
- The constraint on the camera properties used in the image
- The constraint that allows us to use 8 instead of 9 points to solve the fundamental matrix
- The constraint that requires the center of the camera not to be moved
* `[ Option A ]`


---

**Q: Which is NOT a purpose of the SVD in structure from motion**
- To split the transform matrix into Motion and Shape matrices
- To reduce the rank of the measurement matrix
- To solve affine ambiguity
- To extract the camera transform from the correspondences
* `[ Option C ]`


---

**Q: Which of the following does not affect the intrinsic parameters of the camera?**
- Focal length
- Image center point
- Radial distortion
- Rotation
* `[ Option D ]`


---

**Q: What is the degree of freedom of Essential matrix?**
- 4 degrees of freedom, 2 for rotation and 2 for translation
- 5 degrees of freedom, 3 for rotation and 2 for translation
- 5 degrees of freedom, 2 for rotation and 3 for translation
- 6 degrees of freedom, 3 for rotation and 3 for translation
* `[ Option B ]`


---

**Q: What is the Epipolar constraint?**
- An image point in the first view, must occur in the second view on the line carved out by a plane connecting the world point and optical centers.
- An image point in the first view, must occur in the second view on the point carved out by a line connecting the world point and optical centers.
- Images points in two different images can only be matched using polar coordinates
- All epipolar lines intersect at the epipole
* `[ Option A ]`


---

**Q: Why is the eight point algorithm called the eight point algorithm?**
- It uses eight steps to solve the epipolar constraint
- It uses eight points to solve the epipolar constraint
- The determinant of the fundamental matrix should equal 8
- It calculates the fundamental matrix for at most 8 different cameras
* `[ Option B ]`


---

**Q: Which of the following statements are correct?**
-  LEFT epipole is the projection of optical center of RIGHT camera on LEFT image plane
- LEFT epipole is the projection of optical center of LEFT camera on LEFT image plane
- LEFT epipole is the projection of optical center of LEFT camera on RIGHT image plane
- None of the above
* `[ Option A ]`


---

**Q: If the cameras in a stereo setup are diverging then which of the following about the horizontal coordinates of the epipoles is correct?**
- Left epipole has +ve coordinates; right epipole has -ve coordinates
- Left epipole has -ve coordinates; right epipole has -ve coordinates
- Left epipole has +ve coordinates; right epipole has +ve coordinates
- None of the above
* `[ Option D ]`


---

**Q: Which of the element is not included in the epipolar geometry?**
- epipolar plane
- epipole
- epipolar line
- epipolar vertex
* `[ Option D ]`


---

**Q: Which of the following operation is the same between homography and fundamental matrix?**
- camera translation
- optimization formula
- RANSAC algorithm
- point correspondences
* `[ Option C ]`


---

**Q: Epipole defines the **
- Plane containing baseline and world point
- Point of intersection of baseline with the image plane
- Intersection of epipolar plane with the image plane
- Line joining the camera centers
* `[ Option B ]`


---

**Q: Which of the following is not an intrinsic parameter?**
- Focal length
- Image center point
- Translation matrix
- Pixel sizes (mm)
* `[ Option C ]`


---

**Q: How does the epipoles look like in a camera moving forward?**
- Fully horizontal lines
- Fully vertical lines
- Circles around the focus of expansion
- Lines moving outward from the focus of expansion
* `[ Option D ]`


---

**Q: In the eight point algorithm, which step is not included?**
- Compute point correspondances between the two images
- Select the 8 most likely pairs of points.
- Solve F from the selected correspondances
- Identify inliers over all correspondances.
* `[ Option B ]`


---

**Q: Among the following parameters, which ones are intrinsic to a camera?
(i) focal length, (ii) camera rotation, (iii) image centre point, (iv) pixel sizes and (v) translation of camera.**
- (i), (iii) and (iv)
- (i), (ii) and (iii)
- (iii), (iv) and (v)
- None of A, B, C is correct
* `[ Option A ]`


---

**Q: Consider the following statements:
i) For a specific scene, all the epipolar lines intersects at a single point.
ii) In order to perform the 8-point algorithm, the camera intrinsic parameters need to be known.
Which of the following is correct?**
- i)
- ii)
- Both i) and ii)
- Neither i) nor ii)
* `[ Option A ]`


---

**Q: What is the degrees of freedom of the fundamental matrix**
- 1
- 3
- 5
- 7
* `[ Option D ]`


---

**Q: Why do we need an epipolar line?**
- To reduce the correspondence problem to 1D search 
- To reduce the correspondence problem to 2D search 
- To rescale the original image
- None of the above
* `[ Option A ]`


---

**Q: What statement is FALSE about epipolar geometry?**
- The epipolar plane is the plane determined by the point of interest (P) and the two camera centers (0 and OΓÇÖ)
- The epipole is the point of intersection between the line that connects the camera centers and the image plane
- The epipolar line is the intersection between the epipolar plane with the image plane
- The baseline is the line that connects the centers of the two image planes
* `[ Option D ]`


---

**Q: Which of the following is an extrinsic parameter?**
- Focal length
- Pixel sizes
- Radial distortion parameters
- Rotation matrix 
* `[ Option D ]`


---

**Q: Which of the following statement of epipolar is not correct?**
- it reduces correspondence problem to 1D search.
- An epipolar plane intersects images in epipolar lines
- mathematical formulation of epipolar constraint is $x'Fx=0$
- degree of freedom of F is 6
* `[ Option D ]`


---

**Q: Which of the following statement is not right?**
- different from fundamental matrix,no camera translation in homograph
- homograph need at least 4 correspondences while fundamental matrix needs 8
- ransac plays an important role in both homograph and fundamental matrix
- normalization in homograph and fundamental matrix is totally different
* `[ Option D ]`


---

**Q: Epipolar reconstruction works best with:

I. moving objects
II. static objects**
- I
- II
- Both
- Neither
* `[ Option B ]`


---

**Q: For solving in epipolar reconstruction you use:

I. Fundamental matrix estimation
II. eight-point algorithm**
- I
- II
- Both
- Neither
* `[ Option C ]`


---

**Q: What is the degree of freedom for essential matrix?**
- 3
- 4
- 5
- 6
* `[ Option C ]`


---

**Q: What is the degree of freedom for essential matrix?**
- 4
- 5
- 6
- 7
* `[ Option D ]`


---

**Q: In the Lecture 5, the concept of Stereo in CV is introduced, namely, that of shape from "motion" between two views,and estimating depth. There are two main common examples given, namely, that of two cameras with simultaneous views and:**
- Several moving cameras and a variable scene;
- Several static cameras and a variable scene;
- A single static camera and a variable scene;
- A single moving camera and a static scene;
* `[ Option D ]`


---

**Q: On the problem of stereo using two cameras simultaneous view, the problem of matching p in first image to the corresponding p' on the second one remains. The epipolar constraint  enunciates that the image point in the first view, must occur in the second view on the line carved out by a plane (epipolar plane) connecting the world point and optical centers. Why is this formulation useful?**
- It reduces the correspondence problem to a 2D search along an epipolar plane;
- It reduces the correspondence problem to a 1D search along an epipolar line;
- Since the epipole has the same coordinates in both images, the relative search process is easier;
- All epipolar lines meet at the epipoles. This will help the search process;
* `[ Option B ]`


---

**Q: If we have an image in normalized coordinates and the fundamental matrix, what do we still at least need to construct an epipolar constraint**
- Nothing.
- The intrinsic camera parameters.
- The extrinsic camera parameters.
- The intrinsic and the extrinsic camera parameters.
* `[ Option B ]`


---

**Q: If we have the disparity of a matched point in two pictures, the focal length and the translation between the two camera frames, what can we deduce from that?**
- The depth of the matched point.
- The essential matrix.
- The fundamental matrix.
- nothing from the above.
* `[ Option A ]`


---

**Q: Which of the following is correct regarding Homography and fundamental matrix.**
- In both estimations there is a camera translation.
- 4 point correspondences are used in homography whereas 8 for fundamental matrix.
- Ransac in not used for the fundamental matrix.
- All of the above.
* `[ Option B ]`


---

**Q: What is the main difference between the essential and fundamental matrix.**
- For the essential matrix real 2d points are used.
- For the fundamental matrix 3d points are used.
- For the fundamental matrix 2d points are used.
- None of the above.
* `[ Option C ]`


---

