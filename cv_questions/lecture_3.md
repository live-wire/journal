# Questions from `lecture_3` :robot: 

**Q: Consider Affine transformation. What can not be achieved using affine transformations?**
- With affine transformations it is possible to make two parallel lines not parallel
- With affine transformations it is possible to scale the object
- With affine transformations it is possible to translate the object
- With affine transformations it is possible to skew the object
* `[ Option A ]`


---

**Q: Suppose you have to use RANSAC algorithm in order to stitch a panorama together. What statement about RANSAC algorithm is true?**
- RANSAC provides a possible solution to deal with outliers that can hurt the quality of our parameter estima
- RANSAC always provides an optimal solution for panorama images, which were shot from the same y axis (only rotation on y axis)
- RANSAC provides an optimal solution for panorama images, which were shot from the camera at two different points (different (x,y) coordinate pair)
- RANSAC is poor choice for panorama stitching as it does not deal with outliers that can hurt the quality of our parameter estima
* `[ Option A ]`


---

**Q: What is NOT true for good detected features?**
- they can be found in several images
- they occupy a relatively large area of the image
- there are many fewer features than image pixels
- they have a distinctive description
* `[ Option B ]`


---

**Q: How many correspondence pairs are at least needed to find the transformation parameters for a 2D affine transformation?**
- 2
- 3
- 4
- 6
* `[ Option B ]`


---

**Q: Which of the following transformations does not preserve parallel lines:**
- Rotation.
- Projection.
- Translation.
- Reflection.
* `[ Option B ]`


---

**Q: Which of the following is not a characteristic of good features in the context of image stitching:**
- Each feature should be very salient.
- It should be possible to find same features in different images.
- There should be as much as possible features.
- Features should occupy a relatively small portion of the image.
* `[ Option C ]`


---

**Q: How many degrees of freedom has a affine transformation?**
- 3
- 4
- 6
- 8
* `[ Option C ]`


---

**Q: What is the main purpose of RANSAC?**
- It avoids the impact of outliers.
- It minimalizes the Euclidean error.
- It incorporates an extra impact of outliers.
- It translate the epipolar lines to all match in the same epipolar point.
* `[ Option A ]`


---

**Q: How many degrees of freedom does a similarity transformation have?**
- 2
- 3
- 4
- 5
* `[ Option C ]`


---

**Q: RANSAC is a method used to:**
- Find out if a point on one image matches a point on the other image
- Reduce the number of matches for two images
- Find the homography between two images
- Avoid the impact of outliers, by finding a fit with the most inliers
* `[ Option D ]`


---

**Q: Which of the following transformations is a 2D rotation?**
- \[
\begin{bmatrix}
x' \\
y' \\
1
\end{bmatrix}=
  \begin{bmatrix}
    1 & 0 & t_x \\
    0 & 1 & t_y \\
    0 & 0 & 1 \\
    
  \end{bmatrix}
  \begin{bmatrix}
  x \\
  y \\
  1 
  \end{bmatrix}
\]
-     \begin{bmatrix}
x' \\
y' \\
1
\end{bmatrix}=
  \begin{bmatrix}
    s_x & 0 & 0 \\
    0 & s_y & 0 \\
    0 & 0 & 1 \\
  \end{bmatrix}
  \begin{bmatrix}
  x \\
  y \\
 1
  \end{bmatrix}
\]
-     \begin{bmatrix}
x' \\
y' \\
1
\end{bmatrix}=
  \begin{bmatrix}
    \cos\theta & -\sin\theta & 0 \\
    \sin\theta & \cos\theta & 0 \\
    0 & 0 & 1 \\
    
  \end{bmatrix}
  \begin{bmatrix}
  x\\
  y\\
  1
  \end{bmatrix}
\]
- \[
    \begin{bmatrix}
x' \\
y' \\
1
\end{bmatrix}=
  \begin{bmatrix}
    1 & s_x & 0 \\
    s_y & 1 & 0 \\
    0 & 0 & 1 \\
    
  \end{bmatrix}
  \begin{bmatrix}
  x\\
  y\\
  1
  \end{bmatrix}
\]
* `[ Option C ]`


---

**Q: Which of the following is a 2D shear transformation?**
- \[
\begin{bmatrix}
x' \\
y' \\
1
\end{bmatrix}=
  \begin{bmatrix}
    1 & 0 & t_x \\
    0 & 1 & t_y \\
    0 & 0 & 1 \\
    
  \end{bmatrix}
  \begin{bmatrix}
  x \\
  y \\
  1 
  \end{bmatrix}
\]
- \[
    \begin{bmatrix}
x' \\
y' \\
1
\end{bmatrix}=
  \begin{bmatrix}
    s_x & 0 & 0 \\
    0 & s_y & 0 \\
    0 & 0 & 1 \\
  \end{bmatrix}
  \begin{bmatrix}
  x \\
  y \\
 1
  \end{bmatrix}
\]
- \[
    \begin{bmatrix}
x' \\
y' \\
1
\end{bmatrix}=
  \begin{bmatrix}
    \cos\theta & -\sin\theta & 0 \\
    \sin\theta & \cos\theta & 0 \\
    0 & 0 & 1 \\
    
  \end{bmatrix}
  \begin{bmatrix}
  x\\
  y\\
  1
  \end{bmatrix}
\]
- \[
    \begin{bmatrix}
x' \\
y' \\
1
\end{bmatrix}=
  \begin{bmatrix}
    1 & s_x & 0 \\
    s_y & 1 & 0 \\
    0 & 0 & 1 \\
    
  \end{bmatrix}
  \begin{bmatrix}
  x\\
  y\\
  1
  \end{bmatrix}
\]
* `[ Option D ]`


---

**Q: The final affine transformation is a ____________ operation of the rotation, translation, scale and shear**
- Additive 
- Multiplicative
- Both
- None
* `[ Option A ]`


---

**Q: If we only have rotation and translation that have taken place between two images, how many correspondence points do we need to estimate the homography?**
- 1 pt in each image
- 2 pt in each image
- 3 pt in each image
- 4 pt in each image
* `[ Option B ]`


---

**Q: Which of the following features are conserved in an affine transformation?**
- Parallelism
- Lengths
- Angles
- Orientations
* `[ Option A ]`


---

**Q: What is the RANSAC algorithm use for (in image reconstruction)?**
- To find the right transformation from one image to another according to the number of inliers for that transformation
- To exclude portions of the image in order to build a mosaic that is as much as possible resembling to the original scene
- To reshape images that are too big compared to the others, in order to have a better matching between them
- To find the best light conditions in the reconstructed mosaic
* `[ Option A ]`


---

**Q: Which of the following methods is used as a model fitting method for edge detection?**
- SIFT
- Gaussian detection
- RANSAC
- None of the above
* `[ Option C ]`


---

**Q: Which of these is true for a Homography?**
- It transforms a 3D image to obtain a 2D image
- A rectangle always maps to a square and no other quadrilateral
- Straight lines are preserved
- Parallel lines are preserved
* `[ Option C ]`


---

**Q: To stitch two images together with an affine transformation we need to solve the transformation parameters. How many matches are needed to solve for these parameters?**
- a) two, one for each set of three linearly independent equations
- b) three, one for each set of two linearly independent equations
- c) six, one for each parameter
- d) all matches, each provides information
* `[ Option B ]`


---

**Q: An affine transformation can be described as the following matrix: [xΓÇÖ;yΓÇÖ;w] = [a b c; d e f; 0 0 1] * [x; y; z]. Which parameters belong to a shearing effect?**
- a) c and f
- b) b and d
- c) a, b, d and e
- d) a and e
* `[ Option B ]`


---

**Q: Which of the following detected features is wrongly described?**
- Repeatability means that the same feature can be found in several images despite geometric and photometric transformations.
- Saliency means that each feature has a distinctive description.
- Locality means that a feature occupies a relative big area of the image; robust to clutter and occlusion.
- Compactness and efficiency mean that any fewer features than image pixels.
* `[ Option C ]`


---

**Q: Which of the following statements is false?**
- Affine transformations are combinations of linear transformations and translations.
- Parallel lines remain parallel in 2D affine transformations and projective transformations.
- A mapping between any two Projective Planes with the same center of projection is called homography.
- The intuituion behind RANSAC is that if an outlier is chosen to compute the current fit, then the resulting line won't have much support from the rest of the points.
* `[ Option B ]`


---

**Q: Which one is not the property of good detected features?**
- Repeatability
- Sparsity
- Saliency
- Locality
* `[ Option B ]`


---

**Q:  Which statement of homography is not TRUE?**
- A mapping between any two Projective Planes with the same center of projection
- rectangle should map to arbitrary quadrilateral
- parallel lines are preserved
- straight lines are preserved
* `[ Option C ]`


---

**Q: How many correspondence pairs do we at least need in order to solve for the transformation parameters?**
- 1
- 2
- 3
- 4
* `[ Option C ]`


---

**Q: What does homography mean?**
- A mapping between any two projective planes with the same center of projection
- A mapping between two intersecting vectors
- Projection of a shape onto a plane
- The mapping of a shadow onto a plane
* `[ Option A ]`


---

**Q: Which of the following statements is TRUE?**
- The affine 2D planar transformation requires 8 Degrees of Freedom.
- After an affine 2D planar transformation, parallel lines do not remain parallel.
- After an affine 2D planar transformation, straight lines do not remain straight.
- After an affine 2D planar transformation, perpendicular lines do not remain perpendicular.
* `[ Option D ]`


---

**Q: What would be the correct order of steps in order to perform image stitching? (DoG = Difference of Gaussians, HCD = Harris Corner Detector)**
- DoG, HCD, SIFT, RANSAC
- HCD, DoG, SIFT, RANSAC
- RANSAC, HCD, DoG, SIFT
- RANSAC, DoG, HCD, SIFT
* `[ Option A ]`


---

**Q: How many matches(correspondence pairs) do we need to solve for the transformation parameters for a linear system with six unknowns?**
- at least 3
- at least 4
- at least 5
- at least 6
* `[ Option A ]`


---

**Q: Which of the following statement about homography is not true?**
- The homography is an image warp from one 2D image to another 2D image.
- The rectangle should map to arbitrary quadrilateral.
- Parallel line are preserved.
- Homography must preserve straight lines.
* `[ Option C ]`


---

**Q: What is NOT a common application for feature points?**
- Depth estimation
- Robot navigation
- Density estimation
- Object recognition
* `[ Option C ]`


---

**Q: How many feature pairs are necessary to calculate an affine tranformation between two point sets?**
- 2
- 3
- 6
- 8
* `[ Option B ]`


---

**Q: There are many types of image transformations, such as translations or rotations. Taking into account the degrees of freedom of each transformation, one can actually say that**
- a translation is a very specific homography
- a rotation is a very specific translation
- a homography is a very specific shear
- a shear is a very specific affine transformation
* `[ Option A ]`


---

**Q: Homogeneous coordinates (which come from projective geometry in mathematics) are a useful way to present the coordinates of points in an image. In that setting, the point $(100, 50, 2)$ is actually the same as**
- $(200, 100, 4)$
- $(200, 100, 1)$
- $(98, 48, 0)$
- $(102, 52, 0)$
* `[ Option A ]`


---

**Q: In what situation would panaroma stitching still work when the center of the camera is moved?**
- When the object that is captured is flat (is not a 3D structure).
- When the object is very close to the camera.
- When the light source moves between images.
- When a patch of key points are moved by a constant distance.
* `[ Option A ]`


---

**Q: In each iteration, RANSAC picks a subset of points from the input dataset, how does the algorithm do this?**
- Pair by pair, until all are evaluated.
- Randomly selected pairs, until all are evaluated.
- Randomly, until a fixed amount of iterations is reached.
- Pair by pair, until a fixed amount of iterations is reached.
* `[ Option C ]`


---

**Q: What is the number of maximum D.O.F (degree of freedom) of basic 2D transformation?**
- 2
- 5
- 8
- 9
* `[ Option C ]`


---

**Q: RANSAC in computer vision field is mainly used to find?**
- The corner points as much as possible
- The corner points as few as possible
- The optimised transformation between reference and target images
- The matching rule such as SIFT
* `[ Option C ]`


---

**Q: How many degrees of freedom (DoF) are added when moving from an affine to a projective transformation?**
- 1
- 2
- 3
- 4
* `[ Option B ]`


---

**Q: Under what condition is the stitching of panorama photos still possible when changing the center (position) of the camera (as opposed to rotating around the cameraΓÇÖs center of projection)?**
- Under all conditions
- When the scene is close enough to the camera
- When the scene is far away from the camera
- This doesnΓÇÖt work under any circumstance
* `[ Option C ]`


---

**Q: When stitching together a panorama, if the images are taken from a different camera center, the final result:**
- will always appear natural
- will only appear natural if the view is curvy or nearby
- will only appear natural if the view is planar or far away
- will only appear natural if the camera has a high resolution
* `[ Option C ]`


---

**Q: The purpose of RANSAC when used to match features from different images is:**
- to efficiently use all of the matching features
- to minimize the number of matching features which are needed to obtain a satisfying result
- to eliminate matching features which yield a transformation that does not fit a sizeable fraction of matches
- to recreate more matching features based on the existing ones
* `[ Option C ]`


---

**Q: What algorithm can be used to find the pairs the machine point between two images?**
- ASAP
- SIFT
- ALAP
- SSIP
* `[ Option B ]`


---

**Q: Which is NOT a possible cause of outlier?**
- Erroneous pairs of matching
- Noise
- The edge point which does not belong to the line we are fitting
- The sizes of two input images are different
* `[ Option D ]`


---

**Q: What is the main contribution of RANSAC**
- Providing a faster version to check for matches between keypoints
- Finding a matching between keypoints that is not affected by outliers
- Finding keypoints that are rotation, scale and transformation invariant
- Providing a good evaluation metric to score a certain mapping between keypoints
* `[ Option B ]`


---

**Q: Which of the following is not a good property of an image feature?**
- The same feature can be found in several images despite geometric and photometric transformations
- Each feature has a distinctive description
- A feature occupies a relatively small are of the image, robust to clutter and occlusion
- An image contains ideally for every pixel a single feature
* `[ Option D ]`


---

**Q: Which of the following options is not a characteristic of a `good' feature?**
- Saliency
- Efficiency
- Tractability
- Repeatability
* `[ Option C ]`


---

**Q: What is the main motivation for the introduction of the RANSAC algorithm?**
- To minimize the damage done by outliers.
- To increase the execution speed of panorama creation.
- To make image stitching easier.
- To improve feature extraction.
* `[ Option A ]`


---

**Q: How many matching pairs are (at least) required for fitting an affine transformation (or equivalently the degrees of freedom of an affine transform; in that case though the answer is 2*3=6)?**
- 1
- 2
- 3
- 4
* `[ Option C ]`


---

**Q: What will a point $(x, y) = (1,2)$  will map to after applying an affine rotation of $\theta = \pi/4$; in other words what $(x', y')$ is equal to?**
- $(\frac{\sqrt{2}}{2}, \frac{\sqrt{2}}{2})$
- $(\frac{\sqrt{2}}{2}, -\frac{\sqrt{2}}{2})$
- $(-\frac{3\sqrt{2}}{2}, \frac{\sqrt{2}}{2})$
- $(-\frac{\sqrt{2}}{2}, \frac{3\sqrt{2}}{2})$
* `[ Option D ]`


---

**Q: How many degrees of freedom does an affine transormation have?**
- 3
- 4
- 6
- 8
* `[ Option C ]`


---

**Q: What does a projective transormation preserve?**
- Parallel lines
- Straight lines
- Angles of lines
- Lengths of lines
* `[ Option B ]`


---

**Q: Affine transform matrix can be obtained by _________ of the translation, rotation and shear matrix ?**
- Addition
- Subtraction
- Convolution
- Multiplication
* `[ Option D ]`


---

**Q: RANSAC algorithms picks the model which has the most inliers lying inside an arbitrary radius.
RANSAC algorithm matches points in first image to second image. **
- Statement 1 and 2 both are correct
- Statement 1 and 2 both are wrong
- Statement 1 is correct but Statement 2 is wrong
- Statement 1 is wrong but Statement 2 is correct
* `[ Option C ]`


---

**Q: How many unknowns are there in the affine transformation**
- 2
- 4
- 6
- 8
* `[ Option C ]`


---

**Q: Which statement of RANSAC is correct**
- The bigger sampling number of points is chosen, the bigger threshold should be used.
- There could be more than one best lines with same number of inliers
- The line with least number of outliers is the resulting one
- The line with most inliers is the resulting one
* `[ Option D ]`


---

**Q: Which of the following transformation does not necessarily preserve the angles of the original image?**
- Translation
- Similarity
- Scaling
- Affine
* `[ Option D ]`


---

**Q: Why do panoramas generate a synthetic camera view?**
- To avoid undesirable and unnatural visual effects when overlapping areas with multiple distinct planes.
- To account for the 3-dimensional geometry of the scene.
- To get a more natural interpretation in 3-D. 
- All the above
* `[ Option D ]`


---

**Q: While matching features across two images, what is/are the required qualities for a detected feature to be helpful?**
- The feature should be found in both images despite undergoing various transformations from one image to another
- The feature should have a distinctive description
- The feature should occupy a relatively small area of the image
- All of the above
* `[ Option D ]`


---

**Q: Consider the following statement,
A 2D Similarity transformation preserves (1) angles, (2) lengths, (3) orientation.

Choose the correct options from the following**
- Similarity transformation preserves only (1) and (2) 
- Similarity transformation preserves only (1) and (3) 
- Similarity transformation preserves only (2) 
- Similarity transformation preserves (1), (2) and (3) 
* `[ Option D ]`


---

**Q: When can we not generate a synthetic camera view?**
- When we have two images with the same center of projection of a close by scene
- When we have two images with the same center of projection of a far away scene
- When we have two images with a different center of projection of a far away scene
- When we have two images with a different center of projection of a close by scene
* `[ Option D ]`


---

**Q: What statements are correct:
A homograph:
1. preserves straight lines
2. preserves parallel lines**
- 1
- 2
- both
- neither
* `[ Option A ]`


---

**Q: Which of the following is not a characteristic of a good feature?**
- Repeatability
- Locality
- Saliency
- None of the above
* `[ Option D ]`


---

**Q: Which of the following statements about the outlier feature points is true ?**
- Outliers donΓÇÖt affect least squares fit
- Outliers affect least squares fit
- Outliers affect RANSAC algorithm
- None of the above
* `[ Option B ]`


---

**Q: Which of the following is preserved by an affine transformation?**
- Rotation
- Straight lines
- Angles
- Lengths
* `[ Option B ]`


---

**Q: How many equations are needed to solve a linear system for affine transformations? How many matches are needed to generate this amount of equations?**
- 6 and 3 respectively
- 6 and 6 respectively
- 8 and 4 respectively
- 8 and 8 respectively
* `[ Option A ]`


---

**Q: How many degrees of freedom do the following transformations give: (i) sheer (ii) translation (iii) projection?**
- 1, 2, 4
- 1, 2, 8
- 2, 1, 4
- 2, 2, 8
* `[ Option D ]`


---

**Q: The observation of which object is not prone to the translation of view points?**
- A magic cube.
- A metal ball.
- Flat floor with complicate surface pattern.
- Two intersected walls orienting at different angles.
* `[ Option C ]`


---

**Q: In the image stitching proposed in the lecture, when is it allowed to move the camera center?**
- Always, as long as you keep track of the motion
- Never, only rotations are allowed
- When the light rays are parallel 
- When there is not too much light
* `[ Option C ]`


---

**Q: How many matches are needed to fit a similarity transformation **
- 1
- 2
- 3
- 4
* `[ Option B ]`


---

**Q: Why a reliable and distinctive descriptor is needed when building panorama images?**
- To be able to detect the same  feature  in several images despite geometric and photometric transformation.
- To be able to recognise the correct matching point even when there are more than one possible points. 
- To detect the same point independently in different images. 
- All of them.
* `[ Option D ]`


---

**Q: Which of this steps of the RANSAC loop is NOT correct?**
- Randomly select a seed group of points on which to base transformation estimate (e.g., a group of matches).
- Compute transformation from seed group.
- Find inliers to this transformation.
- If the number of outliers is sufficiently large, re-compute least-squares estimate of transformation on all of them.
* `[ Option D ]`


---

**Q: Which of the following transformations has the highest amount of degrees of freedom?**
- translations
- Similarities
- Affine
- Projective
* `[ Option D ]`


---

**Q: Which of the following steps is not a step for RANSAC?**
- Randomly select the number of points required to fit the model.
- Repeat the other steps until a fitting model is found.
- Score by the fraction of inliers within a preset threshold of the model.
- Create a line that runs in between the points. (not through the points)
* `[ Option D ]`


---

**Q: Which of the following transformations does NOT preserve parallel lines?**
- Rotations
- Scaling
- Affine
- Projective
* `[ Option D ]`


---

**Q: Changing the camera parameters ruins the Homography with RANSAC for..**
- Change in the camera direction parameters
- Change in the camera origin position parameters
- Both A and B are true
- Both A and B are false
* `[ Option B ]`


---

**Q: Which relation between the transformation and number of degrees of freedom is wrong?**
- Affine = 6
- Euclidean = 4
- Projective = 8
- Translation = 2
* `[ Option B ]`


---

**Q: If you want to reconstruct a panorama from images taken with a moving camera position: 
1) you have to be far away from the scene
2) it is only possible when you image a flat surface **
- you have to fullfill both rules
- you only have to fullfill one of the rules
-  you only have to fullfill rule 2
- reconstruction is never possible
* `[ Option B ]`


---

**Q: How many degrees of freedom does an affine transformation have?**
- 4
- 6
- 3
- Depends on the image
* `[ Option B ]`


---

**Q: What is the advantage of RANSAC over a least square fit?**
- The negative effect of outliers is usually less pronounced 
- RANSAC uses a weighting function to focus on the most important points to fit
- RANSAC fits more complex lines/trends
- There are no advantages, RANSAC is obsolete
* `[ Option A ]`


---

**Q: Which of the following transformations has a Degree of Freedom equal to 6? **
- Similarity
- Projective
- Affine
- Rigid
* `[ Option C ]`


---

**Q: What is the main advantage of a RANSAC fitting over Least Squares?**
- RANSAC is more computationally efficient
- RANSAC minimizes the effect of outliers on the fit model
- RANSAC uses all the points in order to perform the fitting
- None of the above
* `[ Option B ]`


---

**Q: How many DOF (degrees of freedom) does the 2D projective transformation have?**
- 3
- 4
- 6
- 8
* `[ Option D ]`


---

**Q: Which of the following statements is FALSE?**
- Under affine transformations, circles become ellipses.
- Under affine transformations, parallel lines stay parallel.
- Under projective transformations, parallel lines stay parallel.
- Under projective transformations, equispaced points are not preserved
* `[ Option C ]`


---

**Q: Assuming the 2D Affine transformation is a liner transformation with 6 unknowns, how many matches (correspondence pairs) are needed to solve it?**
- 1
- 2
- 3
- 4
* `[ Option C ]`


---

**Q: In order to stitch together a panorama, we first take a sequence of images, which action can help us generate a synthetic camera view?**
- Rotate the camera about its optical center
- Rotate the camera about the targeted object
- Only move the camera vertically
- Only move the camera horizontally
* `[ Option A ]`


---

**Q: Which of the following two dimensional transformations have respectively 2, 4, and 6 degrees of freedom?**
- Projective, rigid, similarity
- Affine, translation, similarity
- Translation, projective, rigid
- Translation, similarity, affine
* `[ Option D ]`


---

**Q: Which of the following is not a property of a good image feature?**
- Having less features than image pixels 
- In different images of the same thing the same features can be found despite geometric and photometric transformations
- Each feature has a distinctive description
- A feature should occupy a relatively large portion of the image, making it robust to clutter and occlusion
* `[ Option D ]`


---

**Q: Which of the following is NOT a characteristic of a good feature for feature matching?**
- The same feature can be found at multiple places in several images despite geometric and photometric transformations
- Each feature has a distinctive description
- A feature is robust to clutter and occlusion
- Features are fewer than image pixels and occupy a relatively small area of the image.
* `[ Option A ]`


---

**Q: How many correspondence pairs do we need to solve for unknown affine transformation parameters?**
- 2
- 3
- 4
- 5
* `[ Option B ]`


---

**Q: How many degrees of freedom does an affine transformation have?**
- 2
- 3
- 4
- 6
* `[ Option D ]`


---

**Q: Projection on a synthetic image plane when the camera translates works when,**
- The scene is flat
- The scene is far away
- Both A or B
- It never works
* `[ Option C ]`


---

**Q: How many parameters do we need to compute a projective transformations?**
- 3
- 4
- 6
- 8
* `[ Option D ]`


---

**Q: Which of following statements is FALSE regarding RANSAC?**
- It is very fast algorithm.
- It can always get the best result. (e.g. best matching)
- The sample is randomly drawn from the whole data points.
- The number of the sample set could be any positive number.
* `[ Option B ]`


---

**Q: You would like to translate an object in 2D by +1 unit in x direction and -1 unit in y-direction. Moreover, you would like to rotate it by +$\pi/4$. Which operation/transformation matrix would you choose?**
- $\begin{bmatrix}
    x' \\
    y' \\
    1
\end{bmatrix}$
=
$\begin{bmatrix}
    -\pi/4 & \pi/4 & 1 \\
    -\pi/4 & -\pi/4 & -1 \\
    0 & 0 & 1
\end{bmatrix}$
$\begin{bmatrix}
    x \\
    y \\
    1
\end{bmatrix}$
- $\begin{bmatrix}
    x' \\
    y' \\
    1
\end{bmatrix}$
=
$\begin{bmatrix}
    \pi/4 & -\pi/4 & 1 \\
    \pi/4 & \pi/4 & -1 \\
    0 & 0 & 1
\end{bmatrix}$
$\begin{bmatrix}
    x \\
    y \\
    1
\end{bmatrix}$
- $\begin{bmatrix}
    x' \\
    y' \\
    1
\end{bmatrix}$
=
$\begin{bmatrix}
    \pi/4 & 0 & 1 \\
    0 & \pi/4 & -1 \\
    0 & 0 & 1
\end{bmatrix}$
$\begin{bmatrix}
    x \\
    y \\
    1
\end{bmatrix}$
- $\begin{bmatrix}
    x' \\
    y' \\
    1
\end{bmatrix}$
=
$\begin{bmatrix}
    0 & -\pi/4 & 1 \\
    \pi/4 & 0 & 0 \\
    0 & 0 & 1
\end{bmatrix}$
$\begin{bmatrix}
    x \\
    y \\
    1
\end{bmatrix}$
* `[ Option B ]`


---

**Q: Which conditions must be fulfilled so that you can make use of the affine transformation instead of the more complex projective transformation?**
- Scene is far away
- Parallel lines remain parallel
- No object is near the camera
- All of the above
* `[ Option D ]`


---

**Q: What is FALSE about 2D affine transformations?**
- Affine transformations involve linear transformations
- Affine transformations involve translations
- After applying an affine transformation, parallel lines remain parallel
- Affine transformations involve projective warps
* `[ Option D ]`


---

**Q: What is NOT a part of the procedure of stitching together a panorama?**
- Blend the images together to create a mosaic
- Transform the second image to overlap with the first
- Compute transformation between second image and first
- Take a sequence of images from different positions
* `[ Option D ]`


---

**Q: RANSAC stands for:**
- RANdom SAmple Classifier
- RANdom SAmple Consensus
- Regressive Automatic Neural SAmpling Cryptography
- RANdom Sequential Anonymous Consensus
* `[ Option B ]`


---

**Q: Which are true:

I. Homography is the mapping of two projective planes with the same center of projection
II. Affine transformation has 6 degrees of freedom**
- I
- II
- Both
- Neither
* `[ Option C ]`


---

**Q: How many degrees of freedom does an affine transform have?**
- 3
- 4
- 6
- 8
* `[ Option C ]`


---

**Q: What is the objective of the RANSAC algorithm?**
- Finding a random subset of points to fit the model.
- Label each point of the matching as outlier/not outlier.
- Find a model that is robust to outliers in the data.
- Find the best model that works on all matching points.
* `[ Option C ]`


---

**Q: SIFT was mentioned in the lecture but not explained in detail. This is a question of SIFT to test self-study:
Which of the properties of SIFT is wrong?**
- changes in viewpoint can be dealt with SIFT
- It can handle changes in illumination
- It's in general fast and efficient
- It can not handle changes in illumination properly
* `[ Option D ]`


---

**Q: Which of the following statement of RANSAC is wrong?**
- It may need a lot of iterations 
- It is efficient to low inlier ratio
- The parameter needs to be finely tuned
- It's generally simple
* `[ Option B ]`


---

**Q: Which of these options is not a good characteristic for features used for image matching?**
- Saliency
- Locality
- Repeatability
- Globality
* `[ Option D ]`


---

**Q: Which of these steps does not belong to the RANSAC algorithm?**
- Randomly select a seed group of points to base the transformation estimate on
- Compute the transformation from the randomly selected seed group
- Find the inliers to this transformation
- Remove the points that are found as outliers from the dataset
* `[ Option D ]`


---

**Q: how many basic transformations does a 2d plane have?**
- 2
- 3
- 4
- 5
* `[ Option D ]`


---

**Q: what does a translation of a plane preserve?**
- lengths
- angles
- orientation
- straight lines
* `[ Option C ]`


---

**Q: Do in order for the RANSAC caclulation:
1) randomly select 2 poitns 
2) compute the transformation
3) how may points agree
4) kee ptransformation with best alligments**
- 1,2,3,4
- 4,2,3,1
- 2,1,4,3
- 1,4,2,3
* `[ Option A ]`


---

**Q: Symetric points issues of a 3d object.**
- are the points that look the same in the image
- points that are crossing on the imaginary view axis
- points that are sysmetric the same
- none of all the above
* `[ Option B ]`


---

**Q: How many degrees of freedom does an affine and projective transformation have?**
- affine - 6, projective - 8
- affine - 8, projective - 6
- affine - 6, projective - 6
- affine - 8, projective - 8
* `[ Option A ]`


---

**Q: Which of the following property is preserved by an affine transformation but not by a projective transformation?**
- Straight lines
- Parallel lines
- Orientation
- Angles
* `[ Option B ]`


---

**Q: How many degrees of freedom does the 2D Affine Transformation have?**
- 2
- 4
- 6
- 8
* `[ Option C ]`


---

**Q: Which of following is wrong about the basic procedure of stitching together a panorama?**
- Take a sequence of images from the two different positions
- Compute transformation between second image and first
- Transform the second image to overlap with the first
- Blend the two together to create a mosaic
* `[ Option A ]`


---

**Q: What is the correct, simplified order of steps required to create a panorama image.**
- 1) align images 2) find corresponding pairs 3) detect feature points
- 1) find corresponding pairs 2) detect feature points 3) align images
- 1) detect feature points 2) find corresponding pairs 3) align images
- 1) align images 2) detect feature points 3) find corresponding pairs
* `[ Option C ]`


---

**Q: Which transformation can not be described as a 2x3 matrix**
- translation
- rigid
- similarity
- projective
* `[ Option D ]`


---

**Q: which of the following is not a linear transformation?**
- Rotation
- Translation
- Sheer
- Scale
* `[ Option B ]`


---

**Q: what is the role of RANSAC in panorama image building?**
- detecting interest points on both images
- eliminating bad matches between two image's interest points
- finding matches between points of two images.
- Stitching two images together.
* `[ Option B ]`


---

**Q: What is the use for RANSAC in image matching?**
- To prevent incorrect matches from affecting the transformation matrix
- To speed up computation of the transformation matrix
- To find matches that the initial detection step might have missed
- To help compute the matrix inverse which helps us obtain a transformation matrix
* `[ Option A ]`


---

**Q: When does the RANSAC algorithm terminate?**
- Whenever a fit is found with high confidence (high agreement from not selected correspondences)
- After a fixed number of iterations
- After no disagreeing samples can be found
- After all possible subsets have been tried
* `[ Option A ]`


---

**Q: What is not an important property of a good, detectable feature?**
- repeatability 
- locality
- compactness and efficiency
- globality
* `[ Option D ]`


---

**Q: How many more degrees of freedom does a Affine transformation have in comparison to a Euclidian transformation?**
- 2
- 3
- 4
- 5
* `[ Option B ]`


---

**Q: Which of these are affine transformations?**
- Rotation
- Scaling
- Translation
- All of the above
* `[ Option D ]`


---

**Q: What main problem does RANSAC algorithm solve?**
- Robust estimation despite outliers in data
- Handle large amounts of data in small batches
- Allows for distributed optimization
- Efficient optimization of linear system
* `[ Option A ]`


---

**Q: What is a method of solving the problem of outliers in the parameter estimates when building a panorama from a sequence of images?**
- Use one least square line fitting
- Use RANdom SAmple Consensus (RANSAC) 
- Apply a custom homography to the sequence of images
- Apply an affine transformation to the sequence of images
* `[ Option B ]`


---

**Q: Choose the TRUE statement regarding one of the 2D basic transformations?**
- Similarity preserves length
- An affine transformation does not preserve parallelism
- Translation does not preserve orientation
- Similarity preserves angels 
* `[ Option D ]`


---

**Q: How many degrees of freedom does 2d affine transformations have?**
- 2
- 4
- 6
- 8
* `[ Option C ]`


---

**Q: How many matches do we need to solve for the 2D affine transformations parameters?**
- at least 1
- at least 2
- at least 3
- at least 4
* `[ Option C ]`


---

**Q: Which one of the following is an affine transformation**
- Translation
- Rotation
- Scaling
- All of the above
* `[ Option D ]`


---

**Q: Why do we need a reliable distinctive descriptor for interesting points**
- So the matching points on one image can match on the other image
- Is not necessary
- So the images give more information
- None of the above
* `[ Option A ]`


---

**Q: The RANSAC method scores the line fit by the fraction of inliers within a preset threshold of the model. What is likely to happen if the selected threshold is too low?**
- The best model found will be very accurate.
- The algorithm will recognize the line only if it is a perfect line.
- The algorithm will not recognize the line and will select a model that happens to fit a few aligned data points which might be aligned by pure coincidence. 
- The algorithm will take longer to run.
* `[ Option C ]`


---

**Q: What is a known property of a homography?**
- The image resolutions stay the same.
- Straight lines are used as axis for the transformation - to be able to preserve straight lines.
- Parallel lines are not preserved.
- The result is a homogenic transformation of the pixel space.
* `[ Option C ]`


---

**Q: To compute the homography between two images, how many unknown parameters do we need to solve for, and how many matching points do we need to do so?**
- 6 Unknown parameters, 6 matching points.
- 6 Unknown parameters, 3 matching points.
- 8 Unknown parameters, 8 matching points.
- 8 Unknown parameters, 4 matching points.
* `[ Option D ]`


---

**Q: How does RANSAC deal with outliers when applying a transformation?**
- It computes the least squares fit over all matching points of the image, and discards matching points that differ from this line that exceed some threshold.
- After repeatedly subsampling the matching points and computing their transformation, the transformation which has the most inliers is selected.
- After exhaustively searching through the possible transformations, it selects the optimal one with the most inliers.
- Ransac only inspects subsamples of matching points that agree to the best transformation so far. If a new and better transformation is found, it updates this best transformation.
* `[ Option B ]`


---

**Q: How many matches (correspondence pairs) do we need to solve for the transformation parameters for projective transformation?**
- 2
- 3
- 4
- 5
* `[ Option C ]`


---

**Q: Which of the following statements are correct:
A)Projective Transformation only preserves straight lines
B)Similarity transform preserves Straight lines
C)Affine transform preserves the parallelism**
- Only A) and C) are correct
- Only B) is correct
- A),B) and C) all  are correct
- Only A) is correct
* `[ Option C ]`


---

**Q: Which of the following is not an affine transformation?**
- Distortion
- Shear
- Rotate
- Scale
* `[ Option A ]`


---

**Q: What is a necessary condition for generating synthetic camera views?**
- The real cameras should move in a straight line, parallel to the synthetic one.
- The real cameras should have the same center of projection
- The photographed object must be sufficiently far away
- The photographed object must be a planar scene
* `[ Option B ]`


---

**Q: How can a 2D transformation where all angles and lengths are preserved?**
- Affine
- Projective
- Similarity
- rigid
* `[ Option D ]`


---

**Q: Assume two cameras make a picture: When does image reprojection not work?**
- Same centers of projection common projection plane for the scene.
- Different centers of projection, no common projection plane for the scene.
- Different centers of projection, common projection plane for the scene.
- Same centers of projection, rotation of camera, common projection plane for the scene.
* `[ Option B ]`


---

**Q: To generate synthetic camera views for making panoramas we require **
- Multiple images of same object
- 3D geometry of scene
- The same center of projection
- Distance from scene
* `[ Option C ]`


---

**Q: 2D basic transformations in correct increasing order of their Degrees of Freedom is given by option-**
- Translation < Similarity < Projective < Affine
- Translation < Similarity < Affine < Projective
- Similarity < Translation < Projective < Affine
- Similarity < Projective < Translation < Affine
* `[ Option B ]`


---

**Q: For an affine transformation, how many matches are needed at least in order to find the unknown parameters?**
- 2
- 3
- 4
- Cannot be determined.
* `[ Option B ]`


---

**Q: Which of the following statements is true regarding homography.**
- Parallel lines are preserved.
- In order to have a mapping between 2 projective planes the same center of projection is needed.
- Straight lines are not preserved.
- There are 6 parameters in the H - matrix.
* `[ Option B ]`


---

**Q: When generating a panoramic view it is required that the centre of projection of the camera remain the same. Under which condition is this requirement not necessary?**
- When the scene is very far away from the camera, then an accurate panoramic view can still be generated event with a change in the centre of projection.
- When the scene is very close to the camera, then an accurate panoramic view can still be generated event with a change in the centre of projection.
- There is no condition which removes the requirement and still maintains an accurate synthetic view
- None of the above
* `[ Option A ]`


---

**Q: The RANSAC algorithm is used for which of the following reasons**
- It discriminates against inliners in the set of matches 
- It discriminates against outliers in the set of matches 
- It compares two images and determines which feature/interest points can be used for matching.
- None of the above
* `[ Option B ]`


---

**Q: How many degrees of freedom (parameters) are there in a 2D affine model? **
- 6. 4 parameters for 2D affine transformation and 2 for translation.
- 4. 2 parameters for 2D affine transformation and 2 for translation.
- 6. 2 parameters for 2D affine transformation and 4 for translation.
- 6. 3 parameters for 2D affine transformation and 3 for translation.
* `[ Option A ]`


---

**Q: which one is wrong?**
- Using the patch would be invariant to small mismatches in alignment, or changes in lighting, rotation, or other affine transformations.
- In mathematical language, affine transformations are basically linear transformations with translations allowed.
- With linear algebra, we usually handle affine transformations using vector addition
- We can do affine transformations with a neural network by using bias inputs.
* `[ Option A ]`


---

**Q: The basic set of 2D Planar Transformations has some transformations( translation, Euclidean, Similarity, Affine and Projective) which preserve certain properties of the input image. For example, the projective transformation preserves straight lines, but not parallelism. Which of these  properties does the affine (combination of linear transformations and translations) transformation preserve?**
- Parallelism (parallel lines remain that way);
- Angles (angles are not altered after the transformation);
- Lengths (lengths are not altered by the transformation);
- Orientation ( the orientation of the image is not altered by the transformation)
* `[ Option A ]`


---

**Q: The RANSAC, or  Random Sample Concensus Algorithm is used to mute the effect of outliers in a model as much as possible. The algorithm is defined as follows:

First we will sample (randomly) the number of points required to fit the model;
Then, we will solve the model parameters using samples:
And finally, we will score this by the fraction of inliers within a preset threshold of the model.

This is repeated until we find the best model, with confidence.

What is the intuition behind the outlier suppresion this algorithm insures?**
- The random sampling will increase the probability of finding a good set of inputs for the model;
- The outliers are produced consistently, so there exists a consistent method of removing them;
- Inliers are more common than outliers;
- If a fit of the model is  applied to  outliers, then the model will not be consistent with most of the other data points;
* `[ Option D ]`


---

**Q: Which statement is false about affine and projective transformations?**
- Affine transformations are combinations of linear transformations and translations.
- The affine transformation is a special case of the projective transformation.
- The projective transformation preserves parallelism.
- The projective transformation has 8 degrees of freedom.
* `[ Option C ]`


---

**Q: Which statement is false about homography?**
- Another name for homography is projective transform.
- Homography relates the transformation between two planes.
- Homography matrix is a 3 by 3 matrix with 8 degrees of freedom.
- Straight lines are not preserved with a homography.
* `[ Option D ]`


---

