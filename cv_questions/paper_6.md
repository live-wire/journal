# Questions from `paper_6` :robot: 

**Q: Consider image stitching by obtaining global 2D transformations to align one image with the other. What are possible bad artifacts of using this method?**
- Assuming a single global transformation such as a homography in most cases will be incorrect and will lead to misalignments and ghosting effects.
- Assuming a single global transformation is incorrect as images are not necessary the same dimensions
- Assuming a single global transformation such as a homography in most cases will be incorrect and will lead to blurred image.
- Assuming a single global transformation is incorrect as different parts of the images has its own transformations.
* `[ Option A ]`


---

**Q: The proposed stitching method is (1) more robust to parameter selection, but (2) less automated compared with state-of-the-art methods.**
- both statements are false
- statement 1 is true, statement 2 is false
- statement 1 is false, statement 2 is true
- both statements are true
* `[ Option B ]`


---

**Q: Which could be a potential problem of extrapolating a projective transform in no overlapping regions while performing image stitching:**
- It could introduce perspective distortions.
- It could create ghost effects.
- There is no problem. The projective transform can be easily extrapolated from the available points.
- There would be extra computation costs given that the transformation would have to be computed in a larger surface.
* `[ Option A ]`


---

**Q: What are the main error when creating a panorama of 2 (or more images) as stated in the experiments of ΓÇ£Adaptive as-natural-as-possible imageΓÇ¥?**
- Parallax error in overlapping area and perspective distortion in nonoverlapping area.
- Parallax error in nonoverlapping area and perspective distortion in overlapping area.
- transformation errors in overlapping area and homography distortion in nonoverlapping area.
- transformation errors in nonoverlapping area and homography distortion in overlapping area.
* `[ Option A ]`


---

**Q: When stitching together images, extrapolating homographic transformations into regions where an image does not overlap with another image:**
- Produces extreme and unnatural scaling effects;
- Produces no problems;
- Reduces the RANSAC cost;
- Efficiently reduces the number of photoΓÇÖs needed for a panorama.
* `[ Option A ]`


---

**Q: Which of the following is used to compensate for the camera motion?**
- Global similarity transformation
- Global homography transformation
- Homography linearization
- None of the above
* `[ Option A ]`


---

**Q: How can the perspective distortion be reduced in an image-stitching scenario?
A - By extrapolating (using a weighted linear combination) of the homographic matrices to non-overlapping regions 
B - By using a linearized homography for non-overlapping regions (by considering points along the image boundary)
C - By using a global similarity transformation (by estimating camera motion between the two
D - None**
- A and B
- B and C
- A and C
- D
* `[ Option B ]`


---

**Q: State-of-the-art image stitching procedures are based on extrapolating a projective transform in the non-overlapping regions of the given images. Which is the main drawback for this?**
- Introduction of blurrings in the outer part of the reconstructed mosaic
- Creation of errors inside the overlapping regions
- Arising of perspective distortions in regions that are far from the boundary
- Shifting of portions of the images in regions far from the boundary
* `[ Option C ]`


---

**Q: Artifacts seem to occur in mosaics due to:**
- Relative camera motion
- Illumination changes
- Optical aberrations
- All of the above
* `[ Option D ]`


---

**Q: When stitching images together to create panorama images, each image has its own plane in the scene and an intersection angle with the image projection plane (the plane in the scene of the final result). Which homography transformation of which image can be used to derive the optimal similarity transformation that represents the camera motion?**
- a) the plane that is most parallel to the image projection plane
- b) the plane that is most perpendicular to the image projection plane
- c) you have to take the average of all intersection angles of all images
- d) you can take any image from the set
* `[ Option A ]`


---

**Q: In the paper 'Adaptive As-Natural-As-Possible Image Stitching' a novel stitching method is proposed. Which of the following statements is false?**
- In the novel method they used a smooth stitching field over the entire target image, while accounting for all the local transformations.
- Earlier methods focused on obtaining global 2D transformations to align one image with the other.
- In 2D image stitching it is possible to estimate the stitching field accuracy.
- Assuming a single global transformation such as a homography, will be incorrect except under special conditions.
* `[ Option C ]`


---

**Q: Which of the following procedure is NOT included in LO-RANSAC?**
- least squares on inliers
-  iterative least squares
- random sampling of non-minimal samples
- find outliers from samples
* `[ Option D ]`


---

**Q: What is a disadvantage of As-Projective-As-Possible (APAP) stitching?**
- Perspective distortion
- Blur
- Noise
- Image deformation
* `[ Option A ]`


---

**Q: What are the benefits of the Adaptive As-Natural-As-Possible (ANAP) novel stitching method when compared to previous methods?**
- No visible parallax in the overlapping regions.
- Mitigation of the perspective distortion in the non-overlapping regions.
- Less dependent on the parameters chosen and automatic computation of the global similarity transform.	
- All the options presented above.
* `[ Option D ]`


---

**Q: Which of the following approach is not applied to make comparative experiments in this paper?**
- Microsoft image composite editor ICE
- APAP
- SPHP with global homography
- ICE+APAP
* `[ Option D ]`


---

**Q: What is one of the key improvements of the proposed Adaptive ANAP image stitching method?**
- much higher executions speeds
- non-overlapping regions between images are deformed with less artifacts
- the accuracy of matching overlapping regions in images is much higher
- it is more scalable, meaning it can use more images simultaneously compared with previous methods
* `[ Option B ]`


---

**Q: When stitching several images together to create a tile, homographies are often used because**
- they come from the mathematical framework under which the several images can be projected into a common plane
- they are the only way to deal with the fact that pixels are commonly handled in homogeneous coordinates
- they are invariant under changes in the homogeneous coordinate of a pixel
- they are the only transformation that is representable in a $3\times3$ matrix
* `[ Option A ]`


---

**Q: What is the case when referring to a parallax effect?**
- The position or direction of an object appears to differ when viewed from different positions.
- Not all points of interest of all objects in the image are the visible due to light intensity.
- The image contains artifacts due to lighting or movement.
- None of the above.
* `[ Option A ]`


---

**Q: What is(are) the advantage(s) of smooth combination of two stitching fields**
- A fully continuous and smooth stitching field with no bending artifacts 
- A fully continuous and smooth stitching field with no bending artifacts 
- Full benefits of the state-of-the-art alignment accuracy offered by APAP
- All above
* `[ Option D ]`


---

**Q: What is most difficult part in creating a natural looking panorama when using state-of-the-art image stitching techniques?**
- Finding points to match two images
- Parallax effects
- Calculating the homography
- Obtaining a single global 2D transformation
* `[ Option B ]`


---

**Q: A reason why the technique proposed makes image stitching both more adaptive and natural is:**
- combining formerly employed techniques for the first time
- focusing entirely on the non-overlapping regions of the images
- linearizing the homography on the non-overlapping regions
- linearizing the homography over the entire images combined with global similarity
* `[ Option C ]`


---

**Q: What technique does the author do to remove outliers?**
- RANSAC
- LST
- APAP
- SPHP
* `[ Option A ]`


---

**Q: Which of the following is no property of adaptive APAP?**
- The algorithm accounts for differences in brightness or lighting
- It has the same accuracy as APAP
- It doesn't result in any bending artifacts
- A global similarity transform is used to improve the perspective compared to APAP
* `[ Option A ]`


---

**Q: Which of the following options is not a characteristic of the proposed novel stitching method?**
- It compensates for parallax when large motion exists.
- It is more automated compared with state-of-the-art methods.
- It is more robust to parameter selection.
- It allows for obtaining the best perspective in the panorama, based on multiple images.
* `[ Option A ]`


---

**Q: Regarding the extraction of the global similarity transformation, transformations $T_1, \cdots, T_4$ are calculated with corresponding angles $\theta_1, \cdots, \theta_4$ as listed below. Which transformation will be selected as the global similarity transformation?**
- $\pi/6$
- $\pi/4$
- $\pi/3$
- $\pi/2$
* `[ Option A ]`


---

**Q: What is a common problem in image stitching?**
- Visible parallax errors
- Perspective distortions
- Both visible parallax errors and perspective distortions
- None of the above
* `[ Option C ]`


---

**Q: Perspective effect are caused by **
- Applying similarity transformation to overlapping areas
- Applying similarity transformation to non overlapping areas 
- Applying homography transformation to non overlapping areas
- Applying homography transformation to overlapping areas
* `[ Option C ]`


---

**Q: Which statement related to the proposed algorithm is correct**
- Linear homography in overlapping area will reduce perspective distortion
- Homographic transformation in non-overlapping area will reduce perspective distortion
- Applying global similarity transformation only in non-overlapping area will cause unnatural visual effect
- The homographic inlier group with biggest rotation angle will be chosen as the correspondence to compute global similarity transformation
* `[ Option C ]`


---

**Q: Why is the homographic transformation linearised?**
- To reduce the distortions caused due to homographic transformation in the non-overlapping areas.
- To reduce the extrapolation artifacts caused due to homographic transformation in the overlapping areas.
- Neither a or b.
- To achieve both a and b. 
* `[ Option A ]`


---

**Q: Consider the following statements related to the discussion in the paper - "Adaptive As-Natural-As-Possible Image stiching"
(a) One of the advantages of the methods proposed in the paper, is that the global similarity transformation is automatically estimated.
(b) Only over lapping regions are transformed by the calculated global similarity transformation to retain the perspective properties of the target image
(c)The reference image is also transformed to compensate for changes made to the target image 
Choose the correct option **
- All the statements are correct
- All the statements are wrong
- (a) is correct; (b) and (c) are wrong
- (a) and (c) are correct; (b) is wrong
* `[ Option D ]`


---

**Q: What statement is true:
When stitching together two pictures,
1. Global Homography can distort the perspective of non-overlapping regions
2. The true transformations needed differ per area of the picture**
- 1
- 2
- both 
- neither
* `[ Option C ]`


---

**Q: When stitching images how can the perspective distortion be reduced in the overlapping areas?**
- Homography Linearization
- Homography Translation
- Homography Sepaartion
- Homography Transformation
* `[ Option A ]`


---

**Q: Which of the following is true about the Adaptive ANAP Image stitching method?

1. The proposed method is less dependent on the choice of parameters
2. Since the amount of similarity transformations is large, the method chooses the one with the lowest rotation angle**
- Only 1 is correct
- Only 2 is correct
- Both are correct
- Both are incorrect
* `[ Option C ]`


---

**Q: Which of the following is NOT correct for the global similarity transformation?**
- Assumes camera motion can be approximately compensated by similarity transforms
- Not using all the matching points to calculate the transform
- Assumes that there are multiple imaginary planes in the scene that can be used to obtain the transformation
- The optimal plane is the one perpendicular to the camera projection plane
* `[ Option D ]`


---

**Q: Will the method proposed in this paper work nicely on all image stiching problems?**
- Yes, it is demonstrated in the paper on multiple image sets.
- No, It only works on image sets with small intensity differences, since this method seems to have less smoothing than the other methods.
- No, it is overtrained on the sets shown in the paper and does probably not work on other image sets
- No, it only works well for image sets with small rotations, since they automatically choose the one with the lowest rotation angle as the
best candidate.
* `[ Option D ]`


---

**Q: Why are global similarity transformations applied to panoramas?**
- Reduce Distorsions
- Make images look more natural
- Compensate for the camera motion
- All of the previous answers are correct
* `[ Option D ]`


---

**Q: Which of the following troubles is not stated in the paper as a trouble when trying to stitch images?**
- Shape distortion
- Parallax error
- Light emission distortion
- Perspective distortion
* `[ Option C ]`


---

**Q: A 3x3 homographic transformation has how many degrees of freedom?**
- 8
- 9
- 7
- 3
* `[ Option A ]`


---

**Q: What is not an advantage of the proposed method: using local and global homography to interpolate for non-overlapping regions? **
- it reduces parallax
- better perspective distortion
- faster
- more automated
* `[ Option C ]`


---

**Q: This paper presents a more natural-looking method for image stitching. What principle(s) have/has been used to establish this stitching?**
- Local (linearized) homography and a global similarity transformation
- Local (linearized) homogoraphy
- global similarity transformation
- LO-RANSAC
* `[ Option A ]`


---

**Q: Which of the following statements are correct for the algorithm proposed in Paper 6? **
- RANSAC is used to removed outliers after obtaining the feature point matches
- Global similarity transform is used to estimate for camera motion
- The algorithm produces panoramas which are more natural and do not have that wide angle effect
- All of the above
* `[ Option D ]`


---

**Q: The idea behind the paper "Adaptive As-Natural-As-Possible Image Stitching" is that using a unique global transformation to describe the movement of the camera may introduce artifacts that are not present in reality. To prevent this, the authors make use of**
- Linearized/non-linearized homography + global similarity
- Inverse homography + global similarity
- Only global similarity
- Linearized homography + global similarity + brightness correction
* `[ Option A ]`


---

**Q: After obtaining the linearized homography, using a global similarity transformation can help us to further reduce the prospective distortions, what is False for this process?**
- Using all point matches can make the approach more accurate
- It can be used to compensate the effects of camera motion
- To optimize the transformation, we can use RANSAC to remove some outliers
- The corresponeding transformation with the smallest rotation angle should be chosen 
* `[ Option A ]`


---

**Q: Which of the following properties of the novel stitching method provided in ΓÇ£Adaptive As-Natural-As-Possible Image StitchingΓÇ¥  are true?
\begin{enumerate}
    \item Due to the continuous and smooth stitching field, bending of artifacts can be avoided
    \item The novel stitching method benefits of the alignment accuracy provided by SPHP
\end{enumerate}
Which of the statements are true?**
- Statement 1 and 2 are true
- Statement 1 is true, statement 2 is false
- Statement 1 is false, statement 2 is true
- Statement 1 and 2 are false
* `[ Option B ]`


---

**Q: Which of the following is NOT an advantage of LO Ransac?**
- It applies a local optimization (LO) step to promising hypothesis generated from random minimal samples
- Its probability of obtaining a correct solution is close to the theoretical value derived from stopping criterion
- It is fairly complex and increases the computation cost of RANSAC
- It improves the accuracy of RANSAC
* `[ Option C ]`


---

**Q: What is the main problem in using the cylindrical model for image stitching?**
- Ghosting occurs when  the camera is not perfectly rotating around its vertical axis
- Parallax error is still a main source of error and is not minimized
- The perspective is severely warped
- There are no main disadvantages in using the cylindrical model
* `[ Option A ]`


---

**Q: Which of the following statements is NOT the reason for the authors proposing the new stitching method?**
- Moving DLT framework needs to choose a proper offset to avoid extrapolation artifacts.
- The extrapolation of homographic transformation in the
non-overlapping areas produces extreme and unnatural scaling effects.
- Finding a global similarity transformation using all point matches may result in non-optimal solution particularly when the overlapping areas contain distinct image planes.
- The new method has more parameters that can be tuned to make a better stitching result.
* `[ Option D ]`


---

**Q: Which statement about the global similarity transformation used for the presented method is wrong?**
- It describes the camera motion between the target and the reference images.
- It helps reducing distortions.
- After obtaining the feature point matches, RANSAC is used to remove outliers.
- It uses the global homography transformation to uniquely determine the global similarity.
* `[ Option D ]`


---

**Q: What does the smooth combination of the two stitching fields (linearized) homography and global similarity NOT help achieve?**
- A fully continuous and smooth stitching field with no bending artifacts
- Improved perspective in the non-overlapping regions using a global similarity transform
- Full benefits of the state-of-the-art alignment accuracy offered by APAP
- Compensation for parallax when large motion exists
* `[ Option D ]`


---

**Q: Adaptive As-Natural-As-Possible Image Stitching

I. has robust paramater selection and thus is more automated
II. has alignment accuracy of APAP (As-Projective-As-Possible) stitching**
- I
- II
- Both
- Neither
* `[ Option C ]`


---

**Q: Which of the following statements are true about the Adaptive As-Natural-As-Possible Image Stitching method?**
- The method only relies on local image data to compute the warping of the stitched image.
- The proposed method ignores perspective warping effects in order to gain runtime performance.
- The method combines local homography and global transformations.
- A downside of the method is increased dependence on selection of parameters.
* `[ Option C ]`


---

**Q: The procedure of image stitching includes following steps:
a. Take a sequence of images from the same
position
b. If there are more images, repeat
c. Compute transformation between second image
and first
d. Blend the two together to create a mosaic
e. Shift the second image to overlap with the first
which of the following sequence is correct of this procedure?**
- a b c d e
- a c e d b
- a e d b c
- d c b e a
* `[ Option B ]`


---

**Q: Which of these statements is incorrect?**
- If the overlapping areas have multiple distinct planes in two images that are to be stitched together, deriving a single global similarity transformation from the global homography may not be sufficient
- The homography transformation corresponding to the plane that is most parallel to image projection plane can be used to derive the optimal similarity transformation that represents the camera motion
- Using a similarity transformation in the non-overlapping areas of two images that have to be stitched together will introduce perspective distortions
- If the global similarity transformation approximates the camera motion between the target and the reference images, the estimated similarity transform can be used to compensate for the camera motion
* `[ Option C ]`


---

**Q: what does the gloval similarity transformation do?**
- see how similar 2 images are
- correct perspective distortions in stitching 2 images together
-  stitch 2 images together
- none of the above
* `[ Option B ]`


---

**Q: How can RANSAC been used for obtaining feature points**
- canculate a average 
- With a threshold to find a homography of the plane with largest inliers
- Find and remove weak feature points 
- none of the above
* `[ Option B ]`


---

**Q: We want to create a panorama using adaptive as-natural-as-possible image stitching. How are the perspective distortions in the non-overlapping areas reduced?**
- By extrapolating the linearized homography of the non-overlapping areas to the overlapping areas
- By extrapolating the linearized homography of the overlapping areas to the non-overlapping areas
- By incorporating the similarity transformation that has the lowest rotation angle of the overlapping areas in the global similarity transformation
- By incorporating the similarity transformation that has the lowest rotation angle of the non-overlapping areas in the global similarity transformation
* `[ Option D ]`


---

**Q: Which of following statements about the authors proposed algorithm is false?**
- Moving DLT method to estimate the local homography
- Computing global similarity transformation between the reference and the target images.
- Choosing the transformation with the lowest rotation angle as best candidate.
- Combining the linearized homograph with the global similarity   
* `[ Option D ]`


---

**Q: In the paper, the DLT (Direct Linear Transformation) algorithm is used for**
- estimating local homography
- feature detection
- perspective manipulation
- it is not used in the method presented in the paper
* `[ Option A ]`


---

**Q: What is the purpose of linearizing the homography transformation for image stitching?**
- A non linear transformation will result in a very distorted and unnatural looking image 
- A non linear transformation will cause ghosting in the resulting image
- A non linear transformation will cause misalignments when stitching two images.
- To prevent the parallax effect on the images.
* `[ Option A ]`


---

**Q: Why does extrapolating homomorphic transformations outside the overlapping areas of two images produce warped results?**
- The effects of the parameters on the extrapolated points will be nonlinear
- The homomorphic transformation matrix would be non-invertible
- The resulting transform would no longer be homogeneous
- It doesn't, it is just as effective as homography linearization
* `[ Option A ]`


---

**Q: According to the paper, what are most of the problems in 2D image stitching related to?**
- Lighting differences
- Estimating the stitching field in the 3D world
- Determining good features to use for matching
- Scaling
* `[ Option B ]`


---

**Q: How does linearizing the homography help image stitching?**
- Reduce distortion in non-overlapping areas of image
- Simplify calculation of homography
- Allows stitching of more images
- Simplifies finding of an optimal similarity transform
* `[ Option A ]`


---

**Q: Which is NOT used in the novel stitching method proposed by the paper?**
- Local homography
- Homography linearization
- Global similarity transformation
- Global translation transformation
* `[ Option D ]`


---

**Q: Compared to SPHP combined with APAP, in what aspects is the proposed method better?**
- Not sensitive to parameter selection
- No need of manual correction for the rotation angle
- Compute the appropriate global similarity transform automatically
- All of the above
* `[ Option D ]`


---

**Q: What is the problem in image stitching and homography?**
- Stitching needs too many samples for any transformation
- Complex interactions between the 3D world and the camera parameters
- Homography assumption is too complex
- None of the above
* `[ Option B ]`


---

**Q: When stitching images, why is extrapolating the projective transform in the non-overlapping regions not an ideal solution?**
- Non-overlapping regions need to be kept intact.
- This technique adds large amounts of noise in non-overlapping regions.
- This technique introduces severe perspective distortion in regions far from the boundary.
- Non-overlapping regions become blurry when using this approach.
* `[ Option C ]`


---

**Q: Which statement about As-Projective-As-Possible (APAP) or the image stitching technique described in this paper is true?**
- APAP extrapolates the projective transform in the non-overlapping regions
- The paper's technique estimates the local homography and linearizes it in the non-overlapping regions
- Both (A) and (B) are true.
- None of the above.
* `[ Option C ]`


---

**Q: What are the advantages of combining a Global Similarity transform from overlapping and the linearized homography in non-overlapping regions?**
- smooth stitching field with no bending artifacts
- Improved perspective in the non-overlapping regions using
a global similarity transform
- Accuracy as close as to the one offered by the state-of-the-art 'As Projective as Possible Algorithm'
- All of the above.
* `[ Option D ]`


---

**Q: When is finding a global similarity transformation using all point matches more likely to result in a non-optimal solution?**
- When the overlapping areas contain distinct image planes
- When the approximation of camera motion is performed statically
- When not using the global homography transformation to uniquely determine the global similarity
- When not choosing the image with the smallest intersection angle to derive the optimal similarity transformation
* `[ Option A ]`


---

**Q: What can be observed if we use an extrapolation of a homographic transformation in a non overlapping area of two images for image stitching?**
- Unnatural scaling effects
- Increase in noise
- A Perspective swap
- A fine result
* `[ Option A ]`


---

**Q: Which are the global transformations that we need to consider for adaptive As-Natural-As-Possible Image Stitching**
- Gaussian weighting
- Homography linearization
- Global similarity transformation
- All of the above
* `[ Option D ]`


---

**Q: What is the advantage of using homography linerization in the non - overlaping areas?**
- Decreases the execution time.
- Maximizes the clearness of the image.
- Minimizes perspective distortion in 2D.
- All of the above.
* `[ Option C ]`


---

**Q: In the paper: "Adaptive As-Natural-As-Possible Image Stitching", the authors propose a technique which is an improvement over early image stitching techniques. Which of the following is NOT a disadvantage of these earlier stitching techniques?**
- Earlier techniques assumed a global transformation for aligning images and this assumption does not always hold except in special conditions
- Complex interactions in the 3D scene and the camera view make it difficult to estimate the stitching field accuracy which in turn allows for errors such as misalignment 
- Earlier techniques did not factor in advances in computational hardware and as such had become unsuitable to modern computer vision tasks
- None of the above
* `[ Option C ]`


---

**Q: which one is wrong?**
- This┬áproposed an AANAP warping, which linearizes the homography in the non-overlapping regions, mitigate the perspective distortion.
- RANSAC is used to iteratively segment the matching points
- In this paper, AANAP┬ácombined the projective transformation and the global similarity transformation
- It provides stitched panorama results with more visible parallax and perspective distortions
* `[ Option D ]`


---

**Q: Which stitching images , there are several artifacts that can be formed in the final project. What of the following is not an artifact of image stitching?**
- Misalignments;
- Ghosting;
- Perspective distortion;
- Color distortion;
* `[ Option D ]`


---

**Q: Which statement about the paper ΓÇ£Adaptive As-Natural-As-Possible Image StitchingΓÇ¥ is false?**
- Their approach uses RANSAC.
- Compared to other state of the art approaches, it is more robust to parameter selection.
- Their approach linearizes the homography to reduce the perspective distortion in the overlapping areas.
- Compared to other state of the art approaches, their approach results in no visible parallax when large motion exists.
* `[ Option D ]`


---

