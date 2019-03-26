# Questions from `paper_8` :robot: 

**Q: What is NOT a consequence of a bad decision in the ΓÇ£Next best view selectionΓÇ¥ process? **
- Image blurring
- Faulty triangulations
- Drop in quality of pose estimates
- Less accurate triangulation
* `[ Option A ]`


---

**Q: What is bundle adjustment that is used in incremental reconstruction stage of scenes?**
- The joint non-linear refinement of the camera parameters ($P_{c}$) and point parameters ($X_{k}$) that minimizes the projection error
- The verification of potentially overlapping image pairs C
- The detection of specific features in every image
- None of the above
* `[ Option A ]`


---

**Q: What is not a common stage in the Structure-from-Motion pipeline?**
- Matching
- Elaborate search
- Geometric verification
- Feature extraction
* `[ Option B ]`


---

**Q: Which of the following is performed before Structure-from-Motion can be implemented?**
- Feature extraction 
- Matching the points
- Geometric verification
- All of the above
* `[ Option D ]`


---

**Q: Which of the following is not a correct chronological snippet from the incremental structure from motion pipeline?**
- Triangulation, outlier filtering, image reconstruction
- Initialization, image registration, triangulation
- Bundle adjustment, outlier filtering, image registration
- Feature extraction, matching, geometric verification 
* `[ Option A ]`


---

**Q: Structure-from-Motion procedure  is strugling with robustness because...**
- it is heavily contaminated by the presence of outliers
- RANSAC algorithm cannot be used in it
- it is reconstructing large scenes starting from very few images
- it has not been tested enough on real applications so far
* `[ Option A ]`


---

**Q: To decide whether two images correspond to the same scene, a correspondence search algorithm performs three steps. What is the right order for these steps?**
- a) First geometrically verify the two overlapping images, then perform feature extraction, then match the two images
- b) First perform feature extraction, then geometrically verify the two overlapping images, then match the two images
- c) First perform feature extraction, then match the two images, then geometrically verify the two overlapping images
- d) First match the two images, then perform feature extraction, then geometrically verify the two overlapping images
* `[ Option C ]`


---

**Q: The paper revisits the structure-from-motion strategy for 3D reconstruction from unordered image collections. Specifically, its target is to improve the following stage of the 3D reconstruction pipeline:**
- Image Registration
- Triangulation
- Bundle Adjustment
- all of the above
* `[ Option D ]`


---

**Q: While carrying out general SFM, after bundle adjustment how are large re-projection errors handled?**
- Those observations are filtered out
- Those observations are kept the same
- Re-projection errors are ignored
- None of the above 
* `[ Option A ]`


---

**Q: The factorization model proposed in "Shape and Motion from Image Streams under Orthography: a Factorization Method" solves the problem of**
- Noise and partially filled-in measurement matrix (as a result of occlusion)
- Bad illumination in images
- Blurring
- None of the above
* `[ Option A ]`


---

**Q: What will mostly likely cause incremental reconstruction to fail?**
- Initilization of a bad first image pair.
- Projective transformations.
- Underexposed images.
- Images of different resolutions.
* `[ Option A ]`


---

**Q: What is the function of Bundle Adjustment?**
- Mitigate accumulated errors obtained by the propagation of uncertainties in the camera pose
- Mitigate accumulated errors obtained by triangulation
- Mitigate accumulated errors obtained by anomalies in the images (e.g. time stamps)
- All of the above
* `[ Option A ]`


---

**Q: what is Bundle adjustment in the context of shape from motion?**
- A linear process for removing inliers
- A iterative process for removing outliers
- A linear process for removing outliers
- A iterative process for removing inliers
* `[ Option B ]`


---

**Q: Which statement is incorrect?**
- frequent problem in Internet photos are watermarks, timestamps, and frames (WTFs) that incorrectly link images of different landmarks.
- Feature tracks often contain a large number of outliers due to erroneous two-view verification of ambiguous matches along the epipolar line.
- The incremental structure-from-motion pipeline exist out of: images\rightarrow feature  extraction \rightarrow matching \rightarrow geometric verification \rightarrow initialization \rightarrow image registration \rightarrow triangulation \rightarrow bundle adjustment \rightarrow outlier filtering (possible to repeat loop starting at image registration) \rightarrow reconstruction.
- All of the statements are correct. 
* `[ Option D ]`


---

**Q: What is NOT a step in the incremental structure-from-motion pipeline?**
- Feature Extraction
- Matching and Geometric Verification
- Triangulation and Bundle Adjustment
- Coordinate Averaging
* `[ Option D ]`


---

**Q: Which of the following options is NOT a key problem in the use of state-of-the-art incremental SfM as a general-purpose method, according to the authors of the paper?**
- Scalability
- Robustness
- Accuracy
- Computational cost
* `[ Option D ]`


---

**Q: Consider an $8\times 8$ grid; which set of points among the ones given below will render the highest score $S$? Consider the case where $l=3$ levels are utilized and the given $8\times 8$ grid coincides with the image partitioning at the last level.

NOTE: points are given in (x,y) format referring to highest resolution binning of the image.**
- $(3,3), (3,4), (4,3), (4,4)$
- $(3,3), (2,3), (2,2), (3,2)$
- $(3,3), (4,3), (3,2), (4,2)$
- $(3,3), (3,4), (4,3), (3,2)$
* `[ Option B ]`


---

**Q: What is a main challenge to Structure from Motion the paper tries to overcome?**
- Systems fail to register a large fraction of images that empirically should be registrable.
- Systems produce broken models due to mis-registrations or drift.
- Both A and B are correct.
- None of the above.
* `[ Option C ]`


---

**Q: What are some challenges for the state-of-the-art SfM algorithms, as mentioned in the paper?**
- robustness
- training data
- computational time
- all of the above
* `[ Option A ]`


---

**Q: Which element was not key in improving the state of the art performance in Structure from Motion paper?**
- Geometric verification strategy for augmenting the scene graph.
- Next best view selection to improve the incremental reconstruction process.
- Reduced computational cost of the triangulation method.
- Selective prefiltering of image bundles.
* `[ Option D ]`


---

**Q: The scene graph is:**
- A structure that shows the correspondence between features of different frames.
- A descriptor of the transformations that should be carried in order to combine a set of pictures from a common scene.
- A non-directed graph that contains the camera parameters of the different frames. The edges contain nodes whose parameters are the closest.
- A graph in which images are nodes and the edges between them indicate a verified relationship.
* `[ Option D ]`


---

**Q: The components of the proposed SfM algorithm does not improve the start of the art in terms of:**
- accuracy
- efficiency
- scalability
- robustness
* `[ Option C ]`


---

**Q: Why is there no need to apply global Bundle Adjustment after each iteration of incremental SfM?**
- Bundle adjustment only needs to be applied at the end of the algorithm
- Bundle adjustment only affects the model locally
- Global bundle adjustment has no use unless the model is complete
- Incremental SfM does not need bundle adjustment, since errors are not accumulated
* `[ Option B ]`


---

**Q: When finding overlap between 2 images (correspondence search), which step is unusual?**
- Feature matching
- Bundle adjustment
- Geometric verification
- Local feature extraction
* `[ Option B ]`


---

**Q: Choosing a suitable initial image pair is critical in the SfM approach discussed in the paper. What will help in getting more robust results?**
- Initializing from a dense location in the image graph with many overlapping cameras
- Initializing from a dense location in the image graph with few overlapping cameras
- Initializing from a sparse location in the image graph with many overlapping cameras
- Initializing from a sparse location in the image graph with few overlapping cameras
* `[ Option A ]`


---

**Q: Consider general-purpose Structure from mostion (SfM) system. What is NOT the key problems in state of art SfM systems?**
- Good feature selection
- Robustness
- Accuracy
- Completeness
* `[ Option A ]`


---

**Q: which of following is not a procedure of Bundle Adjustment?**
- Parameterization
- Redundant view mining
- Re-Triangulation
- Iterative Refinement
* `[ Option B ]`


---

**Q: Which of these steps is not an standard part of the Incremental Reconstruction, but is proposed in the paper?**
- Image registration
- Triangulation
- Redundant view mining
- Bundle adjustment
* `[ Option C ]`


---

**Q: Choose the correct sequence :**
- Feature Extraction -> Matching -> Geometric Validation ->  Incremental Reconstruction 
- Feature Extraction -> Geometric Validation  -> Matching->  Incremental Reconstruction 
- Feature Extraction ->  Incremental Reconstruction -> Geometric Validation  -> Matching
- Feature Extraction ->  Incremental Reconstruction -> Matching -> Geometric Validation 
* `[ Option A ]`


---

**Q: Which of the following is not the key problem towards building a truly general-purpose pipeline?**
- robustness
- accuracy
- incompleteness
- scalability
* `[ Option C ]`


---

**Q: What factorization method is used to factor the measurement matrix into two matrices?**
- orthographic projection
- rank theorem
- orthography theorem
- singular-value decomposition technique
* `[ Option D ]`


---

**Q: If you have been given two pieces of algorithms for SIFT and RANSAC respectively, how would you use them for correspondence search in incremental SfM? If both are used, what is the most probable ordering?**
- Only SIFT is needed, RANSAC has nothing to do with correspondence search.
- Only RANSAC is needed, SIFT has nothing to do with correspondence search.
- First SIFT, then RANSAC
- First RANSAC, then SIFT
* `[ Option C ]`


---

**Q: Incremental SfM is a sequential processing pipeline with an iterative reconstruction component. Which of the following statements regarding the pipeline is true?**
- The first stage is the correspondence search. The order of this stage is feature selection, geometric verification and matching.
- Triangulation is a crucial step in Sfm, as it increases the stability of the existing model through redundacy and it enables registration of new images by providing additional 2D-3D correspondences.
- Image registration and triangulation are separate procedures and their products are uncorrelated.
- Bundle adjustment (BA) is a linear refinement of camera parameters $P_c$ and point parameters $X_k$ that minimizes the re-projection error.
* `[ Option B ]`


---

**Q: Which of the below options related to the paper 'Structure-from-motion (SfM) revisited' are correct**
- Choosing a initial pair is critical, since the reconstruction may never recover from a bad initialization
- One of the reasons some SfM algorithms suffer the issue of mis-registrations is correspondence search producing an incomplete scene graph
- For sparsely matched image collections, exploiting transitive correspondences boosts triangulation completeness and accuracy
- All of the above are correct
* `[ Option D ]`


---

**Q: What are the usual steps of an incremental Structure from Motion (SfM) pipeline?**
- Images $\Rightarrow$ Correspondence Search $\Rightarrow$ Incremental Reconstruction $\Rightarrow$ Reconstruction
- Images $\Rightarrow$ Triangulation $\Rightarrow$ Matching $\Rightarrow$ Reconstruction
- Images $\Rightarrow$ Bundle Adjustment $\Rightarrow$ Geometric Verification $\Rightarrow$ Reconstruction
- Images $\Rightarrow$ Feature Extraction $\Rightarrow$ Outlier Filtering $\Rightarrow$ Reconstruction
* `[ Option A ]`


---

**Q: What is the correct order of the steps for Incremental reconstruction?**
- Initialization, image registration, triangulation, bundle adjustment
- image registration, initialization, triangulation, bundle adjustment
- Initialization, geometric verification, triangulation, bundle adjustment
- Initialization, image registration, feature matching, bundle adjustment
* `[ Option A ]`


---

**Q: Which statement is false about the paper ΓÇ£Structure-from-Motion RevisitedΓÇ¥?**
- The first stage of an incremental SfM pipeline is correspondence search, which finds scene overlap in the input imaged and identifies projections of the same points in overlapping images.
- A scene graph represents images as nodes and verified pairs of images as edges.
- The second stage of an incremental SfM pipeline, incremental reconstruction, uses the scene graph to generate an estimation of the camera pose for each image and a 3D reconstruction as a sparse point cloud.
- The purpose of the Bundle Adjustment step is to prevent inaccuracies in the correspondence search to propagate in the incremental reconstruction phase and vice versa.
* `[ Option D ]`


---

**Q: What are the main problems that we are aiming to tackle with a structure-from-motion algorithm?**
- Robustness, accuracy, completeness and scalability
- Robustness, accuracy, completeness 
- Robustness, accuracy
- Robustness
* `[ Option A ]`


---

**Q: Which of the following statements is wrong in the context of paper 8?**
- The geometric verification strategy augments the scene graph
with information subsequently improving the robustness of
the initialization and triangulation components. 
- An iterative BA, retriangulation, and outlier filtering strategy significantly
improves completeness and accuracy by mitigating drift effects.
- The next best view selection maximizing the robustness and accuracy of the incremental reconstruction process.
- All of the above
* `[ Option D ]`


---

**Q: What is a higher-level flow for SFM (Structure From Motion) calculation? **
- 1. Keypoint Extraction, 2. Keypoint Matching, 3. Keypoint Matching Verification (via RANSAC, Epipolar Geometry etc), 4. Reconstruction of 3D World 
- 1. Keypoint Matching, 2.Keypoint Extraction , 3. Keypoint Matching Verification (via RANSAC, Epipolar Geometry etc), 4. Reconstruction of 3D World 
- 1. Keypoint Matching, 2.Keypoint Extraction , 3. Reconstruction of 3D World, 4.Keypoint Matching Verification (via RANSAC, Epipolar Geometry etc)  
- None
* `[ Option A ]`


---

**Q: What is not disadvantage of using internet photos for scene reconstruction?**
- There many many photos
- They often contain watermarks and frames
- The camera distribution in the scene is very not-uniform
- The camera parameters are unknown
* `[ Option A ]`


---

**Q: The proposed SfM algorithm improves the state of the art techniques in terms of...**
- Completeness
-  Robustness
- Accuracy
- All the previous ones are correct. 
* `[ Option D ]`


---

**Q: Although incremental structure-from-motion techniques have tremendously advanced, robustness, scalability, completeness and accuracy are key aspects which difficult the process of building a truly general-purpose pipeline. The paper ΓÇ£Structure-from-motion RevisitedΓÇ¥ proposes some improvements to the current state of the art methods, contributing to the improvement of all afore-mentioned key aspects except:**
- Robustness.
- Completeness.
- Scalability.
- Accuracy.
* `[ Option C ]`


---

**Q: Given a very large collection of images from which we want to reconstruct a 3D scene, one cannot apply the basic structure-from-motion algorithm learned in class to the whole dataset. Instead,**
- start by finding which images overlap with which, and what are the projections that relate those pairs of images.
- the images that do not have enough interest points or are too blurry need to be discarded. Then the structure-from-motion algorithm can be applied to the whole remaining dataset.
- apply the SfM algorithm to each unique pair of images.
- all images should firts be downsampled to $128\times 128$.
* `[ Option A ]`


---

**Q: Which of the following is part of the typical SFM pipeline?**
-  feature extraction. 
-  feature matching.
-  triangulating scene points. 
-  all of the above 
* `[ Option D ]`


---

**Q: In an Incremental structure from motion algorithm, correspondence search between images is an important step. Which of the following stages are NOT involved in such a search**
- Feature Extraction
- Feature Matching
- Geometric Verification
- None of the above
* `[ Option D ]`


---

**Q: Which of the following statements is NOT the main contribution of the paper?**
- A geometric verification strategy that augments the scene graph with information subsequently improving the robustness of the initialization and triangulation components.
- A next best view selection maximizing the robustness and accuracy of the incremental reconstruction process.
- A robust triangulation method that produces significantly more complete scene structure.
- Create a Levenberg-Marquardt method for solving bundle adjustment problem.
* `[ Option D ]`


---

**Q: What part of the Incremental Structure-From-Motion pipeline is not improved by the research conducted in the paper ΓÇ£Structure-from-Motion RevisitedΓÇ¥?**
- Geometric Verification
- Triangulation
- Bundle Adjustment
- Feature Extraction 
* `[ Option D ]`


---

**Q: Why is it better to have uniformly distributed feature points in sfm?**
- Uniform distribution leads to better conditioned triangulation
- It isn't better to have uniformly distributed points
- Uniform distribution leads to better matching between images
- Uniform distribution leads to more accurate triangulation in the absence of noise
* `[ Option A ]`


---

**Q: What is NOT a possible reason why the systems fail to register a large fraction of images that empirically should be registrable, or the systems produce broken models due to mis-registrations or drift?**
- Correspondence search producing an incomplete scene graph
- The reconstruction stage failing to register images due to missing or inaccurate scene structure
- Correspondence estimation did not include sufficient redundancy
- Take into account of missing features due to occlusion during reconstruction stage
* `[ Option D ]`


---

**Q: SfM algorithms can fail to produce fully satisfactory results in terms of completeness and robustness. What is NOT a possible cause of this?**
- The correspondence search may be producing an incomplete scene graph
- The reconstruction stage might fail to register images due to missing or inaccurate scene structure
- Image quality might be too low to produce reliable features
- The motion might be too small to create a scene
* `[ Option D ]`


---

**Q: What is most likely the computationally most expensive step in a structure from motion task?**
- Next best view selection
- Triangulation
- Matching
- Bundle adjustment
* `[ Option D ]`


---

**Q: What are the main contributions of the paper?**
- Augmenting the scene graph
- Selecting the best next best view
- Make triangulation in a robust and efficient way
- All of the above
* `[ Option D ]`


---

**Q: What is not a part of correspondence search?**
- Feature extraction
- Matching
- Geometric verification
- Detection
* `[ Option D ]`


---

**Q: SfM is the process of reconstructing 3D structure from its projections into a series of images taken from different viewpoints, and what is Incremental SfM?**
- It is a sequential processing pipeline with an iterative reconstruction component
- It is a discrete processing pipeline with a non-iterative reconstruction component
- It is a sequential processing pipeline with a non-iterative reconstruction 
- component
It is a discrete processing pipeline with a iterative reconstruction component
* `[ Option A ]`


---

**Q: When can we add a new scene point **
- At the moment it appears in a new frame
- 2 seconds after it appears in a new frame
- When it was visible in at least two frames captured from a different view point
- It depends on the camera resolution
* `[ Option C ]`


---

**Q: What is the correct order of the following incremental Structure-from-Motion steps after initialization? 1) Bundle Adjustment 2) Outlier Filtering 3) Triangulation 4) Image Registration**
- 2 - 4 - 3 - 1
- 2 - 4 - 1 - 3
- 4 - 2 - 3 - 1
- 4 - 3 - 1 - 2
* `[ Option D ]`


---

**Q: Which of the following describes the Perspective n Point (PnP) problem?**
- It involves estimating the pose P and instrinsic parameters ( for uncalibrated images) 
- It describes the transformation of a purely rotating or a moving camera capturing a planar scene
- It describes the relation for a moving camera through the essential or fundamental matrix
- It involves estimating the point correspondences between images
* `[ Option A ]`


---

**Q: Which of the following steps is not part of the SfM pipeline?**
- Bundle adjustment
- Triangulation
- Inlier filtering
- Geometric verification
* `[ Option C ]`


---

**Q: What is the downside of internet photos?**
- There is an uncertainty if the place and date are correct
- There is to mutch data to process for a good approchimations
- There are problemns with the cheap camera's 
-  All of the above
* `[ Option A ]`


---

**Q: Which of the following is not a part of correspondence search, which finds scene overlap in provided images?**
- Feature extraction
- Geometric Verification
- Matching
- Image Registration
* `[ Option D ]`


---

**Q: What is the common pipeline for incremental structure from motion?**
- feature extraction -> matching -> geometric verification -> reconstruction
- feature extraction -> geometric verification -> matching -> reconstruction
- feature extraction -> matching -> reconstruction -> geometric verification
- feature extraction -> reconstruction -> matching -> geometric verification
* `[ Option A ]`


---

**Q: Which of the following statements about structure from motion is/are correct?

1.  One of the reasons current state-of-the-art structure from motion systems produce insatisfactory results, is due to the production of incomplete scene graphs
2. Redundant view mining is the clustering of cameras into groups in which they have a high scene overlap**
- Only 1 is correct
- Only 2 is correct
- Both are correct
- Both are incorrect
* `[ Option C ]`


---

**Q: Which statement is true regarding the steps of the correspondence search.**
- Features should be invariant under geometric changes.
- The naive approach for matching is suitable for a large amount of images.
- From the matching step it is guaranteed that the corresponding features do not map to the same point.
- None of the above.
* `[ Option A ]`


---

**Q: Which of these statements concerning structure from motion (SfM) is incorrect?**
- Choosing a suit- able initial pair is critical, since the reconstruction may never recover from a bad initialization
- SfM verifies the matches by trying to estimate a transformation that maps feature points between images using projective geometry
- If a valid transformation maps a sufficient number of features between the images, they are considered geometrically verified
- Additional triangulations do not improve the initial camera pose through increased redundancy
* `[ Option D ]`


---

**Q: What is the function of incremental structure-from-motion?**
- 3D reconstruction from unordered image collections
- 2D reconstruction from unordered image collections
- Object recognition
- Gussian blur
* `[ Option A ]`


---

**Q: Steps involved in Incremental Reconstruction in correct order are given by**
- Initialization -> Image Registration -> Bundle Adjustment -> Outlier Filtering -> Triangulation
- Initialization -> Image Registration -> Outlier Filtering -> Triangulation -> Bundle Adjustment 
- Initialization -> Image Registration -> Triangulation -> Bundle Adjustment -> Outlier Filtering
- Initialization -> Triangulation -> Bundle Adjustment -> Outlier Filtering -> Image Registration
* `[ Option C ]`


---

**Q: Which of the following process is a bottleneck to estimate Structure from motion(Sfm)?**
- Triangulation
- Next Best View Selection
- Bundle Adjustment
- None of the above
* `[ Option C ]`


---

**Q: Can we use the eigenvalue decomposition instead of the SVD for the eight point algorithm?**
- No, the matrix is singular therefore only SVD can be used
- Yes, both techniques are equivalent
- No, it will result into different solutions
- Yes, both techniques can be used with some additional changes
* `[ Option A ]`


---

**Q: Which of the following is true:

I. Challenging components in structure from motion are completeness, robustness, accuracy and efficiency.
II. The open-source structure from motion software COLMAP succeeded in facing the challenges in I. and designed a general-purpose system.**
- I
- II
- Both
- Neither
* `[ Option C ]`


---

**Q: Which one is wrong ?**
- This paper proposes a SfM algorithm that overcomes key challenges to make a further step towards a general-purpose SfM system.
- The proposed components of the algorithm improve the state of the art in terms of completeness, robustness, accuracy, and efficiency
- The author run experiments on datasets only to evaluate the overall system
compared to state-of-the-art incremental and global SfM systems.. 
- A synthetic experiment was conducted in this project to evaluate how well the score S reflects the number and distribution of points.
* `[ Option C ]`


---

**Q: 0**
- 0
- 0
- 0
- 0
* `[ Option A ]`


---

