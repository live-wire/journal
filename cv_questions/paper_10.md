# Questions from `paper_10` :robot: 

**Q: Which criteria doesnΓÇÖt need to be satisfied to connect patches p and q?**
- Spatial proximity
- Temporal overlap
- Common view 
- Epipolar criteria
* `[ Option D ]`


---

**Q: Which of the following can be a problem while reconstructing a scene chronology? **
- Constant illumination.
- Dense temporal sampling.
- Photo manipulation.
- a and c are correct.
* `[ Option D ]`


---

**Q: What is/are assumption made by Kevin Matzen and Noah Snavely in the paper: ΓÇ£Scene ChronologyΓÇ¥?**
- It is possible to reconstruct accurate cameras and points using SfM. This assumption relies on there being sufficient scene content that is constant over time, or at least there being sufficient continuity in a scene from one time to the next. 
- The geometry of the scene is largely static, but that the appearance can change dramatically from one time to the next. Such appearance changes are a key source of dynamism in scenes like Times Square, where storefronts, billboards, and other ads are changing, often from one week to the next.
- The scene geometry can be well-approximated with a set of planes, as in other work on urban reconstruction
- All of the above assumptions are made in the paper.
* `[ Option D ]`


---

**Q: Which of the following is true?**
- We can reconstruct cameras and points using SfM
- A patch p is photoconsistent-visible in view v $\in$ I if p is in front of v, p projects inside the field of view of v, and p passes an occlusion test with other patches
- A patch p is projection-visible in view v $\in$ I if p is projection-visible in view v and further p satisfies a photoconsistency criterion in view v
- Negative observations do not occur when a patch does not exist at a particular point in time
* `[ Option A ]`


---

**Q: Which of the following is not a correct ordering of steps in the 3D reconstruction pipeline in "Scene Chronology" by Matzen et al.?**
- Observation generation -> interval estimation -> under-segmentation
- Under-segmentation -> plane-time RANSAC -> Texture mapping and rendering
- Plane-time RANSAC -> texture mapping and rendering -> 3D reconstruction
- 3D reconstruction -> observation generation -> interval estimation
* `[ Option C ]`


---

**Q: In the paper ΓÇ£Scene ChronologyΓÇ¥, which metric is used to evaluate classification of observations being inside a time interval or outside a time interval?**
- Accuracy.
- True positive rate.
- False positive rate.
- F1-score.
* `[ Option D ]`


---

**Q: In $4D$ reconstruction one wants to turn a collection of $2D$ images into a $3D$ reconstruction of the scene they capture, while also displaying the changes over time. In the paper about scene chronology, the authors discuss a method that aims at determining when in time a specific scene was visible (for example, an advertisement billboard with your favorite soda), which estimated the interval during which the scene existed. For this, the authors pick the smallest interval that**
- maximizes the precision ratio (scenes with the soda)/(all scenes in the interval) and maximizes the recall ratio (scenes with the soda in the interval)/(all scenes with the soda).
- maximizes the precision ratio (scenes with the soda)/(all scenes in the interval).
- maximizes the recall ratio (scenes with the soda in the interval)/(all scenes with the soda).
- contains all timestamps of pictures that contain the soda billboard scene.
* `[ Option A ]`


---

**Q: What is (are) the limitation(s) of the scene chronology method that was introduced by the authors?**
- Photo misregistration can prevent accurate timestamp estimation
- Many structures in the real world cannot be approximated accurately using planar segments
- The assumption that objects come into and leave existence from the landscape only once can be violated.
- All of the above
* `[ Option D ]`


---

**Q: How the 4-D reconstruction find the wrong timestamp of photos?**
- Compare the photo with the existing model
- Check whether the photo has a well-formated timestamp
- Grayscale the photo
- Stiching the photo with the existing ones
* `[ Option A ]`


---

**Q: What is NOT a key issue with the proposed method?**
- Photo misregistration can prevent accurate timestamp estimation
- Many structures in the real world cannot be approximated accurately using planar segments
- Changes in the physical structure of the buildings might violate the static geometry assumption
- Temporal clustering might not work because of sparse data
* `[ Option D ]`


---

**Q: What is NOT a challenge when creating a temporal 3D-reconstruction from online photo databases**
- Enough data
- Photo manipulation
- Illumination changes
- Timestamps
* `[ Option A ]`


---

**Q: Why do negative observations (projection-visible, but not photoconsistent-visible) occur? **
- wrong timestamp
- dynamic occlusion
- patch does not exist at that point in time
- all of the above
* `[ Option D ]`


---

**Q: Which of the following statements is true?**
- Scene chronology attempts to add the dimension of time in 3D reconstruction
- Although scene chronology can make fairly accurate time classification of images, there is a strict upper bound on the number of collected images
- It is not possible to use scene chronology to estimate timestamps on images without time reference
- Scene chronology can be performed even under large-scale geometric changes
* `[ Option A ]`


---

**Q: What is the definition of a positive observation edge $e_{ij+}$?
A positive observation edge**
- encodes a temporal event where $p_j$ is projection-visible to $v_i$
- encodes a temporal event where $v_j$ is projection-visible to $p_i$
- encodes a temporal event where $p_j$ is photoconsistent-visible to $v_i$
- encodes a temporal event where $v_j$ is photoconsistent-visible to $p_i$
* `[ Option C ]`


---

**Q: What is the problem that prevent accurate timestamp estimation**
-  Misregistrtation
- Misalligment
- Both
- None
* `[ Option A ]`


---

**Q: In scene chronology, the images obtained from a dynamic environment are grouped into ΓÇ£plane-time cuboidsΓÇ¥. What does one of those units (ΓÇ£plane-time cuboidΓÇ¥) consists of?**
- All images of a 3D cube or prism (e.g. all images of a building) at a single point in time are grouped together. The ΓÇ£plane-time cuboidsΓÇ¥ have 3 dimensions (3D in space).
- All images of a 3D cube or prism (e.g. all images of a building) taken over time are grouped together. The ΓÇ£plane-time cuboidsΓÇ¥ have 4 dimensions (3D in space + 1D in time).
- All 2D images taken of a planar object over time are grouped together. The ΓÇ£plane-time cuboidsΓÇ¥ have 3 dimensions (2D in space + 1D in time).
- All 2D images taken of a planar object at a single point in time are grouped together. The ΓÇ£plane-time cuboidsΓÇ¥ have 2 dimensions (2D in space).
* `[ Option C ]`


---

**Q: How did the authors estimate the initial time interval of image patches?**
- They took the earliest and latest positive observation timestamps (of the cameras) as the time interval.
- They ordered the positive observations by time and trimmed off the bottom and top $k$-percentile.
- They considered negative observations as well as positive observations, and formulated the problems as a classification problem.
- None of the above.
* `[ Option C ]`


---

**Q: Why and how is under segmentation used in the chronology system?**
- To preprocess the large dataset so that RANSAC can be run
- To preprocess the patches into "surfaces" so that RANSAC is only run on relevants points in 3d space
- A & B
- Neither A nor B.
* `[ Option C ]`


---

**Q: Regarding scene chronology, certain points are detected at multiple timepoints. However, these detections have noise (false positives, incorrect timestamps etc). What is the best way to determine scene chronology?**
- Take the earliest and latest positive observations
- Ordering observations by time and trimming the bottom and top k-percentile
- Using both positive and negative detections and formulate the problem in terms of classification
- Only confirm a detection if the same point is detected again within a certain timestep
* `[ Option C ]`


---

**Q: Which of the following techniques is NOT used in the reconstruction∩╝ƒ**
- RANSAC algorithm
- sfm
- PMVS
- ANN Feature Matching
* `[ Option D ]`


---

**Q: What is the first step in a space-time reconstruction pipeline?**
- Reason about which cameras observe which points.
- Use the observation profile of each 3D point to reason about the time interval in which it existed.
- Segment the scene into sets of points that are spatio-temporally coherent.
- Generate a dense 3D point cloud from a set of images.
* `[ Option D ]`


---

**Q: Which following statement about the method for urban scene reconstruction is false?**
- It is robust and scalable to scenes with noisy, individual point observations. 
- Timespan estimation is much more precise for segments that are clearly limited to some range of time in the middle of the datasetΓÇÖs timeline.
- The automatic and Internet-scalable system for segmenting 3D point cloud reconstructions for scenes containing changing appearance
- It leverages spatial, temporal, and co-visibility cues to segment points into spatio-temporally consistent components 
* `[ Option C ]`


---

**Q: Consider structure from motion, which constructs 3D model from photos. What statement is correct?**
- By segmenting the scene into spatio-temporally consistent cluster, It is possible to visualize in 3D model how particular places changed over time
- It is impossible to add time dimension (fourth dimension) to 3D model inferred from SfM
- SfM is highly robust and can be applied to massive datasets, thus it's relatively easy to add time dimension to 3D model
- None of the above
* `[ Option A ]`


---

**Q: The proposed method is not shown to be:**
- Robust
- Scalable
- Fast
- Fully automated
* `[ Option C ]`


---

**Q: which of the following methods proved to work best for estimating the initial time interval?**
- Take the first and last positive observation timestamps as the bounds of the interval
- Trim off the bottom and top k-percentile positive observation timestamps
- Any method that uses a large portion of both negative and positive observations
- optimize the positive/negative classification by achieving the best precision and recall possible
* `[ Option D ]`


---

**Q: Which assumption was considered for scene chronology?**
- The geometry of the scene remains same
- The geometry of the scene varies 
- The appearance of the scene remains same
- None of the above
* `[ Option A ]`


---

**Q: In the scene chronology paper, which of the following terms is not a criteria for matching patches with high spatio-temporal affinity?**
- Common View
- Temporal Overlap
- Magnitude Matching
- Spatial Proximity
* `[ Option C ]`


---

**Q: In the paper: "Scene Chronology", the authors performed a preprocessing step to sub divide the set of input patches before applying RANSAC on the observed patches for one of the following reasons**
- Direct RANSAC's random sampling approach would be unsuited to the large number of patches (> $10^6$) which the authors were working with due to computational intensity.
- Since RANSAC is a non-deterministic approach, the accuracy reduces with an increase in the points under consideration relative to the number of inliers
- The sub-division step eliminated noisy input and hence would improve RANSAC's accuracy
- None of the above
* `[ Option A ]`


---

**Q: Which of the following is not a criterion for connecting patches?**
- Spatial proximity
- Temporal overlap
- Common view
- Hue similarity
* `[ Option D ]`


---

**Q: Which assumption is NOT made for the time dependent 3D reconstruction?**
- The images are sampled uniformly in time
- The geometry of the scene is static
- The scenes are similar enough to allow a proper reconstruction
- The scene geometry is a set of orthogonal planes
* `[ Option A ]`


---

**Q: Choose the incorrect choice : 
Patches are assumed to be connected if they have**
- High Spatial Proximity
- Temporal Overlap
- Common View
- None
* `[ Option D ]`


---

**Q: Which one of the following is an issue with Scene Chronology development?**
- Image misregistrations
- Poor classification accuracy statistic 
- Large scale databases
- None
* `[ Option A ]`


---

**Q: Why is it useful to compute the spatio-temporal graph?**
- The graph relates spatial and temporal information about the patches.
- By interpolating the patches RANSAC is more likely to find good hypotheses due to a larger input.
- Segmentation methods built on random sampling are easier to scale.
- By under-segmenting the patches RANSAC is more likely to quickly find good hypotheses.
* `[ Option D ]`


---

**Q: Which of the following statements is NOT the issue of "Scene Chronology"?**
- Photo mis-registration can prevent accurate timestamp estimation.
- Many structures in the real world cannot be approximated accurately using planar segments.
- The assumption that objects come into and leave existence only once can be violated.
- Timespan estimation is less precise for segments that are clearly limited to some range of time in the middle of the datasetΓÇÖs timeline.
* `[ Option D ]`


---

**Q: What assumptions are necessary for scene chronology?**
- There is sufficient scene content that is constant
- The geometry of the scene is largely static
- Scene geometry can be well-approximated with a set of planes
- All of the above.
* `[ Option D ]`


---

**Q: Which is the most important assumption for the "Scene Chronology" algorithm to work?**
- Presence of images with just correct timestamps
- The analized scenes should not present 3D changes over time
- Availability of large image sets at different times in the day
- All of the above
* `[ Option B ]`


---

**Q: What are the main assumptions for the scene chronology?**
- Enough constant scene content and mostly static geometry of the scene over time
- Changes over time are constant
- The camera is the same on every scene
- None of the above
* `[ Option A ]`


---

**Q: How do Matzen et Snavely NOT achieve a high-performance 4D reconstruction system?**
- By using Structure-from-Motion
- By using Multi-view Stereo
- By employing RANSAC only a tiny chunks of input data
- By manually time-tagging/geo-tagging the pictures
* `[ Option D ]`


---

**Q: In the paper a pipeline is described for the space-time reconstruction. Which of the following is not a step in the pipeline?**
- Given a dense 3D-construction, expand the visibility graph.
- Estimate initial time intervals.
- Construct a spatio-temporal graph and perform over-segmentation.
- All of these above are steps in the pipeline.
* `[ Option C ]`


---

**Q: How can we make a dense 3D reconstruction of a scene from photos from uncalibrated cameras (such as mobile phones etc.)? **
- By first applying structure-from-motion, and using the computed motions in a multi-view stereo algorithm
- By first applying multi-view stereo, and using the computed motions in a structure-from-motion algorithm
- By using the timestamps that were provide in a plane-time RANSAC
- A dense 3D reconstruction can not be generated from uncalibrated cameras
* `[ Option A ]`


---

**Q: Which of these statements concerning 3D reconstruction methods is incorrect?**
- Spatial and temporal affinity between points can be used to segment a scene into a set of spatio-temporal consistent clusters
- The usage of timestamp data can help improve the reconstruction of 3D environments
- Recall is a measure to evaluate the performance of 3D reconstruction with
- 3D reconstruction can not be used to predict missing data in a photo, because it would interfere with the 3D reconstruction itself
* `[ Option D ]`


---

**Q: What is true about a patch that is photoconsistent-visible?**
- The patch must project inside a field of view 
- The patch must pass an occlusion test
- The cross-correlation score between the patch and the reference patch must be high enough
- All of the above
* `[ Option D ]`


---

**Q: In the scene chronology algorithm, two image patches are connected to each other if they meet certain conditions. Which of the following is not one of them?**
-  Spatial proximity: patch1 and patch2 are in a similar 3d position.
-  Temporal proximity: path1 and patch2 belong to approximately the same time period.
-  Common view: patch1 and patch2 have been seen at least by one common camera.
-  Inlier proximity: Patch1 and Patch2 have a sufficiently large number of inliers between them after running RANSAC.
* `[ Option D ]`


---

**Q: What is NOT a limitation to the technique as described in the paper about Scene Chronology?**
- Photo misregistration can prevent accurate timestamp estimation
- Many structures in the real world cannot be approximated accurately using planar segments
- The estimation of the timespan in the middle of that dataset's timeline is much more accurate than very early or very late in the timeline
- The technique only works for urban scenes
* `[ Option D ]`


---

**Q: Which of the following steps is not a step for scene chronology?**
- Dense 3D reconstruction
- Initial time interval estimation
- Re-evaluating dataset entries
- Plane time clustering
* `[ Option C ]`


---

**Q: The article of scene chronology proposed a method to reconstruct urban scene through time. More specifically, how does it achieve the chronology?**
- It leverages the time dimension as a unique signal for segmenting scenes into meaningful objects
- It provides a model of temporal information, where points from different times simply co-exist
- It explores the availability of vast numbers of photographs from sources
- It actively prunes multiple points that occupy the same spatial location
* `[ Option A ]`


---

**Q: What is the main contribution of this paper? Why is this important?**
- ItΓÇÖs the first full automatic algorithm to detect 3D point clouds and therefore better scalable
- This algorithm can detect point clouds with far greater accuracy than al the algorithms considered before
- The algorithm is more efficient, runs in O(log(n)) instead of O(n^2)
- This is a verification study
* `[ Option A ]`


---

**Q: Which assumption is violated when using structure from motion methods for scenes where photos are taken throughout time?**
- The camera angles are not consistent
- The camera translations for multiple photos are problematic for estimating motion
- The objects that change over time in the scene break the consistency assumption
- New objects in the scene occlude the required 'fixed level markers'
* `[ Option C ]`


---

**Q: What is one of the main assumptions in order for the Scene Chronology Algorithm to work properly?**
- a) that objects in the same scene remain a static geometry
- b) that objects in the same scene dynamically change over time
- c) that objects in the same scene only come or leave once
- d) that objects in the same scene remain the same appearance
* `[ Option A ]`


---

**Q: For scene chronology, some common to traditional () reconstruction systems and some unique to the () reconstruction problem?**
- 3D; 4D
- 2D; 3D
- 3D; 3D
- 3D; 2D
* `[ Option A ]`


---

**Q: what is precision?**
- the ratio of total observation compared to the number of positive observations
- the ratio of the number of positive observations compared to all positive observations ever done
- the ratio of positive observations inside the total number of observations
- the ratio of total observations compared to all observations ever done.
* `[ Option C ]`


---

**Q: Which of the following criteria is NOT used to connect two patches**
- spatial proximity
- temporal overlap
- common view
- pixel intensity histogram similarity
* `[ Option D ]`


---

**Q: Plane-Time RANSAC combines a standard plane-fitting inlier criterion with a new
time interval hypothesis criterion. A patch is considered an inlier to this hypothesis if :**
-  It is within some spatial distance of the hypothesis plane.
-  It is normal is within some cosine distance of the hypothesis plane.
- It is time interval has a high degree of overlap with the hypothesis time interval.
- All of the above.
* `[ Option D ]`


---

**Q: What is 4D modeling?**
- A 3D reconstruction that maps camera settings on a new axis
- Model that estimates when individual 3D points in the scene existed, then uses spatial and temporal affinity
- A technique that considers x, y, z and color to represent an object
- None of the above
* `[ Option B ]`


---

**Q: Which of the following criteria need to be satisfied in order to connect patches $p$ and $q$ in the space-temporal graph?
i) Spatial proximity: $p$ and $q$ are within a 3D distance $\delta_{d}$, and have normals that differ by no more than an angle $\delta_{d}$.
ii) Temporal overlap: $p$ and $q$ have time observations with a high degree of overlap.
iii) Common view. p and q are seen by at least one common camera from the original
PMVS reconstruction $(|V_{PMVS}(p) \cap V_{PMVS}(q)| > 0)$.

iii) Common view. p and q are seen by at least one common camera from the original
PMVS reconstruction $(|V_{PMVS}(p) \cap V_{PMVS}(q)| < 0)$.**
- Only ii)
- Both i) and iii) 
- Only i), ii) and iii)
- All  i), ii), iii) and iv) 
* `[ Option C ]`


---

**Q: What are the problems that might arise while one is constructing a scene and reasoning it's changing appearance over time as described in the paper 'Scene chronology '**
- Structures cannot be approximated by planar segments
- Changes in the physical structure of the building
- Reappearance of the same object in different timeline
- All of the above
* `[ Option D ]`


---

**Q: Which one is wrong?**
- There is a limitation for example: Some semantic elements are periodic. Segmentation granularity dependent on thresholds
- Their system works even though there is no enough redundancy to identify
incorrect timestamps.
- In this paper there are two representations:  Point-based and Semantic segmentation 
- The goal of this project is 1. Rich reconstructions 2. Capturing fine-grained temporal structure 3. Enabling visualizations where they can dial back
to any point in time
* `[ Option B ]`


---

**Q: What are the main advantages of the 3D point reconstructions method?**
- It is first fully automatic and Internet scalable system for segmenting 3D point cloud reconstructions for scenes containing largely static geometry, but changing appearance.   
- It is able to robustly estimate time intervals for each point in a reconstruction by considering both positive and negative observations, despite severe noise.
- The segments produced by the system often are natural units (signs, facades, art, etc.), and are useful for both visualization and time stamping applications
- All of them are correct. 
* `[ Option D ]`


---

**Q: What is the use of the visibility graph in scene chronology**
- To determine patch occlusion and existence of patches in a scene at a point in time
- To aid scene reconstruction
- To provide patch parameters for texture mapping
- To find a subset of points for performing RANSAC on
* `[ Option A ]`


---

**Q: Why is it important to use temporal information in scene reconstruction?
I - Scenes undergo changes over time and can change dramatically
II - If points from multiple time stamps coexist, the reconstructed scene may not be accurate
III - Not utilizing time information may lead to broken reconstructions, veritable chimeras of different objects or appearances conflated in time**
- I, II
- II, III
- I, III
- I, II, III
* `[ Option D ]`


---

**Q: What is not a limitation in current techniques for scene chronology estimation?**
- Photo misregistration can prevent accurate timestamp estimation
- The assumption of static geometry might be violated
- The assumption that objects come into and leave existence only once might be violtated
- Timespan estimation is much more precise for segments that are clearly limited to some range of time at the extremes of the dataset's timeline
* `[ Option D ]`


---

**Q: What are the problems that might arise while one is constructing a scene and reasoning it's changing appearance over time as described in the paper 'Scene chronology '?**
- Structures cannot be approximated by planar segments
- Changes in the physical structure of the building
- Reappearance of the same object in different timeline
- All of the above options are correct
* `[ Option D ]`


---

**Q: Which of the following criteria is not used to satisfy in paper 10 for connecting two patches p and q?**
- Spatial proximity
- Temporal overlap
- Common view
- Spatial overlap
* `[ Option D ]`


---

**Q: Dynamic structure of motion adds the element of**
- Surprise
- Spacial Distortion
- Temporal Distortion
- Time
* `[ Option D ]`


---

**Q: To chronicle a scene over time, some assumptions had to be made, choose the incorrect assumption.**
- The assumption of accurate reconstruction via structure from motion.
- The assumption the geometry of the scene is largely static.
- The scene geometry can be well approximated with a set of planes.
- Each of the objects in the scene should be present at all time.
* `[ Option D ]`


---

**Q: Which of the following factor CANNOT be the possible reason of negative observations when constructing a visibility graph?**
- An occlusion of the scene when a photograph is taken
- A specific scene exists but not is not observed
- The assumed simply did not exist in the past in reality
- Misregistration of data
* `[ Option C ]`


---

**Q: What is not an issue of the system developed in the paper?**
- photo misregistration can prevent accurate timestamp estimation
- many structures in the real world cannot be approximated accurately using planar segments
- changes in the physical structure of the buildings violate the static geometry assumption
- timespan estimation is much less precise for segments that are clearly limited to some range of time in the middle of the datasetΓÇÖs timeline
* `[ Option D ]`


---

**Q: For evaluating initial time interval estimation, it was not used as a metric:**
- Precision
- Recall
- F1
- Accuracy
* `[ Option D ]`


---

**Q: which of the statements is not true**
- time estimation is more for segments that are clearly limited
- photo misregistration can prevent accurate timestamp estimation
- using planar segments is able to approximate all structures in real world
- natural units are produced more often by the algorithm in paper
* `[ Option C ]`


---

**Q: Which of the following is true:

I. Scene Chronology shows changes in a scene over time.
II. Scene Chronology can also be used to estimate the time a picture was taken.**
- I
- II
- Both
- Neither
* `[ Option C ]`


---

**Q: What are the challenges faced during scene construction using screen chronology?**
- Changes in the physical structure of the building
- Structures cannot be approximated by planar segments
- Reappearance of the same object in different timeline
- All of the above
* `[ Option D ]`


---

**Q: State-of-the-art 3D reconstruction methods remain largely agnostic to the time domain. The tenth paper attemps to bring temporal meaning back to these reconstructions. how does it achieve this?**
- Where the output from an existing 3D reconstruction method can be
taken as input and segmented into distinct space-time objects, each with a specific extent in time;
- By providing an output model where points from different time spans simply co-exist;
- by actively pruning multiple points that occupy the same spatial location;
- by  automatically sequencing unstructured images captured over short timescales, based on motion constraints;
* `[ Option A ]`


---

**Q: What is encoded by a negative observation edge in a visibility graph?**
- A temporal event, where a patch is photoconsistent-visible.
- A temporal event, where a patch is not photoconsistent-visible.
- none of the above.
- Visibility graphs don't comprise edges.
* `[ Option B ]`


---

**Q: Which statement is true regarding the chronicle of a scene?**
- The proposed method is based on analyzing the visibility between the images and the timestamps on the same image.
- The geometry of the scene is dynamic.
- The proposed method is planned for a limited amount of images.
- None of the above.
* `[ Option A ]`


---

