# Questions from `paper_4` :robot: 

**Q: By having a video camera pointing to the ground near a corner of a wall,**
- subtle radiance changes were used to count the number of people on the other side of the corner.
- subtle shaking of the image was used to count the number of people on the other side of the image.
- subtle inverted images of the camera (the one recording) were extracted
- nothing was able to be detected, as at least two edges are needed to extract information about the scene on the other side.
* `[ Option A ]`


---

**Q: Which statement is false?**
- The 1-D videos reveal the number of people moving around the corner
- The 1-D videos reveal the angular position of a person moving around the corner
- The method proposed in this paper is an approach using recovery under active illumination
- The method proposed in this paper removes the effect of the sceneΓÇÖs background illumination by subtracting the videoΓÇÖs mean image
* `[ Option C ]`


---

**Q: A person is walking behind a corner, what can one edge camera detect?**
- only that there is someone walking
- the angular position of the person
- the 2D position of the person
- all of the above
* `[ Option B ]`


---

**Q: Wat can be said about these statements: 1: A single edge camera allows us to reconstruct a 90 degree angular image of an occluded scene. 2: Two edge cameras allows us to reconstruct a 180 degree angular image of an occluded scene**
- Statement 1 and 2 are correct.
- Statement 1 and 2 are incorrect.
- Statement 1 is correct and 2 is incorrect.
- Statement 1 is incorrect and 2 is correct.
* `[ Option C ]`


---

**Q: How many edge cameras are present in a doorway?**
- 1
- 2
- 4
- 8
* `[ Option C ]`


---

**Q: By using one corner as a camera to recover hidden scenes it is *not* possible to obtain: **
- The number of people in the scene.
- XZ two dimensional position of the subject over time.
- Angular speed and position. 
- Material properties of the objects such as their color.
* `[ Option B ]`


---

**Q: What is not a disadvantage of ToF cameras?**
- They require specialized and comparatively expensive detectors with fine temporal resolution
- They are limited in how much light they can introduce in the scene to support imaging
- They can only produce very low resolutions
- They are vulnerable to interference from ambient outdoor illumination
* `[ Option C ]`


---

**Q: Which of the following sentences is false? In an indoor controlled setup, by filming the ground behind a single corner (Note: this is NOT the case of a doorway):**
- It is typically possible to obtain the position and speed (in the hidden scene) of a single object in time.
- It is typically possible to count the number of people in the hidden scene.
- A 2-dimensional reconstruction of the hidden scene can be obtained.
- Detection of possible subjects in the hidden scene can be carried out using a normal consumer camera
* `[ Option C ]`


---

**Q: Which of the following requirements is necessary to turn a corner into a camera:**
- At least two cameras
- Multiple images (so e.g. a video) of a corner without the occluded scenery around the corner
- An environment without noise such as rain or low lighting
- The exact geometry of a doorway when multiple cameras are used in a stereo edge camera setup
* `[ Option B ]`


---

**Q: In the paper "Accidental pinhole and pinspeck cameras: revealing the scene outside the picture" it is given the example of a room at night where the only light sources come from outside and go through the window. How can the illumination sources be represented mathematically? Like..**
- Gaussian functions
- Window functions (gate functions)
- Delta functions
- It is not possible to say since the composition of the room is not known
* `[ Option C ]`


---

**Q: Which among these is a part of the edge camera system**
- Visible scenes
- Hidden scenes
- Ground plane
- All of the above
* `[ Option D ]`


---

**Q: Which of the following statements is correct?**
- This method can accurately determine the location of an object, with a single 'edge camera' by looking at the angular width if an object.
- This method can be used on all kinds of surfaces
- This method requires an high quality industrial camera
- This method does not need a reference measurement
* `[ Option A ]`


---

**Q: What is a penumbra?**
- A subtle gradient of light encircling a corner formed by emitted and reflected light from behind the corner.
- A shadow tracing the corner formed by objects from behind the corner.
- The change of a subtle gradient of light encircling a corner.
- None of the above.
* `[ Option A ]`


---

**Q: Which of the following is not a parameter of the function which describes the reflected light from a surface at a point p?**
- The surface's BRDF function $\beta$
- The incoming light $L$
- The surfaceΓÇÖs albedo $a$
- The surface material constant $k$
* `[ Option D ]`


---

**Q: Which of the following component is not applied for an edge camera system?**
- visible scenes
- the occluding edge
- the ground
- the pin hole
* `[ Option D ]`


---

**Q: Which of the following is not used in the calculation of reflected light at a point p?**
- The albedo
- BRDF
- Incoming light
- Direction of light integration
* `[ Option D ]`


---

**Q: In the paper "Turning Corners into Cameras: Principles and Methods", the author can construct a 2D dimensional position of a hidden object using Stereo Edge Cameras. What is the defining characteristic that allows Stereo Edge Cameras to function?**
- Multiple real cameras viewing the occluding edge at slightly different angles, allowing for different ground projections
- Multiple edge cameras allowing for multiple ground projections to be examined due to multiple occluding edges in the scene
- Moving real camera to slow for multiple ground projections to be examined in sequence
- None of the above
* `[ Option B ]`


---

**Q: Stereo edge cameras offer additional information over one edge camera:**
- the number of people moving around the corner
- the absolute position of a subject over time
- a subject's angular size and speed
- a temporal summary of activity
* `[ Option B ]`


---

**Q: Which of the following techniques is NOT used in the implementation of Edge Cameras?**
- Rectify the image using a homography when floor is not parallel to the occluding edge.
- Subtract a background frame to remove the effect of the sceneΓÇÖs background illumination.
- Impose temporal smoothness on MAP estimate to regularize the result in addition to spatial smoothness.
- None of above.
* `[ Option B ]`


---

**Q: What application is not possible with a single edge camera?**
- It can reveal the number of people around the corner.
- The angular speed of a person around the corner with respect to the edge, which is the origin of rotation, can be computed.
- The angular size of a person around the corner can be computed.
- It can triangulate the absolute position over time of a person around the corner.
* `[ Option D ]`


---

**Q: How can we remove the background light from the scene in a passive way?**
- Subtracting to all the frames the average image got by the frames themselves
- Subtracting to all the frames the one taken without the subjects beyond the wall
- Subtracting to all the frames the mean value of the light intensity in the scene
- Subtracting to all the frames a default value according to the illumination conditions
* `[ Option A ]`


---

**Q: Regarding the setting of an edge camera, the reflected light $L_o$ in a certain point $p = (r,\theta)$ of a surface with albedo $\alpha(r,\theta)$ at the point, is given in the following equation. How should we set $\theta$ in order to get the integral of light over the entire visible and hidden scene?

$$ L_o(r, \theta) = \alpha (r, \theta) [L_v  + \int_{\phi=0} ^\theta L_h(\phi)d\phi] $$**
- 0
- $\pi/2$
- $\pi$
- $3\pi/2$
* `[ Option B ]`


---

**Q: Corner cameras work on the principle **
- of a pinspeck camera
- selectively blocking of light
- amplification of selective spectrum
- Differential refractive index
* `[ Option B ]`


---

**Q: An edge camera system incorporates which one of the following components?**
- Visible and hidden scenes
- An occluding edge
- The ground
- All of the above
* `[ Option D ]`


---

**Q: Which one of these is the most in-the-wild technique to use walls as naturally occurring cameras for occluded objects from the viewers point of view.**
- Time of flight cameras since they have good depth recovery properties
- Analyzing subtle variations in the penumbra at the base of a wall edge
- Using accidental pinholes as they have the potential to provide us with actual 2D images of the scene
- None
* `[ Option B ]`


---

**Q: Which 3 basic factors constitute the color of a pixel on a penumbra?**
- Ambient lighting
- Light reflected by hidden objects
- Noise from diffuse reflections
- All of them
* `[ Option D ]`


---

**Q: Which of the following statements is false?**
- Inaccuracy in localizing the projected trajectories, and mis-calibration of the scene can introduce bias into location estimates.
- The observation plane is the plane perpendicular to the occluding edge.
- A penumbra is an emitted and reflected light from behind the corner, hidden from the observer, that has a small effect on the groundΓÇÖs radiance in the form of a subtle gradient of light encircling the corner.
- An edge camera system consists of three components: the visible scene, the occluding edge, and the ground.
* `[ Option D ]`


---

**Q: Change in the intensity of what contains information of things that are hidden from our line of sight by a corner ?**
- Umbra
- Penumbra
- Reflection
- Reflectivity of the surface
* `[ Option B ]`


---

**Q: Using an edge camera with a single corner, what data is the author's method NOT able to retrieve?**
- the angle of the subject with respect to the corner
- the sizes of the subjects in the hidden scene
- the number of subjects in the hidden scene
- occlusion between subjects with respect to the corner
* `[ Option B ]`


---

**Q: What is a penumbra?**
- A shadow casted on the floor by a corner.
- A shadow near a corner due to hidden objects around the corner.
- A gradient of light encircling a corner emitted and reflected from objects around the corner.
- A gradient of light around a corner due to diffraction of different wavelengths of light
* `[ Option C ]`


---

**Q: In using corners as cameras the ... is recorded**
- wall
- floor
- the scene directly
- the space between the door and the wall
* `[ Option B ]`


---

**Q: What is NOT true about a Lambertian surface?**
- The luminance of a Lambertian surface is isotropic
- Lambertian surface models are considered as an extreme scenario where the surface roughness is equal to zero
- A Lambertian surface appears uniformly bright from all directions of view and reflects the entire incident light
- The intensity of images when viewed through a Lambertian surface does not change in proportion with the illumination directions
* `[ Option D ]`


---

**Q: What are the components of an edge camera?**
- The visible and hidden scenes
- The occluding edge
- The ground
- All of the above
* `[ Option D ]`


---

**Q: The introduction of corner cameras could not allow for:**
- better reconstruction of the surrounding environment of a scene.
- improved performance in search and rescue missions.
- new applications for automotive pedestrian safety.
- improved image quality.
* `[ Option D ]`


---

**Q: Given these two statements:
- The hidden scene can be more easily reconstructed in the dark, because then the surrounding light is more dim and the reflections of interest have a higher relative intensity
- For stereo (2D) imaging through a doorway four cameras are needed, 2 for depth determination in the left half of the scene and two for the right half of the scene**
- The first statement is True, the second if False
- The first statement is False, the second if True
- Both statements are True
- Both statements are False
* `[ Option B ]`


---

**Q: Edge cameras look at the ground plane near an edge that is obstructing the view of the observer around that edge (e.g. a wall). From carefully analysing subtle variations in the penumbra (the ΓÇ£shadowΓÇ¥ created by the edge) at the base of the edge, it is possible to deduce a hidden subjectΓÇÖs pattern of motion. From analysing the figure below (insert figure 6 from the paper ΓÇ£Turning Corners Into CamerasΓÇ¥), where variations in the penumbra are depicted as angular position over time, what can you say about the motion pattern of the hidden subject?**
- a) the hidden subject is moving in a square motion
- b) the hidden subject is moving in a circular motion
- c) the hidden subject is moving towards and away from the edge
- d) you cannot say anything about this pattern
* `[ Option B ]`


---

**Q: What is the reason that adjacent walls bring more accurate track of hidden objects?**
- Using offset in angular positions to triangulate location
- Reflection from both walls will cause stronger light intensity
- Bigger observation areas
- Reducing camera noise by multiple cameras
* `[ Option A ]`


---

**Q: Which statement is about looking around corners in different environments is False?**
- In indoor environments, horizontal artifacts can show up from sampling on a square grid.
- In outdoor environments, weather can affect the reconstruction of the image around the corner
- In indoor environments, when the hidden scene is well-lit, the reconstruction becomes hard because the subjects occlude the bright ambient light.
- In outdoor environments, due to changes in the scene that is being filmed, it is hard to distinguish between those changes and changes in the hidden scene.
* `[ Option C ]`


---

**Q: Which of the following methods could not be used to rectify images**
- using a calibration grid
- using regular patterns, such as tiles
- using a known camera calibration
- using a edge camera
* `[ Option D ]`


---

**Q: Which of the following statements about a corner camera is always correct, assuming the ground surface is Lambertian?
(i) If the side that is invisible to the observer does not emits any light, the entire penumbra goes completely dark.
(ii) An ideal open doorway can be considered a composition of two edge cameras.**
- (i)
- (ii)
- (i) and (ii)
- Neither (i) nor (ii)
* `[ Option D ]`


---

**Q: When using corners as cameras to detect hidden scenes which of the following is assumed for background subtraction ?**
- The subjects are stationary
- The subjects motion is uniform
- The subjects motion is random
- All of the above
* `[ Option B ]`


---

**Q: Which of the following information is found from the 1D around the corner reconstructed videos ?**
- The number of people moving around the corner.
- The angular sizes and speeds of the people moving around the corner
- A temporal summary of activity.
- All of the above
* `[ Option D ]`


---

**Q: You would like to measure the absolute position of a subject in a hidden scene behind a doorway. Which from the author proposed ΓÇ£camera principleΓÇ¥ would you use?**
- Camera obscura
- Plenoptic camera
- Stereo edge camera
- Single edge camera
* `[ Option C ]`


---

**Q: How is a camera able to ΓÇÿseeΓÇÖ people around the corner, when it's invisible to the naked eye?**
- Because of the shadows of a person visible on the ground casted by the sun
- Because of mirrors on opposing walls helping the camera see around the corner
- Because of the difference in radiance in the form of a circle, also known as a penumbra
- A camera canΓÇÖt see around the corner, thatΓÇÖs sorcery!
* `[ Option C ]`


---

**Q: How does an object entering the hidden scene effect the visible scene?**
- It doesn't effect the visible scene
- The effects are visible by the eye
- The effects are only detectable by computers.
- None of the above
* `[ Option C ]`


---

**Q: Which of the following are characteristics of penumbra?

I - Its a fuzzy shadow created by objects reflecting small amount of light.
II - Changes in penumbra can help in determining the movement of objects hidden from scene.
III - The intensity of light on penumbra is constant under different illumination conditions of hidden state.**
- I and II
- I and III
- II and III
- I, II and III
* `[ Option A ]`


---

**Q: Which of the following is a property of likelihood ?**
- Ratios of likelihood values measure the relative evidence of one value of the unknown parameter to another
- Given a statistical model and observed data, all of the relevant information contained in the data regarding the unknown parameter is contained in the likelihood
- The Resultant likelihood is multiplication of individual likelihood
- All of the Mentioned
* `[ Option D ]`


---

**Q: Imagine you are in a detective job with one of your clients, both of you are around a corner and you know there is a camera at the other side of the edge and can use light to extract information. Now you don't want the camera to know you are two people and you want the camera to extract signals with almost no strength, what will you do?**
- Both walk in the same angular position when taking the edge of the wall as the origin, and walk far away
- Walk randomly but not cross the edge
- Both walk in the same angular position when taking the edge of the wall as the origin, and walk closer
- Both walk in different angular positions when taking the edge of the wall as the origin, and walk far away
* `[ Option A ]`


---

**Q: Which of the elements is not a part of the edge camera**
- visible scene
- occluding edge
- ground
- laser and detector for ToF measurements
* `[ Option D ]`


---

**Q: What can an edge camera detect(turning vertical walls as cameras to produce 1-D videos)?**
- Number of people behind the occluding wall
- Angular Sizes
- Speeds
- All of the above
* `[ Option D ]`


---

**Q: What is the penumbra?**
- Emmitted light from the observer position
- Reflected ligth from the front position of the corner
- Reflected and emmitted light from the observer corner
- Emmitted and reflected light hidden from behind the corner
* `[ Option D ]`


---

**Q: Choose the correct option**
- Before identifying temporal differences in a hidden scene due to a moving object, one must remove the effect of the scene's background illumination
- While reconstructing hidden scene by observing the ground plane near a open doorway, each side of the doorway contributes to 90 deg view of the hidden scene
- The error in estimated depth does not increase with the actual depth of the object from the doorway
- All of the above statements are correct
* `[ Option A ]`


---

**Q: What would a clearly visible double alternating sinusoid suggest about the environment and the amount of persons around a corner?**
- 2 persons, in foggy rainy conditions.
- 2 persons, in sunny conditions.
- 1 person, in foggy rainy conditions
- 1 person, in sunny conditions.
* `[ Option B ]`


---

**Q: Can people be estimated from shadows**
- No, because the velocity estimation.
- No shadows cannot be constructed to one 1-d image 
- both true
- none, true
* `[ Option D ]`


---

**Q: How many dimensions does the single corner camera output have? (Ignoring time)**
- 1
- 2
- 3
- 4
* `[ Option A ]`


---

**Q: Which of the following is/are not (a) component(s) of a camera edge system?**
- Visible and hidden scenes
- Occluding edge
- Laser
- Ground
* `[ Option C ]`


---

**Q: Select the possible cause, that was not mentioned by the authors, behind the troubling trends:**
- Incentives to gain media attention.
- Absence of rigorous empirical standards.
- The rise of platforms like Kaggle, turning the Machine Learning field into a competition.
- The rapid expansion of the community.
* `[ Option C ]`


---

**Q: Which of the following statements is not an improvement of corner cameras over time-of-flight (ToF) cameras**
- Time-of-flight cameras require specialised and comparatively expensive detectors with fine temporal solutions, whereas corner cameras do not
- Corner cameras can infer the location size and motion of objects in a hidden scene, whereas time-of-flight cameras can not
- Corner cameras are less vulnerable to interference from ambient outdoor illumination than time-of-flight cameras
- Time-of-flight cameras are limited in how much light they can introduce in the scene to support imaging
* `[ Option B ]`


---

**Q: In this paper, we find that walls, and other obstructions with edges, can be exploited as naturally-occurring ΓÇ£camerasΓÇ¥ that reveal the hidden scenes beyond them, as light from obscured portions of a scene is scattered over many of the observable surfaces. This reflected light can then be used to recover information. How is this achieved on the paper?**
- By using specular reflections off of human eyes to image hidden scenes;
- By using structures naturally present in the
real world as cameras, such as naturally occurring pinholes (such
as windows), or pinspecks;
- by using a laser to illuminate a point that is visible to both
the observable and hidden scene, and measuring how long
it takes for the light to return;
- By exploiting the vertical edge at the corner of a wall to construct
a ΓÇ£cameraΓÇ¥ that sees beyond the wall. 
* `[ Option D ]`


---

**Q: The reconstruction of visual patterns around corners comes from**
- the slight changes in gradient of light around the corner
- the shadows being cast on the floor around a corner
- the shadows being cast on the wall of the corner
- the slight changes in gradient on the wall of the corner
* `[ Option A ]`


---

**Q: what information is encoded in the penumbra?**
- It is an encoding of what is around the corner
- It is an encoding of the reflection
- It is an encoding of what is behind the penumbra
- Nothing
* `[ Option A ]`


---

**Q: What is the main idea behind turning a corner into a camera? Why does it work?**
- Emitted and reflected light from behind the corner can be observed outside of the penumbra on the ground near the corner
- Emitted and reflected light from behind the corner can be observed in the penumbra on the ground near the corner
- The speed of emitted and reflected light from behind the corner can be compared with the their speed after hitting the ground near the corner
- The speed of emitted and reflected light from behind the corner can be compared with the their speed before hitting the ground near the corner
* `[ Option B ]`


---

**Q: Which is a source of error when turning corners into cameras**
- Misidentifying the corner of each occluding edge
- The resolution of camera is limited
- The number of photos is not large enough
- Still unknown
* `[ Option A ]`


---

**Q: The 1-D videos of the activity around a corner produced using an edge camera help in measuring which of the following?**
- Number of People around the corner
- Angular size of the objects around the corner
- Angular speed of the objects around the corner
- All of the above
* `[ Option D ]`


---

**Q: How can edge cameras reconstruct a pattern of motion?**
- Analyzing variations of the penumbra of the base of an edge
- Using a bright floor that allows some sort of ΓÇ£mirrorΓÇ¥ effect
- Using extremely high quality cameras that allow to view the reflect of the subjects on the floor
- None of the above
* `[ Option A ]`


---

**Q: The inverse pinhole camera reveal the scenes outside the image by subtracting two images with and without an occlude. Which option can help to improve the SNR of inverse pinhole cameras:**
- Increase the sensitivity of the light sensor
- Using long exposures to capture more lights
- Increase the background illumination against the light blocked by occlude
- Decrease the size of occlude to block less lights
* `[ Option C ]`


---

**Q: In a 1D reconstructed video of an indoor environment it is possible to assume the number of people in the hidden scene. What other information can be extracted?**
- the angular position
- the speed
- the motion's characteristics
- All the above.
* `[ Option D ]`


---

**Q: Select the FALSE statement:**
- Previously non-line-of-sight methods includes the usage of time-of-flight cameras and naturally occurring pinholes
- Edge cameras are used to reconstruct a video of a hidden scene by interpreting small color changes
- Vertical line artifacts present in the 1D reconstructed video of an indoor hidden scene could be caused by additional shadows appearing on the penumbra
- The changes in various weather conditions of an outdoor scene does not introduce artifacts in the 1D reconstructed video of an outdoor hidden scene
* `[ Option D ]`


---

**Q: Which of following is not a component of an edge camera system:**
- Visible scenes
- Hidden scenes
- Occluding edge
- Angular slice
* `[ Option D ]`


---

**Q: Which one is wrong?**
- In addition to identify if something is there, the system can also estimate its speed, but not trajectory.
- When light is reflected off an opaque object onto the ground, it results in  the fuzzy outer region of the shadow which called ΓÇ¥penumbraΓÇ¥
- This work make it possible to see things that are outside the line of human sight without special lasers.
- This system analyzes light reflections in the space it is ΓÇ£shownΓÇ¥ in order to detect if thereΓÇÖs a person or object around the bend.
* `[ Option A ]`


---

**Q: Which component does not an edge camera system consists?**
- Visible and hidden scenes
- Well designed pinhole(s) and/or pinspeck(s)
- Occluding edge
- Ground which meet related requirements
* `[ Option B ]`


---

**Q: What is the main objective of the 'edge cameras' paper?**
- To find edges in the images of a complex scene.
- To reproduce a 1D timeline of objects that are occluded in a scene.
- To compare the performance of consumer grade cameras to professional cameras in the context of edge detection.
- To perform facial recognition of partially occluded objects in the scene.
* `[ Option B ]`


---

**Q: In the paper: "Turning Corners into Cameras: Principles and Methods", Which of the following factors do NOT affect the radiance emanating from a ground in front of a corner**
- Albedo
- Direction of the Light
- BRDF of the ground surface
- None of the above
* `[ Option D ]`


---

**Q: Which of the following statements is true:

1. The penumbra of a corner contains information about the hidden scene around the corner
2. Edge Cameras can currently count the amount of people in a hidden scene.**
- 1
- 2
- both
- neither
* `[ Option C ]`


---

**Q: Corners can be used as cameras. With this, among others, what state variables of moving people around the corner can be extracted?**
- Angular size, speed, number of people 
- Angular size, number of people
- number of people
- none of the above
* `[ Option A ]`


---

