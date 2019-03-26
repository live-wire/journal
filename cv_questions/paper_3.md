# Questions from `paper_3` :robot: 

**Q: Please complete de sentence. A pinhole camera needs two components, the pinhole and the sensor. The pinhole _________, while the sensor____________.**
- "restricts the number of light rays going into the sensor", "captures the image upside down".
- "restricts the number of light rays going into the sensor", "captures the image with the correct orientation".
- "redirects all the incoming light rays to the sensor", "captures the image upside down".
- "redirects all the incoming light rays to the sensor", "captures the image with the correct orientation".
* `[ Option A ]`


---

**Q: Which statement is false?**
- The inverse pinhole camera image requires at least two images or a video from which a reference background is extracted.
- In inverse pinhole cameras, the signal to noise ratio decreases when the background illumination increases with respect to the amount of light blocked by the occluder.
- By making the occlude smaller in inverse pinhole cameras, the image gets sharper, but at a cost of increased noise.
- In inverse pinhole cameras, the main source of distortion comes from the relative orientation between the camera and the surface in which the image gets projected.
* `[ Option B ]`


---

**Q: The SNR of an image from an accidental pinhole camera made by a window is proportional to:**
- $$ A_{window}/2 $$
- $$ sqrt(A_{window}) $$
- $$ A_{window} $$
- $$ A_{window}^2 $$
* `[ Option B ]`


---

**Q: In the formula I(x)=T(x)*S(x) from the paper (Accidental pinhole and pinspeck cameras: revealing the scene outside the picture), the meaning of T(x) is:**
- The amount of light that passes through point x.
- The image formed by a pinhole camera.
- The average amount of illumination at the projection of the pinhole camera.
- The resulting image.
* `[ Option A ]`


---

**Q: What kind of information can be retrieved from an image created by a pinspeck camera?**
- The shape of the aperture
- Recovery of light sources in a room where the lights are on
- The illumination map in an outdoor scene
- All of the above
* `[ Option D ]`


---

**Q: Which of the following statements about pinhole and pinspeck cameras is not correct?**
- Inverse pinhole cameras require at least two images.
- Signal-to-noise ratio in pinhole cameras can be increased with longer exposure.
- In pinspeck cameras, image gets sharper as the occluder gets bigger.
- In pinspeck cameras, one can make the image sharper at the cost of increasing the noise.
* `[ Option C ]`


---

**Q: Which of the following aspects is not making it harder to remove occlusions from an image?**
- High signal to noise ratio of image
- The difficulty of obtaining a reference image
- High intensity difference between occluded image and reference image
- Light on the image coming from a small window 
* `[ Option D ]`


---

**Q: Which of the following sentences is correct? To retrieve an image of what is outside a room...**
- Based on the pinhole camera formulation, at least two different pictures are required
- Based on the pinspeck camera formulation, at least two different pictures and an occluder are required
- A video must be recorded, in both formulations.
- None of the above
* `[ Option B ]`


---

**Q: Which of the following statements about the inverse pinhole camera is FALSE?**
- The inverse pinhole camera can be used to construct an image of what is outside of the room (that is photographed)
- The inverse pinhole camera can be used to determine the shape of a window that projects an inversed image of the 
- The inverse pinhole camera can be used to recover the light sources in an indoor environment
- The inverse pinhole camera can be used to create sharp images of the surrounding environment outside of the image frame
* `[ Option D ]`


---

**Q: In the paper "Turning Corners Into Cameras: Principles and Methods", passive recovery is used to infer presence and position of objects in the hidden part of the scene. It is also said that there is the possibility of using stereo edge cameras. Which is the angle of the reconstructed angular image in the stereo case?**
- 45┬░
- 90┬░
- 135┬░
- 180┬░
* `[ Option D ]`


---

**Q: Which of the following statements is not true for an inverse pinhole?**
- If a picture has no noise and unlimited precision, it is possible to obtain a perfect sharp image
- The pinhole camera helps in seeing light sources
- It requires at least 3 images to extract a reference background
- It needs to be calibrated to use the images to infer the window shape
* `[ Option C ]`


---

**Q: What are the two main limitations of the reverse pinhole camera over the traditional pinhole camera?**
- The signal to noise ratio and the fact that we need more than one image.
-  The fact that we need more than one image and the maximum size
- The amount of parameters we can extract from the result and the signal to noise ratio
- The colors of the result and the amount of processing needed
* `[ Option A ]`


---

**Q: How can traditional pinhole cameras improve the signal to noise ratio?**
- Increasing the sensitivity of the light sensor.
- Use long exposure to capture enough light.
- Both A and B.
- None of the above.
* `[ Option C ]`


---

**Q: Which of the following are not part of the process to obtain an anti pinhole (pinspeck) generated image? **
- Take a picture of an indoor environment with no occlusions 
- Flip the resulting image vertically
- Take a picture of an indoor environment with a small occluder.
- Flip the resulting image horizontally 
* `[ Option D ]`


---

**Q: Which of the following statement of the paper is not ture∩╝ƒ**
- Smaller occluder, sharper image, at the cost of increased noise
- Increase SNR, i.e. increase occluder size
- Best results are small window and small occluder
- For pinhole cameras: images need to be taken from the different point
* `[ Option D ]`


---

**Q: Review the following two statements about inverse pinhole cameras:
\begin{enumerate}
    \item A limitation of the inverse pinhole camera over the pinhole camera is that it requires atleast two images or a video from which a reference can be extracted
    \item If an inverse pinhole camera had no noise and unlimited precision, it would be possible to extract a perfect sharp image from the pinhole
\end{enumerate}
Which of the statements are true?
\maketitle**
- Statement A and B are true
- Statement A is true, statement B is false
- Statement A is false, statement B is true
- Both statement A and B are false
* `[ Option A ]`


---

**Q: In the paper "Accidental pinhole and pinspeck cameras: revealing the scene outside the picture ", where is the main source of distortion in pinspeck cameras?**
- Diffuse light coming from the window
- Ambient light in the room
- Orientation between the camera and the surface where the image gets projected
- Shadows generated by the occluder on the surface of the projected image
* `[ Option C ]`


---

**Q: To obtain the desired picture of a hidden scene from two accidental images, with and without an occluder, we have to:**
- Add the two images and invert them upside-down
- Subtract the two images and invert them upside-down
- Convolve the two images and invert them upside-down
- Cross-correlate the two images and invert them upside-down
* `[ Option B ]`


---

**Q: By inverse pinhole camera, we can:**
- See what is outside the room.
- See light sources when most of the illumination is dominated by direct lighting.
- See the shape of the window.
- A, B and C
* `[ Option D ]`


---

**Q: Which type of camera is different than the other options?**
- Anti-pinhole camera.
- Pinhole camera.
- Pinspeck camera.
- Inverse pinhole camera.
* `[ Option B ]`


---

**Q: Which is the use of the presence of an occluder in the recorded indoor scene?**
- The occluder reduces the brightness of the scene, thus permitting a more effective analysis
- The occluder projects a shadow that can be subtracted from the original image getting a projection of the outdoor scene
- The occluder allows to get more details of the indoor scene that is not in the picture
- The ccluder conveys all the light rays in a certain region of the immage, that gets brighter and clearer to analyse
* `[ Option B ]`


---

**Q: Regarding the setting of a pinspeck camera with light coming into a room via a partially opened window. The image formed in opposite wall is given in the following equation, where $T(x)$ is a parametrization of the shape of the window and $S(x)$ the image formed by a pinhole camera. What should the outdoor illumination $S(x)$ be equal to, so that the following equation holds?

$$ I_{window}(x) - I_{occludedwindow}(x) = T(x) $$**
- $S(x) = 1, \forall x$
- 0
- $\delta(x)$
- $S(x) = I$ (identity matrix)
* `[ Option C ]`


---

**Q: Pinspeck camera operates on **
- two similar images
- image of same scene with occluder and one without an occluder
-  image with occluder and one without an occluder
- images with different lightning conditions
* `[ Option B ]`


---

**Q: What is a possible application of the pinspeck camera?**
- Seeing light sources
- To obtain magnified images
- To obtain sharp images
- To detect objects
* `[ Option A ]`


---

**Q: What affects the image pattern of the outside world on a room wall when a room is used a camera, the wall as the image plane and the window as a pinhole?**
- Shape of the window
- Shape of the opening in the window through which light falls through 
- The amount of diffuse light outside the window
- None
* `[ Option B ]`


---

**Q: What can accidental pinhole images not reveal about their environment?**
- The view outside a room
- The shape of the light aperture into the room
- The illumination map in an outdoor scene
- The location of objects outside the room
* `[ Option D ]`


---

**Q: Which of the following statements is true?**
- To get a sharp image, a big aperture $T(x)$ is needed.
- To improve the signal to noise ratio, traditional pinhole cameras require decreasing the sensitivity of the light sensor.
-  The inverse pinhole has two limitations over the traditional pinhole camera.
- The reference image is the image with the occluder.
* `[ Option C ]`


---

**Q: Which is not one of the possible application of the accidental pinhole image formation ?**
- Seeing things that are outside of the room where the accidental pinhole is formed
- Seeing light sources
- Seeing the shape of the window
- Seeing the occuluder which resulted in the formation of the accidental pinhole
* `[ Option D ]`


---

**Q: What happens to the image of an inverse pinhole camera if the occluder becomes smaller?**
- noise increases, sharpness of the projected image decreases
- noise increases, sharpness of the projected image increases
- noise decreases, sharpness of the projected image increases
- noise decreases, sharpness of the projected image decreases
* `[ Option B ]`


---

**Q: How can a image that would be formed by a pinhole camera be recovered from an image formed by a pinspeck camera with a occlusion with the same size as the pinhole.**
- Subtracting the image with occlusion from the image without occlusion.
- Subtracting the image without occlusion from the image with occlusion.
- Multiplying the image with occlusion with the image without occlusion.
- Convolving the image with occlusion with the image without occlusion.
* `[ Option A ]`


---

**Q: Which of these statements are true:

I. Pinhole and pinspeck are 2 names for the same principle.

II. With a pinspeck camera you can find images not directly available on the original picture.**
- I
- II
- Both
- Neither
* `[ Option B ]`


---

**Q: What is not possible with an inverse pinhole / pinspeck camera?**
- Seeing what is outside a room
- Seeing what is behind a corner
- Seeing light sources outside an image
-  Seeing the shape of the light aperture into a room
* `[ Option B ]`


---

**Q: What information can be revealed from "accidental" images that can be found in scenes?**
- Lighting conditions outside of the scene
- The view outside the room
- The shape of the light aperture into the room
- All of the above
* `[ Option D ]`


---

**Q: Choose the statement that best finishes the following sentence. Accidental images...**
- usually capture phenomena that were not desired in the first place.
- can reveal parts of the scene that were not inside the photograph or video.
- are accidentally taken.
- usually are of good quality.
* `[ Option B ]`


---

**Q: Why is the pinspeck camera also referred to as the ΓÇÖinverse pinholeΓÇÖ technique? **
- In stead of placing the pinhole between the object and the camera, the pinhole-acting part is now essentially placed between the light-source and the object
- In stead of using a small aperture (pinhole), now a small occlusion is used
- In stead of a small aperture, in this case a large opening/window/aperture is used as pinhole
- The pinhole itself is positioned on the same location, but has an inverted orientation
* `[ Option B ]`


---

**Q: A window in a room during daylight can sometimes act as a pinhole and light from outside can enter the room creating light patterns on the walls. If you could make the window smaller and smaller, these light patterns will change. What would the patterns reveal when making the window smaller?**
- a) a blurry right-side up image of the scene outside the room (false because it is upside-down)
- b) a blurry upside-down image of the shape of the window (false because that would be at night)
- c) a blurry upside-down image of the scene outside the room
- d) a blurry right-side up image of the shape of the window (false because that would be at night
* `[ Option C ]`


---

**Q: Which one is not correct when trying to increase SNR of inverse pinhole cameras in order to achieve a sharp image?**
- Increasing the sensitivity of light sensor
- Long exposures to capture more light
- Increasing the background illumination
- Increase the size of occluder
* `[ Option D ]`


---

**Q: How can Pinspeck cameras be used to reveal the scene outside the picture?**
- By flipping the image with the occluder, one can retrieve the scene outside the picture.
- By subtracting the image with the occluder from the reference image, one can retrieve the scene outside the picture.
- By subtracting the image with the occluder from the reference image, and flipping the result, one can retrieve the scene outside the picture.
- By reducing the aperture of the window and flipping the resulting image, one can retrieve the scene outside the picture
* `[ Option C ]`


---

**Q: What could do when we find the light sources, window shape and the scene outside a picture used in computer graphics?**
- provide a better model of the light rays in the scene to render synthetic objects that will be inserted inside the picture.
- find pattern of intensities on the surfaces
- construct a model to produce the blur kernel
- find the difference between origin figure and the figure with shadows
* `[ Option A ]`


---

**Q: Regarding to the inverse pinhole camera, which of the following statements are correct?
(i) The purpose to subtract the two input images is to reduce the noises embedded.
(ii) Reducing the window size will lead to higher precision because the rays are more focused and image looks less blurry.**
- (i)
- (ii)
- (i) and (ii)
- Neither (i) nor (ii)
* `[ Option D ]`


---

**Q: When we have a window in a room and a wall opposite to it which of the following is true ?**
- The window shape doesnΓÇÖt have a strong influence on the blur and gradient statistics of the pattern projected onto the wall.
- The pattern of intensities on the wall corresponds to a convolution between the window shape and the sharp image that would be generated if the window was a perfect pinhole
- The image on the wall has only shadows in them
- None of the above
* `[ Option B ]`


---

**Q: Which of the following statements is incorrect? **
- The inverse pinhole requires at least two images or a video from which to extract a reference. 
- In order to improve SNR in a traditional pinhole camera longer exposures are required. 
- For inverse pinhole cameras the signal to noise ratio increases inversely proportional to the background illumination with respect to the amount of light blocked by the occluder.
- None of the above. 
* `[ Option C ]`


---

**Q: An occluder produces a shadow on the wall of a room. How could the image that would have been produced by a pinhole with the shape of the occluder be recovered?**
- By subtracting the image without occluder from the image with it.
- By dividing the image with occluder by the image without it.
- By inverting the image with occluder.
- None of the above.
* `[ Option A ]`


---

**Q: WhatΓÇÖs NOT true about the inverse pinhole (pinspeck) camera, when comparing it to a traditional pinhole camera?**
- The inverse pinhole makes use of shadows, whereas the pinhole model makes use of light rays
- The Signal to Noise (SNR) is a lot better in the inverse pinhole model than in the pinhole model
- In both models, a small aperture helps with producing a sharp image
- The inverse pinhole requires at least two images, whereas the traditional needs only one
* `[ Option B ]`


---

**Q: We can reveal light pattern in an image by using an aperture. What size should the aperture be?**
- Large
- Small
- Medium
- N.A.
* `[ Option B ]`


---

**Q: Which of the following statements about accidental images is wrong:**
- Accidental images reveal information about the scene outside the image
- Accidental images often occur in scenes without being directly noticeable. However, one way to bring accidental pinhole camera images into focus is by increasing the aperture. 
- Accidental inverse pinhole camera images are formed by subtracting an image with occluder from a reference image.
- In inverse pinhole cameras, the signal to noise ratio increases as the background illumination increases with respect to the light blocked by occluder.
* `[ Option B ]`


---

**Q: what goal can be achieved using images from pinhole and pinspeck camera?**
- explain illumination variations that would otherwise be incorrectly attributed to shadows
- reveal the lighting conditions outside the interior scene
- reveal the view outside a room
- reveal illumination map in an outdoor scene under any signal-to-noise ratio.
* `[ Option D ]`


---

**Q: Which of the following CANNOT be used with a pinhole camera due to its characteristics?**
- Seeing what is outside the room
- Seeing moving objects without motion blur
- Seeing the shape of the window
- Seeing the illumination map in an outdoor scene
* `[ Option B ]`


---

**Q: To create an image with a blurred background and focus only at the object close to the lens we should**
- Decrease the aperture value - bigger pinhole
- Increase the exposure time - short period of time where light rays go through lens
- Decrease the exposure time - longer period of time where light rays go though the lens
- Increase the aperture value - smaller pinhole
* `[ Option A ]`


---

**Q: How can the signal to noise ratio be improved in the image formed by traditional pinhole camera?**
- Increasing the sensitivity of the light sensor
- Long exposures
- Both A) and B)
- Temporal Integration
* `[ Option C ]`


---

**Q: Which one of the following characteristics will have the image when reducing the size of the ocluder ?**
- Sharper and more noisy
- Brighter and sharper
- Inverted and less noisy
- Brigther and inverted
* `[ Option A ]`


---

**Q: Following are two statements about improving signal to noise ratio (SNR) in pinhole cameras.
(a) Increasing sensitivity of the light sensor will improve SNR.
(b) Using long exposures won't improve SNR

Choose the correct option**
- (a) and (b) are both wrong
- (a) and (b) are both correct
- (a) is wrong; (b) is correct
- (a) is correct; (b) is wrong
* `[ Option D ]`


---

**Q: What could one do to increase the performance of an accidental pinhole camera?**
- Increase sensitivity to light of the sensor.
- Use more pictures (or video) and integrate over the data.
- Integrate over the amount of light.
- Use a larger window such that the signal to noice ratio (SNR) decreases.
* `[ Option B ]`


---

**Q: What does inverse pinholes means?**
- Explain illumination variantions that would otherwise be incorrectly attributed to shadows.
- Be lighting conditions outside the interior scene
- the shape of the light appeare in the room 
- All of the above
* `[ Option A ]`


---

**Q: What produces an inverse pinhole camera?**
- Introduction of an occluder between a light source and a surface
- Introduction of an light source between an occluder and a surface
- Introduction of a surface between an occluder and a light source
- Upside down pinhole camera
* `[ Option A ]`


---

**Q: Which of the following statements are limitations of "inverse" pinhole camera images over traditional pinhole cameras?

1. It requires at least 3 images
2. It requires an increase of the background illiumination as the occluder blocks out some of the light.**
- Only 1 is true
- Only 2 is true
- Both are true
- Both are false
* `[ Option B ]`


---

**Q: The main concern motivating this paper is:**
- That the CIFAR-10 dataset may not have been sampled i.i.d. from the ground truth distribution.
- That the the CIFAR-10 dataset may not contain enough images to allow for proper generalization of classification models, leading to overfitting.
- That by adapting model design choices to the
test set over the years, we implicitly fit the model to the test set.
- That model developers are not using the test set results to choose the best hypothesis, leading to biases in those models.
* `[ Option C ]`


---

**Q: Which of these statements about "accidental" images is incorrect?**
- Accidental images can explain illumination variations that would otherwise incorrectly be attributed to shadows
- Accidental images can reveal the view outside of a room
- Accidental images can only occur in an indoor scene
- Accidental images can reveal the illumination map in an outdoor scene
* `[ Option C ]`


---

**Q: Sometimes, what appears to the naked eyes as shadows and patters are really an upside blurry projection of some 'exterior scene', distorted by the particulars of its 'natural' creation -an accidental pinhole 'camera' . Interpreting these 'shadows' as images, and not what they look to be to the naked eye is not intuitive, due to the blurring caused by the window aperture, and:**
- The precise control of the light transport such that images can be viewed;
- The arbitrary geometry of the creation camera obscura effect;
- The Bayesian analysis of diffuse reflections over many different times;
- The specular reflections from the eye;
* `[ Option D ]`


---

**Q: The pinspeck camera can be used to**
- Create an image of the scene
- Create the negative of an image of the scene
- Help filter out noise from the image created by a pinhole camera
- Replace a pinspeck camera
* `[ Option B ]`


---

**Q: How can a camera be created by accident**
- By having a large non-occluded hole go from a light to a dark area
- By having light travel through a pinhole from a dark(ish) area to a light area
- By having light travel through a pinhole from a bright area to a dark area
- None of the above
* `[ Option C ]`


---

**Q: Given are two images of a indoor scene where light enters the room via an open window. The first one has no occlusions ($I_{window}(x) = T_{window}(x) * S(x)$) and the second one has one occluder ($I_{occluded}(x) = T_{occluded}(x) * S(x)$). How can the anti-pinhole/pinspeck camera image be computed?**
- $I_{window}(x) * T_{occluded}(x)$
- $T_{occluded}(x) * I_{window}(x)$
- $I_{window)(x) - I_{occluded}(x)$
- $I_{occluded)(x) - I_{window}(x)$
* `[ Option C ]`


---

**Q: Which option is the limitation of inverse pinhole?**
- Requires at least two image or videos
- Less accurate
- Takes longer time
- Worse resolution
* `[ Option A ]`


---

**Q: While analyzing accidental images using pinspeck cameras, the difference image produced from a fully open window was compared to that of a partially open window. Which of the following statement is true for this particular setup?**
- The signal of the difference image is higher in the case of the partially closed window.
- The intensity magnitude of the difference image produced by the partially opened window and the image obtained by replacing the pinspeck camera by a pinhole camera are similar.
- The Poisson noise in the difference image obtained during a fully open window is significantly lower than that obtained during a partially opened window.
- The state of the window(fully or partially open) has no significant influence over the noise in the difference image.
* `[ Option B ]`


---

**Q: What is one application of an inverse pinhole camera?**
- Bring images from the outside to a darkened room
- To reveal the source of the light on every scene
- To remove blurry images produced by the pinhole camera
- None of the above
* `[ Option A ]`


---

**Q: To reveal the objects behind the corner by an edge camera system, a 1-D video will be reconstructed in both indoor and outdoor scenes. which statement is FALSE:**
- It is possible to count number of people in both scenes.
- It is impossible to count number of people in any scenes.
- It is only possible to count number of people indoor scenes.
- It is only possible to count number of people outdoor scenes.
* `[ Option A ]`


---

**Q: How can an anti - pinhle camera be produced?**
- By having an occluder in front of the aperture.
- By taking the difference of the image with the occluder and the image without the occluder.
- By taking the sum of the image with the occluder and the image without the occluder.
- None of the above.
* `[ Option A ]`


---

**Q: What is NOT an application of the inverse pinhole camera?**
- Recovering the source light that is not directly visible in the original image
- Finding the window shape through which the light is entering the scene
- Finding the scene outside the picture to provide a better model of the light rays used in computer graphics for inserting synthetic object into the scene
- Recognizing objects in a scene with multiple illumination sources 
* `[ Option D ]`


---

**Q: Which of following is not the application of the inverse pinhole camera?**
- Seeing what is outside the room
- Seeing the shape of window
- Seeing illumination map in an outdoor scene
- Seeing what is on the ceiling of the room
* `[ Option D ]`


---

**Q: Build your own pinhole projector from household items, which factors does not affect the image quality and how?(multi choice)**
- light
- aperture size
- focal length
- color of items
* `[ Option D ]`


---

**Q: Which of the following description is correct?**
- The distance between pinhole and surface and the intensity of light
- The size of pinhole and the distance between pinhole and surface
- The relative orientation between camera and surface
- All of A, B and C are not the source (all are wrong)
* `[ Option C ]`


---

**Q: Accidental configurations in photographs can result in pinhole cameras that will:**
- make it impossible to classify objects in a scene
- possible to describe objects that are occluded by the wall in the photograph
- show light maps outside of the image frame
- reduce overall information density in the photograph.
* `[ Option C ]`


---

**Q: In the paper: "Accidental pinhole and pinspeck cameras: revealing the scene outside the picture", which of the following is sufficient for the formation of a Pin Hole Camera**
- The Aperture is far enough from the Image
- The Aperture is small enough to form a sharp image
- The size of the aperture is a linear function of the image size
- None of the above
* `[ Option B ]`


---

**Q: What best describes a pinspeck camera?**
- An array of pinhole camera's combined to form a single image.
- Subtracting an image formed with a large aperture and one formed with an aperture with an occluder
- A camera that can create images at night
- A camera that can see perfectly what is outside a room.
* `[ Option B ]`


---

**Q: If we allow light to enter a room only via a narrow aperture, the room is transformed into a camera obscura. Now, information about the outside world can be obtained via reflections on a wall facing this aperture. What can be done to reduce the noise level of this information?**
- Making the occluder larger
- Further decrease aperture
- Increase aperture
- The noise level canΓÇÖt be influenced.
* `[ Option A ]`


---

