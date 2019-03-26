# Questions from `lecture_2` :robot: 

**Q: Recall that if your camera's projection plane is at $z = f$, then a general point $(a,b,c)$ will be mapped to $g(a,b,c) = (\frac{f}{c}a, \frac{f}{c}b, f)$. This mapping $g$**
- is linear on all coordinates.
- is linear on the first and second coordinates, but not on the third.
- is linear on the third coordinate, but not on the first or second.
- is not linear for any coordinate.
* `[ Option A ]`


---

**Q: Let $M$ be a $2 \times 2$ matrix and let $\lambda_1, \lambda_2$ denote its two real eigenvalues and let $\det M$, $tra M$ denote the determinant and the trace of $M$, respectively. Which of the following is true:**
- $\det M > 0 > tra M \implies \lambda_1, \lambda_2 < 0$
- $\det M = 0 \implies \lambda_1 = \lambda_2 = 0$
- $tra M > 0 \implies \lambda_1 > \lambda_2 > 0$
- $\det M = 1 \implies \lambda_1 = \lambda_2$
* `[ Option A ]`


---

**Q: Which statement A, B or C is false for a Harris Detector? If no statement is false, choose answer D.**
- The corner response $R$ is invariant to image rotation
- The corner response $R$ is partial invariant to image intensity change
- The corner response $R$ is invariant to image scaling
- A, B and C are all true
* `[ Option C ]`


---

**Q: In a Harris Detector, the measure of corner response $R$ is defined as $\lambda_1 \lambda_2 ΓÇô k (lambda_1 + lambda_2)^2$. When is an image point classified as an edge?**
- R is negative with small magnitude
- R is negative with large magnitude
- R is positive with small magnitude
- R is positive with large magnitude
* `[ Option B ]`


---

**Q: For a (pinhole) camera perspective effects occur:**
- almost only close by
- almost only far away
- always
- never
* `[ Option A ]`


---

**Q: The Harris detector is fully invariant to:**
- image rotation
- image scale
- image intensity change
- all of the above
* `[ Option A ]`


---

**Q: What is not a step when you are looking for blobs in a picture (blob detector)?**
- Convolve image with scale normalized Laplacian at several scales.
- Find maxima of squared Laplacian response in scale-space.
- Find minima of squared Laplacian response in scale-space.
- The steps above are all possible steps.
* `[ Option C ]`


---

**Q: Which statement about the Harris detector is not true:**
- Corner response R is invariant to image rotation.
- Partial invariant to multiplicative intensity.
- Not invariant to image scale.
- Invariant to additive intensity.
* `[ Option D ]`


---

**Q: The image recreated by a pinhole camera is:**
- Flipped around the horizontal axis
- Flipped around the vertical axis
- Both
- Neither
* `[ Option C ]`


---

**Q: To which changes is the Harris detector FULLY INVARIANT?**
- Scale
- Image intensity
- Rotation
- None of the above
* `[ Option C ]`


---

**Q: With the Harris detector if the first eigenvalue is small and the second one is big, this means:**
- The window is in a flat region
- A corner has been detected
- The end of a stroke was identified
- The window is located on an edge
* `[ Option D ]`


---

**Q: In the Harris detector, instead of directly using the eigenvalues one could use the relationship between the determinant (D) and the trace (T) of the matrix. Taking this into account, which of the following statements is correct?**
- D >> T suggests that the eigenvalues are different.
- D >> T suggests that one of the eigenvalues is larger.
- D close to T suggests there is an edge.
- Large absolute differences between D and T could indicate that there is a corner or an edge.
* `[ Option D ]`


---

**Q: How does a blob relates to ripples?**
- A blob is the sum of two ripples
- A blob is the convolution of two ripples
- A blob is the superposition of two ripples
- A blob is the mean of two ripples
* `[ Option C ]`


---

**Q: Which of the following statements is false?**
- The Harris detector is invariant to rotation
- The Harris detector is partially invariant to intensity change
- The Harris detector is invariant to image scale
- The Harris detector can be used to detect corners
* `[ Option C ]`


---

**Q: In a pinhole camera, the focal length ΓÇ£fΓÇ¥ represents:**
- the width of the pinhole
- the distance between the pinhole and the object (the object that we want to capture)
- the distance between the pinhole and the image plane (where the projection occurs)
- None of the above
* `[ Option C ]`


---

**Q: Which of the following statements about pinhole cameras is true?**
- When the distance of a far object changes, this change is perceived less than it would have been perceived if the object was closer.
- When the distance of an object is close to the focal length, there is a strong non-linearity in the perceived image.
- All of the above
- None of the above
* `[ Option C ]`


---

**Q: Feature point detection can be used for:**
- 3D reconstruction
- Motion tracking
- Object recognition
- All of the above
* `[ Option D ]`


---

**Q: What is FALSE about the Harris Detector:**
- A region is 'flat' where there is no change when shifting the window in all directions
- An 'edge' is detected when there is no change when shifting the window in the direction of the edge
- A 'corner' is detected when there is change when shifting the window in any direction
- A 'curve' is detected when there is no change when shifting the window along the curve
* `[ Option D ]`


---

**Q: In blob detection, which is the scale normalization factor?**
- $\sigma$
- $\sigma^{2}$
- $\sigma^{3}$
- The scale normalization factor is not used in the blob detection
* `[ Option B ]`


---

**Q: In the Harris detector, the characteristic R value of an edge is:**
- R > 0
- R = 0
- R negative with small magnitude
- R negative with large magnitude
* `[ Option D ]`


---

**Q: Feature point/interest point detection is NOT used for**
- Robot Navigation
- Face recognition
- Indexing and database retrieval
- Noise removal
* `[ Option D ]`


---

**Q: Which of the following conditions must hold true for an edge point? (R is the corner response)**
- Positive value of R with a large magnitude
- Negative value of R with a large magnitude
- Positive value of R with a small magnitude
- $R = 0$
* `[ Option B ]`


---

**Q: In Harris' corner detection, what does it mean if |R| is small?
The input square has:**
- A corner
- a flat region
- an edge
- a blob 
* `[ Option B ]`


---

**Q: What is the focal length in the pinhole camera?**
- The distance between the object and the pinhole.
- The size of the pinhole.
- The distance between the pinhole and the image plane.
- The distance between the object and the image plane.
* `[ Option C ]`


---

**Q: Which of the following statements is true when a set of features is covariant?**
- The image can be transformed without the features changing.
- The sign of the covariance shows the tendency in the linear relationship between two features.
- If we have two transformed versions of the same image, features are detected in corresponding locations.
- The sign of the covariance shows the tendency in the complex relationship between two features.
* `[ Option C ]`


---

**Q: Is the Harris detector invariant to image intensity change?**
- Yes, partially, to additive and multiplicative intensity changes.
- Yes, completely, to additive intensity changes.
- Yes, partially, to additive and multiplicative intensity gains.
- Yes, partially, to additive and multiplicative intensity drops.
* `[ Option A ]`


---

**Q: In the context of Harris Corner Detection, we are given the structure tensor M and its 2 eigenvalues, ($\lambda_1$, $\lambda_2$). If both eigenvalues are small, what does this imply about the "cornerness" of the observed point?**
- The point is a vertical edge
- The point is a flat region
- The point is a corner
- The point is a horizontal edge.
* `[ Option B ]`


---

**Q: What will be the expected behaviour of the scale normalised Laplacian response, when the size of the laplacian approximates the size of a blob?**
- The response takes its maximum value.
- The response will shoot off to infinity. 
- The response takes its minimum value.
- The response will match the shape of the blob.
* `[ Option A ]`


---

**Q: Which of the following for the scale nomalization is wrong?**
- As sigma increases, the response of a derivative of Gaussian filer decreases
- Muliply the laplacian term by sigma
- Laplician of gaussian is circularly symmetric
- Characteristic scale is the scale that produces peak of laplacian response in blob center
* `[ Option B ]`


---

**Q: What of the following changes is sensitive for a ideal harris feature detector?**
- lighting
- perspective imaging
- partial occlusion
- corner region
* `[ Option D ]`


---

**Q: Which of the following is the correct measure of corner response?**
- R=traceM-k(\det{M})^2
- R=\det{M}-k(traceM)^2
- R=\sqrt{(traceM)^2+(\det{M})^2}
- R=traceM-\det{M}
* `[ Option B ]`


---

**Q: When using a harris detector, which of the following eigenvalue-classification combinations is correct?**
- When you have encountered a low \lambda_1 and a low \lambda_2 you have encountered an edge
- When you encounter a high \lambda_1 and a low \lambda_2, you have encountered a corner
- When you encounter a low \lambda_1 and high \lambda_2 you have encountered an edge
- When you encounter a high \lambda_1 and a high \lambda_2, you have encountered a flat surface
* `[ Option C ]`


---

**Q: We find the characteristic scale of a blob by convolving it with multiple Laplacians at different scales and looking for the maximum response. What happens the Laplacian response as the scale increases and how do we fix it?**
- The response amplifies and must be divided by the standard deviation
- The response decays and must be multiplied by the standard deviation 
- The response decays and must be multiplied by the variance
- The response decays and must be divided by the standard deviation
* `[ Option C ]`


---

**Q: We are using a harris detector or detect corners in an image. Let our 2 eigenvalues and their corresponding eigenvectors be L1, L2, V1, and V2, respectively. V1 is an eigenvector parallel to the X-axis, and V2 is an eigenvalue parallel to the y axis. Given that L1 >> L2, how would we classify this window?**
- Vertical Edge
- Horizontal Edge
- Flat Region
- Corner
* `[ Option A ]`


---

**Q: To perform a scale-invariant blob-detection, we need to:**
- convolve the image with a Laplacian kernel and multiply with $\sigma^2$
- convolve the image with a Laplacian kernel and multiply with $\sigma$
- convolve the image with a Gaussian kernel and multiply with $\sigma^2$
- convolve the image with a Gaussian kernel and multiply with $\sigma$
* `[ Option A ]`


---

**Q: The Harris detector:**
- is invariant to image intensity change
- is invariant to image scale
- is partially invariant to rotations
- does not require the computation of the eigenvalues of matrix M, characteristic of the intensity change
* `[ Option D ]`


---

**Q: We could detect blob via:**
- Convolving it with Laplacians at increased scale to see find the one with a maximum.
- Convolving it with Laplacians, which is multiplied by the square of the scale, at increased scale to find the one with a maximum.
- Harris Detector.
- None of above.
* `[ Option B ]`


---

**Q: Harris Detector is sensitive or partial sensitive to changes in:**
- Corner rotation.
- Image scale.
- Image intensity.
- Both B and C.
* `[ Option D ]`


---

**Q: Which of the following statements is True? The Harris detector is invariant to:**
- Image rotation and image additive intensity changes.
- Image additive intensity changes and image scaling.
- Image rotation and image scaling.
- None of the above.
* `[ Option A ]`


---

**Q: What value of the measure corner response (R) corresponds to the presence of a corner?**
- Low positive value.
- Low negative value.
- High positive value.
- High negative value.
* `[ Option C ]`


---

**Q: Can perspective be seen as a linear transformation?**
- Yes, but only if the object is not too far from the observation point
- Yes, it is indeed linear
- No, as it is an intrinsecally cubic non linear transormation
- Yes, but only if the object is far enough from the observation point
* `[ Option D ]`


---

**Q: Is the Harris Detector invariant with respect to rotations?**
- Yes, as the eigenvalues of M are constant
- Yes, as the derivatives of the image with respect to x and y directions do not change
- No, as the eigenvectors of M do not rotate with the image
- No, actually a procedure has to be carried on to make it rotation-invariant
* `[ Option A ]`


---

**Q: Assuming a $1\times 1$ window centered at a pixel with matrix $M = [1, 2; 2, 4]$. Extract corner response $R$; suppose that empirical constant $k=0.04$.
(Bonus question: classify the point based on the extracted $R$ -> edge.)**
- 0.2
- -1
- 1
- -0.2
* `[ Option B ]`


---

**Q: Assuming a focal length $f$ and center of projection $c$, what is the distance between the image plane and the virtual image plane, considering the setting of a pinhole camera?**
- $f$
- $f-c$
- $f+c$
- $2f$
* `[ Option D ]`


---

**Q: Harris detector is used for detecting**
- Edge
- Corner
- Flat region
- All of the above
* `[ Option D ]`


---

**Q: If one of eigen value is small for a region in Harris Detector the region has**
- Corner
- Flat Region
- edge
- edge or a flat region
* `[ Option D ]`


---

**Q: Harris Detector is fully invariant to:**
- Intensity change
- Image rotation
- Image scale
- Partial occlusion
* `[ Option B ]`


---

**Q: If both eigenvalues of a Harris Detector are large, this implies:**
- A blob
- An edge
- A flat region
- A corner
* `[ Option D ]`


---

**Q: Why is a Harris corner detector not scale invariant?**
- Because the Taylor series expansion used in computation of the M matrix (or the structure tensor matrix) is only an approximate
- Because it uses a fixed window size to create the M matrix
- Because the eigenvalues of the energy formulation change at different scales
- None
* `[ Option B ]`


---

**Q: What happens when gaussian filters of increasing sigma are convolved with a 1D blob**
- The responses of the convolution increase as sigma increases
- The responses of the convolution decreases as sigma increases
- The responses of the convolution remains the same as sigma increases
- None
* `[ Option D ]`


---

**Q: What feature is detected with the Harris Detector when the two eigenvalues are large?**
- Flat regions
- Horizontal edges
- Vertical edges
- Corners
* `[ Option D ]`


---

**Q: The Harris Detector is invariant to**
- Rotations
- Intensity changes
- Scaling
- Partial occlusion
* `[ Option A ]`


---

**Q: Where are feature points not used for?**
- Robot navigation
- Indexing and database retrieval
- Point selection
- Motion tracking
* `[ Option C ]`


---

**Q: Which of the following statements is false?**
- The magnitude of the Laplacian response will achieve a maximum at the center of the blob, provided the scale of the Laplacian isΓÇ£matchedΓÇ¥ to the scale of the blob and is called spatial selection.
- The response of a derivative of Gaussian filter to a perfect step edge increase as $\sigma$ increase.
- The characteristic scale of a blob is defined as the scale that produces peak of Laplacian response in the blob center.
- Invariance means that the image is transformed and features do not change. Covariance means that if there are two transformed versions of the same image, features should be detected in corresponding location.
* `[ Option B ]`


---

**Q: Which function can be used to define the perspective effect ? (Note : - z is the distance between the observer and the object)**
- 1
- z
- 1/z
- 1/z^2
* `[ Option C ]`


---

**Q: Corners can be identified through which property of eigen values in Harris Edge Detectors ? (note :- lambda 1 and lambda 2 denote the eigen values)**
- lambda 1 and lambda 2 are both large 
- lambda 1 large and lambda 2 small
- lambda 2 small and lambda 1 small
- lambda 1 and lambda 2 both small
* `[ Option A ]`


---

**Q: What does scale normalization for blob detection mean?**
- The response does not decay when increasing variance (sigma) for the gaussian kernel
- Small blobs can be detected only if the kernel is scale normalized
- With a fixed variance the laplacian response can get a maximum response for multiple sizes of blobs
- It allows for the estimation of the direction (orientation) of a blob
* `[ Option A ]`


---

**Q: Using the two eigenvalues in harris detection, what makes a good feature?**
- When both eigenvalues are very small
- When both eigenvalues are very large
- When one of the eigenvalues is much bigger than the other
- When the eigenvalues are equal to each other
* `[ Option B ]`


---

**Q: Considering a Harris detector, a large eigenvalue of M corresponds to...**
- a large variation of intensity in the direction of the corresponding eigenvector.
- a small variation of intensity in the direction of the corresponding eigenvector.
- no variation of intensity in the direction of the corresponding eigenvector.
- an edge perpendicular to the direction of the corresponding eigenvector.
* `[ Option A ]`


---

**Q: What statement is false?**
- The Harris detector is rotationally invariant.
- The Harris detector is invariant to intensity shift
- The Harris detector is image scale invariant.
- The Harris detector is partially invariant to intensity scale.
* `[ Option C ]`


---

**Q: Which is true:

I. Harris detector look in both x and y direction.

II. Harris detector is scale invariant.**
- I
- II
- Both
- Neither
* `[ Option A ]`


---

**Q: The hole in a barrier that creates an image (pinhole camera) is called a(n):**
- eye
- aperture
- interstice
- fissure
* `[ Option B ]`


---

**Q: What is NOT true about a blob?**
- A blob is a region in a digital image that differs in properties compared to surrounding regions
- A blob is a superposition of two ripples
- A blob is scale-invariant
- A blob is a region of an image in which some properties are constant or approximately constant
* `[ Option C ]`


---

**Q: What is true about the average intensity change E and the eigenvalues $\lambda_1$ and $\lambda_2$ for corner points?**
- $\lambda_1$ and $\lambda_2$ are small and E is almost constant in all directions
- $\lambda_1$ and $\lambda_2$ are large and E is almost constant in all directions
- $\lambda_1$ and $\lambda_2$ are small and E increases in all
directions
- $\lambda_1$ and $\lambda_2$ are large and E increases in all
directions
* `[ Option D ]`


---

**Q: What is the formula for scale-normalized blob detection in 2D?**
- $\nabla^2_{norm}g = \sigma(\frac{\partial^2g}{\partial x^2}+\frac{\partial^2g}{\partial y^2})$
- $\nabla^2_{norm}g = (\frac{\partial^2g}{\partial x^2}+\frac{\partial^2g}{\partial y^2})$
- $\nabla^2_{norm}g = \frac{\partial^2g}{\partial x^2}$
- $\nabla^2_{norm}g = \frac{\partial^2g}{\partial y^2}$
* `[ Option A ]`


---

**Q: Choose the correct relation of the measure of corner response ($R=\det{M}-k(\texttt{trace}M)^2$) and the feature classifications.**
- $(R < 0) \xrightarrow{}$ edge, $(R > 0) \xrightarrow{}$ corner, $|R|$ small $\xrightarrow{}$ flat area
- $(R < 0) \xrightarrow{}$ corner, $(R > 0) \xrightarrow{}$ edge, $|R|$ small $\xrightarrow{}$ flat area
- $(R < 0) \xrightarrow{}$ flat area, $(R > 0) \xrightarrow{}$ corner, $|R|$ small $\xrightarrow{}$ edge
- $(R < 0) \xrightarrow{}$ edge, $(R > 0) \xrightarrow{}$ flat area, $|R|$ small $\xrightarrow{}$ corner
* `[ Option A ]`


---

**Q: Why do we want scale normalization in the context of Gaussian filters?**
- A derivative response decays as sigma increases.
- A derivative response strengthens as sigma increases.
- A derivative response decays as sigma decreases.
- A derivative response strengthens as sigma decreases.
* `[ Option A ]`


---

**Q: To which of the following changes is the Harris Corner Detector sensitive?**
- Rotation
- Scaling
- Shifting
- Intensity scaling
* `[ Option B ]`


---

**Q: Given these two statements:
- The Harris corner detection method is invariant to rotation, since the Gaussian used for blurring is circularly symmetric
- The Harris corner detection method is almost invariant to intensity, since the entire algorithm works, only the threshold R must be fine-tuned for each intensity. **
- The first statement is True, the second if False
- The first statement is False, the second if True
- Both statements are True
- Both statements are False
* `[ Option A ]`


---

**Q: What can we say about the eigenvalues of the Structure Tensor M when there is a ΓÇÖcornerΓÇÖ inside the window?**
- One eigenvalue is big, and one eigenvalue is small
- One of the eigenvalues is approximately zero
- Both eigenvalues are small
- Both eigenvalues are big 
* `[ Option D ]`


---

**Q: How can you make the Laplacian response to a blob, scale-invariant?**
- a) multiply the Laplacian by sigma
- b) divide the Laplacian by sigma squared
- c) multiply the Gaussian derivative by sigma
- d) multiply the second Gaussian derivative by sigma squared
* `[ Option D ]`


---

**Q: The Harris Detector for detecting corners is**
- a) invariant to image rotation, invariant to image scale, and partially variant to intensity change
- b) invariant to image rotation, variant to image scale, and partially invariant to intensity change
- c) variant to image rotation, variant to image scale, and partial invariant to intensity
- d) invariant to image rotation, invariant to image scale, and invariant to intensity change
* `[ Option B ]`


---

**Q: What is the effect of scale normalization of Laplacian filter?**
- Response of filter will decay when increasing the scale
- Response of filter will increase when increasing the scale
- Response of filter will increase when reducing the scale
- Response will keep same when scale changes
* `[ Option D ]`


---

**Q: Which statement of eigenvalues of structure tensor M is not correct?**
- Large eigenvalue means large intensity change
- Image points with two small eigenvalues are in flat region
- Image points with both large eigenvalues are on the edge
- Image points are called corner if intensity changes strongly in all directions
* `[ Option C ]`


---

**Q: Why do we need to apply Scale Normalization when convolving a blob with the Laplacian, and how do we do it (for the Laplacian)?**
- Scale Normalization refers to increasing the sigma of the Laplacian to match the size of the blob. We do so by trying $\sigma$'s on a discrete interval.
- Scale Normalization refers to increasing the sigma of the Laplacian to match the size of the blob. We do so by multiplying the Gaussian derivative by $\sigma$.
- Scale Normalization is needed because the repsonse of a derivative of the Gaussian filter decreases as $\sigma$ increases. We do so by multiplying the Gaussian derivative by $\sigma$.
- Scale Normalization is needed because the repsonse of a derivative of the Gaussian filter decreases as sigma increases. We do so by multiplying the Gaussian derivative by $\sigma^2$
* `[ Option D ]`


---

**Q: To detect corners with Harris Detector, why is it unnecessary to compute the eigenvalues of the structure tensor?**
- Harris Detector detects edges by convolving the image with the Laplacian.
- The determinant of the structure tensor is large in edges and the trace of the structure tensor is large for corners. The values of the eigenvalues themselves are therefore unnecessary.
- The determinant of the structure tensor is large in corners and the trace of the structure tensor is large for edges. The values of the eigenvalues themselves are therefore unnecessary.
- When deciding on interest points, we are thresholding the corner response. The exact eigenvalues therefore don't matter, as long as we have a good estimation of each eigenvalue.
* `[ Option C ]`


---

**Q: What would be the unnormalized Laplacian response happen if the scale increases?**
- advance
- decay
- change unstable
- stay unchanged
* `[ Option B ]`


---

**Q: What will happen if we shift window on the ΓÇ£cornerΓÇ¥?**
- no change as shift window along the edge direction
- no change as shift window in all directions
- significant change as shift window in all directions
- slightly change as shift window along the edge direction
* `[ Option C ]`


---

**Q: Consider an un-normalised blob detector based on a Laplacian of Gaussian kernel with size 6x6. Which of the following blobs gives the highest response magnitude against black background (i.e. pixel values are 0) in a grey valued image? Note that all blobs are uniformly white (i.e. all pixel values are 255).**
- A regular triangular with edge lengths equal 2.
- A regular octagon with edge lengths equal 2.
- A square with edge lengths equal 6.
- A circle with a radius of 6.
* `[ Option B ]`


---

**Q: Which of the following manipulation of images will affect the corner detection by a Harris detector?**
- Rotating the image by 90 degrees clockwise.
- Inverting all the pixel values.
- Adding a constant to all pixel values (no saturation has happened).
- None of the manipulations listed above.
* `[ Option D ]`


---

**Q: The second derivative of Gaussian (laplacian) can be convolved with the image to detect**
- Blobs only
- Corners only
- Edges only
- Blobs, corners and edges
* `[ Option D ]`


---

**Q: How can Harris corner detector be made scale invariant ?**
- Harris detector is scale invariant by default.
- Harris detector cannot be made scale invariant as it is scale covariant.
- By combining the convolution of laplacian of Gaussian and Harris detector results, it can be made scale invariant
- By combining the convolution of Gaussian and Harris detector results, it can be made scale invariant
* `[ Option C ]`


---

**Q: Which of the following properties of the Harris Detector is correct? **
- Image rotation produces a variance in the result.
- Complete invariance to multiplicative intensity changes.
- Not invariant to image scale. 
- None of the above.
* `[ Option C ]`


---

**Q: In the case of a Harris Corner Detector, which of the following statements is incorrect? **
- R is large for a corner
- R depends only on eigenvalues of M
- R is negative with small magnitude for an edge
- None of the above
* `[ Option C ]`


---

**Q: Why is the Harris Detector only partially invariant to changes of the intensity of an image?**
- Because of the fixed window function
- Because of the fixed threshold on local maxima
- The statement is wrong. It is completely invariant.
- The statement is wrong. It is not invariant.
* `[ Option B ]`


---

**Q: If the eigenvalue $\lambda_{1}$ of the structure tensor M is much greater than the eigenvalue $\lambda_{2}$, which characteristic does the observed region probably have?**
- Flat region
- Edge
- Corner
- No statement can be made
* `[ Option B ]`


---

**Q: When doing the transformation of the projection of a pinhole model camera, is this transformation linear or nonlinear? Why?**
- Linear, because of the linear properties of similar triangles
- Linear, because the transformation is from 3D coordinates to 2D coordinates, which is linear by definition
- Nonlinear, because of the transformation of the z world coordinate to the focal length f
- Nonlinear, but only when the aperture is very small
* `[ Option C ]`


---

**Q: The Harris Corner detector is variant/invariant to: Rotation / Intensity shift I > I+b / Scaling**
- Variant / Invariant / Invariant
- Invariant / Variant / Variant
- Invariant / Invariant / Variant
- Invariant / Variant / Invariant
* `[ Option C ]`


---

**Q: A (harris) corner has**
- Two large eigenvalues
- Two small eigenvalues
- One big eigenvalue and one small eigenvalue
- Eigenvalues are not important for detecting corners
* `[ Option A ]`


---

**Q: Increasing sigma by a large margin for a derivative gaussian kernel will change the convolution result how?**
- Output image is more sparse than before
- Output image is less sparse than before
- Output image barely change than before
- None of the above
* `[ Option A ]`


---

**Q: The response of a derivative of Gaussian filter to a perfect step edge changes with sigma. What step should be taken to keep the response scale invariant in Laplacian:**
- Multiply gaussian derivative by sigma
- Multiply gaussian derivative by sigma^2
- Divide gaussian derivative by sigma
- Divide gaussian derivative by sigma^2
* `[ Option B ]`


---

**Q: Which of the following is NOT a property of Harris detector:**
- The detector is invariant to rotation
- Is invariant to change in intensity
- Response in dependent on the eigen values
- Is not invariant to change in scale
* `[ Option B ]`


---

**Q: M is a 2 by 2 Matrix computed from image derivatives. \lambda_1 and \lambda_2 are eignvalues of M. which of the following situation denote it's a corner using Harris Detector?**
- \lambda1 and \lambda2 are large and similar in magnitude
- \lambda1 and \lambda2 are small and similar in magnitude
- \lambda1 is larger than\lambda2
- \lambda1 is smaller than\lambda2
* `[ Option A ]`


---

**Q: Which of the following is not the drawbacks that Harris detector has?**
- sensitive to scale change
- sensitive to signigicant viewpoint change
- sensitive to significant contrast change
- sensitive to rotation
* `[ Option D ]`


---

**Q: How to solve the problem when you try to get a maximum of Laplacians but they decay for bigger scale?**
- Multiply Laplacian by \sigma (note, not \sigma*\sigma)
- Laplacian responses are scale invariant, so nothing needs to be done
-  Scale normalisation
- None of the above.
* `[ Option C ]`


---

**Q: Which of the following statements is FALSE of Harris Detector?**
- Harris detector is invariant to image scale
- Harris detector is partially invariant to intensity change
- Corner location is covariant w.r.t. translation
- Harris detector is covariant to image rotation
* `[ Option A ]`


---

**Q: What is the focal length of a pinhole camera**
- Distance between photographed object and the camera
- Distance between the sensor (film) and the photographed object
- Distance between the pinhole and the sensor (film)
- Distance between the pinhole and the photographed object
* `[ Option C ]`


---

**Q: Considering a Harris Corner Detector, which statement is true**
- Shifting window in all directions on a flat region will cause a large change of intensity
- Shifting a window along the edge will cause a large change of intensity
- Shifting window in all directions on a corner will not cause any intensity change
- Shifting window perpendicularly to the edge will cause a significant intensity change
* `[ Option D ]`


---

**Q: Taking the laplacian(double derivative) of gaussian filter and convolving it with the image gives us an idea about the presence of:**
- Edges
- Blob
- Corner points
- All of the above
* `[ Option D ]`


---

**Q: Which among the following is a property of Harris Detector?**
- Rotational invariance
- Invariant to Scaled intensity
- Invariant to Image scale
- None of the above
* `[ Option A ]`


---

**Q: In a pinhole camera, what is the focal length?**
- Distance between the image plane and the virtual image
- Distance between the image plane and the pinhole
- Distance between the pinhole and the virtual image
- None of them
* `[ Option B ]`


---

**Q: In a Harris detector, if one eigenvalue is much larger than other one, which shape will be detected?**
- Edge
- Corner
- Flat region
- Black region
* `[ Option A ]`


---

**Q: An ideal feature detector is insensitive to**
- Scale
- Intensity variations
- Perspective
- All of the above
* `[ Option D ]`


---

**Q: Choose the correct answer**
- In a pinhole camera, images are better defined as the aperture size is decreased until it's closure
- Light from one single object remains constant as aperture size is decreased
- Aperture shapes can be specially designed to get upright images
- None of the above
* `[ Option D ]`


---

**Q: When two eigenvalues of a second moment matrix are small what does it suggest about a part of the picture.**
- The picture has a strong edge in one direction in that part of the picture.
- The picture has a strong edge or point in both directions in that part of the picture.
- The picture has no big differences around in that part of the picture.
- None of the above answers are correct.
* `[ Option C ]`


---

**Q: Which of the following statements is correct?
I. An image is invariant if the image is transformed and the features do not change. 
II. An image is covariant if two transformed versions of the image have features in corresponding locations.**
- Only I is correct. 
- Only II is correct.
- Both I and II are correct.
- None of the statements are correct. 
* `[ Option C ]`


---

**Q: How is the eigenvulues calculated in Harris corner detection?**
- compute the derivitve of the lines
- compute the trace with the determinate in a function
- compute the determinant
- computer the trace with the derivitive
* `[ Option B ]`


---

**Q: Invariance or convaraiance what is true**
- Invariance both images different points
- Invariance both images same points
- Invariance has the featureres the same way as rotation the harris  
- None of the above
* `[ Option B ]`


---

**Q: Why must multi-scale gaussian filter be normalized?**
- Increasing stdev reduces derivative of an edge 
- Increasing stdev increases derivative of an edge 
- Increasing stdev reduces size of the edge
- Increasing stdev increases size of the edge
* `[ Option A ]`


---

**Q: Why is the structure tensor in Harris detector useful?**
- Allows us to calculate product and sum of eigenvectors easily
- Can be convolved with image for useful properties
- Only needs to be calculated once per image
- Independent of window function
* `[ Option A ]`


---

**Q: Given the structure tensor of a point in an image, what can you conclude when one of the eigenvalues is very large?**
- The point is not a part of a flat region
- The point is not a part of an edge
- The point is not a part of a corner
- None of the above, there lacks information. You would need the other eigenvalue as well.
* `[ Option A ]`


---

**Q: For the Harris corner detector, which of these following properties is it most invariant too?**
- Scale
- Intensity change
- Rotation
- None, they're all equally invariant or not invariant.
* `[ Option C ]`


---

**Q: The bar notation, for example $\bar{x}$, credited to Roger Grosse is meant to represent: **
- $\frac{\partial \mathcal{L}}{\partial x}$
- $\frac{\partial x}{\partial \mathcal{L}}$
- $\frac{\partial \mathcal{L}}{\partial w} \frac{\partial w}{\partial x}$
- $\frac{\partial x}{\partial w} \frac{\partial w}{\partial \mathcal{L}}$
* `[ Option A ]`


---

**Q: The end goal of backpropagation in Feed Forward Neural networks is:**
- To compute the derivative of w  with respect to $\mathcal{L}$.
- To compute the derivatives of w and b with respect to $\mathcal{L}$.
- To compute the derivative of $\mathcal{L}$  with respect to w and b.
- To compute the derivative of $\mathcal{L}$  with respect to w.
* `[ Option C ]`


---

**Q: Which statement about the corner response in a Harris detector is incorrect**
- The corner response depends on the eigenvalues and the null space of the structure tensor (M)
- The corner response is large for a corner
- The corner response is negative with a large magnitude for an edge
- The absolute value of the corner response is small for a flat region
* `[ Option A ]`


---

**Q: Which of these statements is incorrect for a Harris detector?**
- A region is considered 'flat' when there is no change as the window shifts in any direction
- A region is considered an 'edge' when there is no change as the windows shifts along the edge
- A region is considered a 'corner' when there is a significant change as the window shifts in all directions
- A region is considered a 'corner' when there is only a significant change as the window shifts orthogonally to the edge
* `[ Option D ]`


---

**Q: An ideal feature detector would always detect the same point on an image, even if there are changes in the image such as scaling, lighting, perspective imaging, etc. 
The Harris corner detector, however is not ideal. Which of these properties are FALSE about this detector?**
- Harris is rotation invariant.
- Harris is scale invariant
- Harris is not fully invariant to intensity changes
- Harris is partially invariant to intensity changes
* `[ Option B ]`


---

**Q: In the second lecture, we used the Laplacian of the gaussian for 'blob' detection. In this, we found out that the magnitude of the laplacian response is maximum around the center of the 'blob' as long as the laplacian's scale is 'matched' to the scale of the blob. However as the scale increases, the Laplacian response decays. Why does this happen?**
- The response of a derivative of Gaussian filter to a perfect step edge stays the same as ╧â increases
- The response of a derivative of Gaussian filter to a perfect step edge decreases as ╧â decreases.
- The response of a derivative of Gaussian filter to a perfect step edge increases as ╧â increases
- The response of a derivative of Gaussian filter to a perfect step edge decreases as ╧â increases
* `[ Option D ]`


---

**Q: Why do we need to normalize the DoG filters in blob detection**
- As we increase our variance, the DoG peaks become less intense, compensation is needed to detect blobs with same intensity
- We need to compensate for noise being amplified in the derivative
- Otherwise small blobs will vanish 
- We need to compensate for the perspective transform
* `[ Option A ]`


---

**Q: In the harris filter**
- Corners are detected by measuring large changes in vertical and horizontal neighbourhood of a pixel
- Corners are detected by measuring small changes in vertical and horizontal neighbourhood of a pixel
- Corners are detected by measuring a large change in the vertical and a small change in the horizontal neighbourhood of a pixel
- Corners are detected by measuring a small change in the vertical and a large change in the horizontal neighbourhood of a pixel
* `[ Option A ]`


---

**Q: What are feature points used for**
- Image alignment
- Motion tracking
- Robot navigation
- All of the above
* `[ Option D ]`


---

**Q: What can a harris detector be used for**
- detecting edges
- detecting high intensity areas
- detecting low intensity areas
- None of the above
* `[ Option A ]`


---

**Q: Which of the following properties is NOT correct about the Harris corner detector?**
- Invariant to image rotation
- Partial invariance to additive and multiplicative intensity changes
- Invariant to image scaling
- Invariant to image translation
* `[ Option C ]`


---

**Q: What is the characteristic scale of a blob?**
- The scale on which a peak is produced in the Laplacian response at the blob center
- The scale on which a peak is produced in the Gaussian response at the blob center
- The scale on which a peak is produced in the Laplacian response at the blob most left point
- The scale on which a peak is produced in the Laplacian response at the blob most right point
* `[ Option A ]`


---

**Q: What should we do on blob detection if the size of blob is unknown?**
- Try different size of kernel and choose the largest value
- Change the RGB picture to grayscale
- Change the histogram of the image
- Gaussian blur
* `[ Option A ]`


---

**Q: When applying Harris corner detection algorithm, how will the intensity changes when shifting windows on a corner?**
- No large change
- Only change greatly when shifting on one direction
- Only change greatly when shifting on two directions
- Change greatly on every direction
* `[ Option D ]`


---

**Q: Consider an object in front of the pinhole camera. The height of the object is 50cm and it is placed at a distance of 2m from the pinhole camera. A screen is placed on the other side of the pinhole camera and is at a distance of 0.5m from the camera. What is the height of the projected image?**
- 0.25m 
- 0.50m
- 0.125m
- 0.75m
* `[ Option C ]`


---

**Q: Harris Edge Detector is invariant to which of the following?**
- Additive and Multiplicative intensity changes
- Image rotation
- Scale Changes
- None of the above. It is only partially invariant to all the above mentioned changes. 
* `[ Option B ]`


---

**Q: Why is important to perform scale normalization when making blob detection?**
- Because the response of a Gaussian decreases as the standard deviation increases
- Because if it is not done, it is not possible to detect blobs of any scale
- All of the above
- None of the above
* `[ Option A ]`


---

**Q: Which one is not a property of the Harris corner detector?**
- Harris detector is rotation invariant
- Harris detector is intensity change invariant
- Harris detector is scale change invariant
- Harris detector does not have any of the properties above
* `[ Option C ]`


---

**Q: The 2D projection modelling divides both x and y axis by z, which statements about the perspective effects is TRUE?**
- The far away view is nearly linear, but the transform is non-linear
- The close view is nearly linear, but the transform is non-linear
- The far away view is nearly non-linear, but the transform is linear
- The close view is nearly non-linear, but the transform is linear
* `[ Option A ]`


---

**Q: In implementing Harris Detector, what can we tell about the measure of corner response R?**
- To compute R, we need to compute the eigenvalues
- The R is always a positive value
- R is large for detecting an edge
- R is small for detecting a flat region
* `[ Option D ]`


---

**Q: Which of the following statements is correct regarding the Harris detector. Note that M is the structure tensor and ╬╗1, ╬╗2 are eigenvalues of M. **
- If ╬╗1, ╬╗2 are large then the examined point is a corner.
- If ╬╗2 = ╬╗1  then the examined point is a horizontal edge.
- If ╬╗2 < ╬╗1 and ╬╗1, ╬╗2 are small, then the examined point is a  corner.
- If ╬╗1, ╬╗2 are small there is a peak at the examined point.
* `[ Option A ]`


---

**Q: Harris detector is invariant to:**
- rotation
- image scaling
- intensity change
- Options A and C.
* `[ Option A ]`


---

**Q: Consider a Harris Detector where $R=det(M) - k \cdot trace(M)^2$. It detects an edge when:**
- R is positive and large
- |R| is small
- R is negative and large
- floor(R) is a perfect square
* `[ Option C ]`


---

**Q: What is a blob?**
- A superposition of two ripples
- A place of rapid change in the image intensity function
- The magnitude of the Laplacian response
- The corner area covered by the moving window in Harris Corner Detector algorithm
* `[ Option A ]`


---

**Q: In the edge detection, for unnormalized Laplacian response, what change will happen for response with the scale ( the sigma) increasing?**
- Decay
- Enhance
- Decay first, then enhance
- Enhance first, then decay
* `[ Option A ]`


---

**Q: In the Harris detector, if R is large and positive, then we can know what detected area is?**
- Flat
- Edge
- Corner
- Cannot make a judgement
* `[ Option C ]`


---

**Q: Which one is wrong?**
- To find feature points and their correspondences. One approach is to find features in one image that can be accurately tracked using a local search technique. Another is to independently detect features in all the images under consideration and then match features based on their local appearance. 
- The approach to find features in one image that can be accurately tracked using a local search technique is more suitable when a large amount of motion or appearance change is expected. 
- A mathematical way to define the slope and direction of a surface is through its gradient.
- To perform edge detection, simple approach is to combine the outputs of grayscale detectors run on each color band separately.
* `[ Option B ]`


---

**Q: Which one is wrong?**
- As with the Harris operator, pixels where there is strong asymmetry in the local curvature of the indicator function are rejected.  This is implemented by first computing the local Hessian of the difference image D.
- Invariance: image is transformed and features do not change.
- A good (corner) point should have a large intensity change in 
all  directions, i.e.  R should be large negative.
- We define the characteristic scale of a blob as the scale that produces peak of Laplacian response in the blob center
* `[ Option C ]`


---

**Q: Which of the following description is correct?**
- A corner could be recognised as an edge
- Rotation sometimes would change the results of corner detection
- The different orientations of detection would lead to different detection results
- If the gray-scale graph is valued from [0,1], 0 is black and 1 is white, then the Harris detection threshold should be a value belongs to [0,1]
* `[ Option A ]`


---

**Q: Which of the following description is correct?**
- Edge detection performs better on the situation that blob is black and background is white than the contract
- Modelling projection is the linear function of distance of object to center of projection
- An edge could be detected only on some certain orientations
- The ideal pinhole is as small as possible to limit the light
* `[ Option D ]`


---

**Q: A blob detector based on the laplacian is inherently scale invariant**
- True, but only if the blob is viewed from the same perspective
- True, it will always work for different scales.
- False, blob detection is not used on different scales.
- False, but it can work for multiple scales by finding maximum response.
* `[ Option D ]`


---

**Q: Which statement is true about the Harris Detector?**
- Finding accurate corners is only possible when the image is in grayscale
- Harris Detector is computationally infeasible for images with high dimensions
- Harris Detector can be sped up by replacing the eigenvalues with constants
- The corner response is translationally invariant
* `[ Option D ]`


---

**Q: Which of the following is an application of feature detection?**
- Image transformation
- Removing noisy pixels
- Motion Tracking
- None of the above
* `[ Option C ]`


---

**Q: In the Harris detector, if the eigenvalue $\lambda2$ is plotted against $\lambda1$, which region represents a flat region?**
- Bottom Left
- Bottom Right
- Top Right
- Top Left
* `[ Option A ]`


---

**Q: The Laplacian needs to multiplied with what factor to normalize scale:**
- $\sigma$
- $\sigma^2$
- $x$
- $\sqrt{\sigma}$
* `[ Option B ]`


---

**Q: What of the following statements is true:**
- The Harris detector is not rotation invariant
- The Harris detector is fully intensity invariant
- The Harris detector is not invariant to image scale
- The Harris detector cannot detect edges, only corners
* `[ Option C ]`


---

**Q: A Harris detector can detect:**
- Edges only
- Corners only
- Edges and corners
- None of the above
* `[ Option C ]`


---

**Q: A Harris detector is invariant to:**
- Image rotation
- Image intensity
- Image scale
- Threshold for R
* `[ Option A ]`


---

