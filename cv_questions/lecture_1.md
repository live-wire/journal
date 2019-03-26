# Questions from `lecture_1` :robot: 

**Q: We are performing edge detection, by convolving an image F with the second derivative of a Gaussian filter. Where are potential edges located at?**
- maxima
- minima
- zero-crossings
- all of the above
* `[ Option C ]`


---

**Q: Consider the case where an image F of size $28\times 28$ is convolved with a mask H of size $5\times 5$. What is the size of the output image assuming that no padding is utilized?**
- $28\times 28$
- $26\times 26$
- $24\times 26$
- $24\times 24$
* `[ Option D ]`


---

**Q: The convolution operation on an image always results in a different output than the correlation operation output on the same image.**
- TRUE
- FALSE
- N.A.
- N.A.
* `[ Option B ]`


---

**Q: When is the second derivative of a pixel (x,y)  nonzero?**
- Constant intensity areas
- Intensity transition areas
- Constant intensity slope areas
- None of the above
* `[ Option B ]`


---

**Q: If we have a 5 by 5 moving neighbor average filter, how many rows/columns will end up empty?**
- 2
- 1
- 3
- None of the above
* `[ Option A ]`


---

**Q: Which of the following statements about convolution is false:**
- It is linear and shift invariant.
- Differentiating the result of a convolution is equivalent to differentiating the signal and then applying the convolution.
- Is carried out by flipping the filter in the vertical direction and then applying cross correlation.
- Convolving the impulse signal results in the original filter.
* `[ Option C ]`


---

**Q: Let $H_\alpha$ be a kernel whose entries sum up to $\alpha$. Let $I$ be a grayscale image and denote $I_\alpha$ by the convolution of $I$ with $H_\alpha$; how do the intensities of $I_{0.5}$ and $I_{2}$ relate to the intensity of $I$?**
- The intensity of $I_{0.5}$ is lower and the intensity of $I_2$ is higher.
- The intensities of both $I_{0.5}$ and $I_{2}$ are lower.
- The intensity of $I_{0.5}$ is higher and the intensity of $I_2$ is lower.
- The intensities of both $I_{0.5}$ and $I_2$ are higher.
* `[ Option A ]`


---

**Q: To find edges of an image given by a function $f: \mathbb{R}^2 \to \mathbb{R}$, one can look at $|\nabla f|$ and $\Delta f$, respectively looking for:**
- maxima and zero-crossings.
- maxima and minima.
- minima and zero-crossings.
- minima and maxima.
* `[ Option A ]`


---

**Q: When convolving an impulse signal (image) with a kernel H, the result is:**
- An image with in the center the transposed of the kernel function (H).
- An image with in the center the  kernel function (H).
- An image with in the center the inverse of the kernel function (H).
- None of the above 
* `[ Option B ]`


---

**Q: When using a gaussian filter, the rule of thumb is that the variance should be extended to:**
- One sigma
- Two sigma
- Three sigma
- Does not matter because the gaussian function has infinite support.
* `[ Option C ]`


---

**Q: What happens to the image when filtering with kernel [0 1 0; 0 0 0; 0 0 0]?**
- The image is shifted up by 1 pixel
- The image is shifted down by 1 pixel
- The image is shifted left by 1 pixel
- No change
* `[ Option B ]`


---

**Q: Which is not a property of convolution?**
- Commutative
- Associative
- Distributive
- All three are properties of convolution
* `[ Option D ]`


---

**Q: When is the result of convolution and cross-correlation identical?**
- When the signal covariance is negative
- When the filter is symmetrical
- When the filter value add up to 1
- Convolution and cross-correlation are always identical
* `[ Option B ]`


---

**Q: Properties of convolution are:**
- Linear, shift-invariant, associative, commutative, identity, differentiation.
- Non-linear, shift-invariant, associative, commutative, identity, differentiation.
- Linear, translation-invariant, associative, identity, differentiation.
- None of the above.
* `[ Option A ]`


---

**Q: Applying either convolution or cross-correlation on an image would have a different effect when using the following kernel:**
- Derivative filter
- Gaussian filter
- Laplacian filter
- Uniformly weighted "box filter"
* `[ Option A ]`


---

**Q: If g is the Gaussian filter, a possible edge detection can be performed by convolving the image with the following kernel/kernels:**
- g
- $\nabla g$
- $\nabla^2 g$
- $\nabla g$ and $\nabla^2 g$
* `[ Option D ]`


---

**Q: Which properties does convolution have but not correlation?**
- Associative
- Additive
- Invertible
- No difference
* `[ Option A ]`


---

**Q: What does applying Gaussian filter NOT do?**
- Smooth image
- Blur image
- Reduce size of image
- Make it easier to find edges
* `[ Option C ]`


---

**Q: Which of the following statements about Gaussian filtering is true?**
- neighboring pixels generally have all the same weight on the final pixel result
- The choice of the kernel size does not influence computational complexity
- Solutions (such as kernel normalization) need to be employed to avoid obtaining darker images
- the smaller the variance of the Gaussian, the greater the number of neighboring pixels that ΓÇ£matterΓÇ¥ for the final outcome
* `[ Option C ]`


---

**Q: Which of the following statements about filtering is false?**
- Special operations (such as padding) are needed to compute filtering at the borders of the image
- If weights sum up to a value smaller than 1, the resulting image will appear brighter
- Image shifting can be achieved by means of linear filters
- Gaussian smoothing is typically smoother than with using a box filter
* `[ Option B ]`


---

**Q: Which of these functions does not introduce artifacts?**
- The Gaussian
- Correlation filtering
- Averaging kernel
- Moving neighbourhood average
* `[ Option A ]`


---

**Q: Which statement concerning Gaussian filters is incorrect?**
- Gaussian filtered edges of an image can be found by convoluting the image with the Gaussian kernel and then calculating the edges of the result of this transformation
- Gaussian filtered edges of an image can be found by calculating the edges of the image and then convoluting the result of this transformation with the Gaussian kernel
- The size of the mask or kernel of the Gaussian determines the extent of smoothing
- The size of the mask or kernel of the Gaussian does not have to be determined for nondiscrete filters
* `[ Option C ]`


---

**Q: What generally happens if you increase the sigma of a gaussian filter?**
- The image gets blurrier;
- The image gets sharper;
- More random noise is removed;
- Nothing.
* `[ Option A ]`


---

**Q: Which statement is true? A box filter:**
- weighs each pixel evenly;
- takes only the pixels into account that are on the border of the kernel;
- Increases the total intensity of the image;
- Decreases the total intensity of the image;
* `[ Option A ]`


---

**Q: Given the averaging window size is 2k+1$\times$2k+1, the formula of the correlation filtering is:**
- $$G[i,j] = \frac{1}{(2k+1)^2} \sum_{u=-k}^{k} \sum_{v=-k}^{k} F[i+u, j+v]$$
- $$G[i,j] = \frac{1}{(2k+1)^2} \sum_{u=-k}^{k} \sum_{v=k}^{k} F[i-u, j-v]$$
- $$G[i,j] = \frac{1}{(2k+1)^2} \sum_{u=-\infty}^{\infty} \sum_{v=-\infty}^{\infty} F[i+u, j+v]$$
- $$G[i,j] = \frac{1}{(2k+1)^2} \sum_{u=-2k+1}^{2k+1} \sum_{v=-2k+1}^{2k+1} F[i+u, j+v]$$
* `[ Option A ]`


---

**Q: If the parameters of the Gaussian filter are not properly tuned, the resulting image will be darker than the original. Which one of the following techniques is \textbf{NOT} solving the problem?**
- Fix $\sigma^2$, increase window size
- Fix window size, decrease $\sigma^2$
- Fix window size, increase $\sigma^2$
- Increase window size, decrease $\sigma^2$
* `[ Option C ]`


---

**Q: Which of the following is false**
- Convolution is a linear operator
- The derivative operator is a linear operator
- Correlation is associative
- Convolution is associative
* `[ Option C ]`


---

**Q: What is the effect of the derivative operator on the noise level of the image**
- It increases
- It decreases
- It has no effect
- Only the second derivative effects the noise level
* `[ Option A ]`


---

**Q: Which smoothing filter does not introduce artifacts according to koenderink (1984)?**
- Conservative filter
- Gaussian filter
- Mode filter
- Median filter
* `[ Option B ]`


---

**Q: Which of the following is NOT a property of convolution?**
- Linear & shift invariant
- Commutative
- Associative
- Non-distributive
* `[ Option D ]`


---

**Q: Which of the following are NOT properties of convolution?**
- Linear & shift invariant
- Communative (f*g = g*f)
- Associative (f * g) * h = f * (g * h)
- They are all properties of convolution
* `[ Option D ]`


---

**Q: Which gaussian filters below will blur an image the most?**
- Sigma = 0.5, 10x10 kernel
- Sigma = 6, 3x3 kernel
- Sigma = 15, 15x15 kernel
- Sigma = 3, 30x30 kernel
* `[ Option C ]`


---

**Q: Which of the following options is not the reason that vision project is hard for machines?**
- illumination
- view point variation
- occlusion
- color hue
* `[ Option D ]`


---

**Q: What is the result of original figure after filtering which the linear filter has only a ΓÇ£1ΓÇ¥ in row 1, column 2 and others are ΓÇ£0ΓÇ¥ ?**
- shifted left by 1 pixel with correlation
- shifted right by 1 pixel with correlation
- no change
- none of the above is correct
* `[ Option B ]`


---

**Q: What results one may expect filtering image with simple box filter and with Gaussian filter?**
- Using Gaussian filter, pixels that are further away from the center of a kernel does not have a big influence and filtering is more precise.
- Using simple box filter one may expect better results, because box filter kernel takes all pixels into account.
- Using Gaussian filter one may expect better results because Gaussian filter estimates pixel intensities based on Gaussian distribution.
- Both Gaussian and Box filters will give exact same results.
* `[ Option A ]`


---

**Q: Consider cross correlation and convolution operations. Which statement is true?**
- Using convolution operation one may find derivative of a kernel and then convolve the image instead of finding derivative of a whole image.
- Using cross correlation operation one may achieve faster and better results than using convolution.
- Using cross correlation one may find derivative of a kernel and then convolve the image instead of finding derivative of a whole image.
- Both operations have the same properties.
* `[ Option A ]`


---

**Q: For a Gaussian filter, what is a rule of thumb to choose the extent of the kernal for a given sigma?**
- 2*sigma
- 3*sigma
- 4*sigma
- 5*sigma
* `[ Option B ]`


---

**Q: Given an original image, we apply a correlation filter which is [0 1 0; 0 0 0; 0 0 0] on this image, how is the movement of this image?**
- shift up 1 pixel
- shift down 1 pixel
- shift left 1 pixel
- shift right 1 pixel
* `[ Option B ]`


---

**Q: What is NOT a property of convolution?**
- Commutative
- Associative
- Linear & Shift Invariant
- f*(h \cdot g) = (f*h) \cdot g
* `[ Option D ]`


---

**Q: How wide should a kernel be, according to the rule of thumb presented in the lecture?**
- [-3 \sigma ; 3 \sigma]
- [-1.5 \sigma ; 1.5 \sigma]
- [-6 \sigma ; 6 \sigma]
- [-\sigma ; \sigma]
* `[ Option A ]`


---

**Q: Considering a square Gaussian filter with variance equal to s, which is the best dimension for the filter side?**
- s
- 3*s
- 6*s
- 4*s
* `[ Option C ]`


---

**Q: Which of the following filter has to be convolved with a given image in order to perform the best edge detection in the image itself?**
- A second order derivative of a Gaussian filter
- A first order derivative of a Gaussian filter
- A Gaussian filter
- An averaging filter
* `[ Option A ]`


---

**Q: Which is NOT a property of convolution?**
- Commutative
- Associative
- Non-linear and shift invariant
- Distributive
* `[ Option C ]`


---

**Q: Which is the correct formula to compute the edge strength?**
- $\|\nabla f\| = \sqrt{(\frac{\partial f}{\partial x})^2 - (\frac{\partial f}{\partial y})^2}$
- $\|\nabla f\| = \sqrt{(\frac{\partial f}{\partial x})^2 + (\frac{\partial f}{\partial y})^2}$
- $\|\nabla f\| = \sqrt{(\frac{\partial f}{\partial x}) - (\frac{\partial f}{\partial y})}$
- $\|\nabla f\| = \sqrt{(\frac{\partial f}{\partial x}) + (\frac{\partial f}{\partial y})}$
* `[ Option B ]`


---

**Q: Why setting extent to 3 ╧â is a rule of thumb in Gaussain filter?**
- Beyond this extent, the magnitude is almost equal to zero
- Beyond this extent, the phase almost equal to zero
- Beyond this extent, the filter introduce artifact
- Within this range, the extent is smoother
* `[ Option A ]`


---

**Q: Practice with linear filter: an image is correlated by a filter with 9 pixels, only the right middle one is 1 and 0 in elsewhere, what will the resulting image look like:**
- Shift one pixel to left
- Shift one pixel to right
- Shift two pixels to left
- Shift two pixels to right  
* `[ Option A ]`


---

**Q: What would happen if the extent (size) of a gaussian filterΓÇÖs kernel would be chosen to be significantly smaller than 3*sigma? **
- The resulting image of the convolution would appear to be darker.
- The resulting image of the convolution would appear to be brighter
- The resulting image of the convolution would appear to be very blurred
- The convolution will have no effect on the original image
* `[ Option A ]`


---

**Q: Which of the following is false about cross correlation and convolution?**
- convolution and cross correlation produce the same result if the kernels are symmetric in both dimensions
- When applying convolution and cross correlation to an impulse signal image, the resulting image will be the same
- It makes no difference if we convolve and image with a kernel or if we convolve a kernel with an image
- A convolution kernel is the same as a correlation kernel that has been rotated by 180 degrees.
* `[ Option B ]`


---

**Q: Which of the following kernels does not produce an image artifact?**
- \frac{1}{9}\begin{bmatrix}
1 & 1 &1 \\ 
 1& 1 & 1\\ 
 1& 1 & 1
\end{bmatrix}
- \begin{bmatrix}
0 & 0 &0 \\ 
 0& 0 & 1\\ 
 0& 0 & 0
\end{bmatrix}
- \begin{bmatrix}
0 & 0 &0 \\ 
 0& 0 & 0\\ 
 0& 1 & 0
\end{bmatrix}
- \frac{1}{16}\begin{bmatrix}
1 & 2 &1 \\ 
 2& 4 & 2\\ 
 1& 2 & 1
\end{bmatrix}
* `[ Option D ]`


---

**Q: Which of the following is not a property of convolution?**
- Commutativity
- Associativity
- Transitivity
- Identity
* `[ Option C ]`


---

**Q: What is correct about gaussian?**
- Small details can be filterd away, that can have later performace improvements.
- Gaussian has the problem that it introduce artifacts.
- After a gaussian filter when over the image size does not change.
- Gaussian filter uses a gradient magnitude with a gaussian distribution to perform filtering.
* `[ Option A ]`


---

**Q: What is not true about convolution?**
- Commutative f*g = g*f
-  Associative (f*g)*h = f*(g*h)
- It's not linear and shift invariant
- differentiation d(f*g)/dx = df/dx * g
* `[ Option C ]`


---

**Q: For Gaussian filters the rule of thumb is to set the extent to ΓÇª**
- 1 $$\sigma$$
- 2 $$\sigma$$
- 3 $$\sigma$$
- 4 $$\sigma$$
* `[ Option C ]`


---

**Q: What is the result of convolving the impulse signal F with a kernel H?**
- impulse signal F
- kernel H
- kernel H rotated 180 degrees
- kernel H with signs of elements changed
* `[ Option B ]`


---

**Q: Gaussian Filter is which type of filter ?**
- High Pass
- Low Pass
- Band Pass
- Band Stop
* `[ Option B ]`


---

**Q: What is the use of convolving a gaussian filter before performing edge detection ?**
- Counteract the effect of noise
- Enhance the edges
- Blur the Background
- None of the above
* `[ Option A ]`


---

**Q: Which property is not a challenge for a computer vision system?**
- Size
- View point variation
- Occlusion
- Local ambiguity and context
* `[ Option A ]`


---

**Q: What is the result of applying a box filter to an image?**
- No difference
- Sharpened
- Blurred
- Darkened
* `[ Option C ]`


---

**Q: Statement 1: 
With convolution filtering, the Kernel can have any size, even bigger than the original image.
Statment 2: 
With Gaussian filtering, the high frequency content of the image is conserved. **
- Statement 1 is True, Statement 2 is False
- Statement 1 is False, Statement 2 is True
- Both Statements are True
- Both Statements are False 
* `[ Option A ]`


---

**Q: When convolution with a 3x3 Kernel is separated into two convolutions (one with a 1x3 and one with a3x1 Kernel), this saves a computational factor of operation: **
- 1.5 times faster
- 1.7 times faster
- 3 times faster
- They are equally fast 
* `[ Option B ]`


---

**Q: Which statement best describes gaussian filtering of an image?**
- Take the average value of the neighbouring pixels of a certain range
- Take the convolution of the input image and a gaussian kernel
- Take the gradient between a pixel and its direct neighbours
- Determine the mean and variance of a normal distribution that best describes the image
* `[ Option B ]`


---

**Q: What is NOT a property of convolutions?**
- Commutative
- Associative
- Linear & shift invariant
- Convexity
* `[ Option D ]`


---

**Q: Which of the following properties is NOT a property of a convolution?**
-  shift variant
- Identity
- Associative
- Commutative
* `[ Option A ]`


---

**Q: Which sequence of  filter displacements should be done in order to execute a convolution?**
- right to left, bottom to top.
- left to right, top to bottom. 
- bottom to top, right to left.
- bottom to top, left to right.
* `[ Option C ]`


---

**Q: Which kernel would you use if you want to smooth out your image**
- [1 2 1][2 4 2][1 2 1]
- [1 1 1][0 0 0][1 1 1]
- [1 0 1][2 0 2][1 0 1]
- [0 0 0][0 1 0][0 0 0]
* `[ Option A ]`


---

**Q: What approach would you use do mark the gradient orientations of the edges of a picture.**
- Image intensity function
- First derivative of the image intensity function
- Second derivative of the image intensity function
- First derivative of the image histogram function
* `[ Option B ]`


---

**Q: Convolution is a very similar process to Correlation, being that it is exactly like it, except that the filter is flipped before the correlation step. However it has a property over the other which is quite useful.It is associative. What does this property allow for?**
- It allows for the order of several convolutions to be inconsequential;
- It allows for the  the same operation to be performed at every point of  an image;
- Assuming we have several filters to apply to an image, convolution allows for the precomputation of all filters into  a single one, instead of applying each one sucessively;
- It allows for the  convolution of a sum of signals to be equal to the sum of the convolution of each signal. 
* `[ Option C ]`


---

**Q: What type of box filter would you use to shift an image 1 pixel diagonally?**
- \begin{bmatrix}
0 & 0 & 1 \\
0 & 1 & 0  \\
1 & 0 & 0
\end{bmatrix}
\quad
- \begin{bmatrix}
0 & 0 & 0 \\
0 & 1 & 0  \\
0 & 0 & 0
\end{bmatrix}
\quad
- \begin{bmatrix}
1 & 0 & 1  \\
0 & 1 & 0  \\
1 & 0 & 1
\end{bmatrix}
\quad
- \begin{bmatrix}
0 & 0 & 1  \\
0 & 0 & 0  \\
0 & 0 & 0
\end{bmatrix}
\quad
* `[ Option D ]`


---

**Q: Which of statements is true regarding Gaussian function?**
- Gaussian function introduces artifacts as other functions do.
- As Gaussian function is infinite, the larger filter window size the better the results (Gaussian filtering).
- Convolution using Gaussian derivative in x-direction generates edges along the y-direction in the image.
- The second derivative Gaussian convolution is not necessary because the first derivative Gaussian convolution is good enough to detect the edges.
* `[ Option C ]`


---

**Q: Which of the following statement is the most accurate one describing the difference between correlation filtering and convolution?**
- Correlation filtering and convolution have different window sizes.
- Convolution has property of identity while correlation filtering has not.
- The kernel matrix in convolution is just like a transpose of the filtering matrix in correlation filtering.
- Correlation filtering is commutative while convolution is not.
* `[ Option B ]`


---

**Q: Suppose you want to filter an image by use of a gaussian filter with a 3x3 matrix where the nearest neighboring pixel has the most influence on the output. What should be the correct normalization factor and matrix to use?**
- 1/18 and [2 2 2; 2 2 2; 2 2 2]
- 1/16 and [1 2 1; 2 4 2; 1 2 1]
- 1/16 and [2 1 2; 1 4 1; 2 1 2]
- 1/16 and [-1 2 -1; 2 4 2; -1 2 -1]
* `[ Option B ]`


---

**Q: Suppose that at the point (2,2) the image can be described by the following function f(x) = xy^2 + 3x - 5y, compute the gradient direction and the gradient magnitude at this point?**
- M = 7.1 & theta = -8 (wrong gradient calc)
- M = 11.4 & theta = 74.7 (dx/dy instead of dy/dx)
- M = 7.1 & theta = -81 (wrong gradient calc)
- M = 11.4 & theta = 15.3
* `[ Option D ]`


---

**Q: What is a limitation to Gaussian Filtering?**
- The filtering gives less weight to nearby pixels compared to further pixels.
- Because we are only given a single image, it is hard to estimate the parameters of the Gaussian that the kernel is based on accurately.
- When using Gaussian Filtering, the kernel usually becomes very large to capture the entire probability distribution of the Gaussian.
- The kernel is a discretization of the continuous Gaussian probability distribution.
* `[ Option D ]`


---

**Q: Under what condition is the convolution with a kernel equivalent to the cross-correlation with that same kernel?**
- This is equivalent only if the kernel is a Box Filter.
- This is equivalent only if the kernel is a Gaussian Filter.
- This is equivalent for any kernel that is symmetric around its horizontally centered values.
- This is equivalent for any kernel that is point symmetric around its center point.
* `[ Option D ]`


---

**Q: What is the result of the following operation? (cross-correlation of an impulse signal with a kernel)

\begin{equation}
S = \left(\begin{array}{ccccc} 0 & 0 & 0 & 0 & 0\\ 0 & 0 & 0 & 0 & 0\\0 & 0 & 1 & 0 & 0\\0 & 0 & 0 & 0 & 0\\0 & 0 & 0 & 0 & 0 \end{array}\right)
\bigotimes
\left(\begin{array}{ccc} a&b&c \\ d&e&f \\ g&h&i\end{array}\right) 
\end{equation} **
- \begin{equation}
\left(\begin{array}{ccccc} 0 & 0 & 0 & 0 & 0\\ 0 & i & h & j & 0\\0 & f & e & d & 0\\0 & c & b & a & 0\\0 & 0 & 0 & 0 & 0 \end{array}\right)
\end{equation}
- \begin{equation}
\left(\begin{array}{ccccc} 0 & 0 & 0 & 0 & 0\\ 0 & a & b & c & 0\\0 & d & e & f & 0\\0 & g & h & i & 0\\0 & 0 & 0 & 0 & 0 \end{array}\right)
\end{equation}
- \begin{equation}
\left(\begin{array}{ccccc} 0 & 0 & 0 & 0 & 0\\ 0 & g & h & i & 0\\0 & d & e & f & 0\\0 & a & b & c & 0\\0 & 0 & 0 & 0 & 0 \end{array}\right)
\end{equation}
- \begin{equation}
\left(\begin{array}{ccccc} 0 & 0 & 0 & 0 & 0\\ 0 & c & b & a & 0\\0 & f & e & d & 0\\0 & 0 & 0 & 0 & 0\\0 & 0 & 0 & 0 & 0 \end{array}\right)
\end{equation}
* `[ Option A ]`


---

**Q: What is the effect of the following kernel on an image? 

\begin{table}[!ht]
\centering
\begin{tabular}{|l|l|l|}
\hline
0 & 0 & 0 \\ \hline
1 & 0 & 0 \\ \hline
0 & 0 & 0\\ \hline
\end{tabular}
\end{table}**
- moves the image one pixel to the left
- moves the image one pixel to the right
- moves the image one pixel up
- moves the image one pixel down
* `[ Option B ]`


---

**Q: What doe the linear filter [0 0 0;1 0 0;0 0 0] do when applied to an image I?**
- Shifts the image towards Right by 1 pixel
- Shifts the Image  towards Left by 1 pixel
- Blurs the Image
- Both b) and c)
* `[ Option A ]`


---

**Q: How to identify edges in an image after taking the First or Second derivatives of Image**
- Look for zero crossing after taking the First derivative
- Look for zero crossing after taking the Second derivative
- Look for peaks after taking the second derivative
- Both a) and c)
* `[ Option B ]`


---

**Q: Suppose one wants to slightly smooth an image using a method that puts more important to the filters centre pixels. Which of the underlying filters will generally be most appropriate?**
- A moving average filter
- A box filter
- A Gaussian filter with large variance
- A Gaussian filter with small variance
* `[ Option D ]`


---

**Q: Which of the following statements about the gradient is incorrect (if any):**
- The edge strength is given by the magnitude of the gradient
- An image's histogram is build up from gradients
- An image with a gradient running from top to bottom will have a gradient of 0 in the x-direction
- All are correct
* `[ Option C ]`


---

**Q: What happens when the Gaussian filter is not normalized by dividing each value with the sum of all values ?**
- The filtered image becomes brighter.
- The filtered image becomes darker.
- The filtered image will remain the same as input image.
- The filtered image shifts to the right.
* `[ Option A ]`


---

**Q: Which of the following is the best way to address when the Gaussian with standard deviation \sigma doesn't fit within the chosen kernel window ?**
- Reduce \sigma of the Gaussian.
- Increase the kernel size.
- Both A and B are correct.
- Reduce the kernel size.
* `[ Option C ]`


---

**Q: Which of the following can be used for edge detection in an image**
- Maximum point in the intensity function of the image
- Maximum point in the intensity function of the image
- Minimum or Maximum point in the first derivative of the intensity function
- None of the above
* `[ Option D ]`


---

**Q: In edge detection, why is important to use a smoothing filter before taking the derivative?**
- It is less computationally expensive
- Noise in the image make it difficult to locate extrema of the derivative of the intensity function
- It is a mathematically more elegant solution
- None of the above
* `[ Option D ]`


---

**Q: Which pair of the size and variance of a Gaussian kernel is the most appropriate (in the sense that kernels are not cut off too early)?**
- size = 30x30, ╧â = 6
- size = 100x100, ╧â = 50
- size = 20x20, ╧â = 8
- size = 60x60, ╧â = 20 
* `[ Option D ]`


---

**Q: Consider the derivatives of a 2D Gaussian kernel: Dxy (second derivative over x and y) and Dyxx (third derivative over y, x and x). How many lobes are there in Dxy and Dyxx respectively?**
- 4, 8.
- 4, 6.
- 2, 6.
- 2, 8.
* `[ Option C ]`


---

**Q: Assume Gaussian filter. If we increase sigma, ...**
- ... the output image will be less smooth.
- ... the output image will be reversed.
- ... the output image will be smoother.
- ... the output image doesn't change.
* `[ Option A ]`


---

**Q: Which property is not satisfied by convolution?**
- (f*g)*h = f*(g*h)
- f*g=g*f
- f*g=f*h
- d/dx(f*g) = df/dx*g
* `[ Option C ]`


---

**Q: What is the unique property of the Gaussian filter?**
- It is 2D
- The sum of all entries is 1
- It is the only function that does not introduce artifacts
- The image size remains constant after filtering
* `[ Option C ]`


---

**Q: While using a Gaussian filter, what is the most common reason to limit the filter size?**
- Computational effort (multiplication)
- Memory required (storage)
- No reason, make as large as possible
- Computer dependent
* `[ Option A ]`


---

**Q: Which of the following statements is False?**
- The Gaussian is the only function that does not introduce artifacts.
- All the weights of a 4X4 averaging kernel have a value of 1/16.
- As a rule of thumb, it is recommended to set the extent of the Gaussian filter equal to the variance.
- When applying the moving neighborhood average, one pixel is lost on each side.
* `[ Option C ]`


---

**Q: Which of the following statements about edge detection is True?**
- When applying the second derivative of a gaussian filter for edge detection (the filter second derivative is convoluted with the image), each maximum and minimum are considered to be edges.
- When applying the first derivative to a gaussian filter for edge detection (the filter first derivative is convoluted with the image), the maxima and minima are considered to be edges.
- When applying the first derivative to a gaussian filter for edge detection (the filter first derivative is convoluted with the image), only the maxima are considered to be edges.
- When applying the first derivative to a gaussian filter for edge detection (the filter first derivative is convoluted with the image), only the minima are considered to be edges.
* `[ Option B ]`


---

**Q: What is the main purpose of normalizing kernels?**
- To minimize computing power
- To avoid a change in intensity in the resulting image
- To fulfill the necessary condition of the cross-correlation/convolution operation
- To improve the contrast in the resulting image
* `[ Option B ]`


---

**Q: In which direction does the following kernel shift the image considering that the convolution operation is used?
\\\\
\begin{bmatrix}
    0 & 0 & 1 \\
    0 & 0 & 0 \\
    0 & 0 & 0
\end{bmatrix}**
- Bottom left
- Bottom right
- Top left
- Top right
* `[ Option D ]`


---

**Q: Which of the following is a correct 5x5 Averaging kernel?**
- Coefficient: $1/9$ and kernel: 
\[
 \begin{matrix}
  1 & 1 & 1 & 1 & 1 \\
  1 & 1 & 1 & 1 & 1 \\
  1 & 1 & 1 & 1 & 1 \\
  1 & 1 & 1 & 1 & 1 \\
  1 & 1 & 1 & 1 & 1 
 \end{matrix}
\]
- Coefficient: $1/25$
\[
 \begin{matrix}
  1 & 1 & 1 & 1 & 1 \\
  1 & 1 & 1 & 1 & 1 \\
  1 & 1 & 1 & 1 & 1 \\
  1 & 1 & 1 & 1 & 1 \\
  1 & 1 & 1 & 1 & 1 
 \end{matrix}
\]
- Coefficient: $1/25$
\[
 \begin{matrix}
  0 & 1 & 1 & 1 & 0 \\
  1 & 0 & 1 & 0 & 1 \\
  1 & 1 & 1 & 1 & 1 \\
  1 & 0 & 1 & 0 & 1 \\
  0 & 1 & 1 & 1 & 0 
 \end{matrix}
\]
- None of the above
* `[ Option B ]`


---

**Q: Provided you have the following correlation filter \[
 \begin{matrix}
  0 & 0 & 0 & 0 & 0 \\
  0 & 0 & 0 & 0 & 0 \\
  1 & 0& 0 & 0 & 0 \\
  0 & 0 & 0 & 0 & 0 \\
  0 & 0 & 0 & 0 & 0 
 \end{matrix}
\] What do you expect the outcome of the picture to be? **
- All pixels are shifted to the right by two positions
- All pixels are shifted to the left by two positions
- Same as the original image
- None of the above
* `[ Option A ]`


---

**Q: Which image information is used by edge detection algorithms?**
- magnitude of Image Gradient
- First Derivative of Gaussian
- Second Derivative of Gaussian
- Orientation of image gradient
* `[ Option C ]`


---

**Q: Which of the following properties does not belong to a generic convolution operation?**
- Linearity 
- Commutation
- symmetry
- Differentiation
* `[ Option C ]`


---

**Q: Is it necessary for the sum of the filter weights to sum to 1?**
- Yes, to maintain uniformity in output
- Yes, to maintain generalizability across all kernel types
- Yes, to preserve the mean of the filtered image
- No
* `[ Option D ]`


---

**Q: Assume a filter size of 3x3 applied on a 10x10 image.  What is the output of the final image?**
- 10 x 10
- 11 x 11
- 12 x 12
- 13 x 13
* `[ Option C ]`


---

**Q: What is one of the downsides of applying a gaussian filter with a large extent to a picture? **
-  It is very computational expensive
- The resulting picture will get smaller
- The resulting picture will shift to the left
- The resulting picture will shift to the right
* `[ Option B ]`


---

**Q: Given is the gradient $\nabla f = \left[ \frac { \partial f } { \partial x } , \frac { \partial f } { \partial y } \right]$. How can the gradient magnitude be computed?**
- $\sqrt { \left( \frac { \partial f } { \partial x } \right) ^ { 2 } + \left( \frac { \partial f } { \partial y } \right) ^ { 2 } }$
- $\sqrt { \left( \frac { \partial f } { \partial x } \right) + \left( \frac { \partial f } { \partial y } \right) }$
-  $\left( \frac { \partial f } { \partial x } \right) ^ { 2 } + \left( \frac { \partial f } { \partial y } \right) ^ { 2 }$
- $\left( \frac { \partial f } { \partial x } \right) + \left( \frac { \partial f } { \partial y } \right)$
* `[ Option A ]`


---

**Q: Why is it important to convolve an image with a Gaussian filter to be able to find edges on an image?**
- The filter makes the image edges to be more intense 
- The filter smooths the image, making the intensity signal less noisy
- The filter makes the function differentiable everywhere
- It is not important
* `[ Option B ]`


---

**Q: Why do the sum of the values of the filter have to sum to 1?**
- Because the resulting image does not change the intensity
- To not change the quality of the image
- To not change the edges of the resulting image
- It is not important
* `[ Option A ]`


---

**Q: what is the purpose of a filter kernel**
- to apply a filter
- to provide the weights for averaging the pixels of the filter
- to average the neighbouring pixels
-  none of the above
* `[ Option B ]`


---

**Q: How can you, most easily, represent the edges in an image as a function**
-  by looking at the gradient
- by looking at the primitive
- by training a neural network to detect them
- none of the above
* `[ Option A ]`


---

**Q: In which scenario is it most likely that 'object intra-class variation' is an important challenge to take into account?**
- Detecting if an image contains a football in televised footage of a football match.
- Detecting if an image contains a deer in the woods.
- Detecting if an image contains characters written in times new roman.
- Detecting if an image contains food.
* `[ Option D ]`


---

**Q: Which filter is known to be the only filter that does not introduce artifacts?**
- Mean filter
- Gaussian filter
- Box filter
- Sobel filter
* `[ Option B ]`


---

**Q: Which factor determines the extent of smoothing when using Gaussian filter.**
- Number of neighboring pixels
- Variance of Gaussian
- Intensity of pixels
- Shape of image
* `[ Option B ]`


---

**Q: Which one is not true when using first order derivative edge detection **
- Sensitive to noise
- Thicker edges
- Sharper edges
- Maximum detection
* `[ Option C ]`


---

**Q: $ \triangledown f= [3,\sqrt{3}]  $  represents the gradient of an image based on $ \triangledown f= [\frac{\partial f}{\partial x},\frac{\partial f}{\partial y}]  $ . Give the angle of gradient in the image**
- 0 \degree
- 45 \degree
- 30 \degree
- 60 \degree
* `[ Option D ]`


---

**Q: Which of the given kernels is most suitable for detecting a horizontal line?**
- \begin{bmatrix}-1 & -1 & -1 \\ 2 & 2 & 2 \\ -1 & -1 & -1 \end{bmatrix}
- \begin{bmatrix} 2 & -1 & 2 \\ 2 & -1 & 2 \\ 2 & -1 & 2 \end{bmatrix}
- \begin{bmatrix} 2 & 2 & 2  \\ -1 & -1 & -1 \\  2 & 2 & 2  \end{bmatrix}
- \begin{bmatrix}-1 & 2 & -1 \\ -1 & 2 & -1 \\ -1 & 2 & -1 \end{bmatrix}
* `[ Option A ]`


---

**Q: Why is the concept of "vision" hard to implement on machines?**
- Image quality is not high enough
- There is not enough software support
- Convoluted information can be difficult to idenitify
- Pattern matching requires an immense amount of training data
* `[ Option C ]`


---

**Q: What is an appropriate extent of a kernel for a gaussian filter?**
- It should be set to 3 times the value of $\sigma$
- Approximatey to the point where 80% of the main lobe dissipates
- Equal to a quarter of the image size
- As big as possible
* `[ Option A ]`


---

**Q: How to reduce the impact of noise when we calculate the derivatives?**
- Take 2nd order derivatives
- Do smooth first
- Multiply a constant to the signal
- Put a threshold on the derivatives
* `[ Option B ]`


---

**Q: What is the effect of an average kernel to an image?**
- Edge detection
- Blur
- Histogram enhance
- Shift
* `[ Option B ]`


---

**Q: About the Gaussian filter which one is correct?**
- Gaussian filter have infinite support
- Gaussian filter have finite kernel
- Gaussian filter don't have parameter of variance
- Gaussian filter can not smooth the images
* `[ Option A ]`


---

**Q: In the topic of scale normalisation which one is correct?**
- The scale normalisation is the derivative of gaussian filters.
- It is scale variant.
- It is not a laplacian function.
- The derivative of gaussian filter is not a step response.
* `[ Option A ]`


---

**Q: Which of the following linear filter would shift the pixels in the image to top right?**
- All elements 0 except the top right element which should be one
- All elements 0 except the top left element which should be one
- All elements 0 except the bottom left element which should be one
- All elements 0 except the bottom right element which should be one
* `[ Option C ]`


---

**Q: How would you detect an edge in a given image?**
- Find the derivative of the intensities. The edge is where the derivative is higher.
- Find the pixels which have intensities above a certain threshold. The edge is all those pixels.
- Smoothen the image, and find the derivative of the intensities. The edge is where the derivative is higher.
- None of the above
* `[ Option C ]`


---

**Q: What is the key difference between correlation and convolution filtering? 
For convolution,**
- the filter has to be flipped.
- the image has to be flipped.
- the filter or the image has to be flipped.
- the filter and the image have to be flipped.
* `[ Option C ]`


---

**Q: Three of the four following statements are true. Which one is not?
One can detect edges by:**
- looking for sudden intensity changes in an image.
- looking at the largest edge strength values.
- looking at the extremes of the gradient of an image (after smoothing).
- looking at the extremes of the laplacian of an image (after smoothing).
* `[ Option D ]`


---

**Q: Which of the following is not a property of convolution?**
- Linear
- Commutative
- Associative
- None of the above
* `[ Option D ]`


---

**Q: Which of the following filters will result in the same output for both cross-correlation and convolution with any image?**
- Gaussian
- Box
- Laplacian
- All of the above
* `[ Option D ]`


---

**Q: A low pass filter is used for**
- Increasing contrast
- Blurring the image
- Sharpening the image
- Resizing the image
* `[ Option B ]`


---

**Q: Increasing the variance while keeping the kernel constant makes the image**
- More smooth
- Less smooth
- No change in smoothness
- Not enough information
* `[ Option A ]`


---

**Q: Which of the following are a property of convolution**
- Identity
- Associative
- Differentation
- All of them
* `[ Option D ]`


---

**Q: What is the general formula for a gaussian filter?**
- h(u, v) = 1/(2*pi*sigma^2)  *  e^(-(u^2+v^2)/(sigma^2))
- h(u, v) = 1/(2*pi*sigma^2)  *  e^((u^2+v^2)/(sigma^2))
- h(u, v) = 1/(sqrt(2*pi*sigma^2))  *  e^(-(u^2+v^2)/(sigma^2))
- h(u, v) = 1/(sqrt(2*pi*sigma^2))  *  e^((u^2+v^2)/(sigma^2))
* `[ Option A ]`


---

**Q: What determine extent of smoothing for Gaussian filters?**
- u, v in the kernel function.
- Nearest neighboring pixels.
- Variance of Gaussian.
- All of above.
* `[ Option C ]`


---

**Q: Which statement about convolution and cross-correlation is true?**
- Convolution is not a symmetrical operator.
- Cross-correlation is a symmetrical operator.
- Two operators can be same for all filters in some conditions.
- Convolution is linear invariant.
* `[ Option D ]`


---

**Q: Assume the variance of a Gaussian filter in both directions (x and y) are the same is 9, what is the size of the kernel?**
- 9*9
- 3*3
- 27*27
- 18*18
* `[ Option D ]`


---

**Q: Which of the following description is wrong?**
- With image filter processing, a 20*20 (pixel) image could become 16*16, losing 4 layers of boundary.
- With image filter processing, there must be some pixels value changed.
- Discrete filters cannot use infinite kernels.
- Take derivative of Gaussian, the gradient in x and y direction could be different.
* `[ Option B ]`


---

**Q: What is the difference between convolution and cross correlation?**
- Convolution would rotate one matrix 180 by degrees, but cross correlation doesn't
- They are the same in image processing
- Only one dimension of the original image matrix is flipped
- None of the above
* `[ Option A ]`


---

**Q: What's the disadvantage of moving neighborhood average?**
- It's computationally expensive
- It
- In every dimension, 2 pixels lost to boundary
- None of the above
* `[ Option C ]`


---

**Q: Which statement is wrong?**
- Image processing produces information about an image  while computer vision typically produces higher-level information about the content of the image
- The convolution is both commutative and associative.
- A convolution operation is a cross-correlation where the filter is flipped only horizontally.
- Convolution defines how much the signals overlap, whereas correlation tries to find the relation between the signals.
* `[ Option C ]`


---

**Q: Which statement is wrong?**
-  It is impossible to blur an image using a linear filter.
- Laplacian-of-Gaussian edge detector  is 2nd derivative and requires ╧â defining the scale of the Gaussian blurring.
- Variance of Gaussian: determines extent of smoothing
- The derivative of a Gaussian  is used to compute the derivative of the image. It is smoothing with respect to the orthogonal direction, hence it can be used to detect edges. 
* `[ Option A ]`


---

**Q: The variance of a Gaussian filter determines extent of smoothing. What is the rule of thumb for implementing such a filter.**
-  set extent to 3 sigma
- set extend to 1/3 sigma
- set extend to 1
- set extend as large as possible
* `[ Option A ]`


---

**Q: Imagine applying the following kernel to an image. [0,0,0 ; 0,0,1 ; 0,0,0]. What will happen to the image?**
- The image will be shifted to the left by one pixel
- Nothing, the image will stay the same
- The image will be shifted to the right by one pixel
- Not enough information is provided to answer the question
* `[ Option A ]`


---

**Q: Professor F. Ilter has an image $F$ and two distinct kernels $H_1$ and $H_2$. He claims that convolving $F$ with $H_1$ and cross correlating $F$ with $H_2$ has the same result as output. Is this possible?**
- Yes, but $H_1$ and $H_2$ must be gaussian kernels.
- Yes, this means that $H_1$ is equal to $H_2$ flipped in both directions.
- No, this is not possible. The kernels must be exactly the same
- No, this is not possible even when the kernels exactly match.
* `[ Option B ]`


---

**Q: Professor F. Ilter now wants to shift an image $F$ 1 pixel to the right and 1 pixel down using convolution. He's using a single 3x3 kernel $H$ which contains only one 1 in its values, the rest are zeroes. Where is the 1 positioned in the kernel $H$?**
- Top-left corner
- Top-right corner
- Bottom-left corner
- Bottom-right corner
* `[ Option D ]`


---

**Q: What is the difference between a correlation filter and a convolutional filter?**
- Correlation filters are scale invariant compared to convolutional filters.
- They are operations that for the same filter values result in the same end result.
- The correlation filter has better performance than the convolutional filter.
- The convolutional and correlation filter can produce identical results, but not in all cases.
* `[ Option D ]`


---

**Q: The guassian filter is used in Computer Vision because:**
- Due to linearity it has very good performance.
- It causes a uniform distribution in the filter.
- Its derivative has nice properties.
- The filter has trapezoidal constraints for optimization.
* `[ Option C ]`


---

**Q: When performing edge detection, which of the following statements is NOT true:**
- Smoothing can help reduce the unwanted effects of noise
- When convolving with the 2nd derivative of the Gaussian (Laplacian), the edges are found at the peaks of the convoluted function
- The edge strength is given by the gradient magnitude
- Distinguishing objects from the background can be a problem due to edge overlap or 'shallow' edges
* `[ Option B ]`


---

**Q: What is an advantage of using a Gaussian function to filter an image, in comparison with other functions?**
- The Gaussian transforms the image from a discrete image to a continuous image.
- The rule of thumb (set extent to 3*sigma) always gives the optimal solution
- The Gaussian is the only function that doesnΓÇÖt introduce artefacts
- A Gaussian filter will not decrease the image size at the borders
* `[ Option C ]`


---

**Q: Using the following kernel for cross correlation, what will happen to a picture? 

                               1 1 1
 Kernel:  H(u,v)=   1 1 1
                               1 1 1**
- The picture will be averaged with the pixels around a given pixel.
- The picture will be averaged with pixels around a pixel and it will get lighter (higher values)
- The picture will be averaged with pixels around a pixel and it will get darker (lower values)
- The picture will stay the same
* `[ Option B ]`


---

**Q: Which of the following statements is true:
I: Convolution is the flipped version of cross correlation
II: Cross correlation will always lose pixels on the edge of the picture.**
- Only I is true.
- Only II is true.
- Both I and II are true.
- None of the statements are true
* `[ Option A ]`


---

**Q: What is the difference between convolution and cross-correlation?**
- Cross-correlation gives more accurate results
- Convolution flips the kernel in both dimensions
- Convolution uses a bigger kernel size
- Cross-correlation does not need padding for image borders
* `[ Option B ]`


---

**Q: What kernel do we need to get the orientation of an edge in an image?**
- The first derivative of the gaussian in one direction
- The first derivative of the gaussian for both x- and y-direction
- The second derivative of the gaussian in one direction
- The second derivative of the gaussian for both x- and y-direction
* `[ Option B ]`


---

**Q: When is applying cross correlation equivalent to applying convolution between the image and the filter? **
- Only if the filter is Gaussian.
- If the filter is symmetric.
- If the filter is not symmetric.
- None of the above.
* `[ Option B ]`


---

**Q: Which of the following statements is true about the Gaussian filter.**
- If sigma increases then the kernel size increases too.
- If sigma decreases then the smoothness of the filtered image increases.
- If a mask is size of 3*sigma then at the mask's edges the weights of the Gaussian tend to zero.
- Both c and b are correct.
* `[ Option C ]`


---

**Q: How does the value of sigma effect the blurring in Gaussian filter assuming filter size is kept same?**
- Blurring effect increases with increased sigma value
- Blurring effect increases with decreased sigma value
- Blurring effect decreases with increased sigma value
- No effect
* `[ Option A ]`


---

**Q: If a 7x7 averaging filter is applied twice to a 512x512 image, what will be the dimensions of the output image?**
- 500 x 500
- 502 x 502
- 504 x 504
- 506 x 506
* `[ Option A ]`


---

**Q: How much does the rule of thumb tell you to set the extent of Gaussian filters?**
- 2
- 3
- 6
- What you think fits best
* `[ Option B ]`


---

**Q: Which statements are true? 

I. Convolving the impulse signal with a kernel H gives H.

II. Images can not be seen as functions.**
- I
- II
- Both
- Neither
* `[ Option A ]`


---

**Q: 1**
- a.
- b.
- c.
- d.
* `[ Option A ]`


---

**Q: 2**
- a.
- b.
- c.
- d.
* `[ Option A ]`


---

**Q: Which statement is false?**
- The Gaussian is the only function that does not introduce artifacts.
- Gaussian function has infinite support, but discrete filters use finite kernels.
- Variance of a Gaussian determines the extent of the smoothing.
- As a rule of thumb, the extent of the Gaussian should be set to 2 sigma.
* `[ Option D ]`


---

**Q: When an image is convolved with a 3x3 kernel, in which the kernel consists of  zero's except the middle left position is 1. What can be said abount the resulting image?**
- It is shifted to the right by 1 pixel.
- It is shifted to the left by 1 pixel.
- It is shifted to the top by 1 pixel.
- It is shifted to the bottom by 1 pixel.
* `[ Option A ]`


---

**Q: What is the true statement regarding the magnitude of a gradient?**
- It points in the direction of the most rapidly change in intensity
- It gives the direction of the gradient
- It gives the edge strength
- It is a vector of partial derivatives
* `[ Option C ]`


---

**Q: What is one correct solution to keep the original image size when applying an averaging kernel which is smaller in size compared to the original image?**
- Increase the size of the kernel
- Apply zero-padding to the resulted image
- Apply zero-padding to the original image
- Increase the values of the kernel
* `[ Option C ]`


---

**Q: which of the following statements is correct about image convolution**
- You can get convolution filter by rotating correlation filter 90 degree 
- Convolution filter provides the same result/image as correlation does
- Convolution holds associative, commutative like correlation
- To some extend, convolution can reduce computation complexity when a lot of filters are about to be applied on a image.
* `[ Option D ]`


---

**Q: which of the following statements is wrong about edge dection**
- Edge is a place of rapid change in the image intensity function.
- First and second derivative both cannot provide sound solution to edge dection problems without processing image first.
- Zero crossing of first derivative means edge.
- Different gradient direction will lead to different detected edges.
* `[ Option C ]`


---

**Q: Which of the following is true:
1. Gaussian functions can be used as infinite kernels for discrete filters
2. Gaussian functions give most influence to pixels closest to the current pixel**
- 1
- 2
- both
- neither
* `[ Option B ]`


---

**Q: Convolutions:**
- are Commutative and associative
- are Sensitive for shifts
- Flip the image in both dimensions
- Can not be applied after being diffentiated
* `[ Option A ]`


---

**Q: Consider a noisy image: How can an edge in x direction be detected the easiest?**
- By inspection of the derivative w.r.t. x.
- By inspecting the convolution of a gaussian with the derivative w.r.t. x.
- By inspecting the convolution of a gaussian with the image.
- By inspection of the inverse of the derivative w.r.t. x.
* `[ Option B ]`


---

**Q: What is specified by the following term in a 2D image (f denotes intensity): arctan(\frac{\frac{df}{dy}}{ \frac{df}{dx}})**
- The edge strength
- The noise
- The edge direction, if an edge is at the inspected point
- Image stability
* `[ Option C ]`


---

**Q: Which of the following statements is False?**
- The Gaussian is the only function that does not introduce artifacts.
- All the weights of a 4X4 averaging kernel have a value of 1/16.
- As a rule of thumb, it is recommended to set the extent of the Gaussian filter equal to the variance.
- When applying the moving neighborhood average, one pixel is lost on each side.
* `[ Option C ]`


---

**Q: Which of the following statements about edge detection is True?**
- When applying the second derivative of a gaussian filter for edge detection (the filter second derivative is convoluted with the image), each maximum and minimum are considered to be edges.
- When applying the first derivative to a gaussian filter for edge detection (the filter first derivative is convoluted with the image), the maxima and minima are considered to be edges.
- When applying the first derivative to a gaussian filter for edge detection (the filter first derivative is convoluted with the image), only the maxima are considered to be edges.
- When applying the first derivative to a gaussian filter for edge detection (the filter first derivative is convoluted with the image), only the minima are considered to be edges.
* `[ Option B ]`


---

