# Papers ArXiv :notebook_with_decorative_cover:

`Summaries of all the papers I read` :pencil2:

These notes are best viewed with MathJax [extension](https://chrome.google.com/webstore/detail/github-with-mathjax/ioemnmodlmafdkllaclgeombjnmnbima) in chrome.

---
`Nov 25, 2018`
#### Pixel Recurrent Neural Networks
- [Link](https://arxiv.org/abs/1601.06759)
- The aim of this paper is to estimate the distribution of images to compute the likelihood and generate new ones.
- Important obstacle in Generative Modeling is building something that is `tractable` (easy to understand) and `scalable`. 
- _Previous works_:
    - Techniques like AutoEncoders make use of Latent Variables (for dimensioanlity reduction) but are not tractable.
    - Tractable models involve modeling using products of conditional distributions, but are not sophisticated enough to model long-range correlation between pixels.
    - Enter, RNNs! They (2d-RNNs) have been awesome at modelling gray-scale images and textures before this paper.
- _Contributions_:


