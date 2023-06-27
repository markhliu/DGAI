# Demystify Generative AI

“Technology advanced enough is indistinguishable from magic.”

--Arthur C. Clarke (author of 2001: A Space Odyssey)


## A simple introduction to generative AI for people without computer science background.

A 16-chapter series to introduce generative AI on how to generate images, text, music, figures, patterns, and more. The series show how to:

* Create a ChatGPT-style large language model from scratch to generate text that can pass as human-written.
* Generate images that are indistinguishable from real photos
* Compose music that anyone would think it’s real
* Create patterns such as a sequence of odd numbers, or numbers that are multiples of five
* Generate data that mimic certain shapes: sine curve, cosine shape, hyperbola graph
* Control the latent space to generate images with certain attributes: men with glasses, women with glasses, transitioning gradually from men with glasses to men without glasses, or from women without glases to women with glasses...
* Style transfer: convert an horse image to a zebra... 

## Chapter 10: Attribute-Control in Variational Autoencoders
Train a variational autoencoder (VAE) to generate color images of human faces. Control encodings to generate images with certain attributes: e.g., images that gradually transition from images with glasses to images without glasses. Take the encodings of men with glasses, minus encodings of men without glasses, and add in the encodings of women without glasses, you'll generate images of women with glasses. The whole experience seems like straight out of science fiction, hence the opening quote by the science fiction writer Arthur Clarke: “Technology advanced enough is indistinguishable from magic.” 

To give you an idea what the chapter will accomplish, here is the transition from women with glasses to women without glasses:
<img src="https://gattonweb.uky.edu/faculty/lium/ml/wgwng6.png" />
Transition from women without glasses to men without glasses
<img src="https://gattonweb.uky.edu/faculty/lium/ml/wngmng6.png" />
Two examples of encoding arithmetic:
<img src="https://gattonweb.uky.edu/faculty/lium/ml/vectorArithmetic1.png" />

<img src="https://gattonweb.uky.edu/faculty/lium/ml/vectorArithmetic2.png" />

