# Demystify Generative AI

“Technology advanced enough is indistinguishable from magic.”

--Arthur C. Clarke (author of 2001: A Space Odyssey)


## text, music, image, figure, and pattern generation in PyTorch

A 16-chapter series to create images, text, music, figures, and patterns in PyTorch. The series show how to:

* Create a ChatGPT-style large language model from scratch to generate text that can pass as human-written
* Generate images that are indistinguishable from real photos
* Compose music that anyone would think it’s real
* Create patterns such as a sequence of odd numbers, multiples of five, ...
* Generate data that mimic certain shapes: sine curve, cosine shape, hyperbola graph
* Control the latent space to generate images with certain attributes: men with glasses, women with glasses, transitioning gradually from men with glasses to men without glasses, or from women without glases to women with glasses...
* Style transfer: convert a horse image to a zebra... 

## Chapter 1: Introduction to PyTorch
## Chapter 2: Deep Learning with PyTorch
## Chapter 3: Generative Adversarial Networks (GANs)
Most of the generative models in this book belong to a framework called Generative Adversarial Networks (GANs). This chapter introduces you to the basic idea behind GANs and you'll learn to use the framework to generate data samples that form an inverted-U shape. At the end of this chapter, you'll be able to generate data to mimic any shape: sine, cosine, quadratic, and so on. 
![invertedU](https://github.com/markhliu/DGAI/assets/50116107/9da4fdab-d852-4f9e-b6bf-a184928d2885)

## Chapter 4: Pattern Generation with GANs
## Chapter 5: Image Generation with GANS
## Chapter 6: High Resolution Image Generation with Deep Convolutional GANs
## Chapter 7: Conditional GAN and Wasserstein GAN
## Chapter 8: CycleGAN
## Chapter 9: Introduction to Variational Autoencoders
## Chapter 10: Attribute-Control in Variational Autoencoders
Train a variational autoencoder (VAE) to generate color images of human faces. Control encodings to generate images with certain attributes: e.g., images that gradually transition from images with glasses to images without glasses. Take the encodings of men with glasses, minus encodings of men without glasses, and add in the encodings of women without glasses, you'll generate images of women with glasses. The whole experience seems like straight out of science fiction, hence the opening quote by the science fiction writer Arthur Clarke: “Technology advanced enough is indistinguishable from magic.” 

To give you an idea what the chapter will accomplish, here is the transition from women with glasses to women without glasses:
<img src="https://gattonweb.uky.edu/faculty/lium/ml/wgwng6.png" />
Transition from women without glasses to men without glasses
<img src="https://gattonweb.uky.edu/faculty/lium/ml/wngmng6.png" />
Two examples of encoding arithmetic:
<img src="https://gattonweb.uky.edu/faculty/lium/ml/vectorArithmetic1.png" />

<img src="https://gattonweb.uky.edu/faculty/lium/ml/vectorArithmetic2.png" />


## Chapter 11: Text Generation with Character-Level LSTM
## Chapter 12: Text Generation with Word-Level LSTM
## Chapter 13: Create A GPT from Scratch
## Chapter 14: Train a ChatGPT style Transformer
## Chapter 15: MuseGAN
Train a generative adversarial network (GAN) to produce music. here is a sample of the generated music:
https://gattonweb.uky.edu/faculty/lium/ml/MuseGAN_song.mp3

## Chapter 16: Music Transformer
Train a ChatGPT-style transformer to generate music. here is a sample of the generated music:
https://gattonweb.uky.edu/faculty/lium/ml/musicTrans.mp3





