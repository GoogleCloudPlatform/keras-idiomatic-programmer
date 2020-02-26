[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![License](https://i.creativecommons.org/l/by/4.0/80x15.png)](LICENSE)



This repository is origanized as follows:

|File       | Description|
|-----------|------------|
| `Primer` | Section 1 (chapters I, II and II) made freely available in PDF|
| [`Workshops`](Workshops) |Workshop content bundles for each chapter in the book (FREE)|

### About the book.

*What is the technology or idea that you’re writing about?*

Coding computer vision models using design patterns that are AutoML friendly -- bring your own model to automl and guide the search space.

Teaches junior to advanced techniques for coding models in Tensorflow 2.0 with Keras Functional API.

Readers will learn concepts of design patterns (object oriented programming) and how design patterns speed up model design and understandability to users.

Starting with section 2 (Intermediate) the composable design pattern will be used to demonstrate Automatic Learning concepts and how using the design pattern one can incorporate automatic learning concepts at a level higher of abstraction and guide the search space.

*Why is it important now?*

 I find as my role at Google interfacing with Enterprise clients, that decision makers are looking to move their data science/ML teams to automatic learning, such that their talents/skills can be redirected to more challenging tasks.

To make that transition, they need automatic learning to be demystified and not a black box, where ML practitioners and ML ops can guide the search space and effectively evaluate the models for production-scale use.


## Topics

### Preface
	The Machine Learning Steps
	Classical vs. Narrow AI

### Novice (Primer)
	Chapter I - Deep Neural Networks
		I.I. Input Layer
		I.II. Deep Neural Networks (DNN)
		I.III. Feed Forward
		I.IV. DNN Binary Classifier
		I.V. DNN Multi-Class Classifier
		I.VI. DNN Multi-Label Multi-Class Classifier
		I.VII. Simple Image Classifier
	Chapter II - Convolutional and ResNet Neural Networks
		II.I Convolutional Neural Networks
		II.II. CNN Classifier
		II.III. Basic CNN
		II.IV. VGG
		II.V. Residual Networks (ResNet)
		II.VI. ResNet50
	Chapter III -  Training Foundation
		III.I. Feed Forward and Backward Propagation
		III.II. Dataset Splitting
		III.III. Normalization
		III.IV. Validation & Overfitting
		III.V. Convergence
		III.VI.  Checkpointing & Earlystopping
		III.V. Hyperparameters
		III.VI. Invariance
		III.VII. Raw (Disk) Datasets
		III.VIII. Model Save/Restore

### Junior
	Chapter 1 - Introduction
		1.1 How I Teach
		1.2. What You Will Take Away
		1.3. The Machine Learning Steps
	Chapter 2  - Procedural Design Pattern
		2.1. Procedural Design Pattern
		2.2. Stem Component
		2.3. Learner Component
		2.4. Classifier Component
 	Chapter 3 -  Wide Convolutional Neural Networks
		3.1. Inception V1
 		3.2. Inception V2
		2.3. Inception V3
		3.4. ResNeXt
		3.5. Wide Residual Network (WRN)
	Chapter 4 - Alternative ConnectivityPatterns
		4.1. Densely Connected CNN (DenseNet)
 		4.2. SE-Net
		4.3. Xception
	Chapter 5 - Mobile Convolutional Neural Networks
		5.1. MobileNet
		5.2 SqueezeNet
		5.3. ShuffleNet
		5.4. Quantization
	Chapter 5 - AutoEncoders
		6.1. AutoEncoder
		6.2. Convolutional AutoEncoder
		6.3. Sparse AutoEncoder
		6.4. Denoising AutoEncoder
		6.5. Pre-text Tasks
	Chapter 7 - Hyperparameter Tuning
		7.1. Warmup (Numerical Stability)
		7.2. Grid Search
		7.3. Random Search
		7.4. Learning Rate Scheduler
		7.5. Regularization
        Chapter 8 - Transfer Learning
		8.1. Overview
		8.2. Fine-Tuning
		8.3. Full-Tuning
	Chapter 9 - Training Pipeline
		9.1. Data Formats & Storage
 		9.2. Dataset Curation
		9.3. Data Preprocessing
		9.4. Label Smoothing
 		9.5. Model Feeding
		9.6. Training Schedulers
		9.7. Model Evaluations
	Chapter 10 - Data Augmentation
		10.1. Crop / Flip / Rotate
		10.2. Augmentation Pipeline


### Intermediate

	Chapter 11  - AutoML by Design (Patterns)
		11.1. Metaparameters
		11.2. Embeddings
		11.3. Macro-Architecture
		11.4. Model Configuration
		11.5. Dynamic Regularization
	Chapter 12 - Multi-Task Models
		12.1. Object Detection - (Region Based)
		12.2. Object Detection - (Pyramid Based)
		12.3. Object Detection - (Single Shot)
		12.4. Generative Adversarial Networks (GAN)
		12.5. Style Transfer
	Chapter 13 - Network Architecture Search
		13.1. NasNet
		13.2. MNasNet
		13.3. MobileNet V3
	Chapter 14 - Automatic Hyperparameter Search
		14.1. Bayesian Optimization
		14.2. Data Augmentation Strategies
	Chapter 15 - Production Foundation (Training at Scale)
		15.1. Data Schema
		15.2. Data Validation
		15.3. Model Versioning
		15.4. Model Deployment
		15.5. A/B Testing
		15.6. Continuous Evaluation
		15.7. Distribution Skew / Data Drift

### Advanced

	Chapter 16  - Model Amalgamation
		16.1. Inter-Model Connectivity
		16.2. Factory Pattern
		16.3. Abstract Factory Pattern
		16.4. Intelligent Automation
	Chapter 17 - Automatic Macro-Architecture Search
		17.1. Dynamic Group/Block configuration
		17.2. Guiding the Search Space
	Chapter 18 - Knowledge Distillation (Student/Teacher)
		18.1. Student/Teacher Networks
		18.2. Label Distillation
		18.3. Weight Transfusion
	Chapter 19 - Semi/Weakly Supervised Learning
		19.1. Pseudo Labeling / Noisy Data
		19.2. Synthetic Data
		19.3. Optical Flow
	Chapter 20 - Self Supervised Learning
		20.1. Pre-text Training (Learn Essential Features)
		20.2. Adversarial Training (Robustness)
		20.3. Data Weighting (Reinforcement Learning)
