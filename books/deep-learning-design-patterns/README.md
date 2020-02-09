[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![License](https://i.creativecommons.org/l/by/4.0/80x15.png)](LICENSE)



This repository is origanized as follows:

|File       | Description|
|-----------|------------|
| `Primer` | Section 1 (chapters I, II and II) made freely available in PDF|
| [`Workshops`](workshops) |Workshop content bundles for each chapter in the book (FREE)|

### About the book.

*What is the technology or idea that you’re writing about?*

Coding computer vision models using design patterns that are AutoML friendly -- bring your own model to automl and guide the search space.

Teaches junior to advanced techniques for coding models in Tensorflow 2.0 with Keras Functional API.

Readers will learn concepts of design patterns (object oriented programming) and how design patterns speed up model design and understandability to users.

Starting with section 2 (Intermediate) the composable design pattern will be used to demonstrate Automatic Learning concepts and how using the design pattern one can incorporate automatic learning concepts at a level higher of abstraction and guide the search space.

*Why is it important now?*

 I find as my role at Google interfacing with Enterprise clients, that decision makers are looking to move their data science/ML teams to automatic learning, such that their talents/skills can be redirected to more challenging tasks.

To make that transition, they need automatic learning to be demystified and not a black box, where ML practitioners and ML ops can guide the search space and effectivelyevaluate the models for production-scale use.


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
	Chapter 1  - Models by Design (Patterns)
		1.1. Procedural Design Pattern
		1.2. Stem Component
		1.3. Learner Component
		1.4. Classifier Component
 	Chapter 2 -  Wide Convolutional Neural Networks
		2.1. Inception V1
 		2.2. Inception V2
		2.3. Inception V3
		2.4. ResNeXt
		2.5. Wide Residual Network (WRN)
	Chapter 3 - Alternative ConnectivityPatterns
		3.1. Densely Connected CNN (DenseNet)
 		3.2. SE-Net
		3.3. Xception
	Chapter 4 - Mobile Convolutional Neural Networks
		4.1. MobileNet
		4.2 SqueezeNet
		4.3. ShuffleNet
	Chapter 5 - AutoEncoders
		5.1. <TODO>
	Chapter 6 - Hyperparameter Tuning
		6.1. Warmup (Numerical Stability)
		6.2. Grid Search
		6.3. Random Search
		6.4. Learning Rate Scheduler
		6.5. Regularization
        Chapter 7 - Transfer Learning
		7.1. Overview
		7.2. Fine-Tuning
		7.3. Full-Tuning
	Chapter 8 - Training Pipeline
		8.1. Data Formats & Storage
 		8.2. Dataset Curation
		8.3. Data Preprocessing
		8.4. Label Smoothing
 		8.5. Model Feeding
		8.6. Training Schedulers
		8.7. Model Evaluations
	Chapter 9 - Data Augmentation
		9.1. Crop / Flip / Rotate
		9.2. Augmentation Pipeline


### Intermediate

	Chapter 10  - AutoML by Design (Patterns)
		10.1. Metaparameters
		10.2. Embeddings
		10.3. Macro-Architecture
		10.4. Model Configuration
		10.5. Dynamic Regularization
	Chapter 11 - Multi-Task Models
		11.1. Object Detection - (Region Based)
		11.2. Object Detection - (Pyramid Based)
		11.3. Object Detection - (Single Shot)
		11.4. Generative Adversarial Networks (GAN)
		11.5. Style Transfer
            Chapter 12 - Automatic Hyperparameter Search
		12.1. Bayesian Optimization
		12.2. Data Augmentation Strategies
	Chapter 13 - Production Foundation (Training at Scale)
		13.1. Data Schema
		13.2. Data Validation
		13.3. Model Versioning
		13.4. Model Deployment
		13.5. A/B Testing
		13.6. Continuous Evaluation
		13.7. Distribution Skew / Data Drift

### Advanced

	Chapter 14  - Model Amalgamation
		14.1. <TODO>
	Chapter 15 - Automatic Macro-Architecture Search
		15.1. Dynamic Group/Block configuration
		15.2. Guiding the Search Space
	Chapter 16 - Knowledge Distillation (Student/Teacher)
		16.1. Student/Teacher Networks
		16.2. Label Distillation
		16.3. Weight Transfusion
	Chapter 17 - Semi/Weakly Supervised Learning
		17.1. Pseudo Labeling / Noisy Data
		17.2. Synthetic Data
		17.3. Optical Flow
	Chapter 18 - Self Supervised Learning
		18.1. Pre-text Training (Learn Essential Features)
		18.2. Adversarial Training (Robustness)
		18.3. Data Weighting (Reinforcement Learning)
