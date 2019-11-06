# Notebooks

These notebooks address *goto production (GTP)* questions that I receive as a member of Google AI Developer Relations. 

| Notebook                              | Description   |
| ------------------------------------- | ------------- |
| `Prediction with Example Key`         | Adding Unique Identifier to Predictions in asynchronous distributed batch prediction |
| `Pre-Stem Deconvolution`              | Using deconvolution (transpose) to learn optimal transformations for different input sizes to existing model |
| `Building Data preprocessing into Graph` | Using TF 2.0 Subclassing and @tf.function decorator to put the data preprocessing as part of the model graph |
| `Estimating the CPU/GPU utilization for training` | Using pre-warmpup methods to estimate utilization across compute resources, for the purpose of planning the optimal utilization prior to full training. |
| `Fun with ML in Production`            | A composition of fun things (tricks) that one may try in ML production environment |
| `community-labs`                      | notebooks for community-based research experiments` |

### Prediction with Example Key

#### Problem

You have an existing model that does prediction which is put into production. Due to the load, the serving side uses a distributed (load balanced) batch prediction. 

How does one identity which prediction goes with which request when the predictions are returned "real-time" asynchronous, where you don't know the order that they will be returned in?

#### Solution

The solution is very trival. We simply create a wrapper model around the existing pretrained model using tf.keras multiple inputs/outputs functionality in the Functional API to add an identity link which passes a unique identifier per request to the output layer; whereby, each prediction returns the prediction result and the unique identifier.

### Pre-Stem Deconvolution

#### Problem

You have an existing model architecture optimized for an input shape which is put into production. The model is repurposed to take inputs of a substantially smaller image size. 

How does one use a small input size on a model designed for substantially larger input size and maintain comparable performance?

#### Solution

The solution is very trival. We simply create an additional group that is added to the input layer of the existing model, which consists of deconvolution layers to learn the optimal method to upsample the images to match the input shape of the input layer of the existing model.

### Building Data Preprocessing into the Graph

#### Problem

TF 2.0 has a ton of new rich features and power. Before TF 2.0, preprocessing of your data was done upstream on the CPU, which could starve the GPU's from running at full throttle. Think of this upstream preprocessing as the fuel, and if constrained, the GPU is choking waiting for fuel.

#### Solution

In this notebook, we show you how easy it is to do this, by creating a new input layer using subclassing of tf.keras.layers, and the @tf.function decorator to convert the Python code for preprocessing into TF graph operations. Another benefit to this, is that if you later want to change the preprocessing w/o re-training the model, you can simple replace the new preprocessing layer in your existing model, since it has no trainable weights.

### Estimating your CPU/GPU utilization for Training

#### Problem

Currently training infrastructure does not do auto-scaling (unlike batch prediction). Instead, you sent your utilization strategy as part of starting your training job.

If your training on the cloud, a poor utilization may result in an under or over utilization. In under utilization, you're leaving compute power (money) on the table. In over utilization, the training job may become bottleneck or excessively interrupted by other processes.

Things you might consider when under utilizing. Do I scale up (larger instances) or do I scale out (distributed training).

#### Solution

In this notebook, we use short training runs (warm-start) combined with the `psutil` module to see what our utilization will be when we do a full training run. Since we are only interested in utilization, we don't care what the accuracy is --we can just use a defacto (best guess) on hyperparameters.

In my experience, I find the sweetspot for utilization on a single instance is 70%. That leaves enough compute power from background processes pre-empting the training and if training headless, to be able to ssh in and monitor the system.

### Fun with ML in Production

#### Topics

This notebook demonstrates a compilation of techniques one might consider when training in a modern production environment. Covered are:

  * Model Aggregation & Model Cutout
  * Adding Layers to Trained Models
  * Auxilliary Classifiers (why not to use them)
