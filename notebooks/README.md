# Notebooks

These notebooks address *goto production (GTP)* questions that I receive as a member of Google AI Developer Relations. 

| Notebook                              | Description   |
| ------------------------------------- | ------------- |
| `Prediction with Example Key`         | Adding Unique Identifier to Predictions in asynchronous distributed batch prediction |
| `Pre-Stem Deconvolution`              | Using deconvolution (transpose) to learn optimal transformations for different input sizes to existing model |

### Prediction with Example Key

#### Problem

You have an existing model that does prediction which is put into production. Due to the load, the serving side uses a distributed (load balanced) batch prediction. 

How does one identity which prediction goes with which request when the predictions are returned "real-time" asynchronous, where you don't know the order that they will be returned in?

#### Solution

The solution is very trival. We simple create a wrapper model around the existing pretrained model using tf.keras multiple inputs/outputs functionality in the Functional API to add an identity link which passes a unique identifier per request to the output layer; whereby, each prediction returns the prediction result and the unique identifier.

### Pre-Stem Deconvolution

#### Problem

You have an existing model architecture optimized for an input shape which is put into production. The model is repurposed to take inputs of a substantially smaller image size. 



