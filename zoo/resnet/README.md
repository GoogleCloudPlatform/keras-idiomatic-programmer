
# ResNet v1.0

[Paper](https://arxiv.org/pdf/1512.03385.pdf)

## Macro-Architecture

<img src='macro.jpg'>

Macro-architecture for ResNet50:

```python
# The input tensor
inputs = layers.Input(shape=(224, 224, 3))

# The stem convolutional group
x = stem(inputs)

# First Residual Block Group of 64 filters
x = residual_group(64, 2, x, strides=(1, 1))

# Second Residual Block Group of 128 filters
x = residual_group(128, 3, x)

# Third Residual Block Group of 256 filters
x = residual_group(256, 5, x)

# Fourth Residual Block Group of 512 filters
x = residual_group(512, 2, x)

# The classifier for 1000 classes
outputs = classifier(x, 1000)

# Instantiate the Model
model = Model(inputs, outputs)
```

## Micro-Architecture

<img src='micro.jpg'>

### Stemp Group

<img src="stem.jpg">

### ResNet Block with Identity Shortcut

<img src='identity-block.jpg'>

### ResNet Block with Projection Shortcut

<img src='projection-block.jpg'>
