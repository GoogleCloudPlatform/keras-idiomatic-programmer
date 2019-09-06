
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
# Double the size of filters to fit the first Residual Group
x = projection_block(64, x, strides=(1,1))

# Identity residual blocks
for _ in range(2):
    x = bottleneck_block(64, x)

# Second Residual Block Group of 128 filters
# Double the size of filters and reduce feature maps by 75% (strides=2, 2) to fit the next Residual Group
x = projection_block(128, x)

# Identity residual blocks
for _ in range(3):
    x = bottleneck_block(128, x)

# Third Residual Block Group of 256 filters
# Double the size of filters and reduce feature maps by 75% (strides=2, 2) to fit the next Residual Group
x = projection_block(256, x)

# Identity residual blocks
for _ in range(5):
    x = bottleneck_block(256, x)

# Fourth Residual Block Group of 512 filters
# Double the size of filters and reduce feature maps by 75% (strides=2, 2) to fit the next Residual Group
x = projection_block(512, x)

# Identity residual blocks
for _ in range(2):
    x = bottleneck_block(512, x)

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
