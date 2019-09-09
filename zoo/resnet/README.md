
# ResNet

[Paper](https://arxiv.org/pdf/1512.03385.pdf)

## Macro-Architecture

<img src='macro.jpg'>

Macro-architecture for ResNet50:

```python
def learner(x):
    """ Construct the Learner
        x  : input to the learner
    """
    # First Residual Block Group of 64 filters
    x = residual_group(x, 64, 2, strides=(1, 1))

    # Second Residual Block Group of 128 filters
    x = residual_group(x, 128, 3)

    # Third Residual Block Group of 256 filters
    x = residual_group(x, 256, 5)

    # Fourth Residual Block Group of 512 filters
    x = residual_group(x, 512, 2)
    return x

# The input tensor
inputs = layers.Input(shape=(224, 224, 3))

# The learner
x = learner(x)

# The classifier for 1000 classes
outputs = classifier(x, 1000)

# Instantiate the Model
model = Model(inputs, outputs)
```

## Micro-Architecture

<img src='micro.jpg'>

```python
def residual_group(x, n_filters, n_blocks, strides=(2, 2)):
    """ Construct a Residual Group
        x         : input into the group
        n_filters : number of filters for the group
        n_blocks  : number of residual blocks with identity link
        strides   : whether the projection block is a strided convolution
    """
    # Double the size of filters to fit the first Residual Group
    x = projection_block(x, n_filters, strides=strides)

    # Identity residual blocks
    for _ in range(n_blocks):
        x = identity_block(x, n_filters)
    return x
```

### Stem Group

<img src="stem.jpg">

```python
def stem(inputs):
    """ Construct the Stem Convolutional Group
        inputs : the input vector
    """
    # The 224x224 images are zero padded (black - no signal) to be 230x230 images prior to the first convolution
    x = layers.ZeroPadding2D(padding=(3, 3))(inputs)

    # First Convolutional layer which uses a large (coarse) filter
    x = layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='valid', use_bias=False, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Pooled feature maps will be reduced by 75%
    x = layers.ZeroPadding2D(padding=(1, 1))(x)
    x = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2))(x)
    return x
```

### ResNet Block with Identity Shortcut

<img src='identity-block.jpg'>

```python
def identity_block(x, n_filter):
    """ Construct a Bottleneck Residual Block with Identity Link
        x        : input into the block
        n_filters: number of filters
    """
    # Save input vector (feature maps) for the identity link
    shortcut = x

    ## Construct the 1x1, 3x3, 1x1 residual block (fig 3c)

    # Dimensionality reduction
    x = layers.Conv2D(n_filters, (1, 1), strides=(1, 1), use_bias=False, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Bottleneck layer
    x = layers.Conv2D(n_filters, (3, 3), strides=(1, 1), padding="same", use_bias=False, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Dimensionality restoration - increase the number of output filters by 4X
    x = layers.Conv2D(n_filters * 4, (1, 1), strides=(1, 1), use_bias=False, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)

    # Add the identity link (input) to the output of the residual block
    x = layers.add([shortcut, x])
    x = layers.ReLU()(x)
    return x
```

#### v2.0

In v2.0, the BatchNormalization and ReLU activation function is moved from after the convolution to before.

```python
def identity_block(x, n_filters):
    """ Construct a Bottleneck Residual Block with Identity Link
        x        : input into the block
        n_filters: number of filters
    """

    # Save input vector (feature maps) for the identity link
    shortcut = x

    ## Construct the 1x1, 3x3, 1x1 convolution block

    # Dimensionality reduction
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(n_filters, (1, 1), strides=(1, 1), use_bias=False, kernel_initializer='he_normal')(x)

    # Bottleneck layer
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(n_filters, (3, 3), strides=(1, 1), padding="same", use_bias=False, kernel_initializer='he_normal')(x)

    # Dimensionality restoration - increase the number of output filters by 4X
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(n_filters * 4, (1, 1), strides=(1, 1), use_bias=False, kernel_initializer='he_normal')(x)

    # Add the identity link (input) to the output of the residual block
    x = layers.add([shortcut, x])
    return x
```

### ResNet Block with Projection Shortcut

#### v1.0
<img src='projection-block.jpg'>

```python
def projection_block(x, n_filters, strides=(2,2)):
    """ Construct Bottleneck Residual Block with Projection Shortcut
        Increase the number of filters by 4X
        x        : input into the block
        n_filters: number of filters
        strides  : whether entry convolution is strided (i.e., (2, 2) vs (1, 1))
    """
    # Construct the projection shortcut
    # Increase filters by 4X to match shape when added to output of block
    shortcut = layers.Conv2D(4 * n_filters, (1, 1), strides=strides, use_bias=False, kernel_initializer='he_normal')(x)
    shortcut = layers.BatchNormalization()(shortcut)

    ## Construct the 1x1, 3x3, 1x1 residual block (fig 3c)

    # Dimensionality reduction
    # Feature pooling when strides=(2, 2)
    x = layers.Conv2D(n_filters, (1, 1), strides=strides, use_bias=False, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Bottleneck layer
    x = layers.Conv2D(n_filters, (3, 3), strides=(1, 1), padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Dimensionality restoration - increase the number of filters by 4X
    x = layers.Conv2D(4 * n_filters, (1, 1), strides=(1, 1), use_bias=False, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)

    # Add the projection shortcut link to the output of the residual block
    x = layers.add([x, shortcut])
    x = layers.ReLU()(x)
    return x
```
#### v1.5
<img src='projection-block-v1.5.jpg'>

In v1.5, the strided convolution is moved from the 1x1 convolution to the 3x3 bottleneck convolution.

```python
def projection_block(x, n_filters, strides=(2,2)):
    """ Construct Bottleneck Residual Block of Convolutions with Projection Shortcut
        Increase the number of filters by 4X
        x        : input into the block
        n_filters: number of filters
        strides  : whether the first convolution is strided
    """
    # Construct the projection shortcut
    # Increase filters by 4X to match shape when added to output of block
    shortcut = layers.Conv2D(4 * n_filters, (1, 1), strides=strides, use_bias=False, kernel_initializer='he_normal')(x)
    shortcut = layers.BatchNormalization()(shortcut)

    ## Construct the 1x1, 3x3, 1x1 residual block

    # Dimensionality reduction
    x = layers.Conv2D(n_filters, (1, 1), strides=(1,1), use_bias=False, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Bottleneck layer
    # Feature pooling when strides=(2, 2)
    x = layers.Conv2D(n_filters, (3, 3), strides=strides, padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Dimensionality restoration - increase the number of output filters by 4X
    x = layers.Conv2D(4 * n_filters, (1, 1), strides=(1, 1), use_bias=False, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)

    # Add the projection shortcut to the output of the residual block
    x = layers.add([x, shortcut])
    x = layers.ReLU()(x)
    return x
```

#### v2.0

In v2.0, the BatchNormalization and ReLU activation function is moved from after the convolution to before.

```python
def projection_block(x, n_filters, strides=(2,2)):
    """ Construct a Bottleneck Residual Block of Convolutions with Projection Shortcut
        Increase the number of filters by 4X
        x        : input into the block
        n_filters: number of filters
        strides  : whether the first convolution is strided
    """
    # Construct the projection shortcut
    # Increase filters by 4X to match shape when added to output of block
    shortcut = layers.BatchNormalization()(x)
    shortcut = layers.Conv2D(4 * n_filters, (1, 1), strides=strides, use_bias=False, kernel_initializer='he_normal')(shortcut)

    ## Construct the 1x1, 3x3, 1x1 convolution block

    # Dimensionality reduction
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(n_filters, (1, 1), strides=(1,1), use_bias=False, kernel_initializer='he_normal')(x)

    # Bottleneck layer
    # Feature pooling when strides=(2, 2)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(n_filters, (3, 3), strides=strides, padding='same', use_bias=False, kernel_initializer='he_normal')(x)

    # Dimensionality restoration - increase the number of filters by 4X
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(4 * n_filters, (1, 1), strides=(1, 1), use_bias=False, kernel_initializer='he_normal')(x)

    # Add the projection shortcut to the output of the residual block
    x = layers.add([x, shortcut])
    return x
```

### Classifier

<img src="classifier.jpg">

```python
def classifier(x, n_classes):
  """ Construct the Classifier Group 
      x         : input to the classifier
      n_classes : number of output classes
  """
  # Pool at the end of all the convolutional residual blocks
  x = layers.GlobalAveragePooling2D()(x)

  # Final Dense Outputting Layer for the outputs
  outputs = layers.Dense(n_classes, activation='softmax')(x)
  return outputs
```
