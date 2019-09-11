
# SE-Net

[Paper](https://arxiv.org/pdf/1709.01507.pdf)

## Macro-Architecture

<img src="macro.jpg">

Macro-architectures for SE-ResNet50 and SE-ResNext50:

*SE-ResNet*
```python
def learner(x, ratio):
    """ Construct the Learner
        x    : input to the learner
        ratio: amount of filter reduction in squeeze
    """
    # First Residual Block Group of 64 filters
    x = se_group(x, 3, 64, ratio, strides=(1, 1))

    # Second Residual Block Group of 128 filters
    # Double the size of filters and reduce feature maps by 75% (strides=2, 2) to fit the next Residual Group
    x = se_group(x, 4, 128, ratio)

    # Third Residual Block Group of 256 filters
    # Double the size of filters and reduce feature maps by 75% (strides=2, 2) to fit the next Residual Group
    x = se_group(x, 6, 256, ratio)

    # Fourth Residual Block Group of 512 filters
    # Double the size of filters and reduce feature maps by 75% (strides=2, 2) to fit the next Residual Group
    x = se_group(x, 3, 512, ratio)
    return x

# Meta-parameter: Amount of filter reduction in squeeze operation
ratio = 16

# The input tensor
inputs = Input(shape=(224, 224, 3))

# The Stem Group
x = stem(inputs)

# The Learnet
x = learner(x, ratio)

# The Classifier for 1000 classes
outputs = classifier(x, 1000)

# Instantiate the Model
model = Model(inputs, outputs)
```

*SE-ResNeXt*

```python
def learner(x, ratio):
    """ Construct the Learner
        x     : input to the learner
        ratio : amount of filter reduction during squeeze
    """
    # First ResNeXt Group
    # Double the size of filters to fit the first Residual Group
    x = se_group(x, 3, 128, 256, ratio=ratio, strides=(1, 1))

    # Second ResNeXt
    # Double the size of filters and reduce feature maps by 75% (strides=2, 2) to fit the next Residual Group
    x = se_group(x, 4, 256, 512, ratio=ratio)

    # Third ResNeXt Group
    # Double the size of filters and reduce feature maps by 75% (strides=2, 2) to fit the next Residual Group
    x = se_group(x, 6, 512, 1024, ratio=ratio)

    # Fourth ResNeXt Group
    # Double the size of filters and reduce feature maps by 75% (strides=2, 2) to fit the next Residual Group
    x = se_group(x, 3, 1024, 2048, ratio=ratio)
    return x
    
# Meta-parameter: Amount of filter reduction in squeeze operation
ratio = 16

# The input tensor
inputs = Input(shape=(224, 224, 3))

# The Stem Group
x = stem(inputs)

# The Learner
x = learner(x, ratio)

# The Classifier for 1000 classes
outputs = classifier(x, 1000)

# Instantiate the Model
model = Model(inputs, outputs)
```

## Micro-Architecture

<img src="micro.jpg">

```python
def se_group(x, n_blocks, n_filters, ratio, strides=(2, 2)):
    """ Construct the Squeeze-Excite Group
        x        : input to the group
        n_blocks : number of blocks
        n_filters: number of filters
        ratio    : amount of filter reduction during squeeze
        strides  : whether projection block is strided
    """
    # first block uses linear projection to match the doubling of filters between groups
    x = projection_block(x, n_filters, strides=strides, ratio=ratio)

    # remaining blocks use identity link
    for _ in range(n_blocks-1):
        x = identity_block(x, n_filters, ratio=ratio)
    return x
```

### Residual Block and Identity Shortcut w/SE Link

<img src="identity-block.jpg">

```python
def identity_block(x, n_filters, ratio=16):
    """ Create a Bottleneck Residual Block with Identity Link
        x        : input into the block
        n_filters: number of filters
        ratio    : amount of filter reduction during squeeze
    """
    # Save input vector (feature maps) for the identity link
    shortcut = x

    ## Construct the 1x1, 3x3, 1x1 residual block (fig 3c)

    # Dimensionality reduction
    x = Conv2D(n_filters, (1, 1), strides=(1, 1), use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Bottleneck layer
    x = Conv2D(n_filters, (3, 3), strides=(1, 1), padding="same", use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Dimensionality restoration - increase the number of output filters by 4X
    x = Conv2D(n_filters * 4, (1, 1), strides=(1, 1), use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)

    # Pass the output through the squeeze and excitation block
    x = squeeze_excite_block(x, ratio)

    # Add the identity link (input) to the output of the residual block
    x = Add()([shortcut, x])
    x = ReLU()(x)
    return x
```

### Residual Block and Projection Shortcut w/SE Link

<img src="projection-block.jpg">

```python
def projection_block(x, n_filters, strides=(2,2), ratio=16):
    """ Create Bottleneck Residual Block with Projection Shortcut
        Increase the number of filters by 4X
        x        : input into the block
        n_filters: number of filters
        strides  : whether entry convolution is strided (i.e., (2, 2) vs (1, 1))
        ratio    : amount of filter reduction during squeeze
    """
    # Construct the projection shortcut
    # Increase filters by 4X to match shape when added to output of block
    shortcut = layers.Conv2D(4 * n_filters, (1, 1), strides=strides, use_bias=False, kernel_initializer='he_normal')(x)
    shortcut = layers.BatchNormalization()(shortcut)

    ## Construct the 1x1, 3x3, 1x1 residual block (fig 3c)

    # Dimensionality reduction
    # Feature pooling when strides=(2, 2)
    x = Conv2D(n_filters, (1, 1), strides=strides, use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Bottleneck layer
    x = Conv2D(n_filters, (3, 3), strides=(1, 1), padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Dimensionality restoration - increase the number of filters by 4X
    x = Conv2D(4 * n_filters, (1, 1), strides=(1, 1), use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)

    # Pass the output through the squeeze and excitation block
    x = squeeze_excite_block(x, ratio)

    # Add the projection shortcut link to the output of the residual block
    x = Add()([x, shortcut])
    x = ReLU()(x)
    return x
```

### Squeeze-Excitation Block

<img src="se-block.jpg">

```python
def squeeze_excite_block(x, ratio=16):
    """ Create a Squeeze and Excite block
        x    : input to the block
        ratio : amount of filter reduction during squeeze
    """
    # Remember the input
    shortcut = x

    # Get the number of filters on the input
    filters = x.shape[-1]

    # Squeeze (dimensionality reduction)
    # Do global average pooling across the filters, which will the output a 1D vector
    x = GlobalAveragePooling2D()(x)

    # Reshape into 1x1 feature maps (1x1xC)
    x = Reshape((1, 1, filters))(x)

    # Reduce the number of filters (1x1xC/r)
    x = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(x)

    # Excitation (dimensionality restoration)
    # Restore the number of filters (1x1xC)
    x = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(x)

    # Scale - multiply the squeeze/excitation output with the input (WxHxC)
    x = Multiply()([shortcut, x])
    return x
```
