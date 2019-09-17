
# SE-Net

    se_resnet.py - academic - procedural
    se_resnext.py - academic - procedural
    se_resnet_c.py - composable - OOP
    se_resnext_c.py - composable - OOP

[Paper](https://arxiv.org/pdf/1709.01507.pdf)

## Macro-Architecture

<img src="macro.jpg">

Macro-architectures for SE-ResNet50 and SE-ResNeXt50:

*SE-ResNet*
```python
def learner(x, groups, ratio):
    """ Construct the Learner
        x     : input to the learner
	groups: list of groups: number of filters and blocks
        ratio : amount of filter reduction in squeeze
    """
    # First Residual Block Group (not strided)
    n_filters, n_blocks = groups.pop(0)
    x = group(x, n_filters, n_blocks, ratio, strides=(1, 1))

    # Remaining Residual Block Groups (strided)
    for n_filters, n_blocks in groups:
    	x = group(x, n_filters, n_blocks, ratio)
    return x	

# Meta-parameter: # Meta-parameter: list of groups: filter size and number of blocks
groups = { 50 : [ (64, 3), (128, 4), (256, 6),  (512, 3) ],		# SE-ResNet50
           101: [ (64, 3), (128, 4), (256, 23), (512, 3) ],		# SE-ResNet101
           152: [ (64, 3), (128, 8), (256, 36), (512, 3) ]		# SE-ResNet152
         }

# Meta-parameter: Amount of filter reduction in squeeze operation
ratio = 16

# The input tensor
inputs = Input(shape=(224, 224, 3))

# The Stem Group
x = stem(inputs)

# The Learnet
x = learner(x, group[50], ratio)

# The Classifier for 1000 classes
outputs = classifier(x, 1000)

# Instantiate the Model
model = Model(inputs, outputs)
```

*SE-ResNeXt*

```python
def learner(x, groups, cardinality, ratio):
    """ Construct the Learner
        x          : input to the learner
        groups     : list of groups: filters in, filters out, number of blocks
	cardinality: width of group convolution
        ratio      : amount of filter reduction during squeeze
    """
    # First ResNeXt Group (not strided)
    filters_in, filters_out, n_blocks = groups.pop(0)
    x = group(x, n_blocks, filters_in, filters_out, cardinality, ratio, strides=(1, 1))

    # Remaining ResNeXt Groups
    for filters_in, filters_out, n_blocks in groups:
        x = group(x, n_blocks, filters_in, filters_out, cardinality, ratio)
    return x

# Meta-parameter: # Meta-parameter: list of groups: filter size and number of blocks
groups = { 50 : [ (64, 3), (128, 4), (256, 6),  (512, 3) ],		# SE-ResNet50
           101: [ (64, 3), (128, 4), (256, 23), (512, 3) ],		# SE-ResNet101
           152: [ (64, 3), (128, 8), (256, 36), (512, 3) ]		# SE-ResNet152
         }
	 
# Meta-parameter: Width of group convolution
cardinality = 32

# Meta-parameter: Amount of filter reduction in squeeze operation
ratio = 16

# The input tensor
inputs = Input(shape=(224, 224, 3))

# The Stem Group
x = stem(inputs)

# The Learner
x = learner(x, cardinality, ratio)

# The Classifier for 1000 classes
outputs = classifier(x, 1000)

# Instantiate the Model
model = Model(inputs, outputs)
```

## Micro-Architecture

<img src="micro.jpg">

*SE-ResNet*
```python
def group(x, n_filters, n_blocks, ratio, strides=(2, 2)):
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

*SE-ResNeXt*
```python
def group(x, n_blocks, filters_in, filters_out, cardinality, ratio, strides=(2, 2)):
    """ Construct a Squeeze-Excite Group
        x          : input to the group
        n_blocks   : number of blocks in the group
        filters_in : number of filters  (channels) at the input convolution
        filters_out: number of filters  (channels) at the output convolution
        ratio      : amount of filter reduction during squeeze
	cardinality: width of group convolution
        strides    : whether projection block is strided
    """
    # First block is a linear projection block
    x = projection_block(x, filters_in, filters_out, strides=strides, cardinality=cardinality, ratio=ratio)

    # Remaining blocks are identity links
    for _ in range(n_blocks-1):
        x = identity_block(x, filters_in, filters_out, cardinality=cardinality, ratio=ratio)
    return x
```

### Residual Block and Identity Shortcut w/SE Link

<img src="identity-block.jpg">

*SE-ResNet*
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

*SE-ResNeXt*
```python
def identity_block(x, filters_in, filters_out, cardinality=32, ratio=16):
    """ Construct a ResNeXT block with identity link
        x          : input to block
        filters_in : number of filters  (channels) at the input convolution
        filters_out: number of filters (channels) at the output convolution
        cardinality: width of cardinality layer
        ratio      : amount of filter reduction during squeeze
    """

    # Remember the input
    shortcut = x

    # Dimensionality Reduction
    x = Conv2D(filters_in, kernel_size=(1, 1), strides=(1, 1),
                      padding='same', kernel_initializer='he_normal')(shortcut)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Cardinality (Wide) Layer (split-transform)
    filters_card = filters_in // cardinality
    groups = []
    for i in range(cardinality):
        group = Lambda(lambda z: z[:, :, :, i * filters_card:i *
                              filters_card + filters_card])(x)
        groups.append(Conv2D(filters_card, kernel_size=(3, 3),
                                    strides=(1, 1), padding='same', kernel_initializer='he_normal')(group))

    # Concatenate the outputs of the cardinality layer together (merge)
    x = Concatenate()(groups)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Dimensionality restoration
    x = Conv2D(filters_out, kernel_size=(1, 1), strides=(1, 1),
                      padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)

    # Pass the output through the squeeze and excitation block
    x = squeeze_excite_block(x, ratio)

    # Identity Link: Add the shortcut (input) to the output of the block
    x = Add()([shortcut, x])
    x = ReLU()(x)
    return x
```

### Residual Block and Projection Shortcut w/SE Link

<img src="projection-block.jpg">

*SE-ResNeXt*
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

*SE-ResNeXt*
```python
def projection_block(x, filters_in, filters_out, cardinality=32, strides=1, ratio=16):
    """ Construct a ResNeXT block with projection shortcut
        x          : input to the block
        filters_in : number of filters  (channels) at the input convolution
        filters_out: number of filters (channels) at the output convolution
        cardinality: width of cardinality layer
        strides    : whether entry convolution is strided (i.e., (2, 2) vs (1, 1))
        ratio      : amount of filter reduction during squeeze
    """

    # Construct the projection shortcut
    # Increase filters by 2X to match shape when added to output of block
    shortcut = Conv2D(filters_out, kernel_size=(1, 1), strides=strides,
                                 padding='same', kernel_initializer='he_normal')(x)
    shortcut = BatchNormalization()(shortcut)

    # Dimensionality Reduction
    x = Conv2D(filters_in, kernel_size=(1, 1), strides=(1, 1),
                      padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Cardinality (Wide) Layer (split-transform)
    filters_card = filters_in // cardinality
    groups = []
    for i in range(cardinality):
        group = Lambda(lambda z: z[:, :, :, i * filters_card:i *
                              filters_card + filters_card])(x)
        groups.append(Conv2D(filters_card, kernel_size=(3, 3),
                                    strides=strides, padding='same', kernel_initializer='he_normal')(group))

    # Concatenate the outputs of the cardinality layer together (merge)
    x = Concatenate()(groups)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Dimensionality restoration
    x = Conv2D(filters_out, kernel_size=(1, 1), strides=(1, 1),
                      padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)

    # Pass the output through the squeeze and excitation block
    x = squeeze_excite_block(x, ratio)

    # Add the projection shortcut (input) to the output of the block
    x = Add()([shortcut, x])
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

## Composable

*Example Instantiate a SE-ResNet model*

```python
# SE-ResNet50 from research paper
senet = SEResNet(50)

# ResNet50 custom input shape/classes
senet = SEResNet(50, input_shape=(128, 128, 3), n_classes=50)

# getter for the tf.keras model
model = senet.model
```

*Example: Composable Group/Block*/

```python
inputs = Input((32, 32, 3))
x = Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
# SE Residual group: 2 blocks, 128 filters
x = SEResNet.group(2, 128)(x)
# SE Residual block with projection, 256 filters
x = SEResNet.projection_block(256)
# Residual block with identity, 256 filters
x = SEResNet.identity_block(256)
x = Flatten()(x)
x = Dense(100, activation='softmax')
