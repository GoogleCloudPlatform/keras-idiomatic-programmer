
# ResNet

    resnet(v1/v1.5/v2).py - academic - procedural
    resnet_cifar10(v1/v2).py - academic - procedural
    resnet(v1/v1.5/v2)_c.py - composable - OOP

[Paper](https://arxiv.org/pdf/1512.03385.pdf)

## Macro-Architecture

<img src='macro.jpg'>

```python
def learner(x, groups):
    """ Construct the Learner
        x     : input to the learner
        groups: list of groups: number of filters and blocks
    """
    # First Residual Block Group (not strided)
    n_filters, n_blocks = groups.pop(0)
    x = group(x, n_filters, n_blocks, strides=(1, 1))

    # Remaining Residual Block Groups (strided)
    for n_filters, n_blocks in groups:
        x = group(x, n_filters, n_blocks)
    return x

# Meta-parameter: list of groups: filter size and number of blocks
groups = { 50 : [ (64, 2), (128, 3), (256, 5),  (512, 2) ],           # ResNet50
           101: [ (64, 2), (128, 3), (256, 22), (512, 2) ],           # ResNet101
           152: [ (64, 2), (128, 7), (256, 35), (512, 2) ]            # ResNet152
         }

# The input tensor
inputs = Input(shape=(224, 224, 3))

# The stem convolutional group
x = stem(inputs)

# The learner
x = learner(x, groups[50])

# The classifier for 1000 classes
outputs = classifier(x, 1000)

# Instantiate the Model
model = Model(inputs, outputs)
```

## Micro-Architecture

<img src='micro.jpg'>

```python
def group(x, n_filters, n_blocks, strides=(2, 2)):
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
    x = ZeroPadding2D(padding=(3, 3))(inputs)

    # First Convolutional layer which uses a large (coarse) filter
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='valid', use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Pooled feature maps will be reduced by 75%
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
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

    # Add the identity link (input) to the output of the residual block
    x = Add()([shortcut, x])
    x = ReLU()(x)
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
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(n_filters, (1, 1), strides=(1, 1), use_bias=False, kernel_initializer='he_normal')(x)

    # Bottleneck layer
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(n_filters, (3, 3), strides=(1, 1), padding="same", use_bias=False, kernel_initializer='he_normal')(x)

    # Dimensionality restoration - increase the number of output filters by 4X
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(n_filters * 4, (1, 1), strides=(1, 1), use_bias=False, kernel_initializer='he_normal')(x)

    # Add the identity link (input) to the output of the residual block
    x = Add()([shortcut, x])
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

    # Add the projection shortcut link to the output of the residual block
    x = Add()([x, shortcut])
    x = ReLU()(x)
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
    shortcut = Conv2D(4 * n_filters, (1, 1), strides=strides, use_bias=False, kernel_initializer='he_normal')(x)
    shortcut = BatchNormalization()(shortcut)

    ## Construct the 1x1, 3x3, 1x1 residual block

    # Dimensionality reduction
    x = Conv2D(n_filters, (1, 1), strides=(1,1), use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Bottleneck layer
    # Feature pooling when strides=(2, 2)
    x = Conv2D(n_filters, (3, 3), strides=strides, padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Dimensionality restoration - increase the number of output filters by 4X
    x = Conv2D(4 * n_filters, (1, 1), strides=(1, 1), use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)

    # Add the projection shortcut to the output of the residual block
    x = Add()([x, shortcut])
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
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(n_filters, (1, 1), strides=(1,1), use_bias=False, kernel_initializer='he_normal')(x)

    # Bottleneck layer
    # Feature pooling when strides=(2, 2)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(n_filters, (3, 3), strides=strides, padding='same', use_bias=False, kernel_initializer='he_normal')(x)

    # Dimensionality restoration - increase the number of filters by 4X
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(4 * n_filters, (1, 1), strides=(1, 1), use_bias=False, kernel_initializer='he_normal')(x)

    # Add the projection shortcut to the output of the residual block
    x = Add()([x, shortcut])
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
  x = GlobalAveragePooling2D()(x)

  # Final Dense Outputting Layer for the outputs
  outputs = Dense(n_classes, activation='softmax')(x)
  return outputs
```
## Composable

*Example Instantiate a ResNet model*

```python
# ResNet50 v1.0 from research paper
resnet = ResNetV1(50)

# ResNet50 v1.0 custom input shape/classes
resnet = ResNetV1(50, input_shape=(128, 128, 3), n_classes=50)

# getter for the tf.keras model
model = resnet.model
```

*Example: Composable Group/Block*/

```python
inputs = Input((32, 32, 3))
x = Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
# Residual group: 2 blocks, 128 filters
x = ResNetV1.group(2, 128)(x)
# Residual residual block with projection, 256 filters
x = ResNetV1.projection_block(256)
# Residual residual block with identity, 256 filters
x = ResNetV1.identity_block(256)
x = Flatten()(x)
x = Dense(100, activation='softmax')
```
