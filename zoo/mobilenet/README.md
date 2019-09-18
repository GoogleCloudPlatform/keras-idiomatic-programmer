
# MobileNet v1.0

[Paper](https://arxiv.org/pdf/1704.04861.pdf)

## Macro-Architecture

<img src='macro.jpg'>

Macro-architecture code for MobileNet v1 (224x224 input):

```python
def learner(x, alpha):
    """ Construct the Learner
        x      : input to the learner
        alpha  : width multiplier
    """
    # First Depthwise Separable Convolution Group
    x = depthwise_group(x, 128, 2, alpha)

    # Second Depthwise Separable Convolution Group
    x = depthwise_group(x, 256, 2, alpha)

    # Third Depthwise Separable Convolution Group
    x = depthwise_group(x, 512, 6, alpha)

    # Fourth Depthwise Separable Convolution Group
    x = depthwise_group(x, 1024, 2, alpha)
    return x
    
# Meta-parameter: width multiplier (0 .. 1) for reducing number of filters.
alpha      = 1   

# Meta-parameter: resolution multiplier (0 .. 1) for reducing input size
pho        = 1

# Meta-parameter: dropout rate
dropout    = 0.5 

inputs = Input(shape=(int(224 * pho), int(224 * pho), 3))

# The Stem Group
x = stem(inputs, alpha)    

# The Learner
x = learner(x, alpha)

# The classifier for 1000 classes
outputs = classifier(x, alpha, dropout, 1000)

# Instantiate the Model
model = Model(inputs, outputs)
```

## Micro-Architecture

<img src="micro.jpg">

```python
def depthwise_group(x, n_filters, n_blocks, alpha):
    """ Construct a Depthwise Separable Convolution Group
        x         : input to the group
        n_filters : number of filters
        n_blocks  : number of blocks in the group
        alpha     : width multiplier
    """   
    # In first block, the depthwise convolution is strided - feature map size reduction
    x = depthwise_block(x, n_filters, alpha, strides=(2, 2))
    
    # Remaining blocks
    for _ in range(n_blocks - 1):
        x = depthwise_block(x, n_filters, alpha, strides=(1, 1))
    return x
```

### Stem Group

<img src="stem.jpg">

```python
def stem(inputs, alpha):
    """ Construct the Stem Group
        inputs : input tensor
        alpha  : width multiplier
    """
    # Convolutional block
    x = ZeroPadding2D(padding=((0, 1), (0, 1)))(inputs)
    x = Conv2D(32, (3, 3), strides=(2, 2), padding='valid')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Depthwise Separable Convolution Block
    x = depthwise_block(x, 64, alpha, (1, 1))
    return x
```

### Depthwise Separable Block

<img src="depthwise-block.jpg">

```python
def depthwise_block(x, n_filters, alpha, strides):
    """ Construct a Depthwise Separable Convolution block
        x         : input to the block
        n_filters : number of filters
        alpha     : width multiplier
        strides   : strides
    """
    # Apply the width filter to the number of feature maps
    filters = int(n_filters * alpha)

    # Strided convolution to match number of filters
    if strides == (2, 2):
        x = layers.ZeroPadding2D(padding=((0, 1), (0, 1)))(x)
        padding = 'valid'
    else:
        padding = 'same'

    # Depthwise Convolution
    x = DepthwiseConv2D((3, 3), strides, padding=padding)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Pointwise Convolution
    x = Conv2D(filters, (1, 1), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x
```

### Strided Depthwise Separable Block

<img src="strided-depthwise-block.jpg">

```python
```

### Classifier Group

<img src="classifier.jpg">

```python
def classifier(x, alpha, dropout, n_classes):
    """ Construct the classifier group
        x         : input to the classifier
        alpha     : width multiplier
        dropout   : dropout percentage
        n_classes : number of output classes
    """
    # Flatten the feature maps into 1D feature maps (?, N)
    x = layers.GlobalAveragePooling2D()(x)

    # Reshape the feature maps to (?, 1, 1, 1024)
    shape = (1, 1, int(1024 * alpha))
    x = Reshape(shape)(x)
    # Perform dropout for preventing overfitting
    x = Dropout(dropout)(x)

    # Use convolution for classifying (emulates a fully connected layer)
    x = Conv2D(n_classes, (1, 1), padding='same', activation='softmax')(x)
    # Reshape the resulting output to 1D vector of number of classes
    x = Reshape((n_classes, ))(x)
    return x
```


# MobileNet v2.0

[Paper](https://arxiv.org/pdf/1801.04381.pdf)

## Macro-Architecture

<img src='macro-v2.jpg'>

Macro-architecture code for MobileNet v2 (224x224 input):

```python
def learner(x, alpha, expansion=6):
    """ Construct the Learner
        x        : input to the learner
        alpha    : width multiplier
        expansion: multiplier to expand number of filters
    """
    # First Inverted Residual Convolution Group
    x = group(x, 16, 1, alpha, expansion=1, strides=(1, 1))
    
    # Second Inverted Residual Convolution Group
    x = group(x, 24, 2, alpha, expansion)

    # Third Inverted Residual Convolution Group
    x = group(x, 32, 3, alpha, expansion)
    
    # Fourth Inverted Residual Convolution Group
    x = group(x, 64, 4, alpha, expansion)
    
    # Fifth Inverted Residual Convolution Group
    x = group(x, 96, 3, alpha, expansion, strides=(1, 1))
    
    # Sixth Inverted Residual Convolution Group
    x = group(x, 160, 3, alpha, expansion)
    
    # Seventh Inverted Residual Convolution Group
    x = group(x, 320, 1, alpha, expansion, strides=(1, 1))
    
    # Last block is a 1x1 linear convolutional layer,
    # expanding the number of filters to 1280.
    x = Conv2D(1280, (1, 1), use_bias=False, kernel_initializer='glorot_uniform')(x)
    x = BatchNormalization()(x)
    x = ReLU(6.)(x)
    return x

# Meta-parameter: width multiplier (0 .. 1) for reducing number of filters.
alpha = 1

# Meta-parameter: multiplier to expand number of filters
expansion = 6

inputs = Input(shape=(224, 224, 3))

# The Stem Group
x = stem(inputs, alpha)    

# The Learner
x = learner(x, alpha, expansion)

# The classifier for 1000 classes
outputs = classifier(x, 1000)

# Instantiate the Model
model = Model(inputs, outputs)
```

## Micro-Architecture

<img src="micro-v2.jpg">

```python
def inverted_group(x, n_filters, n_blocks, alpha, expansion=6, strides=(2, 2)):
    """ Construct an Inverted Residual Group
        x         : input to the group
        n_filters : number of filters
        n_blocks  : number of blocks in the group
        alpha     : width multiplier
        expansion : multiplier for expanding the number of filters
        strides   : whether first inverted residual block is strided.
    """   
    # In first block, the inverted residual block maybe strided - feature map size reduction
    x = inverted_block(x, n_filters, alpha, expansion, strides=strides)
    
    # Remaining blocks
    for _ in range(n_blocks - 1):
        x = inverted_block(x, n_filters, alpha, expansion, strides=(1, 1))
    return x
```

### Stem Group

<img src="stem-v2.jpg">

```python
def stem(inputs, alpha):
    """ Construct the Stem Group
        inputs : input tensor
        alpha  : width multiplier
    """
    # Calculate the number of filters for the stem convolution
    # Must be divisible by 8
    n_filters = max(8, (int(32 * alpha) + 4) // 8 * 8)
    
    # Convolutional block
    x = ZeroPadding2D(padding=((0, 1), (0, 1)))(inputs)
    x = Conv2D(n_filters, (3, 3), strides=(2, 2), padding='valid', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU(6.)(x)

    return x
```

### Inverted Residual Block

<img src='inverted-block.jpg'>

```python
def inverted_block(x, n_filters, alpha, expansion=6, strides=(1, 1)):
    """ Construct an Inverted Residual Block
        x         : input to the block
        n_filters : number of filters
        alpha     : width multiplier
        strides   : strides
        expansion : multiplier for expanding number of filters
    """
    # Remember input
    shortcut = x

    # Apply the width filter to the number of feature maps for the pointwise convolution
    filters = int(n_filters * alpha)
    
    n_channels = int(x.shape[3])
    
    # Dimensionality Expansion (non-first block)
    if expansion > 1:
        # 1x1 linear convolution
        x = Conv2D(expansion * n_channels, (1, 1), padding='same', use_bias=False)(x)
        
        x = BatchNormalization()(x)
        x = ReLU(6.)(x)

    # Strided convolution to match number of filters
    if strides == (2, 2):
        x = ZeroPadding2D(padding=((0, 1), (0, 1)))(x)
        padding = 'valid'
    else:
        padding = 'same'

    # Depthwise Convolution
    x = DepthwiseConv2D((3, 3), strides, padding=padding, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU(6.)(x)

    # Linear Pointwise Convolution
    x = Conv2D(filters, (1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    
    # Number of input filters matches the number of output filters
    if n_channels == filters and strides == (1, 1):
        x = Add()([shortcut, x]) 
    return x
 ```

### Strided Inverted Residual Block

<img src='strided-inverted-block.jpg'>

### Classifier

<img src="classifier-v2.jpg">

```python

def classifier(x, n_classes):
    """ Construct the classifier group
        x         : input to the classifier
        n_classes : number of output classes
    """
    # Flatten the feature maps into 1D feature maps (?, N)
    x = GlobalAveragePooling2D()(x)

    # Dense layer for final classification
    x = Dense(n_classes, activation='softmax')(x)
    return x
```
