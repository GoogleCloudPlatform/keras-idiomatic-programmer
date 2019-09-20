
# DenseNet

    densenet.py - academic - procedural
    densenet_c.py - composable - OOP


[Paper](https://arxiv.org/pdf/1608.06993.pdf)

## Macro-Architecture

<img src='macro.jpg'>

Macro-architecture code for DenseNet 121:

```python
def learner(x, groups, n_filters, reduction):
    """ Construct the Learner
        x         : input to the learner
        groups    : set of number of blocks per group
        n_filters : number of filters (growth rate)
        reduction : the amount to reduce (compress) feature maps by
    """
    # pop off the list the last dense block
    last = groups.pop()

    # Create the dense groups and interceding transition blocks
    for n_blocks in groups:
        x = group(x, n_blocks, n_filters, reduction)

    # Add the last dense group w/o a following transition block
    x = group(x, last, n_filters)
    return x
    
# Meta-parameter: amount to reduce feature maps by (compression factor) during transition blocks
reduction = 0.5

# Meta-parameter: number of filters in a convolution block within a residual block (growth rate)
n_filters = 32

# Meta-parameter: number of residual blocks in each dense group
groups = { 121 : [6, 12, 24, 16],     # DenseNet 121
           169 : [6, 12, 32, 32],     # DenseNet 169
           201 : [6, 12, 48, 32] }    # DenseNet 201

# The input vector
inputs = Input(shape=(224, 224, 3))

# The Stem Convolution Group
x = stem(inputs, n_filters)

# The Learner
x = learner(x, groups[121], n_filters, reduction)

# Classifier for 1000 classes
outputs = classifier(x, 1000)

# Instantiate the model
model = Model(inputs, outputs)
```

## Micro-Architecture

<img src='micro.jpg'>

```python
def group(x, n_blocks, n_filters, reduction=None):
    """ Construct a Dense Block
        x         : input to the block
        n_blocks  : number of residual blocks in dense block
        n_filters : number of filters in convolution layer in residual block
        reduction : amount to reduce feature maps by
    """
    # Construct a group of densely connected residual blocks
    for _ in range(n_blocks):
        x = dense_block(x, n_filters)

    # Construct interceding transition block
    if reduction is not None:
        x = trans_block(x, reduction)
    return x
```

### Stem Group

<img src="stem.jpg">

```python
def stem(inputs, n_filters):
    """ Construct the Stem Convolution Group
        inputs   : input tensor
        n_filters: number of filters for the dense blocks (k)
    """
    # Pads input from 224x224 to 230x230
    x = ZeroPadding2D(padding=((3, 3), (3, 3)))(inputs)
    
    # First large convolution for abstract features for input 224 x 224 and output 112 x 112
    # Stem convolution uses 2 * k (growth rate) number of filters
    x = Conv2D(2 * n_filters, (7, 7), strides=(2, 2), use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    # Add padding so when downsampling we fit shape 56 x 56
    x = ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = MaxPooling2D((3, 3), strides=2)(x)
    return x
```

### Dense Block

<img src="dense-block.jpg">

```python
def dense_block(x, n_filters):
    """ Construct a Densely Connected Residual Block
        x        : input to the block
        n_filters: number of filters in convolution layer in residual block
    """
    # Remember input tensor into residual block
    shortcut = x

    # BN-RE-Conv pre-activation form of convolutions

    # Dimensionality expansion, expand filters by 4 (DenseNet-B)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(4 * n_filters, (1, 1), strides=(1, 1), use_bias=False, kernel_initializer='he_normal')(x)

    # Bottleneck convolution
    # 3x3 convolution with padding=same to preserve same shape of feature maps
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(n_filters, (3, 3), strides=(1, 1), padding='same', use_bias=False, kernel_initializer='he_normal')(x)

    # Concatenate the input (identity) with the output of the residual block
    # Concatenation (vs. merging) provides Feature Reuse between layers
    x = Concatenate()([shortcut, x])
    return x
```

### Transitional Block

<img src="trans-block.jpg">

```python
def trans_block(x, reduction):
    """ Construct a Transition Block
        x        : input layer
        reduction: percentage of reduction of feature maps
    """

    # Reduce (compress) the number of feature maps (DenseNet-C)
    # shape[n] returns a class object. We use int() to cast it into the dimension size
    n_filters = int( int(x.shape[3]) * reduction)
    
    # BN-LI-Conv pre-activation form of convolutions
    
    # Use 1x1 linear projection convolution
    x = BatchNormalization()(x)
    x = Conv2D(n_filters, (1, 1), strides=(1, 1), use_bias=False, kernel_initializer='he_normal')(x)

    # Use mean value (average) instead of max value sampling when pooling reduce by 75%
    x = AveragePooling2D((2, 2), strides=(2, 2))(x)
    return x
```

### Residual Block

<img src="residual-block.jpg">


### Classifier

<img src="classifier.jpg">

```python
def classifier(x, n_classes):
    """ Construct the Classifier Group
        x         : input to the classifier
        n_classes : number of output classes
    """
    # Global Average Pooling will flatten the 7x7 feature maps into 1D feature maps
    x = GlobalAveragePooling2D()(x)
    # Fully connected output layer (classification)
    x = Dense(n_classes, activation='softmax', kernel_initializer='he_normal')(x)
    return x
```

## Composable

*Example Instantiate a DenseNet model*

```python
# DenseNet121 from research paper
densenet = DenseNet(121)

# DenseNet121 custom input shape/classes
densenet = DenseNet(121, input_shape=(128, 128, 3), n_classes=50)

# getter for the tf.keras model
model = densenet.model
```

*Example: Composable Group/Block*/

```python
inputs = Input((32, 32, 3))
x = Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
# DenseNet group: 6 blocks, 32 filters
x = DenseNet.group(x, 6, 32)
# Residual block with 32 filters
x = DensetNet.dense_block(x, 32)
x = Flatten()(x)
x = Dense(100, activation='softmax')
```
