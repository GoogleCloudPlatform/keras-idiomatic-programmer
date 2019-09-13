
# DenseNet

[Paper](https://arxiv.org/pdf/1608.06993.pdf)

## Macro-Architecture

<img src='macro.jpg'>

Macro-architecture code for DenseNet 121:

```python
# Meta-parameter: amount to reduce feature maps by (compression factor) during transition blocks
reduction = 0.5

# Meta-parameter: number of filters in a convolution block within a residual block (growth rate)
n_filters = 32

# Meta-parameter: number of residual blocks in each dense group
groups = { '121' : [6, 12, 24, 16],     # DenseNet 121
           '169' : [6, 12, 32, 32],     # DenseNet 169
           '201' : [6, 12, 48, 32] }    # DenseNet 201

# The input vector
inputs = Input(shape=(224, 224, 3))

# The Stem Convolution Group
x = stem(inputs, n_filters)

# The Learner
x = learner(x, groups['50'], n_filters, reduction)

# Classifier for 1000 classes
outputs = classifier(x, 1000)

# Instantiate the model
model = Model(inputs, outputs)
```

## Micro-Architecture

<img src='micro.jpg'>

```python
def learner(x, blocks, n_filters, reduction):
    """ Construct the Learner
        x         : input to the learner
    """
    # pop off the list the last dense block
    last = blocks.pop()

    # Create the dense groups and interceding transition blocks
    for n_blocks in blocks:
        x = dense_group(x, n_blocks, n_filters)
        x = trans_block(x, reduction)

    # Add the last dense group w/o a following transition block
    x = dense_block(x, last, n_filters)
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
    x = Conv2D(2 * n_filters, (7, 7), strides=(2, 2), use_bias=False)(x)
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
def dense_group(x, n_blocks, n_filters):
    """ Construct a Dense Block
        x         : input to the block
        n_blocks  : number of residual blocks in dense block
        n_filters : number of filters in convolution layer in residual block
    """
    # Construct a group of residual blocks
    for _ in range(n_blocks):
        x = residual_block(x, n_filters)
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
    x = Conv2D(n_filters, (1, 1), strides=(1, 1), use_bias=False)(x)

    # Use mean value (average) instead of max value sampling when pooling reduce by 75%
    x = AveragePooling2D((2, 2), strides=(2, 2))(x)
    return x
```

### Residual Block

<img src="residual-block.jpg">

```python
def residual_block(x, n_filters):
    """ Construct a Residual Block
        x        : input to the block
        n_filters: number of filters in convolution layer in residual block
    """
    # Remember input tensor into residual block
    shortcut = x 
    
    # BN-RE-Conv pre-activation form of convolutions

    # Dimensionality expansion, expand filters by 4 (DenseNet-B)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(4 * n_filters, (1, 1), strides=(1, 1), use_bias=False)(x)
    
    # Bottleneck convolution
    # 3x3 convolution with padding=same to preserve same shape of feature maps
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(n_filters, (3, 3), strides=(1, 1), padding='same', use_bias=False)(x)

    # Concatenate the input (identity) with the output of the residual block
    # Concatenation (vs. merging) provides Feature Reuse between layers
    x = Concatenate()([shortcut, x])
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
    # Global Average Pooling will flatten the 7x7 feature maps into 1D feature maps
    x = GlobalAveragePooling2D()(x)
    # Fully connected output layer (classification)
    x = Dense(n_classes, activation='softmax')(x)
    return x
```
