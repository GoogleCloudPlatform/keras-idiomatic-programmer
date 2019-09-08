
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

# number of residual blocks in each dense block
blocks = [6, 12, 24, 16]

# The input vector
inputs = Input(shape=(224, 224, 3))

# The Stem Convolution Group
x = stem(inputs, n_filters)

# The Learner
x = learner(x, blocks, n_filters, reduction)

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

    # Create the dense blocks and interceding transition blocks
    for n_blocks in blocks:
        x = dense_block(x, n_blocks, n_filters)
        x = trans_block(x, reduction)

    # Add the last dense block w/o a following transition block
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
    x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)))(inputs)
    
    # First large convolution for abstract features for input 224 x 224 and output 112 x 112
    # Stem convolution uses 2 * k (growth rate) number of filters
    x = layers.Conv2D(2 * n_filters, (7, 7), strides=(2, 2), use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # Add padding so when downsampling we fit shape 56 x 56
    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = layers.MaxPooling2D((3, 3), strides=2)(x)
    return x
```

### Dense Block

<img src="dense-block.jpg">

### Transitional Block

<img src="trans-block.jpg">

### Residual Block

<img src="residual-block.jpg">
