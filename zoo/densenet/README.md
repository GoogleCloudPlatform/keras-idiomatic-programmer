
# DenseNet

[Paper](https://arxiv.org/pdf/1608.06993.pdf)

## Macro-Architecture

<img src='macro.jpg'>

Macro-architecture code for DenseNet 121:

```python
# Meta-parameter: amount to reduce feature maps by (compression) during transition blocks
reduce_by = 0.5

# Meta-parameter: number of filters in a convolution block within a residual block
n_filters = 32

# number of residual blocks in each dense block
blocks = [6, 12, 24, 16]

# pop off the list the last dense block
last   = blocks.pop()

# The input vector
inputs = Input(shape=(230, 230, 3))

# The Stem Convolution Group
x = stem(inputs)

# The learner
x = learner(x, blocks, n_filters, reduce_by)

# Classifier for 1000 classes
outputs = classifier(x, 1000)

# Instantiate the model
model = Model(inputs, outputs)
```

## Micro-Architecture

<img src='micro.jpg'>

```python
def learner(x, blocks, n_filters, reduce_by):
    """ Construct the Learner
        x         : input to the learner
    """
    # pop off the list the last dense block
    last = blocks.pop()

    # Create the dense blocks and interceding transition blocks
    for n_blocks in blocks:
        x = dense_block(x, n_blocks, n_filters)
        x = trans_block(x, reduce_by)

    # Add the last dense block w/o a following transition block
    x = dense_block(x, last, n_filters)
    return x
```

### Stem Group

<img src="stem.jpg">

```python
def stem(inputs):
    """ Construct the Stem Convolution Group
        inputs : input tensor
    """
    # First large convolution for abstract features for input 230 x 230 and output 112 x 112
    x = layers.Conv2D(64, (7, 7), strides=2)(inputs)
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
