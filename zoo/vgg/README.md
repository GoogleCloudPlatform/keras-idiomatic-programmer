
# VGG

[Paper](https://arxiv.org/pdf/1409.1556.pdf)

## Macro-Architecture

<img src='macro.jpg'>

Macro-architecture for VGG16:

```python
def learner(x):
    """ Construct the (Feature) Learner
        x        : input to the learner
    """
    # The convolutional blocks
    x = conv_block(x, 1, 64)
    x = conv_block(x, 2, 128)
    x = conv_block(x, 3, 256)
    x = conv_block(x, 3, 512)
    x = conv_block(x, 3, 512)
    return x
    
# The input vector 
inputs = Input( (224, 224, 3) )

# The stem group
x = stem(inputs)

# The learner
x = learner(x)

# The classifier
outputs = classifier(x, 1000)
```

## Micro-Architecture 

### Convolutional Group (Block)

<img src='micro-conv.jpg'>

```python
def conv_block(x, n_layers, n_filters):
    """ Construct a Convolutional Block
        x        : input to the block
        n_layers : number of convolutional layers
        n_filters: number of filters
    """
    # Block of convolutional layers
    for n in range(n_layers):
        x = Conv2D(n_filters, (3, 3), strides=(1, 1), padding="same", activation="relu")(x)
        
    # Max pooling at the end of the block
    x = MaxPooling2D(2, strides=(2, 2))(x)
    return x
```

### Stem Group

<img src='stem.jpg'>

```python
def stem(inputs):
    """ Construct the Stem Convolutional Group
        inputs : the input vector
    """
    x = Conv2D(64, (3, 3), strides=(1, 1), padding="same", activation="relu")(inputs)
    return x
```

### Classifier Group

<img src='classifier.jpg'>

```python
def classifier(x, n_classes):
    """ Construct the Classifier
        x         : input to the classifier
        n_classes : number of output classes
    """
    # Flatten the feature maps
    x = Flatten()(x)
    
    # Two fully connected dense layers
    x = Dense(4096, activation='relu')(x)
    x = Dense(4096, activation='relu')(x)

    # Output layer for classification 
    x = Dense(n_classes, activation='softmax')(x)
    return x
```
