
# VGG

    vgg.py - academic - procedural
    vgg_c.py - composable - OOP

[Paper](https://arxiv.org/pdf/1409.1556.pdf)

## Macro-Architecture

<img src='macro.jpg'>

```python
def learner(x, blocks):
    """ Construct the (Feature) Learner
        x        : input to the learner
        blocks   : list of groups: filter size and number of conv layers
    """
    # The convolutional groups
    for n_layers, n_filters in blocks:
        x = group(x, n_layers, n_filters)
    return x
    
# Meta-parameter: list of groups: number of layers and filter size
groups = { 16 : [ (1, 64), (2, 128), (3, 256), (3, 512), (3, 512) ],          # VGG16
           19 : [ (1, 64), (2, 128), (4, 256), (4, 256), (4, 256) ] }         # VGG19

# The input vector
inputs = Input( (224, 224, 3) )

# The stem group
x = stem(inputs)

# The learner
x = learner(x, groups[16])

# The classifier
outputs = classifier(x, 1000)

# Instantiate the Model
model = Model(inputs, outputs)
```

## Micro-Architecture 

### Convolutional Group (Block)

<img src='micro-conv.jpg'>

```python
def group(x, n_layers, n_filters):
    """ Construct a Convolutional Group
        x        : input to the group
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

## Composable

*Example Instantiate a VGG model*

```python
# VGG16 from research paper
vgg = VGG(16)

# VGG16 custom input shape/classes
vgg = VGG(16, input_shape=(128, 128, 3), n_classes=50)

# getter for the tf.keras model
model = vgg.model
```

*Example: Composable Group*/

```python
inputs = Input((32, 32, 3))
x = Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
# VGG group: 1 conv layer, 128 filters
x = VGG.group(x, 1, 128)
# VGG group: 2 conv layers, 256 filters
x = VGG.group(x, 2, 256)
x = Flatten()(x)
x = Dense(50, activation='softmax')
```

