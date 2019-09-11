
# SqueezeNet

[Paper](https://arxiv.org/pdf/1602.07360.pdf)

## Macro-Architecture

<img src='macro.jpg'>

```python
def learner(x):
    ''' Construct the Learner
        x    : input to the learner
    '''
    # First fire group, progressively increase number of filters
    x = fire_group(x, [16, 16, 32])

    # Second fire group
    x = fire_group(x, [32, 48, 48, 64])

    # Last fire block (module)
    x = fire_block(x, 64)

    # Dropout is delayed to end of fire blocks (modules)
    x = Dropout(0.5)(x)
    return x

# The input shape
inputs = Input((224, 224, 3))

# The Stem Group
x = stem(inputs)

# The Learner
x = learner(x)

# The classifier
outputs = classifier(x, 1000)

# Instantiate the Model
model = Model(inputs, outputs)
```

## Micro-Architecture

<img src='micro.jpg'>

```python
def fire_group(x, filters):
    ''' Construct a Fire Group
        x     : input to the group
        filters: list of number of filters per fire block (module)
    '''
    # Add the fire blocks (modules) for this group
    for n_filters in filters:
        x = fire_block(x, n_filters)

    # Delayed downsampling
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    return x
```

### Stem Group

<img src="stem.jpg">

```python
def stem(inputs):
    ''' Construct the Stem Group  '''
    x = Conv2D(96, (7, 7), strides=2, padding='same', activation='relu',
               kernel_initializer='glorot_uniform')(inputs)
    x = MaxPooling2D(3, strides=2)(x)
    return x
```

### Fire Block

<img src="fire.jpg">

```python
def fire_block(x, n_filters):
    ''' Construct a Fire Module
        x        : input to the module
        n_filters: number of filters
    '''
    # squeeze layer
    squeeze = Conv2D(n_filters, (1, 1), strides=1, activation='relu',
                     padding='same', kernel_initializer='glorot_uniform')(x)

    # branch the squeeze layer into a 1x1 and 3x3 convolution and double the number
    # of filters
    expand1x1 = Conv2D(n_filters * 4, (1, 1), strides=1, activation='relu',
                      padding='same', kernel_initializer='glorot_uniform')(squeeze)
    expand3x3 = Conv2D(n_filters * 4, (3, 3), strides=1, activation='relu',
                      padding='same', kernel_initializer='glorot_uniform')(squeeze)

    # concatenate the feature maps from the 1x1 and 3x3 branches
    x = Concatenate()([expand1x1, expand3x3])
    return x
```

### Classifier Group

<img src="classifier.jpg">

```python
def classifier(x, n_classes):
    ''' Construct the Classifier
        x        : input to the classifier
        n_classes: number of output classes
    '''
    # set the number of filters equal to number of classes
    x = Conv2D(n_classes, (1, 1), strides=1, activation='relu', padding='same',
               kernel_initializer='glorot_uniform')(x)

    # reduce each filter (class) to a single value
    x = GlobalAveragePooling2D()(x)
    x = Activation('softmax')(x)
    return x
```

## Fire Bypass Micro-Archirecture

<img src="micro-bypass.jpg">

### Fire Bypass Block

<img src="bypass-block.jpg">

## Fire Complex Bypass Micro-Archirecture

<img src="micro-complex.jpg">

### Fire Complex Bypass Block

<img src="complex-block.jpg">



