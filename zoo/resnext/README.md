
# ResNeXt

[Paper](https://arxiv.org/pdf/1611.05431.pdf)

## Macro-Architecture

<img src='macro.jpg'>

```python
# The input tensor
inputs = layers.Input(shape=(224, 224, 3))

# The Stem Group
x = stem(inputs)

# First ResNeXt Group
x = residual_group(x, 128, 256, 2, strides=(1, 1))

# Second ResNeXt
x = residual_group(x, 256, 512, 3)

# Third ResNeXt Group
x = residual_group(x, 512, 1024, 5)

# Fourth ResNeXt Group
x = residual_group(x, 1024, 2048, 2)

# The Classifier for 1000 classes
outputs = classifier(x, 1000)

# Instantiate the Model
model = Model(inputs, outputs)
```

## Micro-Architecture

<img src='micro.jpg'>

```python
# The input tensor
inputs = layers.Input(shape=(224, 224, 3))

# The Stem Group
x = stem(inputs)

# First ResNeXt Group
x = residual_group(x, 128, 256, 2, strides=(1, 1))

# Second ResNeXt
x = residual_group(x, 256, 512, 3)

# Third ResNeXt Group
x = residual_group(x, 512, 1024, 5)

# Fourth ResNeXt Group
x = residual_group(x, 1024, 2048, 2)

# The Classifier for 1000 classes
outputs = classifier(x, 1000)

# Instantiate the Model
model = Model(inputs, outputs)
```

### Stem Group

<img src="stem.jpg">

```python
def stem(inputs):
    """ Construct the Stem Convolution Group
        inputs : input vector
    """
    x = layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same', kernel_initializer='he_normal', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    return x
```

### ResNeXt Block with Identity Shortcut

<img src='identity-block.jpg'>

```python
def bottleneck_block(x, filters_in, filters_out, cardinality=32):
    """ Construct a ResNeXT block with identity link
        x          : input to block
        filters_in : number of filters  (channels) at the input convolution
        filters_out: number of filters (channels) at the output convolution
        cardinality: width of cardinality layer
    """

    # Remember the input
    shortcut = x

    # Dimensionality Reduction
    x = layers.Conv2D(filters_in, kernel_size=(1, 1), strides=(1, 1),
                      padding='same', kernel_initializer='he_normal', use_bias=False)(shortcut)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Cardinality (Wide) Layer (split-transform)
    filters_card = filters_in // cardinality
    groups = []
    for i in range(cardinality):
        group = layers.Lambda(lambda z: z[:, :, :, i * filters_card:i *
                              filters_card + filters_card])(x)
        groups.append(layers.Conv2D(filters_card, kernel_size=(3, 3), strides=(1, 1),
                                    padding='same', kernel_initializer='he_normal', use_bias=False)(group))

    # Concatenate the outputs of the cardinality layer together (merge)
    x = layers.concatenate(groups)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Dimensionality restoration
    x = layers.Conv2D(filters_out, kernel_size=(1, 1), strides=(1, 1),
                      padding='same', kernel_initializer='he_normal', use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    # Identity Link: Add the shortcut (input) to the output of the block
    x = layers.add([shortcut, x])
    x = layers.ReLU()(x)
    return x
```

### ResNeXt Block with Projection Shortcut

<img src='projection-block.jpg'>

### Cardinality

<img src='cardinality.jpg'>
