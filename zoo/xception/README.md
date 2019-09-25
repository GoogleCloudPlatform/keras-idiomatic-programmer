
# Xception

    xception.py (procedural - academic)
    xception_c.py (OOP - composable)

[Paper](https://arxiv.org/pdf/1610.02357.pdf)

## Macro-Architecture

<img src='macro.jpg'>

```python
# Create the input vector
inputs = Input(shape=(299, 299, 3))

# Create entry section
x = entryFlow(inputs)

# Create the middle section
x = middleFlow(x)

# Create the exit section for 1000 classes
outputs = exitFlow(x, 1000)

# Instantiate the model
model = Model(inputs, outputs)
```

## Micro-Architecture - Entry Flow

<img src='micro-entry.jpg'>

```python
def entryFlow(inputs):
    """ Create the entry flow section
        inputs : input tensor to neural network
    """

    def stem(inputs):
        """ Create the stem entry into the neural network
            inputs : input tensor to neural network
        """
        # Strided convolution - dimensionality reduction
        # Reduce feature maps by 75%
        x = Conv2D(32, (3, 3), strides=(2, 2))(inputs)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        # Convolution - dimensionality expansion
        # Double the number of filters
        x = Conv2D(64, (3, 3), strides=(1, 1))(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        return x

    # Create the stem to the neural network
    x = stem(inputs)

    # Create three residual blocks using linear projection
    for n_filters in [128, 256, 728]:
        x = projection_block(x, n_filters)

    return x
```

### Entry Flow Stem Group

<img src="stem.jpg">

### Entry Flow Block

<img src="block-projection.jpg">

```python
def projection_block(x, n_filters):
    """ Create a residual block using Depthwise Separable Convolutions with Projection shortcut
        x        : input into residual block
        n_filters: number of filters
    """
    # Remember the input
    shortcut = x
    
    # Strided convolution to double number of filters in identity link to
    # match output of residual block for the add operation (projection shortcut)
    shortcut = Conv2D(n_filters, (1, 1), strides=(2, 2), padding='same')(shortcut)
    shortcut = BatchNormalization()(shortcut)

    # First Depthwise Separable Convolution
    x = SeparableConv2D(n_filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Second depthwise Separable Convolution
    x = SeparableConv2D(n_filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Create pooled feature maps, reduce size by 75%
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # Add the projection shortcut to the output of the block
    x = Add()([x, shortcut])

    return x
```

## Micro-Architecture - Middle Flow

<img src="micro-middle.jpg">

```python
def middleFlow(x):
    """ Create the middle flow section
        x : input tensor into section
    """
    # Create 8 residual blocks
    for _ in range(8):
        x = residual_block(x, 728)
    return x
```

### Middle Flow Block

<img src="block-middle.jpg">

```python
def residual_block(x, n_filters):
    """ Create a residual block using Depthwise Separable Convolutions
        x        : input into residual block
        n_filters: number of filters
    """
    # Remember the input
    shortcut = x

    # First Depthwise Separable Convolution
    x = SeparableConv2D(n_filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Second depthwise Separable Convolution
    x = SeparableConv2D(n_filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Third depthwise Separable Convolution
    x = SeparableConv2D(n_filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    # Add the identity link to the output of the block
    x = Add()([x, shortcut])
    return x
```

## Micro-Architecture - Exit Flow

<img src="micro-exit.jpg">

```python
def exitFlow(x, n_classes):
    """ Create the exit flow section
        x         : input to the exit flow section
        n_classes : number of output classes
    """
    def classifier(x, n_classes):
        """ The output classifier
            x         : input to the classifier
            n_classes : number of output classes
        """
        # Global Average Pooling will flatten the 10x10 feature maps into 1D
        # feature maps
        x = GlobalAveragePooling2D()(x)

        # Fully connected output layer (classification)
        x = Dense(n_classes, activation='softmax')(x)
        return x

    # Remember the input
    shortcut = x

    # Strided convolution to double number of filters in identity link to
    # match output of residual block for the add operation (projection shortcut)
    shortcut = Conv2D(1024, (1, 1), strides=(2, 2), padding='same')(shortcut)
    shortcut = BatchNormalization()(shortcut)

    # First Depthwise Separable Convolution
    # Dimensionality reduction - reduce number of filters
    x = SeparableConv2D(728, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)

    # Second Depthwise Separable Convolution
    # Dimensionality restoration
    x = SeparableConv2D(1024, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Create pooled feature maps, reduce size by 75%
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # Add the projection shortcut to the output of the pooling layer
    x = Add()([x, shortcut])

    # Third Depthwise Separable Convolution
    x = SeparableConv2D(1556, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Fourth Depthwise Separable Convolution
    x = SeparableConv2D(2048, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Create classifier section
    x = classifier(x, n_classes)

    return x
```

### Exit Flow Residual Block

<img src="block-exit-residual.jpg">

### Exit Flow Convolutional Block

<img src="block-exit-conv.jpg">

### Exit Flow Classifier

<img src="classifier.jpg">

## Composable

*Example Instantiate a Xception model*

```python
from xception_c import Xception

# Xception from research paper
xception = Xception()

# Xception custom input shape/classes
xception = Xception(input_shape=(128, 128, 3), n_classes=50)

# getter for the tf.keras model
model = xception.model
```

*Example: Composable Group/Block*

```python
# Make a mini-xception for CIFAR-10
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, Flatten, Dense

# Stem
inputs = Input((32, 32, 3))
x = Conv2D(32, (3, 3), strides=1, padding='same', activation='relu')(inputs)

# Learner
# Xception entry: 
# Xception middle:
# Xception exit:
x = Xception.entry(x, [64, 128, 256])
x = Xception.middle(x, [256, 256, 512])


# Classifier
outputs = Xception.exit(x, ??)
model = Model(inputs, outputs)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.summary()
```

```python
```
