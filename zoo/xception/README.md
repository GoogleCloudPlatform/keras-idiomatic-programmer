
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
x = Xception.entry(x, [16, 32, 64])
x = Xception.middle(x, [64, 64, 64])


# Classifier
outputs = Xception.exit(x, n_classes=10)
model = Model(inputs, outputs)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.summary()
```

```python
# REMOVED for Brevity

batch_normalization_117 (BatchN (None, 2, 2, 1024)   4096        separable_conv2d_85[0][0]        
__________________________________________________________________________________________________
re_lu_94 (ReLU)                 (None, 2, 2, 1024)   0           batch_normalization_117[0][0]    
__________________________________________________________________________________________________
conv2d_32 (Conv2D)              (None, 1, 1, 1024)   66560       add_32[0][0]                     
__________________________________________________________________________________________________
max_pooling2d_19 (MaxPooling2D) (None, 1, 1, 1024)   0           re_lu_94[0][0]                   
__________________________________________________________________________________________________
batch_normalization_115 (BatchN (None, 1, 1, 1024)   4096        conv2d_32[0][0]                  
__________________________________________________________________________________________________
add_33 (Add)                    (None, 1, 1, 1024)   0           max_pooling2d_19[0][0]           
                                                                 batch_normalization_115[0][0]    
__________________________________________________________________________________________________
separable_conv2d_86 (SeparableC (None, 1, 1, 1556)   1604116     add_33[0][0]                     
__________________________________________________________________________________________________
batch_normalization_118 (BatchN (None, 1, 1, 1556)   6224        separable_conv2d_86[0][0]        
__________________________________________________________________________________________________
re_lu_95 (ReLU)                 (None, 1, 1, 1556)   0           batch_normalization_118[0][0]    
__________________________________________________________________________________________________
separable_conv2d_87 (SeparableC (None, 1, 1, 2048)   3202740     re_lu_95[0][0]                   
__________________________________________________________________________________________________
batch_normalization_119 (BatchN (None, 1, 1, 2048)   8192        separable_conv2d_87[0][0]        
__________________________________________________________________________________________________
re_lu_96 (ReLU)                 (None, 1, 1, 2048)   0           batch_normalization_119[0][0]    
__________________________________________________________________________________________________
global_average_pooling2d_2 (Glo (None, 2048)         0           re_lu_96[0][0]                   
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 10)           20490       global_average_pooling2d_2[0][0] 
==================================================================================================
Total params: 5,801,314
Trainable params: 5,786,538
Non-trainable params: 14,776
```

from tensorflow.keras.datasets import cifar10
import numpy as np

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = (x_train / 255.0).astype(np.float342)
x_test  = (x_test  / 255.0).astype(np.float342)
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=1)
```python

```python
Epoch 1/5
45000/45000 [==============================] - 467s 57ms/sample - loss: 1.5985 - acc: 0.4127 - val_loss: 1.8141 - val_acc: 0.3506
Epoch 2/5
45000/45000 [==============================] - 461s 10ms/sample - loss: 1.3191 - acc: 0.5251 - val_loss: 1.3030 - val_acc: 0.5358
Epoch 3/5
45000/45000 [==============================] - 463s 28ms/sample - loss: 1.1777 - acc: 0.5784 - val_loss: 1.3208 - val_acc: 0.5198
```
