
# SqueezeNet

    squeezenet(_bypass/_complex).py - academic - procedural
    squeezenet(_bypass/_complex)_c.py - composable - OOP

[Paper](https://arxiv.org/pdf/1602.07360.pdf)

## Macro-Architecture

<img src='macro.jpg'>

```python
def learner(x):
    ''' Construct the Learner
        x    : input to the learner
    '''
    # First fire group, progressively increase number of filters
    x = group(x, [16, 16, 32])

    # Second fire group
    x = group(x, [32, 48, 48, 64])

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
def group(x, filters):
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

```python
def fire_group(x, filters):
    ''' Construct the Fire Group
        x       : input to the group
        filters : list of number of filters per fire block in group
    '''
    for n_filters, bypass in filters:
        x = fire_block(x, n_filters)

    # Delayed downsampling
    x = MaxPooling2D((3, 3), strides=2)(x)
    return x
```

### Fire Bypass Block

<img src="bypass-block.jpg">

```python
def fire_block(x, n_filters, bypass=False):
    ''' Construct a Fire Block
        x        : input to the block
        n_filters: number of filters in the block
        bypass   : whether block has an identity shortcut
    '''
    # remember the input
    shortcut = x

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

    # if identity link, add (matrix addition) input filters to output filters
    if bypass:
        x = Add()([x, shortcut])

    return x
```

## Fire Complex Bypass Micro-Archirecture

<img src="micro-complex.jpg">

```python
def fire_group(x, filters):
    ''' Construct a Fire Group
        x      : input to the group
        filters: list of number of filters per block in group
    '''
    for n_filters in filters:
        x = fire_block(x, n_filters)

    # Delayed downsampling
    x = MaxPooling2D((3, 3), strides=2)(x)
    return x
```

### Fire Complex Bypass Block

<img src="complex-block.jpg">

```python
def fire_block(x, n_filters):
    ''' Construct a Fire Block  with complex bypass
        x        : input to the block
        n_filters: number of filters in block
    '''
    # remember the input (identity)
    shortcut = x

    # if the number of input filters does not equal the number of output filters, then use
    # a transition convolution to match the number of filters in identify link to output
    if shortcut.shape[3] != 8 * n_filters:
        shortcut = Conv2D(n_filters * 8, (1, 1), strides=1, activation='relu',
                          padding='same', kernel_initializer='glorot_uniform')(shortcut)

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

    # if identity link, add (matrix addition) input filters to output filters
    if shortcut is not None:
        x = Add()([x, shortcut])
    return x
```

## Composable

*Example Instantiate a SqueezeNet model*

```python
from squeezenet_c import SqueezeNet
# SqueezeNet from research paper
squeezenet = SqueezeNet()

# ResNeXt50 custom input shape/classes
squeezenet = SqueezeNet(input_shape=(128, 128, 3), n_classes=50)

# getter for the tf.keras model
model = squeezenet.model
```

*Example: Composable Group/Block*

```python
# Make mini-squeezenet for CIFAR-10
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, Flatten, Dense

# Stem
inputs = Input((32, 32, 3))
x = Conv2D(16, (3, 3), strides=1, padding='same', activation='relu')(inputs)

# Learner
# SqueezeNet group: 2 blocks, 16 and 32 filters
# SqueezeNet fire block with 64 filters
# SqueezeNet fire block with 128 filters
x = SqueezeNet.group(x, [16, 32])
x = SqueezeNet.fire_block(x, 64)
x = SqueezeNet.fire_block(x, 128)

# Classifier
x = Flatten()(x)
outputs = Dense(10, activation='softmax')(x)
model = Model(inputs, outputs)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.summary()
```

```python
# Removed for brevity
__________________________________________________________________________________________________
concatenate_68 (Concatenate)    (None, 32, 32, 128)  0           conv2d_1691[0][0]                
                                                                 conv2d_1692[0][0]                
__________________________________________________________________________________________________
conv2d_1693 (Conv2D)            (None, 32, 32, 32)   4128        concatenate_68[0][0]             
__________________________________________________________________________________________________
conv2d_1694 (Conv2D)            (None, 32, 32, 128)  4224        conv2d_1693[0][0]                
__________________________________________________________________________________________________
conv2d_1695 (Conv2D)            (None, 32, 32, 128)  36992       conv2d_1693[0][0]                
__________________________________________________________________________________________________
concatenate_69 (Concatenate)    (None, 32, 32, 256)  0           conv2d_1694[0][0]                
                                                                 conv2d_1695[0][0]                
__________________________________________________________________________________________________
max_pooling2d_1 (MaxPooling2D)  (None, 15, 15, 256)  0           concatenate_69[0][0]             
__________________________________________________________________________________________________
conv2d_1696 (Conv2D)            (None, 15, 15, 64)   16448       max_pooling2d_1[0][0]            
__________________________________________________________________________________________________
conv2d_1697 (Conv2D)            (None, 15, 15, 256)  16640       conv2d_1696[0][0]                
__________________________________________________________________________________________________
conv2d_1698 (Conv2D)            (None, 15, 15, 256)  147712      conv2d_1696[0][0]                
__________________________________________________________________________________________________
concatenate_70 (Concatenate)    (None, 15, 15, 512)  0           conv2d_1697[0][0]                
                                                                 conv2d_1698[0][0]                
__________________________________________________________________________________________________
conv2d_1699 (Conv2D)            (None, 15, 15, 128)  65664       concatenate_70[0][0]             
__________________________________________________________________________________________________
conv2d_1700 (Conv2D)            (None, 15, 15, 512)  66048       conv2d_1699[0][0]                
__________________________________________________________________________________________________
conv2d_1701 (Conv2D)            (None, 15, 15, 512)  590336      conv2d_1699[0][0]                
__________________________________________________________________________________________________
concatenate_71 (Concatenate)    (None, 15, 15, 1024) 0           conv2d_1700[0][0]                
                                                                 conv2d_1701[0][0]                
__________________________________________________________________________________________________
flatten_6 (Flatten)             (None, 230400)       0           concatenate_71[0][0]             
__________________________________________________________________________________________________
dense_522 (Dense)               (None, 10)           2304010     flatten_6[0][0]                  
==================================================================================================
Total params: 3,263,994
Trainable params: 3,263,994
Non-trainable params: 0
```

```python
from tensorflow.keras.datasets import cifar10
import numpy as np

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = (x_train / 255.0).astype(np.float32)
x_test  = (x_test  / 255.0).astype(np.float32)
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=1)
```

```python
Epoch 1/10
45000/45000 [==============================] - 380s 8ms/sample - loss: 1.6933 - acc: 0.3904 - val_loss: 1.3306 - val_acc: 0.5358
Epoch 2/10
45000/45000 [==============================] - 366s 8ms/sample - loss: 1.1897 - acc: 0.5774 - val_loss: 1.0908 - val_acc: 0.6182
Epoch 3/10
45000/45000 [==============================] - 371s 8ms/sample - loss: 0.9708 - acc: 0.6584 - val_loss: 0.9541 - val_acc: 0.6626
Epoch 4/10
45000/45000 [==============================] - 391s 9ms/sample - loss: 0.8118 - acc: 0.7144 - val_loss: 0.9033 - val_acc: 0.6830
Epoch 5/10
45000/45000 [==============================] - 390s 9ms/sample - loss: 0.6768 - acc: 0.7644 - val_loss: 0.9297 - val_acc: 0.6876
Epoch 6/10
45000/45000 [==============================] - 394s 9ms/sample - loss: 0.5428 - acc: 0.8097 - val_loss: 1.0250 - val_acc: 0.6898
Epoch 7/10
45000/45000 [==============================] - 394s 9ms/sample - loss: 0.4018 - acc: 0.8588 - val_loss: 1.0728 - val_acc: 0.6838
Epoch 8/10
45000/45000 [==============================] - 466s 10ms/sample - loss: 0.2890 - acc: 0.8977 - val_loss: 1.2819 - val_acc: 0.6714
Epoch 9/10
45000/45000 [==============================] - 391s 9ms/sample - loss: 0.2152 - acc: 0.9231 - val_loss: 1.4663 - val_acc: 0.6716
Epoch 10/10
45000/45000 [==============================] - 386s 9ms/sample - loss: 0.1622 - acc: 0.9431 - val_loss: 1.6893 - val_acc: 0.6702
```
