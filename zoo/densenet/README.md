
# DenseNet

    densenet.py - academic - procedural
    densenet_c.py - composable - OOP


[Paper](https://arxiv.org/pdf/1608.06993.pdf)

## Macro-Architecture

<img src='macro.jpg'>

Macro-architecture code for DenseNet 121:

```python
def learner(x, groups, n_filters, reduction):
    """ Construct the Learner
        x         : input to the learner
        groups    : set of number of blocks per group
        n_filters : number of filters (growth rate)
        reduction : the amount to reduce (compress) feature maps by
    """
    # pop off the list the last dense block
    last = groups.pop()

    # Create the dense groups and interceding transition blocks
    for n_blocks in groups:
        x = group(x, n_blocks, n_filters, reduction)

    # Add the last dense group w/o a following transition block
    x = group(x, last, n_filters)
    return x
    
# Meta-parameter: amount to reduce feature maps by (compression factor) during transition blocks
reduction = 0.5

# Meta-parameter: number of filters in a convolution block within a residual block (growth rate)
n_filters = 32

# Meta-parameter: number of residual blocks in each dense group
groups = { 121 : [6, 12, 24, 16],     # DenseNet 121
           169 : [6, 12, 32, 32],     # DenseNet 169
           201 : [6, 12, 48, 32] }    # DenseNet 201

# The input vector
inputs = Input(shape=(224, 224, 3))

# The Stem Convolution Group
x = stem(inputs, n_filters)

# The Learner
x = learner(x, groups[121], n_filters, reduction)

# Classifier for 1000 classes
outputs = classifier(x, 1000)

# Instantiate the model
model = Model(inputs, outputs)
```

## Micro-Architecture

<img src='micro.jpg'>

```python
def group(x, n_blocks, n_filters, reduction=None):
    """ Construct a Dense Group
        x         : input to the block
        n_blocks  : number of residual blocks in dense group
        n_filters : number of filters in convolution layer in residual block
        reduction : amount to reduce feature maps by
    """
    # Construct a group of densely connected residual blocks
    for _ in range(n_blocks):
        x = residual_block(x, n_filters)

    # Construct interceding transition block
    if reduction is not None:
        x = trans_block(x, reduction)
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
    x = Conv2D(2 * n_filters, (7, 7), strides=(2, 2), use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    # Add padding so when downsampling we fit shape 56 x 56
    x = ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = MaxPooling2D((3, 3), strides=2)(x)
    return x
```

### Dense Block

<img src="dense-block.jpg">

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
    x = Conv2D(n_filters, (1, 1), strides=(1, 1), use_bias=False, kernel_initializer='he_normal')(x)

    # Use mean value (average) instead of max value sampling when pooling reduce by 75%
    x = AveragePooling2D((2, 2), strides=(2, 2))(x)
    return x
```

### Residual Block

<img src="residual-block.jpg">

```python
def residual_block(x, n_filters):
    """ Construct a Densely Connected Residual Block
        x        : input to the block
        n_filters: number of filters in convolution layer in residual block
    """
    # Remember input tensor into residual block
    shortcut = x

    # BN-RE-Conv pre-activation form of convolutions

    # Dimensionality expansion, expand filters by 4 (DenseNet-B)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(4 * n_filters, (1, 1), strides=(1, 1), use_bias=False, kernel_initializer='he_normal')(x)

    # Bottleneck convolution
    # 3x3 convolution with padding=same to preserve same shape of feature maps
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(n_filters, (3, 3), strides=(1, 1), padding='same', use_bias=False, kernel_initializer='he_normal')(x)

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
    x = Dense(n_classes, activation='softmax', kernel_initializer='he_normal')(x)
    return x
```

## Composable

*Example Instantiate a DenseNet model*

```python
from densenet_c import DenseNet

# DenseNet121 from research paper
densenet = DenseNet(121)

# DenseNet121 custom input shape/classes
densenet = DenseNet(121, input_shape=(128, 128, 3), n_classes=50)

# getter for the tf.keras model
model = densenet.model
```

*Example: Composable Group/Block*

```python
# Make a mini-densenet for CIFAR-10
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, Flatten, Dense

# Stem
inputs = Input((32, 32, 3))
x = Conv2D(32, (3, 3), strides=1, padding='same', activation='relu')(inputs)

# Learner
# DenseNet group: 6 blocks, 32 filters, 50% reduction
# Residual block with 32 filters
# Transitional block with 50% reduction
# Residual block with 32 filters
x = DenseNet.group(x, n_blocks=6, n_filters=32, reduction=0.5)
x = DensetNet.residual_block(x, n_filters=32)
x = DenseNet.trans_block(x, reduction=0.5)
x = DensetNet.residual_block(x, n_filters=32)

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
batch_normalization_495 (BatchN (None, 16, 16, 128)  512         conv2d_815[0][0]                 
__________________________________________________________________________________________________
re_lu_485 (ReLU)                (None, 16, 16, 128)  0           batch_normalization_495[0][0]    
__________________________________________________________________________________________________
conv2d_816 (Conv2D)             (None, 16, 16, 32)   36864       re_lu_485[0][0]                  
__________________________________________________________________________________________________
concatenate_41 (Concatenate)    (None, 16, 16, 144)  0           average_pooling2d_2[0][0]        
                                                                 conv2d_816[0][0]                 
__________________________________________________________________________________________________
batch_normalization_496 (BatchN (None, 16, 16, 144)  576         concatenate_41[0][0]             
__________________________________________________________________________________________________
conv2d_817 (Conv2D)             (None, 16, 16, 72)   10368       batch_normalization_496[0][0]    
__________________________________________________________________________________________________
average_pooling2d_3 (AveragePoo (None, 8, 8, 72)     0           conv2d_817[0][0]                 
__________________________________________________________________________________________________
batch_normalization_497 (BatchN (None, 8, 8, 72)     288         average_pooling2d_3[0][0]        
__________________________________________________________________________________________________
re_lu_486 (ReLU)                (None, 8, 8, 72)     0           batch_normalization_497[0][0]    
__________________________________________________________________________________________________
conv2d_818 (Conv2D)             (None, 8, 8, 128)    9216        re_lu_486[0][0]                  
__________________________________________________________________________________________________
batch_normalization_498 (BatchN (None, 8, 8, 128)    512         conv2d_818[0][0]                 
__________________________________________________________________________________________________
re_lu_487 (ReLU)                (None, 8, 8, 128)    0           batch_normalization_498[0][0]    
__________________________________________________________________________________________________
conv2d_819 (Conv2D)             (None, 8, 8, 32)     36864       re_lu_487[0][0]                  
__________________________________________________________________________________________________
concatenate_42 (Concatenate)    (None, 8, 8, 104)    0           average_pooling2d_3[0][0]        
                                                                 conv2d_819[0][0]                 
__________________________________________________________________________________________________
flatten_2 (Flatten)             (None, 6656)         0           concatenate_42[0][0]             
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 10)           66570       flatten_2[0][0]                  
==================================================================================================
Total params: 516,394
Trainable params: 511,898
Non-trainable params: 4,496
__________________________________________________________________________________________________
```

```python
from tensorflow.keras.datasets import cifar10
import numpy as np

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = (x_train / 255.0).astype(np.float342)
x_test  = (x_test  / 255.0).astype(np.float342)
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=1)
```

```python
Epoch 1/10
45000/45000 [==============================] - 1833s 41ms/sample - loss: 1.4802 - acc: 0.5183 - val_loss: 1.2519 - val_acc: 0.5730
Epoch 2/10
45000/45000 [==============================] - 1930s 43ms/sample - loss: 0.9954 - acc: 0.6553 - val_loss: 1.0739 - val_acc: 0.6402
Epoch 3/10
45000/45000 [==============================] - 1928s 43ms/sample - loss: 0.8538 - acc: 0.7030 - val_loss: 0.9468 - val_acc: 0.6790
Epoch 4/10
45000/45000 [==============================] - 1812s 41ms/sample - loss: 0.7683 - acc: 0.7322 - val_loss: 1.1035 - val_acc: 0.6482
Epoch 5/10
45000/45000 [==============================] - 1738s 39ms/sample - loss: 0.6916 - acc: 0.7593 - val_loss: 0.9849 - val_acc: 0.6752
Epoch 6/10
45000/45000 [==============================] - 1746s 39ms/sample - loss: 0.6264 - acc: 0.7814 - val_loss: 0.8333 - val_acc: 0.7194
Epoch 7/10
45000/45000 [==============================] - 1667s 37ms/sample - loss: 0.5744 - acc: 0.7964 - val_loss: 0.7589 - val_acc: 0.7414
Epoch 8/10
45000/45000 [==============================] - 1738s 39ms/sample - loss: 0.5237 - acc: 0.8176 - val_loss: 0.7885 - val_acc: 0.7404
Epoch 9/10
45000/45000 [==============================] - 1743s 39ms/sample - loss: 0.4742 - acc: 0.8330 - val_loss: 0.8664 - val_acc: 0.7236
Epoch 10/10
45000/45000 [==============================] - 1681s 37ms/sample - loss: 0.4296 - acc: 0.8509 - val_loss: 0.7641 - val_acc: 0.7666
```
