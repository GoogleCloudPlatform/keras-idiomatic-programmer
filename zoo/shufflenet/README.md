
# ShuffleNet v1.0

    shufflenet.py - academic - procedural
    shufflenet_c.py - composable - OOP

[Paper](https://arxiv.org/pdf/1707.01083.pdf)

## Macro-Architecture

<img src='macro.png'>

```python
def learner(x, groups, n_partitions, filters, reduction):
    ''' Construct the Learner
	x           : input to the learner
        groups      : number of shuffle blocks per shuffle group
        n_partitions: number of groups to partition feature maps (channels) into.
        filters     : number of filters per shuffle group
        reduction   : dimensionality reduction on entry to a shuffle block
    '''
    # Assemble the shuffle groups
    for i in range(3):
        x = group(x, n_partitions, groups[i], filters[i+1], reduction)
    return x
    
# meta-parameter: The number of groups to partition the filters (channels)
n_partitions=2

# meta-parameter: number of groups to partition feature maps (key), and
# corresponding number of output filters (value)
filters = {
        1: [24, 144, 288, 576],
        2: [24, 200, 400, 800],
        3: [24, 240, 480, 960],
        4: [24, 272, 544, 1088],
        8: [24, 384, 768, 1536]
}

# meta-parameter: the dimensionality reduction on entry to a shuffle block
reduction = 0.25

# meta-parameter: number of shuffle blocks per shuffle group
groups = [4, 8, 4 ]

# input tensor
inputs = Input( (224, 224, 3) )

# The Stem convolution group (referred to as Stage 1)
x = stem(inputs)

# The Learner
x = learner(x, groups, n_groups, filters[n_partitions], reduction)
```

## Micro-Architecture

<img src='micro.png'>

```python
def group(x, n_partitions, n_blocks, n_filters, reduction):
    ''' Construct a Shuffle Group 
        x           : input to the group
        n_partitions: number of groups to partition feature maps (channels) into.
        n_blocks    : number of shuffle blocks for this group
        n_filters   : number of output filters
        reduction   : dimensionality reduction
    '''
    
    # first block is a strided shuffle block
    x = strided_shuffle_block(x, n_partitions, n_filters, reduction)
    
    # remaining shuffle blocks in group
    for _ in range(n_blocks-1):
        x = shuffle_block(x, n_partitions, n_filters, reduction)
    return x
```

### Stem Group

<img src="stem.jpg">

```python
def stem(inputs):
    ''' Construct the Stem Convolution Group 
        inputs : input image (tensor)
    '''
    x = Conv2D(24, (3, 3), strides=2, padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D((3, 3), strides=2, padding='same')(x)
    return x
```

### Shuffle Block

<img src='block.png'>

```python

    
def shuffle_block(x, n_partitions, n_filters, reduction):
    ''' Construct a shuffle Shuffle block  
        x           : input to the block
        n_partitions: number of groups to partition feature maps (channels) into.
        n_filters   : number of filters
        reduction   : dimensionality reduction factor (e.g, 0.25)
    '''
    # identity shortcut
    shortcut = x
    
    # pointwise group convolution, with dimensionality reduction
    x = pw_group_conv(x, n_partitions, int(reduction * n_filters))
    x = ReLU()(x)
    
    # channel shuffle layer
    x = channel_shuffle(x, n_partitions)
    
    # Depthwise 3x3 Convolution
    x = DepthwiseConv2D((3, 3), strides=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    
    # pointwise group convolution, with dimensionality restoration
    x = pw_group_conv(x, n_paritions, n_filters)
    
    # Add the identity shortcut (input added to output)
    x = Add()([shortcut, x])
    x = ReLU()(x)
    return x

def pw_group_conv(x, n_partitions, n_filters):
    ''' A Pointwise Group Convolution  
        x           : input tensor
        n_partitions: number of groups to partition feature maps (channels) into.
        n_filers    : number of filters
    '''
    # Calculate the number of input filters (channels)
    in_filters = x.shape[3]

    # Derive the number of input filters (channels) per group
    grp_in_filters  = in_filters // n_partitions
    # Derive the number of output filters per group (Note the rounding up)
    grp_out_filters = int(n_filters / n_partitions + 0.5)
      
    # Perform convolution across each channel group
    groups = []
    for i in range(n_groups):
        # Slice the input across channel group
        group = Lambda(lambda x: x[:, :, :, grp_in_filters * i: grp_in_filters * (i + 1)])(x)

        # Perform convolution on channel group
        conv = Conv2D(grp_out_filters, (1,1), padding='same', strides=1, use_bias=False)(group)
        # Maintain the point-wise group convolutions in a list
        groups.append(conv)

    # Concatenate the outputs of the group pointwise convolutions together
    x = Concatenate()(groups)
    # Do batch normalization of the concatenated filters (feature maps)
    x = BatchNormalization()(x)
    return x
    
def channel_shuffle(x, n_partitions):
    ''' Implements the channel shuffle layer 
        x           : input tensor
        n_partitions: number of groups to partition feature maps (channels) into.
    '''
    # Get dimensions of the input tensor
    batch, height, width, n_filters = x.shape

    # Derive the number of input filters (channels) per group
    grp_in_filters  = n_filters // n_partitions

    # Separate out the channel groups
    x = Lambda(lambda z: K.reshape(z, [-1, height, width, n_partitions, grp_in_filters]))(x)
    # Transpose the order of the channel groups (i.e., 3, 4 => 4, 3)
    x = Lambda(lambda z: K.permute_dimensions(z, (0, 1, 2, 4, 3)))(x)
    # Restore shape
    x = Lambda(lambda z: K.reshape(z, [-1, height, width, n_filters]))(x)
    return x
```

### Strided Shuffle Block

<img src='strided-block.png'>

```python
def strided_shuffle_block(x, n_partitions, n_filters, reduction):
    ''' Construct a Strided Shuffle Block 
        x           : input to the block
        n_partitions: number of groups to partition feature maps (channels) into.
        n_filters   : number of filters
        reduction   : dimensionality reduction factor (e.g, 0.25)
    '''
    # projection shortcut
    shortcut = x
    shortcut = AveragePooling2D((3, 3), strides=2, padding='same')(shortcut)   
    
    # On entry block, we need to adjust the number of output filters
    # of the entry pointwise group convolution to match the exit
    # pointwise group convolution, by subtracting the number of input filters
    n_filters -= int(x.shape[3])
    
    # pointwise group convolution, with dimensionality reduction
    x = pw_group_conv(x, n_partitions, int(reduction * n_filters))
    x = ReLU()(x)
    
    # channel shuffle layer
    x = channel_shuffle(x, n_partitions)

    # Depthwise 3x3 Strided Convolution
    x = DepthwiseConv2D((3, 3), strides=2, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)

    # pointwise group convolution, with dimensionality restoration
    x = pw_group_conv(x, n_paritions, n_filters)
    
    # Concatenate the projection shortcut to the output
    x = Concatenate()([shortcut, x])
    x = ReLU()(x)
    return x
```

### Classifier

<img src='classifier.jpg'>

```python
def classifier(x, n_classes):
    ''' Construct the Classifier Group 
        x         : input to the classifier
        n_classes : number of output classes
    '''
    # Use global average pooling to flatten feature maps to 1D vector, where
    # each feature map is a single averaged value (pixel) in flatten vector
    x = GlobalAveragePooling2D()(x)
    x = Dense(n_classes, activation='softmax')(x)
    return x
```

## Composable

*Example Instantiate a ShuffleNet model*

```python
from shufflenet_c import ShuffleNet

# ShuffleNet v1 from research paper
shufflenet = ShuffleNet()

# ShuffleNet v1 custom input shape/classes
shufflenet = ShuffleNet(input_shape=(128, 128, 3), n_classes=50)

# getter for the tf.keras model
model = shufflenet.model
```

*Example: Composable Group/Block*

```python
# Make a mini-shufflenet for CIFAR-10
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, Flatten, Dense

# Stem
inputs = Input((32, 32, 3))
x = Conv2D(32, (3, 3), strides=1, padding='same', activation='relu')(inputs)

# Learner
# Shuffle Group: 2 partitions, 4 blocks, 256 filters
x = ShuffleNet.group(x, 2, 4, 128, reduction=0.25)
# Shuffle Block: 2 partitions, 256 filters, strided
# Shuffle Block: 2 partitions, 256 filters, non-strided
x = ShuffleNet.strided_shuffle_block(x, 2, 256, reduction=0.25)
x = ShuffleNet.shuffle_block(x, 2, 256, reduction=0.25)

# Classifier
x = Flatten()(x)
outputs = Dense(10, activation='softmax')(x)
model = Model(inputs, outputs)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.summary()
```

```python
batch_normalization_1373 (Batch (None, 8, 8, 64)     256         concatenate_64[0][0]             
__________________________________________________________________________________________________
re_lu_1316 (ReLU)               (None, 8, 8, 64)     0           batch_normalization_1373[0][0]   
__________________________________________________________________________________________________
lambda_385 (Lambda)             (None, 8, 8, 2, 32)  0           re_lu_1316[0][0]                 
__________________________________________________________________________________________________
lambda_386 (Lambda)             (None, 8, 8, 32, 2)  0           lambda_385[0][0]                 
__________________________________________________________________________________________________
lambda_387 (Lambda)             (None, 8, 8, 64)     0           lambda_386[0][0]                 
__________________________________________________________________________________________________
depthwise_conv2d_43 (DepthwiseC (None, 8, 8, 64)     576         lambda_387[0][0]                 
__________________________________________________________________________________________________
batch_normalization_1374 (Batch (None, 8, 8, 64)     256         depthwise_conv2d_43[0][0]        
__________________________________________________________________________________________________
lambda_388 (Lambda)             (None, 8, 8, 32)     0           batch_normalization_1374[0][0]   
__________________________________________________________________________________________________
lambda_389 (Lambda)             (None, 8, 8, 32)     0           batch_normalization_1374[0][0]   
__________________________________________________________________________________________________
conv2d_1678 (Conv2D)            (None, 8, 8, 128)    4096        lambda_388[0][0]                 
__________________________________________________________________________________________________
conv2d_1679 (Conv2D)            (None, 8, 8, 128)    4096        lambda_389[0][0]                 
__________________________________________________________________________________________________
concatenate_65 (Concatenate)    (None, 8, 8, 256)    0           conv2d_1678[0][0]                
                                                                 conv2d_1679[0][0]                
__________________________________________________________________________________________________
batch_normalization_1375 (Batch (None, 8, 8, 256)    1024        concatenate_65[0][0]             
__________________________________________________________________________________________________
add_436 (Add)                   (None, 8, 8, 256)    0           re_lu_1315[0][0]                 
                                                                 batch_normalization_1375[0][0]   
__________________________________________________________________________________________________
re_lu_1317 (ReLU)               (None, 8, 8, 256)    0           add_436[0][0]                    
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 16384)        0           re_lu_1317[0][0]                 
__________________________________________________________________________________________________
dense_521 (Dense)               (None, 10)           163850      flatten_5[0][0]                  
==================================================================================================
Total params: 206,178
Trainable params: 203,586
Non-trainable params: 2,59
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
```
