
# ResNeXt

	resnext.py - academic - procedural
	resnext_cifar10.py - academic - procedural
	resnext_c.py - composable - OOP

[Paper](https://arxiv.org/pdf/1611.05431.pdf)

## Macro-Architecture

<img src='macro.jpg'>

```python
def learner(x, groups, cardinality=32):
    """ Construct the Learner
        x          : input to the learner
        groups     : list of groups: filters in, filters out, number of blocks
        cardinality: width of group convolution
    """
    # First ResNeXt Group (not-strided)
    filters_in, filters_out, n_blocks = groups.pop(0)
    x = group(x, filters_in, filters_out, n_blocks, strides=(1, 1), cardinality=cardinality)

    # Remaining ResNeXt groups
    for filters_in, filters_out, n_blocks in groups:
        x = group(x, filters_in, filters_out, n_blocks, cardinality=cardinality)
    return x

# Meta-parameter: number of filters in, out and number of blocks
groups = { 50 : [ (128, 256, 3), (256, 512, 4), (512, 1024, 5),  (1024, 2048, 3)],  # ResNeXt 50
           101: [ (128, 256, 3), (256, 512, 4), (512, 1024, 22), (1024, 2048, 3)],  # ResNeXt 101
           152: [ (128, 256, 3), (256, 512, 8), (512, 1024, 35), (1024, 2048, 3)]   # ResNeXt 152
         }

# Meta-parameter: width of group convolution
cardinality = 32

# The input tensor
inputs = Input(shape=(224, 224, 3))

# The Stem Group
x = stem(inputs)

# The Learner
x = learner(x, groups[50], cardinality)
# The Classifier for 1000 classes
outputs = classifier(x, 1000)

# Instantiate the Model
model = Model(inputs, outputs)
```

## Micro-Architecture

<img src='micro.jpg'>

```python
def group(x, filters_in, filters_out, n_blocks, cardinality=32, strides=(2, 2)):
    """ Create a Residual group
	x          : input to the group
        filters_in : number of filters  (channels) at the input convolution
        filters_out: number of filters (channels) at the output convolution
        cardinality: width of group convolution
	strides    : whether its a strided convolution
    """
    # Double the size of filters to fit the first Residual Group
    # Reduce feature maps by 75% (strides=2, 2) to fit the next Residual Group
    x = projection_block(x, filters_in, filters_out, cardinality=cardinality, strides=strides)

    # Remaining blocks
    for _ in range(n_blocks):
        x = identity_block(x, filters_in, filters_out, cardinality=cardinality)	
    return x
```

### Stem Group

<img src="stem.jpg">

```python
def stem(inputs):
    """ Construct the Stem Convolution Group
        inputs : input vector
    """
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', kernel_initializer='he_normal', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    return x
```

### ResNeXt Block with Identity Shortcut

<img src='identity-block.jpg'>

```python
def identity_block(x, filters_in, filters_out, cardinality=32):
    """ Construct a ResNeXT block with identity link
        x          : input to block
        filters_in : number of filters  (channels) at the input convolution
        filters_out: number of filters (channels) at the output convolution
        cardinality: width of group convolution
    """

    # Remember the input
    shortcut = x

    # Dimensionality Reduction
    x = Conv2D(filters_in, (1, 1), strides=(1, 1),
                      padding='same', kernel_initializer='he_normal', use_bias=False)(shortcut)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Cardinality (Wide) Layer (split-transform)
    filters_card = filters_in // cardinality
    groups = []
    for i in range(cardinality):
        group = Lambda(lambda z: z[:, :, :, i * filters_card:i *
                              filters_card + filters_card])(x)
        groups.append(Conv2D(filters_card, (3, 3), strides=(1, 1),
                                    padding='same', kernel_initializer='he_normal', use_bias=False)(group))

    # Concatenate the outputs of the cardinality layer together (merge)
    x = Concatenate()(groups)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Dimensionality restoration
    x = Conv2D(filters_out, (1, 1), strides=(1, 1),
                      padding='same', kernel_initializer='he_normal', use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    # Identity Link: Add the shortcut (input) to the output of the block
    x = Add()([shortcut, x])
    x = ReLU()(x)
    return x
```

### ResNeXt Block with Projection Shortcut

<img src='projection-block.jpg'>

```python
def projection_block(x, filters_in, filters_out, cardinality=32, strides=(2, 2)):
    """ Construct a ResNeXT block with projection shortcut
        x          : input to the block
        filters_in : number of filters  (channels) at the input convolution
        filters_out: number of filters (channels) at the output convolution
        cardinality: width of group convolution
        strides    : whether entry convolution is strided (i.e., (2, 2) vs (1, 1))
    """

    # Construct the projection shortcut
    # Increase filters by 2X to match shape when added to output of block
    shortcut = Conv2D(filters_out, (1, 1), strides=strides,
                                 padding='same', kernel_initializer='he_normal')(x)
    shortcut = BatchNormalization()(shortcut)

    # Dimensionality Reduction
    x = Conv2D(filters_in, (1, 1), strides=(1, 1),
                      padding='same', kernel_initializer='he_normal', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Cardinality (Wide) Layer (split-transform)
    filters_card = filters_in // cardinality
    groups = []
    for i in range(cardinality):
        group = Lambda(lambda z: z[:, :, :, i * filters_card:i *
                              filters_card + filters_card])(x)
        groups.append(Conv2D(filters_card, (3, 3), strides=strides,
                                    padding='same', kernel_initializer='he_normal', use_bias=False)(group))

    # Concatenate the outputs of the cardinality layer together (merge)
    x = Concatenate()(groups)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Dimensionality restoration
    x = Conv2D(filters_out, (1, 1), strides=(1, 1),
                      padding='same', kernel_initializer='he_normal', use_bias=False)(x)
    x = BatchNormalization()(x)

    # Identity Link: Add the shortcut (input) to the output of the block
    x = Add()([shortcut, x])
    x = ReLU()(x)
    return x
```

### Cardinality

<img src='cardinality.jpg'>

### Classifier

<img src="classifier.jpg">

```python
def classifier(x, n_classes):
  """ Construct the Classifier Group 
      x         : input to the classifier
      n_classes : number of output classes
  """
  # Pool at the end of all the convolutional residual blocks
  x = GlobalAveragePooling2D()(x)

  # Final Dense Outputting Layer for the outputs
  outputs = Dense(n_classes, activation='softmax', kernel_initializer='he_normal')(x)
  return outputs
```

## Composable

*Example Instantiate a ResNeXt model*

```python
from resnext_c import ResNeXt
# ResNeXt50 from research paper
resnext = ResNeXt(50)

# ResNeXt50 custom input shape/classes
resnext = ResNeXt(50, input_shape=(128, 128, 3), n_classes=50)

# getter for the tf.keras model
model = resnext.model
```

*Example: Composable Group/Block*/

```python
# Make mini-ResNeXt for CIFAR-10
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, Flatten, Dense
inputs = Input((32, 32, 3))

# Stem
x = Conv2D(32, (3, 3), strides=1, padding='same', activation='relu')(inputs)

# Learner
# Residual Next group: 2 blocks, 64 to 128 filters
# Residual Next block with projection, 128 to 256 filters
# Residual Next block with identity, 256 filters
x = ResNeXt.group(x, 64, 128, 2)
x = ResNeXt.projection_block(x, 128, 256)
x = ResNeXt.identity_block(x, 256, 256)

# Classifier
x = Flatten()(x)
outputs = Dense(10, activation='softmax')(x)
model = Model(inputs, outputs)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.summary()
```

```python
# Removed for brevity

batch_normalization_427 (BatchN (None, 8, 8, 256)    1024        concatenate_9[0][0]              
__________________________________________________________________________________________________
re_lu_420 (ReLU)                (None, 8, 8, 256)    0           batch_normalization_427[0][0]    
__________________________________________________________________________________________________
conv2d_743 (Conv2D)             (None, 8, 8, 256)    65536       re_lu_420[0][0]                  
__________________________________________________________________________________________________
batch_normalization_428 (BatchN (None, 8, 8, 256)    1024        conv2d_743[0][0]                 
__________________________________________________________________________________________________
add_140 (Add)                   (None, 8, 8, 256)    0           re_lu_418[0][0]                  
                                                                 batch_normalization_428[0][0]    
__________________________________________________________________________________________________
re_lu_421 (ReLU)                (None, 8, 8, 256)    0           add_140[0][0]                    
__________________________________________________________________________________________________
flatten_1 (Flatten)             (None, 16384)        0           re_lu_421[0][0]                  
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 10)           163850      flatten_1[0][0]                  
==================================================================================================
Total params: 461,450
Trainable params: 456,586
Non-trainable params: 4,864
```

```python
from tensorflow.keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = (x_train / 255.0).astype(np.float32)
x_test  = (x_test  / 255.0).astype(np.float32)

model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=1)
```

```python
Epoch 1/10
45000/45000 [==============================] - 579s 13ms/sample - loss: 1.9641 - acc: 0.4374 - val_loss: 1.5466 - val_acc: 0.4746
Epoch 2/10
45000/45000 [==============================] - 561s 12ms/sample - loss: 1.0568 - acc: 0.6242 - val_loss: 1.0967 - val_acc: 0.6214
```
