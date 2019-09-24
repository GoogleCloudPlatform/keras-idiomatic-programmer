
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
from vgg_c import VGG

# VGG16 from research paper
vgg = VGG(16)

# VGG16 custom input shape/classes
vgg = VGG(16, input_shape=(128, 128, 3), n_classes=50)

# getter for the tf.keras model
model = vgg.model
```

*Example: Composable Group*

```python
# make mini-VGG for CIFAR-10
from tensorflow.keras import Input, Model
from tensorflow.keras import Conv2D, Flatten, Dense

# Stem
inputs = Input((32, 32, 3))
x = Conv2D(32, (3, 3), strides=1, padding='same', activation='relu')(inputs)

# Learner
# VGG group: 1 conv layer, 128 filters
# VGG group: 2 conv layers, 256 filters
x = VGG.group(x, 1, 128)
x = VGG.group(x, 2, 256)

# Classifier
x = Flatten()(x)
outputs = Dense(10, activation='softmax')(x)
model = Model(inputs, outputs)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.summary()
```

```python
    Layer (type)                 Output Shape              Param #   
=================================================================
input_4 (InputLayer)         [(None, 32, 32, 3)]       0         
_________________________________________________________________
conv2d_26 (Conv2D)           (None, 28, 28, 32)        2432      
_________________________________________________________________
conv2d_27 (Conv2D)           (None, 28, 28, 64)        18496     
_________________________________________________________________
max_pooling2d_11 (MaxPooling (None, 14, 14, 64)        0         
_________________________________________________________________
conv2d_28 (Conv2D)           (None, 14, 14, 128)       73856     
_________________________________________________________________
conv2d_29 (Conv2D)           (None, 14, 14, 128)       147584    
_________________________________________________________________
max_pooling2d_12 (MaxPooling (None, 7, 7, 128)         0         
_________________________________________________________________
flatten_3 (Flatten)          (None, 6272)              0         
_________________________________________________________________
dense_5 (Dense)              (None, 10)                62730     
=================================================================
Total params: 305,098
Trainable params: 305,098
Non-trainable params: 0
```

```python
# Train the model on CIFAR-10
from tensorflow.keras.datasets import cifar10
import numpy as np
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = (x_train / 255.0).astype(np.float32)
y_train = (y_train / 255.0).astype(np.float32)
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=1)
```

```python
Train on 45000 samples, validate on 5000 samples
Epoch 1/10
45000/45000 [==============================] - 73s 2ms/sample - loss: 1.4581 - acc: 0.4728 - val_loss: 1.1139 - val_acc: 0.6030
Epoch 2/10
45000/45000 [==============================] - 77s 2ms/sample - loss: 0.9879 - acc: 0.6559 - val_loss: 0.8826 - val_acc: 0.6948
Epoch 3/10
45000/45000 [==============================] - 433s 10ms/sample - loss: 0.7916 - acc: 0.7264 - val_loss: 0.8561 - val_acc: 0.7152
Epoch 4/10
45000/45000 [==============================] - 81s 2ms/sample - loss: 0.6645 - acc: 0.7689 - val_loss: 0.7758 - val_acc: 0.7362
Epoch 5/10
45000/45000 [==============================] - 82s 2ms/sample - loss: 0.5571 - acc: 0.8058 - val_loss: 0.7687 - val_acc: 0.7568
Epoch 6/10
45000/45000 [==============================] - 82s 2ms/sample - loss: 0.4691 - acc: 0.8349 - val_loss: 0.7511 - val_acc: 0.7558
Epoch 7/10
45000/45000 [==============================] - 82s 2ms/sample - loss: 0.3811 - acc: 0.8669 - val_loss: 0.8617 - val_acc: 0.7520
Epoch 8/10
45000/45000 [==============================] - 82s 2ms/sample - loss: 0.3132 - acc: 0.8897 - val_loss: 0.9241 - val_acc: 0.7468
Epoch 9/10
45000/45000 [==============================] - 81s 2ms/sample - loss: 0.2583 - acc: 0.9076 - val_loss: 1.0457 - val_acc: 0.7438
Epoch 10/10
45000/45000 [==============================] - 83s 2ms/sample - loss: 0.2174 - acc: 0.9221 - val_loss: 1.1191 - val_acc: 0.7428
```

