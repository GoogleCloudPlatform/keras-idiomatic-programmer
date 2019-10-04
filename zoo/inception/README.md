
# Inception

[Paper V1/V2](https://arxiv.org/pdf/1409.4842.pdf)<br/>
[Paper V3](https://arxiv.org/pdf/1512.00567.pdf)
[Corrected Paper V3](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.pdf)

   The paper submitted to ARXIV, and last revised on Dec 11, 2015, has some typos in it that were addressed with a revision    that is stored on CV-Foundation. Mostly notably, correctly the reference to the model submitted to the ILSVRC 2015 image    classification (1st runner up), from V2 to V3.
   
   It is generally agreed that V2 is the same as V1 with the additional of batch normalization, but no additional   
   factorization.
   
   The later paper continues to have the typos in Table 1. Per my discussion with two of the paper's authors:
   
   Sergey Ioffe: *The inception v3 model has been opensourced. Please see         
   https://github.com/tensorflow/models/blob/master/research/inception/inception/slim/inception_model.py (which also cites 
   the paper where this model was described).*
   
   Christian Szegedy: *I agree with Sergey that the implementation serves as the best reference.*

    ```python
    inception_(v1/v2/v3).py - academic - procedural
    inception_(v1.v2/v3)_c.py - composable - OOP
    ```

## Macro-Architecture v1.0 and v2.0

<img src="macro.jpg">

```python
def learner(x, n_classes):
    """ Construct the Learner
        x        : input to the learner
        n_classes: number of output classes
    """
    aux = [] # Auxiliary Outputs

    # Group 3
    x, o = group(x, [((64,),  (96,128),   (16, 32), (32,)),  # 3a
                     ((128,), (128, 192), (32, 96), (64,))]) # 3b
    aux += o

    # Group 4
    x, o = group(x, [((192,),  (96, 208), (16, 48), (64,)), # 4a
                     None,                               # auxiliary classifier
                     ((160,), (112, 224), (24, 64), (64,)), # 4b
                     ((128,), (128, 256), (24, 64), (64,)), # 4c
                     ((112,), (144, 288), (32, 64), (64,)), # 4d
                     None,                                  # auxiliary classifier
                     ((256,), (160, 320), (32, 128), (128,))], # 4e
                     n_classes=n_classes)
    aux += o

    # Group 5
    x, o = group(x, [((256,), (160, 320), (32, 128), (128,)), # 5a
                     ((384,), (192, 384), (48, 128), (128,))],# 5b
                     pooling=False)
    aux += o
    return x, aux

# Meta-parameter: dropout percentage
dropout = 0.4

# The input tensor
inputs = Input(shape=(224, 224, 3))

# The stem convolutional group
x = stem(inputs)

# The learner
x, aux = learner(x, 1000)

# The classifier for 1000 classes
outputs = classifier(x, 1000, dropout)

# Instantiate the Model
model = Model(inputs, [outputs] + aux)
```

## Macro-Architecture v3.0

<img src=macro-v3.jpg>

## Micro-Architecture v1.0 and v2.0

<img src="micro-v1.jpg">

```python
def group(x, blocks, pooling=True, n_classes=1000):
    """ Construct an Inception group
        x         : input into the group
        blocks    : filters for each block in the group
        pooling   : whether to end the group with max pooling
        n_classes : number of classes for auxiliary classifier
    """
    aux = [] # Auxiliary Outputs

    # Construct the inception blocks (modules)
    for block in blocks:
        # Add auxiliary classifier
        if block is None:
           aux.append(auxiliary(x, n_classes))
        else:
            x = inception_block(x, block[0], block[1], block[2], block[3])

    if pooling:
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = MaxPooling2D((3, 3), strides=2)(x)
    return x, aux
```
### Stem v1.0

<img src="stem-v1.jpg">

```python
def stem(inputs):
    """ Construct the Stem Convolutional Group 
        inputs : the input vector
    """
    # The 224x224 images are zero padded (black - no signal) to be 230x230 images prior to the first convolution
    x = ZeroPadding2D(padding=(3, 3))(inputs)

    # First Convolutional layer which uses a large (coarse) filter
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='valid', activation='relu', kernel_initializer='glorot_uniform')(x)

    # Pooled feature maps will be reduced by 75%
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # Second Convolutional layer which uses a mid-size filter
    x = Conv2D(64, (1, 1), strides=(1, 1), padding='same', activation='relu', kernel_initializer='glorot_uniform')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(192, (3, 3), strides=(1, 1), padding='valid', activation='relu', kernel_initializer='glorot_uniform')(x)

    # Pooled feature maps will be reduced by 75%
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    return x
```

### Stem v2.0

Adds batch normalization to the convolutional layers and uses the common convention to drop biases in the convolutional layer when it is followed by batch normalization.

```python
def stem(inputs):
    """ Construct the Stem Convolutional Group
        inputs : the input vector
    """
    # The 224x224 images are zero padded (black - no signal) to be 230x230 images prior to the first convolution
    x = ZeroPadding2D(padding=(3, 3))(inputs)

    # First Convolutional layer which uses a large (coarse) filter
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='valid', use_bias=False, kernel_initializer='glorot_uniform')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Pooled feature maps will be reduced by 75%
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # Second Convolutional layer which uses a mid-size filter
    x = Conv2D(64, (1, 1), strides=(1, 1), padding='same', use_bias=False, kernel_initializer='glorot_uniform')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(192, (3, 3), strides=(1, 1), padding='valid', use_bias=False, kernel_initializer='glorot_uniform')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Pooled feature maps will be reduced by 75%
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    return x
```

### Stem v3.0

<img src="stem-v3.jpg">

```python
def stem(inputs):
    """ Construct the Stem Convolutional Group 
        inputs : the input vector
    """
    # Coarse filter of V1 (7x7) factorized into 3 3x3.
    # First 3x3 convolution is strided
    x = Conv2D(32, (3, 3), strides=(2, 2), padding='valid', use_bias=False, kernel_initializer='glorot_uniform')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(32, (3, 3), strides=(1, 1), padding='valid', use_bias=False, kernel_initializer='glorot_uniform')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    # Third 3x3, filters are doubled and padding added
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', use_bias=False, kernel_initializer='glorot_uniform')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    # Pooled feature maps will be reduced by 75%
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # 3x3 reduction
    x = Conv2D(80, (3, 3), strides=(1, 1), padding='valid', use_bias=False, kernel_initializer='glorot_uniform')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    # Dimensionality expansion
    x = Conv2D(192, (3, 3), strides=(1, 1), padding='valid', use_bias=False, kernel_initializer='glorot_uniform')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Pooled feature maps will be reduced by 75%
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    return x
```

### Stem v4.0

<img src="stem-v4.jpg">

```python
```

### Inception Block v1.0

Adds batch normalization to the convolutional layers and uses the common convention to drop biases in the convolutional layer when it is followed by batch normalization.

<img src="block-v1.jpg">

```python
def inception_block(x, f1x1, f3x3, f5x5, fpool):
    """ Construct an Inception block (module)
        x    : input to the block
        f1x1 : filters for 1x1 branch
        f3x3 : filters for 3x3 branch
        f5x5 : filters for 5x5 branch
        fpool: filters for pooling branch
    """
    # 1x1 branch
    b1x1 = Conv2D(f1x1[0], (1, 1), strides=1, padding='same', activation='relu', kernel_initializer='glorot_uniform')(x)

    # 3x3 branch
    # 3x3 reduction
    b3x3 = Conv2D(f3x3[0], (1, 1), strides=1, padding='same', activation='relu', kernel_initializer='glorot_uniform')(x)
    b3x3 = ZeroPadding2D((1,1))(b3x3)
    b3x3 = Conv2D(f3x3[1], (3, 3), strides=1, padding='valid', activation='relu', kernel_initializer='glorot_uniform')(b3x3)

    # 5x5 branch
    # 5x5 reduction
    b5x5 = Conv2D(f5x5[0], (1, 1), strides=1, padding='same', activation='relu', kernel_initializer='glorot_uniform')(x)
    b5x5 = ZeroPadding2D((1,1))(b5x5)
    b5x5 = Conv2D(f5x5[1], (3, 3), strides=1, padding='valid', activation='relu', kernel_initializer='glorot_uniform')(b5x5)

    # Pooling branch
    bpool = MaxPooling2D((3, 3), strides=1, padding='same')(x)
    # 1x1 projection
    bpool = Conv2D(fpool[0], (1, 1), strides=1, padding='same', activation='relu', kernel_initializer='glorot_uniform')(bpool)

    # Concatenate the outputs (filters) of the branches
    x = Concatenate()([b1x1, b3x3, b5x5, bpool])
    return x
```

### Inception Block v2.0

```python
def inception_block(x, f1x1, f3x3, f5x5, fpool):
    """ Construct an Inception block (module)
        x    : input to the block
        f1x1 : filters for 1x1 branch
        f3x3 : filters for 3x3 branch
        f5x5 : filters for 5x5 branch
        fpool: filters for pooling branch
    """
    # 1x1 branch
    b1x1 = Conv2D(f1x1[0], (1, 1), strides=1, padding='same', use_bias=False, kernel_initializer='glorot_uniform')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # 3x3 branch
    # 3x3 reduction
    b3x3 = Conv2D(f3x3[0], (1, 1), strides=1, padding='same', use_bias=False, kernel_initializer='glorot_uniform')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    b3x3 = ZeroPadding2D((1,1))(b3x3)
    b3x3 = Conv2D(f3x3[1], (3, 3), strides=1, padding='valid', use_bias=False, kernel_initializer='glorot_uniform')(b3x3)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # 5x5 branch
    # 5x5 reduction
    b5x5 = Conv2D(f5x5[0], (1, 1), strides=1, padding='same', use_bias=False, kernel_initializer='glorot_uniform')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    b5x5 = ZeroPadding2D((1,1))(b5x5)
    b5x5 = Conv2D(f5x5[1], (3, 3), strides=1, padding='valid', use_bias=False, kernel_initializer='glorot_uniform')(b5x5)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Pooling branch
    bpool = MaxPooling2D((3, 3), strides=1, padding='same')(x)
    # 1x1 projection
    bpool = Conv2D(fpool[0], (1, 1), strides=1, padding='same', use_bias=False, kernel_initializer='glorot_uniform')(bpool)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Concatenate the outputs (filters) of the branches
    x = Concatenate()([b1x1, b3x3, b5x5, bpool])
    return x
```
### Inception Blocks v3.0

<img src='block-35.jpg'>

```python
def inception_block_A(x, f1x1, f3x3, f5x5, fpool):
    """ Construct an Inception block (module)
        x    : input to the block
        f1x1 : filters for 1x1 branch
        f3x3 : filters for double 3x3 branch
        f5x5 : filters for 5x5 branch 
        fpool: filters for pooling branch
    """
    # 1x1 branch
    b1x1 = Conv2D(f1x1[0], (1, 1), strides=1, padding='same', use_bias=False, kernel_initializer='glorot_uniform')(x)
    b1x1 = BatchNormalization()(b1x1)
    b1x1 = ReLU()(b1x1)

    # 3x3 double branch
    # 3x3 reduction
    b3x3 = Conv2D(f3x3[0], (1, 1), strides=1, padding='same', use_bias=False, kernel_initializer='glorot_uniform')(x)
    b3x3 = BatchNormalization()(b3x3)
    b3x3 = ReLU()(b3x3)
    b3x3 = Conv2D(f3x3[1], (3, 3), strides=1, padding='same', use_bias=False, kernel_initializer='glorot_uniform')(b3x3)
    b3x3 = BatchNormalization()(b3x3)
    b3x3 = ReLU()(b3x3)
    b3x3 = Conv2D(f3x3[1], (3, 3), strides=1, padding='same', use_bias=False, kernel_initializer='glorot_uniform')(b3x3)
    b3x3 = BatchNormalization()(b3x3)
    b3x3 = ReLU()(b3x3)

    # 5x5 branch
    # 5x5 reduction
    b5x5 = Conv2D(f5x5[0], (1, 1), strides=1, padding='same', use_bias=False, kernel_initializer='glorot_uniform')(x)
    b5x5 = BatchNormalization()(b5x5)
    b5x5 = ReLU()(b5x5)
    b5x5 = Conv2D(f5x5[1], (3, 3), strides=1, padding='same', use_bias=False, kernel_initializer='glorot_uniform')(b5x5)
    b5x5 = BatchNormalization()(b5x5)
    b5x5 = ReLU()(b5x5)

    # Pooling branch
    bpool = AveragePooling2D((3, 3), strides=1, padding='same')(x)
    # 1x1 projection
    bpool = Conv2D(fpool[0], (1, 1), strides=1, padding='same', use_bias=False, kernel_initializer='glorot_uniform')(bpool)
    bpool = BatchNormalization()(bpool)
    bpool = ReLU()(bpool)

    # Concatenate the outputs (filters) of the branches
    x = Concatenate()([b1x1, b3x3, b5x5, bpool])
    return x
```

<img src='block-17.jpg'>

<img src='block-8.jpg'>

### Classifier

<img src='classifier.jpg'>

```python
def classifier(x, n_classes, dropout=0.4):
    """ Construct the Classifier Group
        x         : input to the classifier
        n_classes : number of output classes
        dropout   : percentage for dropout rate
    """
    # Pool at the end of all the convolutional residual blocks
    x = AveragePooling2D((7, 7))(x)
    x = Flatten()(x)
    x = Dropout(dropout)(x)

    # Final Dense Outputting Layer for the outputs
    outputs = Dense(n_classes, activation='softmax', kernel_initializer='glorot_uniform')(x)
    return outputs
```

### Auxiliary Classifier v1 & v2

<img src='auxiliary.jpg'>

```python
def auxiliary(x, n_classes):
    """ Construct the auxiliary classier
        x        : input to the auxiliary classifier
        n_classes: number of output classes
    """
    x = AveragePooling2D((5, 5), strides=(3, 3))(x)
    x = Conv2D(128, (1, 1), strides=(1, 1), padding='same', activation='relu', kernel_initializer='glorot_uniform')(x)
    x = Flatten()(x)
    x = Dense(1024, activation='relu', kernel_initializer='glorot_uniform')(x)
    x = Dropout(0.7)(x)
    output = Dense(n_classes, activation='softmax', kernel_initializer='glorot_uniform')(x)
    return output
```

## Composable

Example Instantiate a Inception V1 model

```python
from inception_v1_c import InceptionV1

# Inception V1 from research paper
inception = InceptionV1()

# InceptionV1 custom input shape/classes
inception = InceptionV1(input_shape=(128, 128, 3), n_classes=50)
```

# getter for the tf.keras model
model = inception.model

