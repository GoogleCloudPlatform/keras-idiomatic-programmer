
# AutoEncoder

    autoencoder.py - academic - procedural
    autoencoder_c.py - composable - OOP


## Macro-Architecture

<img src='macro.jpg'>

Macro-architecture code for AutoEncoder

```python
# metaparameter: number of filters per layer in encoder
layers = [64, 32, 32]

# The input tensor
inputs = Input(shape=(32, 32, 3))

# The encoder
x = encoder(inputs, layers)

# The decoder
outputs = decoder(x, layers)

# Instantiate the Model
model = Model(inputs, outputs)
```

## Micro-Architecture 

### Encoder

<img src='encoder.jpg'>

```python
def encoder(inputs, layers):
    """ Construct the Encoder
        inputs : the input vector
        layers : number of filters per layer
    """
    x = inputs

    # Feature pooling by 1/2H x 1/2W
    for n_filters in layers:
        x = Conv2D(n_filters, (3, 3), strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

    return x
```

### Decoder

<img src="decoder.jpg">

```python
def decoder(x, layers):
    """ Construct the Decoder
      x      : input to decoder
      layers : the number of filters per layer (in encoder)
    """

    # Feature unpooling by 2H x 2W
    for _ in range(len(layers)-1, 0, -1):
        n_filters = layers[_]
        x = Conv2DTranspose(n_filters, (3, 3), strides=(2, 2), padding='same', use_bias=False,         
                            kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

    # Last unpooling, restore number of channels
    x = Conv2DTranspose(3, (3, 3), strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    return x
```

## Composable

*Example Instantiate a AutoEncoder model*

```python
from autoencoder_c import AutoEncoder

# Create an AutoEncoder for 32x32x3 (CIFAR-10)
autoencoder = AutoEncoder()

# Create a custom AutoEncoder
autoencoder = AutoEncoder(input_shape=(128, 128, 3), layers=[128, 64, 32])

```
