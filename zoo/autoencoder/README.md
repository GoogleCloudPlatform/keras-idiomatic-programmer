
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
```

### Decoder

<img src="decoder.jpg">

```python
```

## Composable

*Example Instantiate a AutoEncoder model*

```python
from autoencoder_c import AutoEncoder

# 
autoencoder = AutoEncoder()

