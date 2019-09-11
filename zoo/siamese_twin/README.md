# Siamese Neural Network

[Paper](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)

## Macro-Architecture

### Siamese Neural Network

<img src="macro.jpg">

```python
# Input shape for the Omniglot dataset
input_shape = (105, 105, 3)

# Create the twin model using the Sequential API  
model = twin(input_shape)

# Create input tensors for the left and right side (twins) of the network.
left_input  = Input(input_shape)
right_input = Input(input_shape)

# Create the encoders for the left and right side (twins)
left  = model( left_input )
right = model( right_input )

# Use Lambda method to create a custom layer for implementing a L1 distance layer.
L1Distance = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))

# Connect the left and right twins (via encoders) to the layer that calculates the
# distance between the encodings.
connected = L1Distance([left, right])

# Create the output layer for predicting the similarity from the distance layer
outputs = Dense(1,activation='sigmoid', kernel_initializer=dense_weights, bias_initializer=biases)(connected)
    
# Create the Siamese Network model
# Connect the left and right inputs to the outputs
model = Model(inputs=[left_input,right_input],outputs=out
```

### Twin Micro-Architecture

<img src="micro.jpg">

```python
def twin(input_shape):
    ''' Construct the model for both twins of the Siamese (connected) Network
        input_shape : input shape for input vector
    '''
    global dense_weights, biases
    
    model = Sequential()
    
    # The weights for the convolutional layers are initialized from a normal distribution
    # with a zero_mean and standard deviation of 10e-2
    conv_weights = RandomNormal(mean=0.0, stddev=10e-2)
    
    # The weights for the dense layers are initialized from a normal distribution
    # with a mean of 0 and standard deviation of 2 * 10e-1
    dense_weights = RandomNormal(mean=0.0, stddev=(2 * 10e-1))
    
    # The biases for all layers are initialized from a normal distribution
    # with a mean of 0.5 and standard deviation of 10e-2
    biases = RandomNormal(mean=0.5, stddev=10e-2)

    # Build the model
    stem(input_shape)
    block()
    encoder()
    return model
```

### Stem Group

<img src="stem.jpg">

```python
```

### Convolutional Block

<img src="block-conv.jpg">

### Encoder 

<img src="encoder.jpg">

### Classifier Group

<img src="classifier.jpg">

