
# Inception v1

[Paper]()

## Macro-Architecture

<img src="macro.jpg">

```python
def learner(x, n_classes):
    """ Construct the Learner
        x        : input to the learner
        n_classes: number of output classes
    """
    # Dimensiionality Expansion Groups
    x = group(x, 3, 64)
    x = group(x, 1, 128)
    # Auxiliary Classifier
    x = auxiliary(x, n_classes) 
    x = group(x, 2, 192)

    # Dimensionality Reduction Groups
    x = group(x, 1, 160)
    # Auxiliary Classifier
    x = auxiliary(x, n_classes)
    x = group(x, 2, 128)
    return x
    
# Meta-parameter: dropout percentage
dropout = 0.4

# The input tensor
inputs = Input(shape=(224, 224, 3))

# The stem convolutional group
x = stem(inputs)

# The learner
x = learner(x, 1000)

# The classifier for 1000 classes
outputs = classifier(x, 1000, dropout)

# Instantiate the Model
model = Model(inputs, outputs)
```

## Micro-Architecture

<img src="micro.jpg">

### Stem v4.0

<img src="stem-v4.jpg">

### Inception Block v2.0

<img src="block-v2.jpg">

