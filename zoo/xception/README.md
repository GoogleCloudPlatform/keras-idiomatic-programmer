
# Xception

[xception.py](xception.py) - academic (idiomatic)<br/>
[xception_c.py](xception_c.py) - production (composable)<br/>

[Paper](https://arxiv.org/pdf/1610.02357.pdf)

## Macro-Architecture

<img src='macro.jpg'>

## Micro-Architecture - Entry Flow

<img src='micro-entry.jpg'>

### Entry Flow Stem Group

<img src="stem.jpg">

### Entry Flow Block

<img src="block-projection.jpg">

## Micro-Architecture - Middle Flow

<img src="micro-middle.jpg">

### Middle Flow Block

<img src="block-middle.jpg">

## Micro-Architecture - Exit Flow

<img src="micro-exit.jpg">

### Exit Flow Residual Block

<img src="block-exit-residual.jpg">

### Exit Flow Convolutional Block

<img src="block-exit-conv.jpg">

### Exit Flow Classifier

<img src="classifier.jpg">

## Composable

*Example: Instantiate a stock Xception model*

```python
from xception_c import Xception

# Xception from research paper
xception = Xception()

# Xception custom input shape/classes
xception = Xception(input_shape=(128, 128, 3), n_classes=50)

# getter for the tf.keras model
model = xception.model
```

*Example: Compose and Train an Xception model*

```python
    ''' Example for constructing/training a Xception model on CIFAR-10
    '''
    # Example of constructing a mini-Xception
    entry  = [{ 'n_filters' : 128 }, { 'n_filters' : 728 }]
    middle = [{ 'n_filters' : 728 }, { 'n_filters' : 728 }, { 'n_filters' : 728 }]

    xception = Xception(entry=entry, middle=middle, input_shape=(32, 32, 3), n_classes=10)
    xception.model.summary()
    xception.cifar10()
```
