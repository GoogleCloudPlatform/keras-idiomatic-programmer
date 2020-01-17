# Siamese Neural Network

[siamese_twin.py](siamese_twin.py) - academic (idiomatic)
[siamese_twin_c.py](siamese_twin_c.py) - production (composable)

[Paper](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)

## Macro-Architecture

### Siamese Neural Network

<img src="macro.jpg">

### Twin Micro-Architecture

<img src="micro.jpg">

### Stem Group

<img src="stem.jpg">

### Convolutional Block

<img src="block-conv.jpg">

### Encoder 

<img src="encoder.jpg">

### Classifier Group

<img src="classifier.jpg">

## Composable

*Example: Instantiate a stock Siamese Twin model*/

```python
from siamese_twin_c import SiameseTwin

# Siamese Twin from research paper
siam = SiameseTwin()

# getter for the tf.keras model
model = siam.model
```

