
# WRN (Wide Residual Network)

    wrn_c.py - composable - OOP

[Paper](https://arxiv.org/pdf/1605.07146.pdf)

## Macro-Architecture

<img src='macro.jpg'>

```python
    def __init__(self, groups=None, depth=16, k=8, dropout=0, input_shape=(32, 32, 3), n_classes=10, reg=None):
        """ Construct a Wids Residual (Convolutional Neural) Network
            depth      : number of layers
            k          : width factor
            groups     : number of filters per group
            input_shape: input shape
            n_classes  : number of output classes
            reg        : kernel regularization
        """
        if groups is None:
            groups = self.groups

        # The input tensor
        inputs = Input(input_shape)

        # The stem convolutional group
        x = self.stem(inputs, reg=reg)

        # The learner
        x = self.learner(x, groups=groups, depth=depth, k=k, dropout=dropout, reg=reg)

        # The classifier
        outputs = self.classifier(x, n_classes, reg=reg)

        # Instantiate the Model
        self._model = Model(inputs, outputs)

    def learner(self, x, **metaparameters):
        """ Construct the Learner
            x     : input to the learner
            groups: number of filters per group
            depth : number of convolutional layers
        """
        groups = metaparameters['groups']
        depth  = metaparameters['depth']

        # calculate the number of blocks from the depth
        n_blocks = (depth - 4) // 6

        # first group, the projection block is not strided
        x = WRN.group(x, n_blocks=n_blocks, strides=(1, 1), **groups.pop(0), **metaparameters)

        # remaining groups
        for group in groups:
            x = WRN.group(x, n_blocks=n_blocks, strides=(2, 2), **group, **metaparameters)
        return x
```

## Micro-Architecture

<img src='micro.jpg'>

```python
@staticmethod
    def group(x, init_weights=None, **metaparameters):
        """ Construct a Wide Residual Group
            x         : input into the group
            n_blocks  : number of residual blocks with identity link
        """
        n_blocks  = metaparameters['n_blocks']

        # first block is projection to match the number of input filters to output fitlers for the add operation
        x = WRN.projection_block(x, init_weights=init_weights, **metaparameters)

        # wide residual blocks
        for _ in range(n_blocks-1):
            x = WRN.identity_block(x, init_weights=init_weights, **metaparameters)
        return x
```

### Stem Group

<img src="stem.jpg">

```python
def stem(self, inputs, **metaparameters):
        """ Construct the Stem Convolutional Group
            inputs : the input vector
            reg    : kernel regularizer
        """
        reg = metaparameters['reg']

        # Convolutional layer
        x = Conv2D(16, (3, 3), strides=(1, 1), padding='same', use_bias=False,
                   kernel_initializer=self.init_weights, kernel_regularizer=reg)(inputs)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        return x
```
