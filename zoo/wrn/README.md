
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
