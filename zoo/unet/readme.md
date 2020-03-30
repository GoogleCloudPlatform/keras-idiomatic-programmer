
# U-Net

[unet_c.py](unet_c.py) - production (composable)

[Paper](https://arxiv.org/pdf/1505.04597.pdf)

## Macro-Architecture

<img src='macro.jpg'>

<img src='contract-macro.jpg'>

<img src='expand-macro.jpg'>

## Micro-Architecture

### Contracting Group

<img src='contract-block.jpg'>

### Expanding Group

<img src='expand-block.jpg'>

### Classifier

<img src="classifier.jpg">

## Composable

*Example: Instantiate a stock U-Net model*

```python
from unet_c import UNet

# U-Net from research paper
resnet = UNet()

# getter for the tf.keras model
model = unet.model
```
