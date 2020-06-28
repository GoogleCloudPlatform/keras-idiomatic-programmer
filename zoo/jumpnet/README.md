# JumpNet

[jumpnet.py](jumpnet.py) - academic (idiomatic)<br/>
[jumpnet_c.py](jumpnet_c.py) - production (composable)

[Paper](tba)

## Macro-Architecture

<img src='macro.jpg'>

## Micro-Architecture

<img src='micro.jpg'>

### Stem Group

<img src="stem.jpg">

### Residual Block



## Draft for Paper / Abalation Study (on-going)

Introduction: A human crafted (without machine assistance) model design for improving residual convolution networks accuracy with fewer parameters and less depth requirements. I compare my design, and subsequent study, to three previous SOTA residual convolution networks, based on deep layers: ResNet v2 (2016), DenseNet (2016), and SparseNet (2018).

Each of these former architectures established new best practices for increasing accuracy, increasing capacity with deeper layers, minimizing parameters and matrix multiple operations, above best practices of former SOTA residual convolution network architectures.

The Jump-Net architecture is designed to incorporate the prior best practised by ResNet v2, DenseNet and SparseNet, and introduces a new skip connection, which I refer to as a group jump link. This new skip connection further minimizes the number of parameters and matmul operations, while preserving accuracy and without going deeper in layers.
