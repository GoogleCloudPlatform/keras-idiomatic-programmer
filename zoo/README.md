# Model Zoo

All the models here are coded using the idiomatic design pattern for models. The models are based on their corresponding research paper and presented in two styles:

  1. Academic - procedural, documented in comments and w/o production wrapping.
  2. Composable (end in _c.py) - OOP with composable groups and blocks for reuse.

| Model       | Paper |<br/>
*Deep Convolutional Neural Networks*<br/>
| `VGG16`     | [Very Deep Convolutional Networks for Large-Scale Image Recognition, 2014](https://arxiv.org/pdf/1409.1556.pdf) |<br/>
| `VGG19`     | [Very Deep Convolutional Networks for Large-Scale Image Recognition, 2014](https://arxiv.org/pdf/1409.1556.pdf) |<br/>
*Residual Convolutional Neural Networks*<br/>
| `ResNet34`  | [Deep Residual Learning for Image Recognition, 2015](https://arxiv.org/pdf/1512.03385.pdf) |<br/>
| `ResNet50`  | [Deep Residual Learning for Image Recognition, 2015](https://arxiv.org/pdf/1512.03385.pdf) |<br/>
| `ResNet101` | [Deep Residual Learning for Image Recognition, 2015](https://arxiv.org/pdf/1512.03385.pdf) |<br/>
| `ResNet152` | [Deep Residual Learning for Image Recognition, 2015](https://arxiv.org/pdf/1512.03385.pdf) |<br/>
| `ResNet_cifar10` | [Deep Residual Learning for Image Recognition, 2015](https://arxiv.org/pdf/1512.03385.pdf) |<br/>
| `ResNet50_v1.5`  | [Deep Residual Learning for Image Recognition, 2015](https://arxiv.org/pdf/1512.03385.pdf) |<br/>
| `ResNet50_v2.0`  | [Identity Mappings in Deep Residual Networks, 2016](https://arxiv.org/pdf/1603.05027.pdf) |<br/>
| `ResNet_cifar10_v2.0`  | [Identity Mappings in Deep Residual Networks, 2016](https://arxiv.org/pdf/1603.05027.pdf) |<br/>
| `SE-ResNet50`    | [Squeeze-and-Excitation Networks, 2017](https://arxiv.org/pdf/1709.01507.pdf) |<br/>
| `SE-ResNet101`   | [Squeeze-and-Excitation Networks, 2017](https://arxiv.org/pdf/1709.01507.pdf) |<br/>
| `SE-ResNet152`   | [Squeeze-and-Excitation Networks, 2017](https://arxiv.org/pdf/1709.01507.pdf) |<br/>
*Wide Convolutional Neural Networks*<br/>
| `Inception_v1`   | [Going Deeper with Convolutions, 2015](https://arxiv.org/pdf/1409.4842.pdf)   |<br/>
| `Inception_v2`   | [Going Deeper with Convolutions, 2015](https://arxiv.org/pdf/1409.4842.pdf)   |<br/>
| `Inception_v3`   | [Rethinking the Inception Architecture for Computer Vision, 2015](https://arxiv.org/pdf/1512.00567.pdf) |<br/>
| `ResNeXt50`  | [Aggregated Residual Transformations for Deep Neural Networks, 2016](https://arxiv.org/pdf/1611.05431.pdf) |<br/>
| `ResNeXt101` | [Aggregated Residual Transformations for Deep Neural Networks, 2016](https://arxiv.org/pdf/1611.05431.pdf) |<br/>
| `ResNeXt152` | [Aggregated Residual Transformations for Deep Neural Networks, 2016](https://arxiv.org/pdf/1611.05431.pdf) | <br/>
| `ResNeXt_cifar10` | [Aggregated Residual Transformations for Deep Neural Networks, 2016](https://arxiv.org/pdf/1611.05431.pdf) |<br/>
| `Xception`   | [Xception: Deep Learning with Depthwise Separable Convolutions, 2016](https://arxiv.org/pdf/1610.02357.pdf) |<br/>
| `SE-ResNeXt50`    | [Squeeze-and-Excitation Networks, 2017](https://arxiv.org/pdf/1709.01507.pdf) |<br/>
| `SE-ResNeXt101`   | [Squeeze-and-Excitation Networks, 2017](https://arxiv.org/pdf/1709.01507.pdf) |<br/>
| `SE-ResNeXt152`   | [Squeeze-and-Excitation Networks, 2017](https://arxiv.org/pdf/1709.01507.pdf) |<br/>
*Densely Connected Convolutional Neural Networks*<br/>
| `DenseNet121` | [Densely Connected Convolutional Networks, 2016](https://arxiv.org/pdf/1608.06993.pdf) |<br/>
| `DenseNet169` | [Densely Connected Convolutional Networks, 2016](https://arxiv.org/pdf/1608.06993.pdf) |<br/>
| `DenseNet201` | [Densely Connected Convolutional Networks, 2016](https://arxiv.org/pdf/1608.06993.pdf) |<br/>
*Mobile Networks*<br/>
| `MobileNet v1` | [MobileNets: Efficient Convolutional Neural Networks for Mobile VisionApplications, 2017](https://arxiv.org/pdf/1704.04861.pdf) |<br/>
| `MobileNet v2` | [MobileNetV2: Inverted Residuals and Linear Bottlenecks, 2019](https://arxiv.org/pdf/1801.04381.pdf) |<br/>
| `SqueezeNet` |  [SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size, 2016](https://arxiv.org/pdf/1602.07360.pdf) |<br/>
| `SqueezeNet_bypass` |  [SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size, 2016](https://arxiv.org/pdf/1602.07360.pdf) |<br/>
| `SqueezeNet_complex` |  [SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size, 2016](https://arxiv.org/pdf/1602.07360.pdf) |<br/>
| `ShuffleNet` | [ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices, 2017](https://arxiv.org/pdf/1707.01083.pdf) |<br/>
*One-Shot Classification Networks*</br>
| `Siamese Twin` | [Siamese Neural Networks for One-shot Image Recognition, 2015](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf) |<br/>
*AutoEncoders*<br/>
| `Auto Encoder` | |<br/>

## Architecture Representation

The architecture representation of models consists of an overall macro-architecture and a micro-architecture.

### Macro-Architecture

The macro architecture consists of a stem (entry) group, a collection of groups (middle), and a classifier group (exit). The number of groups is defined by the macro architecture. The macro architecture may optionally contain a pre-stem, which perform additional operations, such as data preprocessing, model aggregation, and prediction post-processing.

<img src='macro.jpg'>

### Micro-Architecture

The micro architecture consists of a collection of blocks, which collectively form a group. A block consists of an input and an output, and within the block is a set of layers connected by the block pattern. The number of blocks and the block pattern is defined by the meta-parameters (where parameters are the weights/biases learned, and hyper-parameters are the parameters used to train the model, but not part of the model).

<img src='micro.jpg'>

