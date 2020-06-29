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

**Abstract**: A human crafted (without machine assistance) model design for improving residual convolution networks accuracy with fewer parameters and less depth requirements. I compare my design, and subsequent study, to three previous SOTA residual convolution networks, based on deep layers: ResNet v2 (2016) [7], DenseNet (2016) [8], and SparseNet (2018) [9].

Each of these former architectures established new best practices for increasing accuracy, increasing capacity with deeper layers, minimizing parameters and matrix multiple operations, above best practices of former SOTA residual convolution network architectures.

The Jump-Net architecture is designed to incorporate the prior best practised by ResNet v2 [7], DenseNet [8] and SparseNet [9], and introduces a new skip connection, which I refer to as a group jump link. This new skip connection further minimizes the number of parameters and matmul operations, while preserving accuracy and without going deeper in layers.

<TODO- Summarize ablation study>

**Introduction**:

We propose a new architecture that fine tunes improvements proposed by SparseNet for residual neural networks. The authors, Liu and Zeng, made observations on inefficiencies in feature reuse between residual blocks by the ResNet v2 and DenseNet designs. They observed that using the matrix add operation to reuse features between residual blocks, that the earlier features become more and more diluted, and as the network goes deeper, information from the early layers is effectively lost. They observed that the approach in DenseNet of using a matrix concatenation operation solved the dilution, but they observed that as the concatenated feature maps, from earlier layers, are aggressively downsized, that in later layers the feature maps become sparse. That is, the values tend towards zero; thereby losing all reusable information.

SparseNet uses the DenseNet approach of concatenating feature maps for reuse, but differs by proposing using an exponential selection for reuse between residuals blocks. That is, instead of concatenating the feature map input to the output of each block, they exponentially (powers of 2) increased the distance between blocks for concatenation. For example, the first concatenated feature maps occur one block away, the second at two blocks away, the third at 4 blocks away, the fourth at 8 blocks away, etc. They refer to this style of feature map concatenation as sparse. They theorized that by making the connections increasingly sparse, it would alleviate the need for aggressive feature map reductions between groups, as in the transitional block in DenseNet, and prevent the early concatenated feature maps from becoming sparse, tending to zeros, in deeper layers.

The choice of using an exponential selection appears arbitrary, and may in fact limit the usefulness of the technique proposed in SqueezeNet. For example, since the number of blocks in a group is configurable, one would have feature map concatenation links breaking out of arbitrary blocks inside groups into arbitrary blocks in subsequent groups. This appears to potentially undermine configuring the optimal number of blocks per group before group pooling of the feature maps.

Further this raises the question of how friendly this selection algorithm would be to macro architecture search. Currently, machine design of network architectures are going down two paths, macro and micro architecture search. In the latter (micro), also known as network architecture search (NAS), we explore the search space with convolutional/residual blocks. The former (macro), is a substantially lesser expensive form of search, where the search space is limited to configurable attributes, also known as metaparameters, within convolutional/residual groups based on an architecture template. In macro-architecture search, the effectiveness is based on the expertise of the data scientist in selecting a template and guiding the search space relating to the metaparameters.

I propose a different approach to the benefits of the sparse concatenated feature map reuse, whereby the sparse connections are between group boundaries, herein referred to as group jump links. As such, the number of blocks within the group can be optimized during macro-architecture search independent of the group jump link.


**Related Work**:

AlexNet [1], winner of the 2012 ILSVRC challenge for image classification using a convolutional neural network brought both deep learning and convolutional neural networks into the research mainstay for solving imaging related tasks. <TODO - how it solved it>

ZFNet [2], winner of the 2013 ILSVRC challenge for image classification, made further improvements to AlexNet. <TODO - what were the improvements>

GoogLeNet (Inception v1) [4] and VGG [3], 1st and 2nd place winners of the 2013 ILSVRC challenge for image classification, introduced using a repeating convolutional block pattern, delayed pooling. VGG experimented using the convolutional block patterns to go deeper in layers for more capacity, while GoogLeNet experimented with both going deeper and wider in layers for more capacity. <TODO - more details>

<TODO - ResNet v1>

<TODO - Batch Norm>

<TODO - DenseNet >

<TODO - Inception v3 redesign>

<TODO - SparseNet >

<TODO - NiN, different direction>

**Architecture**:

**Experiments**:

**Conclusion**:

**References**:

[1] ImageNet Classification with Deep Convolutional Neural Networks, Krizhesky, et. al., 2012

[2] Visualizing and Understanding Convolutional Networks, Zeiler, et. al., 2014

[3] Very Deep Convolutional Neural Networks for Large Scale Image Recognition, Simonyan, et. al., 2014

[4] Going Deeper with Convolutions, Szegedy, et. al., 2014

[5] Deep Residual Learning for Image Recognition, He, et. al., 2015

[6] Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift, Ioffe, et. al., 2015

[7] Identity Mappings in Deep Residual Networks, He, et. al., 2016

[8] Densely Connected Convolutional Networks, Huang, et. al., 2016

[9] SparseNet: A Sparse DenseNet for Image Classification, Liu, et. al., 2018

