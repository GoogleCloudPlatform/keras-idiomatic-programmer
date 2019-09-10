
# ShuffleNet v1.0

[Paper](https://arxiv.org/pdf/1707.01083.pdf)

## Macro-Architecture

<img src='macro.png'>

```python
def learner(x, blocks, n_groups, filters, reduction):
    ''' Construct the Learner
	      x        : input to the learner
        blocks   : number of shuffle blocks per shuffle group
        n_groups : number of groups to partition feature maps (channels) into.
        filters  : dict that maps n_groups to list of output filters per block
        reduction: dimensionality reduction on entry to a shuffle block
    '''
    # Assemble the shuffle groups
    for i in range(3):
        x = shuffle_group(x, n_groups, blocks[i], filters[n_groups][i+1], reduction)
    return x
    
# meta-parameter: The number of groups to partition the filters (channels)
n_groups=2

# meta-parameter: number of groups to partition feature maps (key), and
# corresponding number of output filters (value)
filters = {
        1: [24, 144, 288, 576],
        2: [24, 200, 400, 800],
        3: [24, 240, 480, 960],
        4: [24, 272, 544, 1088],
        8: [24, 384, 768, 1536]
}

# meta-parameter: the dimensionality reduction on entry to a shuffle block
reduction = 0.25

# meta-parameter: number of shuffle blocks per shuffle group
blocks = [4, 8, 4 ]

# input tensor
inputs = Input( (224, 224, 3) )

# The Stem convolution group (referred to as Stage 1)
x = stem(inputs)

# The Learner
x = learner(x, blocks, n_groups, filters, reduction)
```

## Micro-Architecture

<img src='micro.png'>

```python
def shuffle_group(x, n_groups, n_blocks, n_filters, reduction):
    ''' Construct a Shuffle Group 
        x        : input to the group
        n_groups : number of groups to partition feature maps (channels) into.
        n_blocks : number of shuffle blocks for this group
        n_filters: number of output filters
        reduction: dimensionality reduction
    '''
    
    # first block is a strided shuffle block
    x = strided_shuffle_block(x, n_groups, n_filters, reduction)
    
    # remaining shuffle blocks in group
    for _ in range(n_blocks-1):
        x = shuffle_block(x, n_groups, n_filters, reduction)
    return x
```

### Stem Group

<img src="stem.jpg">

```python
def stem(inputs):
    ''' Construct the Stem Convolution Group 
        inputs : input image (tensor)
    '''
    x = Conv2D(24, (3, 3), strides=2, padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D((3, 3), strides=2, padding='same')(x)
    return x
```

### Shuffle Block

<img src='block.png'>

### Strided Shuffle Block

<img src='strided-block.png'>

### Classifier

<img src='classifier.jpg'>

```python
```
