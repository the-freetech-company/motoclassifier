*notes*   


what to change:
- adadelta optimizer seemed to perform well but there was significant overfitting



what has been attempted:
- SGD Optimizer does not work well


what to add:
- change batch size, try 64 instead of 32
- try weight decay
- experiment with fine grain dropout between .2-.35
- try implementing some of the conv layers below


Conv2D: This is a standard convolutional layer that applies a set of filters to the input image, creating a feature map.

SeparableConv2D: This is a variant of the standard Conv2D layer that applies a set of depth-wise filters followed by point-wise filters, which can reduce the number of parameters and improve computation efficiency.

ConvolutionalTranspose2D: Also known as a transposed convolutional layer or a "deconvolution" layer, this layer applies a set of filters to upsample the input image.

AtrousConv2D (or "dilated convolution"): This is a variant of the standard Conv2D layer that applies filters with a larger "reception field" (i.e., the area of the input image that the filters are applied to) by increasing the spacing between the filter elements (or "dilation rate").

DepthwiseConv2D: This is a depthwise separable convolutional layer that applies a set of depth-wise filters to each channel of the input image.

Conv2D with strides: This is a standard convolutional layer with a stride argument, which can be used to reduce the spatial dimension of the output feature maps.

Conv2D with padding: This is a standard convolutional layer with a padding argument, which can be used to preserve the spatial dimensions of the input image.

| Convolution Layer | Advantages | Disadvantages |
| --- | --- | --- |
| Conv2D | - Can learn spatial hierarchies of features  - Can be used to learn spatial translations invariance | - Can be computationally expensive - Large number of parameters to learn |
| SeparableConv2D | - Reduced computation cost - Reduced number of parameters to learn | - Can have limited representational power - Not suitable for all types of data |
| DepthwiseConv2D | - Reduced computation cost - Reduced number of parameters to learn | - Can have limited representational power - Not suitable for all types of data |
| TransposedConv2D | - Can be used for upsampling - Can learn spatial hierarchies of features | - Can be computationally expensive - Large number of parameters to learn |
| AtrousConv2D | - Can learn spatial hierarchies of features - Can be used to learn spatial translations invariance - Can be used to increase the receptive field of the model | - Can be computationally expensive - Large number of parameters to learn |
| Conv1D | - Can learn temporal hierarchies of features | - Can be computationally expensive - Large number of parameters to learn |