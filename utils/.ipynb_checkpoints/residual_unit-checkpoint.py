import tensorflow as tf
from tensorflow.keras import layers, Sequential

# modified form ResidualBlock as ResidualUnit
class ResidualUnit(layers.Layer):

    """
    This is class for creating residual unit with varying output size
    Inputs:
        inputs: tf.Tensor of shape (batch, height, width, channels)
    Arguments:
        filter: Integer, the dimensionality of the output space(i.e. the number of output filter in the convolution).
        stride: Integer, specifying the strides of the first convolution layer along the height and width.
    Return:
        output: tf.Tensor of shape (batch, height/stride, width/stride, filter). 
    """

    def __init__(self, filter, stride=1):
        super().__init__()
        
        self.conv1 = layers.Conv2D(filter*2, (3, 3), strides=stride, padding='same')
        self.bn1 = layers.BatchNormalization()
        
        self.conv2 = layers.Conv2D(filter*2, (3, 3), strides=1, padding='same')
        self.bn2 = layers.BatchNormalization()
        
        self.conv3 = layers.Conv2D(filter, (3, 3), strides=1, padding='same')
        self.bn3 = layers.BatchNormalization()
        
        self.downsample = layers.Conv2D(filter, (1, 1), strides=stride)
        
    def call(self, inputs, training=None):
        
        out = tf.nn.relu(self.bn1(self.conv1(inputs)))
        out = tf.nn.relu(self.bn2(self.conv2(out)))
        out = tf.nn.relu(self.bn3(self.conv3(out)))
        
        # this is not the identity exactly, 1 by 1 convolution is used on channels dimension
        identity = self.downsample(inputs)

        output = layers.add([out, identity])
        output = tf.nn.relu(output)
        
        return output


class ResidualUnitIdentity(layers.Layer):

    """
    This is class for creating residual unit requires the input channels equals output channels if stride = 1
    Inputs:
        inputs: tf.Tensor of shape (batch, height, width, channels)
    Arguments:
        stride: Integer, specifying the strides of the first convolution layer along the height and width.
    Return:
        output: tf.Tensor of shape (batch, height/stride, width/stride, filter). 
    """

    def __init__(self, filter, stride=1):
        super().__init__()
        self.filter = filter
        self.stride = stride

        self.conv1 = layers.Conv2D(filter*2, (3, 3), strides=stride, padding='same')
        self.bn1 = layers.BatchNormalization()
        
        self.conv2 = layers.Conv2D(filter*2, (3, 3), strides=1, padding='same')
        self.bn2 = layers.BatchNormalization()
        
        self.conv3 = layers.Conv2D(filter, (3, 3), strides=1, padding='same')
        self.bn3 = layers.BatchNormalization()
        
        if stride != 1:
            self.downsample = Sequential()
            self.downsample.add(layers.Conv2D(filter, (1, 1), strides=stride))
        else:
            self.downsample = lambda x:x
        
    def call(self, inputs, training=None):
        if self.stride == 1:
            assert inputs.shape[-1] == self.filter

        out = tf.nn.relu(self.bn1(self.conv1(inputs)))
        out = tf.nn.relu(self.bn2(self.conv2(out)))
        out = tf.nn.relu(self.bn3(self.conv3(out)))
        
        # this is identity exactly if stride = 1
        identity = self.downsample(inputs)

        output = layers.add([out, identity])
        output = tf.nn.relu(output)
        
        return output


class DownSampleUnit(layers.Layer):

    """
    This is class for downsampling structure like pool-residual-residulal... 
    Inputs:
        inputs: tf.Tensor of shape (batch, height, width, channels)
    Arguments:
        r: Integer, number of residual units between adjacent pooling layer in the mask branch
        filters_residual: List, filter size with length equal to r
        strides_residual: List, stride size with length equal to r
        stride_pool: Integer, stride size of the pooling layer
    Return:
        output: tf.Tensor
    """

    def __init__(self, r, filters_residual, strides_residual, stride_pool=2):
        super().__init__()
        assert len(strides_residual) == r
        self.r = r

        self.maxpool2d = layers.MaxPool2D(strides=stride_pool, padding='same')
        self.residualunits = dict()
        for i, stride in enumerate(strides_residual):
            self.residualunits["residual_unit_%d" % i] = ResidualUnit(filters_residual[i], stride=stride)
        
    def call(self, inputs, training=None):

        output = self.maxpool2d(inputs)
        for i in range(self.r):
            residual_unit = self.residualunits["residual_unit_%d" % i]
            output = residual_unit(output)
        
        return output


class UpSampleUnit(layers.Layer):

    """
    This is class for upsampling structure like ...residual-residulal-interpolation 
    Inputs:
        inputs: tf.Tensor of shape (batch, height, width, channels)
    Arguments:
        r: Integer, number of residual units between adjacent interpolation layer in the mask branch
        filters_residual: List, filter size with length equal to r
        strides_residual: List, stride size with length equal to r
        up_size: Integer, upsampling size
    Return:
        output: tf.Tensor
    """

    def __init__(self, r, filters_residual, strides_residual, up_size=2):
        super().__init__()
        assert len(strides_residual) == r
        self.r = r

        self.residualunits = dict()
        for i, stride in enumerate(strides_residual):
            self.residualunits["residual_unit_%d" % i] = ResidualUnit(filters_residual[i], stride=stride)

        self.interpolation = layers.UpSampling2D(size=up_size, interpolation='nearest')
        
    def call(self, inputs, training=None):

        output = inputs
        for i in range(self.r):
            residual_unit = self.residualunits["residual_unit_%d" % i]
            output = residual_unit(output)
        output = self.interpolation(output)

        return output