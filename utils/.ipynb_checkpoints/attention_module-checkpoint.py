import tensorflow as tf
from tensorflow.keras import layers, Sequential

from .residual_unit import ResidualUnit, ResidualUnitIdentity, DownSampleUnit, UpSampleUnit

# modified form ResidualBlock as ResidualUnit
class TrunkBranch(layers.Layer):

    """
    This is class for create the trunk branck in an attention module
    Inputs:
        inputs: tf.Tensor of shape (batch, height, width, channels)
    Arguments:
        filters: List, filter size with length equal to t 
        strides: List, stride size with length equal to t 
        t: Integer, the number of residual unit in a trunk branch, this is actually redundant.
    Return:
        output: tf.Tensor
    """

    def __init__(self, filters, strides, t=2):
        super().__init__()

        assert len(filters) == t
        assert len(strides) == t
        self.t = t

        self.residualunits = dict()
        for i, filter in enumerate(filters):
            self.residualunits["residual_unit_%d" % i] = ResidualUnit(filter, stride=strides[i])
        
    def call(self, inputs, training=None):
        
        output = inputs
        for i in range(self.t):
            residual_unit = self.residualunits["residual_unit_%d" % i]
            output = residual_unit(output)
        
        return output


class MaskBranch(layers.Layer):
    
    """
    This is class for create the soft mask branck in an attention module
    Inputs:
        inputs: tf.Tensor of shape (batch, height, width, channels)
    Arguments:
        filter: Integer, the dimensionality of the output space
        s: Integer, number of downsampling/upsampling units used, differing from stage to stage 
        r: Integer, number of residual units between adjacent pooling layer in the mask branch
        down: List of Dictionary, the dictionary contains the filters_residual, strides_residual, stride_pool
        up: List of Dictionary, the dictionary contains the filters_residual, strides_residual, up_size
    Return:
        output: tf.Tensor
    """

    def __init__(self, filter, s, r, down, up):
        super().__init__()
        assert len(down) == s
        assert len(up) == s

        self.s = s
        self.r = r

        self.downsampleunits = dict()
        for i, param in enumerate(down):
            self.downsampleunits["downsample_unit_%d" % i] = DownSampleUnit(r, param['filters_residual'], param['strides_residual'], param['stride_pool'])
        self.upsampleunits = dict()
        for i, param in enumerate(up):
            self.upsampleunits["upsample_unit_%d" % (s-i-1)] = UpSampleUnit(r, param['filters_residual'], param['strides_residual'], param['up_size'])

        self.identities = dict()
        for i in range(s-1):
            self.identities["identity_%d" % i] = ResidualUnitIdentity(filter=up[i]['filters_residual'][-1], 
                                                                      stride=up[i]['strides_residual'][-1])

        self.conv1 = layers.Conv2D(filter, (1, 1), padding='same')
        self.conv2 = layers.Conv2D(filter, (1, 1), padding='same')

    def call(self, inputs, training=None):
        
        output = inputs
        identities = dict()
        for i in range(self.s):
            output = self.downsampleunits["downsample_unit_%d" % i](output)
            if ("identity_%d" % i) in self.identities:
                identities["identity_%d" % i] = self.identities["identity_%d" % i](output)
        
        for i in range(self.s-1, -1, -1):
            if ("identity_%d" % i) in identities:
                print(identities["identity_%d" % i].shape, output.shape)
                output += identities["identity_%d" % i]
            output = self.upsampleunits["upsample_unit_%d" % i](output)
        
        output = self.conv1(output)
        output = self.conv2(output)
        
        mask = tf.math.sigmoid(output)
        
        return mask
