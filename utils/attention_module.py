import tensorflow as tf
from tensorflow.keras import layers, Sequential, Model

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
        if s >= 2:
            # this part needs more effort
            # I want the identities can output the desired shape smartly
            for i in range(s-1):
                self.identities["identity_%d" % i] = ResidualUnitIdentity(filter=up[s-2-i]['filters_residual'][-1], 
                                                                            stride=up[s-2-i]['strides_residual'][-1])

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
                output += identities["identity_%d" % i]
            output = self.upsampleunits["upsample_unit_%d" % i](output)
        
        output = self.conv1(output)
        output = self.conv2(output)
        
        mask = tf.math.sigmoid(output)
        
        return mask


class AttentionModule(Model):

    """
    This is class for an attention module
    Inputs:
        inputs: tf.Tensor of shape (batch, height, width, channels)
    Arguments:
        p: Integer, the number of residual units before and after two branches
        t: Integer, the number of residual units in the trunk branch
        r: Integer, number of residual units between adjacent pooling layer in the mask branch
        
        filter_side: List of Integer, filter size with length equal to p
        stride_side: List of Integer, stride size with length equal to p
        filters_trunk: List of Integer, filter size with length equal to t
        strides_trunk: List of Integer, stride size with length equal to t
        filter_mask: Integer, the dimensionality of the output space of mask branch

        s: Integer, number of downsampling/upsampling units used, differing from stage to stage 
        down: List of Dictionary, the dictionary contains the filters_residual, strides_residual, stride_pool
        up: List of Dictionary, the dictionary contains the filters_residual, strides_residual, up_size
    Return:
        output: tf.Tensor
    """
    def __init__(self, filter_side, stride_side, filters_trunk, strides_trunk,
                 filter_mask, s, down, up, p = 1, t = 2, r = 1):
        super().__init__()
 
        # ResidualUnit before and after the two branches
        assert len(filter_side) == p
        assert len(stride_side) == p

        self.before = Sequential()
        self.after = Sequential()
        for (filter, stride) in zip(filter_side, stride_side):
            self.before.add(layers.Conv2D(filter, kernel_size=(3, 3), strides=1, padding='same'))
            self.before.add(layers.Conv2D(filter, kernel_size=(3, 3), strides=stride, padding='same'))

        # Trunk Branch
        self.trunk = TrunkBranch(filters=filters_trunk, strides=strides_trunk, t = t)

        # Mask Branch
        self.mask = MaskBranch(filter=filter_mask, s=s, r=r, down=down, up=up)

        
    def call(self, inputs, training=None):
        
        output = self.before(inputs)
        # Trunk branch
        trunk_out = self.trunk(output)

        # Mask branch
        mask_out = self.mask(output)

        # Attention
        output = (mask_out+1) * trunk_out
        output = self.after(output)
        
        return output