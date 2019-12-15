import tensorflow as tf
from tensorflow.keras import layers, Sequential, Model

from .residual_unit import ResidualUnit, ResidualUnitIdentity
from .attention_module import AttentionModule

# 32*32*3 -> 32*32*128
down_1 = [
    {'filters_residual':[32], 'strides_residual':[1], 'stride_pool':2},
    {'filters_residual':[64], 'strides_residual':[1], 'stride_pool':2},
    {'filters_residual':[128], 'strides_residual':[1], 'stride_pool':2}
]
up_1 = [
    {'filters_residual':[64], 'strides_residual':[1], 'up_size':2},
    {'filters_residual':[32], 'strides_residual':[1], 'up_size':2},
    {'filters_residual':[16], 'strides_residual':[1], 'up_size':2}
]

# 32*32*128 -> 16*16*256
down_2 = [
    {'filters_residual':[64], 'strides_residual':[1], 'stride_pool':2},
    {'filters_residual':[128], 'strides_residual':[1], 'stride_pool':2}
]
up_2 = [
    {'filters_residual':[64], 'strides_residual':[1], 'up_size':2},
    {'filters_residual':[64], 'strides_residual':[1], 'up_size':2}
]

# 16*16*256 -> 8*8*512
down_3 = [{'filters_residual':[128], 'strides_residual':[1], 'stride_pool':2}]
up_3 = [{'filters_residual':[128], 'strides_residual':[1], 'up_size':2}]


class Attention56(Model):
    def __init__(self, config):
        super().__init__()
                
        # 32*32*3 -> 32*32*128
        self.conv1 = layers.Conv2D(filters = 4, kernel_size = 3, padding = 'same')
        
        self.stage_1 = AttentionModule(
            filter_side=[8], stride_side=[2],                     # 32*32*3 -> 16*16*8
            filters_trunk=[8, 16], strides_trunk=[1, 1],          # 16*16*8 -> 16*16*8 -> 16*16*16
            filter_mask=16, s=3, down=down_1, up=up_1,
            p=1, t=2, r=1)                                                  
        
        self.residualunitidentity_1 = ResidualUnitIdentity(filter=16, stride=1)      # 16*16*16
        
        self.stage_2 = AttentionModule(
            filter_side=[32], stride_side=[2],                    # 16*16*16 -> 8*8*32
            filters_trunk=[32, 64], strides_trunk=[1, 1],         # 8*8*32 -> 8*8*32 -> 8*8*64
            filter_mask=64, s=2, down=down_2, up=up_2,            # 
            p = 1, t = 2, r=1)                                                       
        
        self.residualunitidentity_2 = ResidualUnitIdentity(filter=64, stride=1)     # 32*32*256
        
        self.stage_3 = AttentionModule(
            filter_side=[128], stride_side=[2],                   # 8*8*64 -> 4*4*128
            filters_trunk=[128, 256], strides_trunk=[1, 1],       # 4*4*128 -> 4*4*128 -> 4*4*256
            filter_mask=256, s=1, down=down_3, up=up_3,
            p = 1, t = 2, r=1)
        
        self.residualunit_1 = ResidualUnit(filter=1024, stride=2)                    # 2*2*512
        self.bn1 = layers.BatchNormalization()
        #self.residualunit_2 = ResidualUnit(filter=1024, stride=2)                   # 1*1*1024
        #self.bn2 = layers.BatchNormalization()
        
        self.avepooling = layers.AveragePooling2D(2,2)

        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(512)    
        self.fc2 = layers.Dense(256)

        if 'CIFAR10' in config:
            self.out = layers.Dense(10)                            
        elif 'CIFAR100' in config:
            self.out = layers.Dense(100)
        
    def call(self, inputs, training=None):
        
        out = self.conv1(inputs)
        out = self.residualunitidentity_1(self.stage_1(out))
        out = self.residualunitidentity_2(self.stage_2(out))
        out = self.stage_3(out)
        
        out = self.bn1(self.residualunit_1(out))
        #out = self.bn2(self.residualunit_2(out))
        
        out = self.avepooling(out)

        flatten = self.flatten(out)
        logits = self.out(self.fc2(self.fc1(flatten)))
        
        return logits
        

            
        

