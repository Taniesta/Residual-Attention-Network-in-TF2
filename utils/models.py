import tensorflow as tf
from tensorflow.keras import layers, regularizers, Sequential, Model

from .residual_unit import ResidualUnit, ResidualUnitBetween
from .attention_module import AttentionModule

l2 = regularizers.l2(0.001)
l1 = regularizers.l2(0.001)

# 16*16*16 -> 16*16*16
down_1 = [
    {'filters_residual':[32], 'strides_residual':[1], 'stride_pool':2},
    {'filters_residual':[32], 'strides_residual':[1], 'stride_pool':2},
    {'filters_residual':[32], 'strides_residual':[1], 'stride_pool':2}
]
up_1 = [
    {'filters_residual':[32], 'strides_residual':[1], 'up_size':2},
    {'filters_residual':[32], 'strides_residual':[1], 'up_size':2},
    {'filters_residual':[32], 'strides_residual':[1], 'up_size':2}
]

# 8*8*64 -> 8*8*64
down_2 = [
    {'filters_residual':[128], 'strides_residual':[1], 'stride_pool':2},
    {'filters_residual':[128], 'strides_residual':[1], 'stride_pool':2}
]
up_2 = [
    {'filters_residual':[128], 'strides_residual':[1], 'up_size':2},
    {'filters_residual':[128], 'strides_residual':[1], 'up_size':2}
]

# 4*4*256 -> 4*4*256
down_3 = [{'filters_residual':[512], 'strides_residual':[1], 'stride_pool':2}]
up_3 = [{'filters_residual':[512], 'strides_residual':[1], 'up_size':2}]


class Attention56(Model):
    def __init__(self, config):
        super().__init__()
                
        # 32*32*3 -> 32*32*4
        self.conv0 = layers.Conv2D(filters=4, kernel_size=5, padding = 'same')
        # 32*32*4 -> 16*16*16
        self.residualunit_0 = ResidualUnitBetween(filter=4, stride=2)

        self.stage_1 = AttentionModule(
            filter_side=[32], stride_side=[1],                     
            filters_trunk=[32, 32], strides_trunk=[1, 1],          
            filter_mask=32, s=3, down=down_1, up=up_1,
            p=1, t=2, r=1)                                                  
        # 16*16*16 -> 8*8*64
        self.residualunit_1 = ResidualUnitBetween(filter=16, stride=2)      

        self.stage_2 = AttentionModule(
            filter_side=[128], stride_side=[1],                    
            filters_trunk=[128, 128], strides_trunk=[1, 1],         
            filter_mask=128, s=2, down=down_2, up=up_2,            
            p = 1, t = 2, r=1)                                                       

        # 8*8*64 -> 4*4*256
        self.residualunit_2 = ResidualUnit(filter=64, stride=2)     

        self.stage_3 = AttentionModule(
            filter_side=[512], stride_side=[1],                   
            filters_trunk=[512, 512], strides_trunk=[1, 1],       
            filter_mask=512, s=1, down=down_3, up=up_3,
            p = 1, t = 2, r=1)

        # 4*4*256 -> 2*2*1024
        self.residualunit_3 = ResidualUnit(filter=256, stride=2)     
        self.bn = layers.BatchNormalization()   
        self.avepooling = layers.AveragePooling2D(2,2)

        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(256, kernel_regularizer=l2)    
        self.fc2 = layers.Dense(64, kernel_regularizer=l2)

        if 'CIFAR10' in config:
            self.out = layers.Dense(10, kernel_regularizer=l2)                            
        elif 'CIFAR100' in config:
            self.out = layers.Dense(100, kernel_regularizer=l2)
        
    def call(self, inputs, training=None):
        
        out = self.residualunit_0(self.conv0(inputs))
        
        out = self.residualunit_1(self.stage_1(out))
        out = self.residualunit_2(self.stage_2(out))
        out = self.residualunit_3(self.stage_3(out))

        out = self.avepooling(self.bn(out))

        flatten = self.flatten(out)
        logits = self.out(self.fc2(self.fc1(flatten)))
        
        return logits
        

            
        

