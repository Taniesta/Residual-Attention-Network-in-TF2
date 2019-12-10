# Residual Attention Network for Image Classification in TensorFlow 2
### Instructions of running code:

- **Environment**: Tensorflow 2.0.0

- **Datasets**: CIFAR10 & CIFAR100, loaded automatically during running

- **Entry**: Just run the main notebook file named as `residual-attention.ipynb` from beginning to the end. 

### Organization of the files:

- `ref/`: the folder for the reference papers
- `report/`: the folder for the report writing
- `outputs/`: the folder for the outputs during the running time
  - `outputs/checkpoints/`: the folder for the model saving
  - `outputs/images/`: the folder for the images generated if any
  - `outputs/logs/`: the folder for the tensorboard
- `utils/`: the folder for the $\texttt{classes}$ and $\texttt{functions}$ defined for the notebook file
- `residual-attention.ipynb`: the main UI file containing instructors, comments, results shown both by running histories and figures 

### Key functions and classes

![attention_module_stage2](C:\Users\sheng\OneDrive\CU Third Semester\Neural Networks and Deep Learning\Assignment\e4040-2019Fall-Project-SJST-ss5593-fs2658-yt2633\report\imgs\attention_module_stage2.png)

the diagram for the paper: [Residual Attention Network for Image Classification](https://arxiv.org/abs/1704.06904) (marked as Figure 2 in the original paper)

#### `residual_unit.py`

1. $\texttt{classes}$ `ResidualUnit`: this is the residual unit shown as small write rectangle in the figure above
2. $\texttt{classes}$ `ResidualUnitIdentity`: similar as above but require the exactly identity mapping for the skip connection
3. $\texttt{classes}$ `DownSampleUnit`&`UpSampleUnit`: this is the down&up sample step shown as a triangle in the figure above

#### `attention_module.py`

1. $\texttt{classes}$ `TrunkBranch`: this is the trunk branch in the attention module which is indicated by the above branch in the figure above
2. $\texttt{classes}$ `MaskBranch`: this is the mask branch in the attention module which is indicated by the below branch in the figure above
3. $\texttt{classes}$ `AttentionModule`: this is attention module which has a slightly different structure for each stage

#### `models.py`

1. $\texttt{classes}$ `Attention56`: this is the model which is called attention56 in the original paper 


