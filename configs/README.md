# Configuration Document
This document explains each component of configuration file.
You can find example configuration files in `config_examples` folder.
## Train configuration
### max_iterate
Maximum iteration number. Default is 100000.
### batch_size
Batch size. Default is 16.
### dataset
Target dataset. Currently only 'car', 'chair', 'kitti' and 'synthia'
is allowed.
### dataset_format
(Not used) Only `npy` format is allowed.
### is_pose_matrix
If this is true, use pose matrix as pose input. Default is false.
Only used for scene data.
### lr
Learning rate. Default is 5e-5.
### export_image_per
How frequently export image during training. Default is max_iterate / 10.
### available_gpu_ids
If you have multiple gpus, use multiple gpu ids, e.g. [0,1,2,3].
If you have only single gpu, set this value to [0].
### multiprocess_max
Maximum number of multiprocessing. This is not needed be same with 
`available_gpu_ids`. Even you have only single gpu, you can train
multiple models at once. 
But be careful not to cause overflow.
### image_size
Image size. Default is 256.
### parent_folder
Export models to this folder.
### model_list
A list of models to train.
#### model_type
Only two types are allowed.
* `t` for Tatarchenko15 (Pixel Generation)
* `z` for Zhou16 (Appearance Flow)
#### attention_strategy
Attention/skip connection strategy.
Followings are allowed.
* `no` : Vanilla
* `u_net` : U-Net without attention.
* `u_attn` : Attention U-Net.
* `h_attn` or `h` : Flow based hard attention.
* `cr_attn` or `cr` : Cross attention.
* `mixed` : You can use different strategy for 
each layer by setting following `attention_strategy_details`.
#### attention_strategy_details
A dictionary of (layer_size, strategy). 
If `attention_strategy` is `mixed`, this is used.
## Test configuration
Because many parameters are overlapped with training,
we only explain the others.
Test is done in exhaustive way.
### parent_folder
Load models from this folder.
### result_export_folder
Export results to this folder.
### target_scene_infos
Scene number to export. It should be in form of 
* objects : [model id, input azimuth, input elevation, target azimuth, target elevation].
* scenes : [scene id, input frame, target frame]