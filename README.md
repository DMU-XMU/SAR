# Sequential Action-Induced Invariant Representation for Reinforcement Learning


This is the code of paper: **Sequential Action-Induced Invariant Representation for Reinforcement Learning**.

# Carla images
![Demo img](https://github.com/DMU-XMU/SAR/blob/main/img/carla.png)
#  Training videos in Carla autonomous driving
![demo1](https://github.com/DMU-XMU/SAR/blob/main/videos/demo1.gif)
![demo2](https://github.com/DMU-XMU/SAR/blob/main/videos/demo2.gif)

## install CARLA
Please firstly install UE4.26.

Download CARLA from https://github.com/carla-simulator/carla/releases, e.g.:
1. https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.6.tar.gz
2. https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/AdditionalMaps_0.9.6.tar.gz

Add to your python path:
```
export PYTHONPATH=$PYTHONPATH:/home/rmcallister/code/bisim_metric/CARLA_0.9.6/PythonAPI
export PYTHONPATH=$PYTHONPATH:/home/rmcallister/code/bisim_metric/CARLA_0.9.6/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:/home/rmcallister/code/bisim_metric/CARLA_0.9.6/PythonAPI/carla/dist/carla-0.9.8-py3.5-linux-x86_64.egg
```

Install:
```
pip install pygame
pip install networkx
```
# run DMControl tasks
```
./run.sh --agent drq --auxiliary sar  --batch_size 256 --num_sample 256 --targ_extr 0 --env dmc.cheetah.run
```

# run CARLA tasks

Terminal 1:
```
cd CARLA_0.9.6
bash CarlaUE4.sh --RenderOffScreen --carla-rpc-port=2000 --fps=20
```

Terminal 2:
```
cd CARLA_0.9.6
./run.sh --agent curl --auxiliary sar --batch_size 128 --num_sample 128 --targ_extr 1 --port 2000 --env carla.highway.map04
```

# You can attach a tensorboard to monitor training by running:
```
tensorboard --logdir ./
```
Our code is modified based on: 
1. https://github.com/MIRALab-USTC/RL-CRESP.git 
2. https://github.com/facebookresearch/deep_bisim4control.git 

