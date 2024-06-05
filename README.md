## Exploring Transition Space with Causality Discovering and Re-composing

## Setup

Create the conda environment by running : 

```
conda env create -f environment.yml
```

Clone the [lexa-benchmark](https://github.com/orybkin/lexa-benchmark) repo, and modify the python path   
`export PYTHONPATH=<path to lexa-training>/lexa:<path to lexa-benchmark>`  

Export the following variables for rendering  
`export MUJOCO_RENDERER=egl; export MUJOCO_GL=egl`

**WARNING!** Make sure to use the right python and mujoco version. The robobin environment code is known to break with other versions. Other environments might or might not work.

## Training
For training, run : 

```
export CUDA_VISIBLE_DEVICES=<gpu_id>  
python train.py --configs defaults <method> --task <task> --logdir <log path> --time_limit <time limit>
```

where method can be `CDR_temporal`, `CDR_cosine`, `ddl`, `diayn` or `gcsl`   
Supported tasks are `dmc_walker_walk`, `dmc_quadruped_run`, `robobin`, `kitchen`, `joint`. The time limit should be 1000 for DMC and default otherwise.

To view the graphs and gifs during training, run `tensorboard --logdir <log path>`


