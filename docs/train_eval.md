# Prerequisites

**Please ensure you have prepared the environment and the nuScenes dataset.**

# Train and Test

Train VAD with 8 GPUs 
```shell
cd /path/to/VAD
conda activate vad
python -m torch.distributed.run --nproc_per_node=8 --master_port=2333 tools/train.py projects/configs/VAD/VAD_base.py --launcher pytorch --deterministic --work-dir path/to/save/outputs
```

**NOTE**: We release two types of training configs: the end-to-end configs and the two-stage (stage-1: Perception & Prediction; stage-2: Planning) configs. They should produce similar results. The two-stage configs are recommended because you can just train the stage-1 model once and use it as a pre-train model for stage-2.

Eval VAD with 1 GPU
```shell
cd /path/to/VAD
conda activate vad
CUDA_VISIBLE_DEVICES=0 python tools/test.py projects/configs/VAD/VAD_base.py /path/to/ckpt.pth --launcher none --eval bbox --tmpdir tmp
```

**NOTE**: Using distributed mode (multi GPUs) for evaluation will lead to inaccurate results, so make sure to use non-distributed mode (1 GPU) for evaluation.
