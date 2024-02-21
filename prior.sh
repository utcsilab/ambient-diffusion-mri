R=4
EXPERIMENT_NAME=brainMRI_prior_R=$R
GPUS_PER_NODE=1
GPU=3
MODEL_PATH=/home/asad/ambient-diffusion-mri/models/brainMRI_ksp_384_$R
MAPS_PATH=/home/asad/ambient-diffusion-mri/data/fastMRI/numpy/ksp_brainMRI_384.zip
SEEDS=4
BATCH=4

torchrun --standalone --nproc_per_node=$GPUS_PER_NODE \
    prior.py --gpu=$GPU --network=$MODEL_PATH/  --maps_path=$MAPS_PATH\
    --outdir=results/$EXPERIMENT_NAME \
    --experiment_name=$EXPERIMENT_NAME \
    --ref=$MODEL_PATH/stats.jsonl \
    --seeds=$SEEDS --batch=$BATCH \
    --mask_full_rgb=True --num_masks=1 --guidance_scale=0.0 \
    --training_options_loc=$MODEL_PATH/training_options.json \
    --num=$SEEDS --img_channels=2 --with_wandb=False