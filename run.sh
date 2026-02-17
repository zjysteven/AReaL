export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_MODE=online

uv run examples/aigise/aigise_rl_mt.py \
    --config examples/aigise/aigise_grpo_mt.yaml \
    scheduler.type=local \
    experiment_name=aigise-grpo-mt \
    trial_name=trial0 \
    allocation_mode=sglang:d2p1t1+d2p1t1 \
    cluster.n_nodes=1 \
    cluster.n_gpus_per_node=4 \
    gconfig.max_new_tokens=2048 \
    train_dataset.batch_size=4