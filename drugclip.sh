

data_path="/data/protein/local_data/drug_clip_pdb_general/train_no_test_af/"


save_dir="/drug/save_dir/affinity/$(date +"%Y-%m-%d_%H-%M-%S")/"

tmp_save_dir="/drug/save_dir/affinity/tmp_save/$(date +"%Y-%m-%d_%H-%M-%S")/d"
tsb_dir="./tsbs/affinity_$(date +"%Y-%m-%d_%H-%M-%S")_tsb"

n_gpu=1
MASTER_PORT=10055
finetune_mol_model="/data/protein/molecule/pretrain/mol_pre_no_h_220816.pt"
finetune_pocket_model="/data/protein/molecule/pretrain/pocket_pre_220816.pt"
weight_path="/data/protein/save_dir/affinity/2023-05-06_22-08-56/checkpoint_best.pt"


batch_size=48
batch_size_valid=64
batch_size_valid=128
epoch=200
dropout=0.0
warmup=0.06
update_freq=1
dist_threshold=8.0
recycling=3
lr=1e-3

export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
CUDA_VISIBLE_DEVICES="1" python -m torch.distributed.launch --nproc_per_node=$n_gpu --master_port=$MASTER_PORT $(which unicore-train) $data_path --user-dir ./unimol --train-subset train --valid-subset valid \
       --num-workers 8 --ddp-backend=c10d \
       --task drugclip --loss in_batch_softmax --arch drugclip  \
       --max-pocket-atoms 256 \
       --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-8 --clip-norm 1.0 \
       --lr-scheduler polynomial_decay --lr $lr --warmup-ratio $warmup --max-epoch $epoch --batch-size $batch_size --batch-size-valid $batch_size_valid \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256 --update-freq $update_freq --seed 1 \
       --tensorboard-logdir $tsb_dir \
       --log-interval 100 --log-format simple \
       --validate-interval 1 \
       --best-checkpoint-metric valid_bedroc --patience 2000 --all-gather-list-size 2048000 \
       --save-dir $save_dir --tmp-save-dir $tmp_save_dir --keep-last-epochs 5 \
       --find-unused-parameters \
       --maximize-best-checkpoint-metric \
       --finetune-pocket-model $finetune_pocket_model \
       --finetune-mol-model $finetune_mol_model \
       --finetune-from-model $weight_path \