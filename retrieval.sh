results_path="./test"  # replace to your results path
batch_size=8
weight_path="checkpoint_best.pt"
MOL_PATH="mols.lmdb" # path to the molecule file
POCKET_PATH="pocket.lmdb" # path to the pocket file
EMB_DIR="./data/emb" # path to the cached mol embedding file

CUDA_VISIBLE_DEVICES="1" python ./unimol/retrieval.py --user-dir ./unimol $data_path "./data" --valid-subset test \
       --results-path $results_path \
       --num-workers 8 --ddp-backend=c10d --batch-size $batch_size \
       --task drugclip --loss in_batch_softmax --arch drugclip  \
       --max-pocket-atoms 256 \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256  --seed 1 \
       --path $weight_path \
       --log-interval 100 --log-format simple \
       --mol-path $MOL_PATH \
       --pocket-path $POCKET_PATH \
       --emb-dir $EMB_DIR \