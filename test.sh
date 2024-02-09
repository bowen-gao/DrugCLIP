results_path="./test"  # replace to your results path
batch_size=8
weight_path="checkpoint_best.pt"

TASK="DUDE" # DUDE or PCBA

CUDA_VISIBLE_DEVICES="4" python ./unimol/test.py --user-dir ./unimol $data_path "./data" --valid-subset test \
       --results-path $results_path \
       --num-workers 8 --ddp-backend=c10d --batch-size $batch_size \
       --task drugclip --loss in_batch_softmax --arch drugclip  \
       --max-pocket-atoms 256 \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256  --seed 1 \
       --path $weight_path \
       --log-interval 100 --log-format simple \
       --max-pocket-atoms 511 \
       --test-task $TASK \