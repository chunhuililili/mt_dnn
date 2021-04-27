DATA_DIR="expr/clue/data/hfl/chinese-roberta-wwm-ext-large"
TASK_DEF="expr/clue/task_def.yml"
#TRAIN_DATASET="afqmc,cmnli,csl,iflytek,tnews,wsc,lcqmc,bq,pawsx"
#TEST_DATASET="afqmc,cmnli,csl,iflytek,tnews,wsc,lcqmc,bq,pawsx"
TRAIN_DATASET="lcqmc"
TEST_DATASET="lcqmc"
SEED=-1
#BERT_PATH="hfl/chinese-roberta-wwm-ext-large"
BERT_PATH="expr/clue/clue_classifiers/model_2_162000.pt"
tstr=$(date +"%FT%H%M")
OUT_PATH="expr/clue/clue_classifiers1/"$TRAIN_DATASET$tstr
BATCH_SIZE=16
MAX_SEQ_SIZE=128
EPOCH_NUM=5
export CUDA_VISIBLE_DEVICES=7
LOG=$OUT_PATH/$tstr/log.log

python3 -m train \
    --name clue_multi_task \
    --data_dir $DATA_DIR \
    --task_def $TASK_DEF \
    --train_datasets $TRAIN_DATASET \
    --test_datasets $TEST_DATASET \
    --seed $SEED \
    --do_lower_case \
    --init_checkpoint $BERT_PATH \
    --output_dir $OUT_PATH \
    --log_file $LOG \
    --tensorboard \
    --batch_size $BATCH_SIZE \
    --max_seq_len $MAX_SEQ_SIZE \
    --epochs $EPOCH_NUM \
    --save_per_updates_on \
    --save_per_updates 3000 \
    --learning_rate 5e-5 \
    --global_grad_clipping 1 \
    --grad_clipping 0 \
    --optimizer adamax \
    #--bert_model_type bert-base-chinese \
