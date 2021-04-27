#!/bin/bash
if [[ $# -ne 2 ]]; then
  echo "train.sh <batch_size> <gpu>"
  exit 1
fi
#prefix="mt-dnn-rte"
BATCH_SIZE=$1
gpu=$2
echo "export CUDA_VISIBLE_DEVICES=${gpu}"
export CUDA_VISIBLE_DEVICES=${gpu}
tstr=$(date +"%FT%H%M")

#train_datasets="mnli,rte,qqp,qnli,mrpc,sst,cola,stsb"
train_datasets="bq,pawsx,lcqmc"
#test_datasets="mnli_matched,mnli_mismatched,rte"
test_datasets="bq,pawsx,lcqmc"
MODEL_ROOT="checkpoints"
#BERT_PATH="mt_dnn_models/bert_model_base_chinese.pt"
#CONFIG_PATH="nezha-large/bert_config.json"
BERT_PATH="hfl/chinese-roberta-wwm-ext-large"
#BERT_PATH="mt_dnn_models/bert_base/pytorch_model.bin"
DATA_DIR="data/qianyan/bert-base-chinese/"
#DATA_DIR="data/canonical_data/bert-pawsx/"

answer_opt=1
optim="adamax"
grad_clipping=0
global_grad_clipping=1
lr="5e-5"
resume_model_ckpt="checkpoints/mt-dnn-rte_adamax_answer_opt1_gc0_ggc1_2021-04-02T0834/model_2_36000.pt"
task_def="experiments/glue/glue_simq_def.yml"

model_dir="checkpoints/${tstr}"
log_file="${model_dir}/log.log"
encoder_type=1
python train.py --data_dir ${DATA_DIR} --init_checkpoint ${BERT_PATH} --batch_size ${BATCH_SIZE} --output_dir ${model_dir} --log_file ${log_file} --answer_opt ${answer_opt} --optimizer ${optim} --train_datasets ${train_datasets} --test_datasets ${test_datasets} --grad_clipping ${grad_clipping} --global_grad_clipping ${global_grad_clipping} --learning_rate ${lr} --save_per_updates_on --save_per_updates 3000  --task_def $task_def  
#--init_config  $CONFIG_PATH --encoder_type $encoder_type
#--resume --model_ckpt $resume_model_ckpt --save_per_updates_on --save_per_updates 1000
