export CUDA_VISIBLE_DEVICES=5
python -m predict \
	--task_def expr/clue/task_def.yml \
        --task bq \
	--task_id 0 \
	--prep_input expr/clue/data/hfl/chinese-roberta-wwm-ext-large/bq_dev.json \
	--batch_size_eval 16 \
	--init_checkpoint expr/clue/clue_classifiers1/2021-04-24T0235/model_2_114000.pt \
	--score expr/clue/clue_classifiers1/2021-04-24T0235/bq_dev.tsv \
	--cuda false \
	--with_label
head -100 expr/clue/clue_classifiers1/2021-04-24T0235/bq_dev.tsv 
