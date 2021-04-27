python -m prepro_std \
    --model hfl/chinese-roberta-wwm-ext-large \
    --do_lower_case \
    --root_dir expr/clue/data/ \
    --task_def expr/clue/task_qianyan.yml \
    --task_type qianyan
