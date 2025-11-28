PRE_SEQ_LEN=128 # 256
LR=2e-2

# bash BaseModels/ChatGLM-6B/ptuning/train.sh
# 在 P-tuning v2 训练时模型只保存 PrefixEncoder 部分的参数：
# - 模型参数：
#   - 基础模型BaseModels/chatglm-6b的权重会被微调
#   - P-Tuning特有的前缀嵌入（prefix embeddings），其长度由$PRE_SEQ_LEN决定
# - 训练配置：
#   - 最大训练步数：3000步
#   - 学习率：由$LR变量指定
#   - 使用混合精度训练（--fp16参数）
# - 输出相关：
#   - 模型检查点会保存在output/chatglm-6b-pt-$PRE_SEQ_LEN-$LR/目录下
#   - 每1000步保存一次检查点
#   - 每10步输出一次训练日志
# - 数据处理：
#   - 会从训练集和验证集文件创建缓存
#   - 根据指定的prompt_column和response_column处理数据
#=================================================================#
#=================================================================#
    # --model_name_or_path THUDM/chatglm-6b \
    # --model_name_or_path BaseModels/chatglm-6b \

# --predict_with_generate 每次验证都要执行 auto-regressive decoding，GPU 占用会波动，导致训练平均 GPU 利用率下降。

# # 4090 24GB Usage: 60%
# CUDA_VISIBLE_DEVICES=1 python BaseModels/ChatGLM-6B/ptuning/main.py \
#     --do_train \
#     --train_file datas/ShenNong_TCM_Dataset/ChatMed_TCM_train.json \
#     --validation_file datas/ShenNong_TCM_Dataset/ChatMed_TCM_test.json \
#     --prompt_column query \
#     --response_column response \
#     --overwrite_cache \
#     --model_name_or_path BaseModels/chatglm-6b \
#     --output_dir output/chatglm-6b-pt-$PRE_SEQ_LEN-$LR \
#     --overwrite_output_dir \
#     --max_source_length 128 \
#     --max_target_length 128 \
#     --per_device_train_batch_size 4 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 4 \
#     --predict_with_generate \
#     --max_steps 3000 \
#     --logging_steps 10 \
#     --save_steps 1000 \
#     --learning_rate $LR \
#     --pre_seq_len $PRE_SEQ_LEN

# # 4090 24GB Usage: 80.7% <8h
# CUDA_VISIBLE_DEVICES=1 python BaseModels/ChatGLM-6B/ptuning/main.py \
#     --do_train \
#     --train_file datas/ShenNong_TCM_Dataset/ChatMed_TCM_train.json \
#     --validation_file datas/ShenNong_TCM_Dataset/ChatMed_TCM_test.json \
#     --prompt_column query \
#     --response_column response \
#     --overwrite_cache \
#     --model_name_or_path BaseModels/chatglm-6b \
#     --output_dir output/chatglm-6b-pt-$PRE_SEQ_LEN-$LR \
#     --overwrite_output_dir \
#     --max_source_length 256 \
#     --max_target_length 256 \
#     --per_device_train_batch_size 4 \
#     --gradient_accumulation_steps 8 \
#     --max_steps 3000 \
#     --logging_steps 10 \
#     --save_steps 1000 \
#     --learning_rate $LR \
#     --pre_seq_len $PRE_SEQ_LEN \
#     --fp16 \
#     --predict_with_generate False

# 4090 24GB Usage: 80.7% <2h
CUDA_VISIBLE_DEVICES=1 python BaseModels/ChatGLM-6B/ptuning/main.py \
    --do_train \
    --train_file datas/ShenNong_TCM_Dataset/ChatMed_TCM_train.json \
    --validation_file datas/ShenNong_TCM_Dataset/ChatMed_TCM_test.json \
    --prompt_column query \
    --response_column response \
    --overwrite_cache \
    --model_name_or_path BaseModels/chatglm-6b \
    --output_dir output/chatglm-6b-pt-$PRE_SEQ_LEN-$LR \
    --overwrite_output_dir \
    --max_source_length 256 \
    --max_target_length 256 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --max_steps 3000 \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN \
    --fp16 \
    --predict_with_generate False