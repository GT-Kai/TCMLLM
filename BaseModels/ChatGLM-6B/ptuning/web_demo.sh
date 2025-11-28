PRE_SEQ_LEN=128

# bash BaseModels/ChatGLM-6B/ptuning/web_demo.sh

CUDA_VISIBLE_DEVICES=1 python BaseModels/ChatGLM-6B/ptuning/web_demo.py \
    --model_name_or_path BaseModels/chatglm-6b \
    --ptuning_checkpoint output/chatglm-6b-pt-128-2e-2/checkpoint-3000 \
    --pre_seq_len $PRE_SEQ_LEN

