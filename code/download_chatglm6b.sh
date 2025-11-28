#!/bin/bash

MODEL_DIR="chatglm-6b"
# BASE_URL="https://huggingface.co/zai-org/chatglm-6b/resolve/main"
BASE_URL="https://hf-mirror.com/zai-org/chatglm-6b/resolve/main"

mkdir -p $MODEL_DIR
cd $MODEL_DIR

echo "开始下载 ChatGLM-6B 全部分片..."

# 分片文件列表
FILES=(
    "pytorch_model-00001-of-00008.bin"
    "pytorch_model-00002-of-00008.bin"
    "pytorch_model-00003-of-00008.bin"
    "pytorch_model-00004-of-00008.bin"
    "pytorch_model-00005-of-00008.bin"
    "pytorch_model-00006-of-00008.bin"
    "pytorch_model-00007-of-00008.bin"
    "pytorch_model-00008-of-00008.bin"
    "pytorch_model.bin.index.json"
    "config.json"
    "tokenizer.model"
    "tokenizer_config.json"
    "ice_text.model"
)

# 多线程下载函数
download_file () {
    local file=$1
    echo "下载: $file"
    aria2c -x 16 -s 16 -k 1M -c \
      "$BASE_URL/$file" \
      -o "$file"
}

# 下载所有文件
for f in "${FILES[@]}"; do
    download_file "$f"
done

echo "所有文件下载完成！"
