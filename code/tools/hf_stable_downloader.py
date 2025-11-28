#!/usr/bin/env python3
import os
import hashlib
from huggingface_hub import HfApi, hf_hub_url

# ---------------------------
# 可配置参数
# ---------------------------
REPO_ID = "zai-org/chatglm-6b"    # 模型或数据集
LOCAL_DIR = "chatglm-6b"          # 保存路径
REPO_TYPE = "model"               # "model" 或 "dataset"
RESUME = True                     # 是否断点续传

# ---------------------------
# 工具函数
# ---------------------------

def check_file_correct(filename, expected_size):
    """校验文件大小是否一致"""
    if expected_size is None:
        # 小文件（如 README、.gitattributes）不校验
        return os.path.exists(filename)

    if not os.path.exists(filename):
        return False
    return os.path.getsize(filename) == expected_size


def download_file_with_retry(repo_id, filename, local_path, expected_size, repo_type="model"):
    """下载单个文件，损坏则自动删除重下"""
    from huggingface_hub import hf_hub_download

    retry = 0
    while retry < 5:
        try:
            print(f"\n➡️  正在下载：{filename} (尝试 {retry+1}/5)")
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=LOCAL_DIR,
                repo_type=repo_type,
                force_download=not RESUME
            )

            if check_file_correct(local_path, expected_size):
                print(f"✅ 文件正确：{filename}")
                return True
            else:
                print(f"❌ 文件损坏：{filename}，删除重下...")
                if os.path.exists(local_path):
                    os.remove(local_path)
                retry += 1

        except Exception as e:
            print(f"⚠️ 下载失败：{e}，重试中...")
            if os.path.exists(local_path):
                os.remove(local_path)
            retry += 1

    print(f"💥 多次失败：{filename}")
    return False


# ---------------------------
# 主逻辑
# ---------------------------

def main():
    print("🔍 获取远端文件列表...")
    api = HfApi()

    repo_info = api.repo_info(REPO_ID, repo_type=REPO_TYPE)
    files = repo_info.siblings

    os.makedirs(LOCAL_DIR, exist_ok=True)

    for file in files:
        filename = file.rfilename
        expected_size = file.size
        local_path = os.path.join(LOCAL_DIR, filename)

        print("\n==============================")
        print(f"📦 文件：{filename}")
        if expected_size is None:
            print(f"📏 目标大小：未知（Git 小文件）")
        else:
            print(f"📏 目标大小：{expected_size/1024/1024:.2f} MB")
        print("==============================")

        # 如果文件已存在且正确 → 跳过
        if check_file_correct(local_path, expected_size):
            print(f"✔ 已存在且正确：{filename}")
            continue

        # 下载
        success = download_file_with_retry(
            repo_id=REPO_ID,
            filename=filename,
            local_path=local_path,
            expected_size=expected_size,
            repo_type=REPO_TYPE
        )

        if not success:
            print(f"❌ 下载失败：{filename}，跳过剩余文件。")
            break

    print("\n🎉 全部文件处理完成！")


if __name__ == "__main__":
    main()
