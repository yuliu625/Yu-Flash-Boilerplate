from huggingface_hub import hf_hub_download, snapshot_download

import os
from pathlib import Path


# def download_from_huggingface(repo_id, local_dir):
#     """
#     下载Hugging Face Hub上的模型或数据集到本地指定文件夹。

#     参数:
#     repo_id (str): Hugging Face Hub上的模型或数据集的ID。
#     local_dir (str): 要下载到的本地文件夹路径。
#     """
#     try:
#         # 下载模型或数据集
#         hf_hub_download(repo_id=repo_id, local_dir=local_dir)
#         print(f"下载完成: {repo_id} 已保存到 {local_dir}")
#     except Exception as e:
#         print(f"下载失败: {e}")


def download_model_from_huggingface(repo_id, local_dir, is_only_torch=True):
    try:
        # 下载模型或数据集
        if is_only_torch:
            snapshot_download(
                repo_id=repo_id, local_dir=local_dir,
                allow_patterns=[
                    '*.pt', '*.pth', '*.bin',
                    '*.json', '*.txt', '*.md',
                    # '*.safetensors',
                ]
            )
        else:
            snapshot_download(repo_id=repo_id, local_dir=local_dir)
        print(f"下载完成: {repo_id} 已保存到 {local_dir}")
    except Exception as e:
        print(f"下载失败: {e}")


def download_dataset_from_huggingface(repo_id, local_dir):
    try:
        # 下载模型或数据集
        snapshot_download(repo_id=repo_id, local_dir=local_dir, repo_type="dataset")  # 如果是数据集
        print(f"下载完成: {repo_id} 已保存到 {local_dir}")
    except Exception as e:
        print(f"下载失败: {e}")


def main(repo_id, local_dir_str):
    repo_id = repo_id  # 模型或数据集的ID
    local_dir = Path(local_dir_str) / repo_id  # 本地文件夹路径
    download_model_from_huggingface(repo_id, local_dir)


if __name__ == "__main__":
    os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
    # os.environ['HF_HOME'] = "~/.cache/huggingface"

    # 仓库id。
    repo_id = r"openai/clip-vit-base-patch32"
    # 本地路径，服务器上需要选择一下类型。
    local_dir_str = r"D:\dcmt\model\hf"
    main(repo_id, local_dir_str)
    