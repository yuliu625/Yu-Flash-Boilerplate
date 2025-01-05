"""
从huggingface上下载模型和数据集的方法。

如果不是因为网络原因，我并不会写这个工具。

单独配置和运行这个文件，这里会将指定的仓库下载到指定的本地位置。
"""

# TODO: 将这里的方法以对象化进行重构。同时考虑workflow。

from huggingface_hub import hf_hub_download, snapshot_download

import os
from pathlib import Path


# 下面这个被注释的方法只能一次下载一个文件，很不实用。
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


def download_model_from_huggingface(repo_id: str, local_dir: Path, is_only_torch=True):
    """
    封装了从huggingface上下载模型的api的方法。
    :param repo_id: huggingface上仓库的id，一般是 "用户名/仓库名" ，可以自动复制的。
    :param local_dir: 本地存储仓库这个文件夹的路径。这里配套的main方法会自动处理一部分。
    :param is_only_torch: 是否仅下载torch相关的文件。默认为了空间和速度，仅下载torch相关。
    """
    try:
        # 下载模型
        if is_only_torch:
            snapshot_download(
                repo_id=repo_id, local_dir=local_dir,  # 基础选项。
                allow_patterns=[
                    '*.pt', '*.pth', '*.bin',
                    '*.json', '*.txt', '*.md',
                    '*.safetensors',
                    # '*.tar'
                ]  # 我不断在检查和总结的torch相关的文件。
            )
        else:
            # 服务器上可以选择这个，完全避免出错。
            snapshot_download(repo_id=repo_id, local_dir=local_dir)
        print(f"下载完成: {repo_id} 已保存到 {local_dir}")
    except Exception as e:
        print(f"下载失败: {e}")


def download_dataset_from_huggingface(repo_id: str, local_dir: Path):
    """
    封装了从huggingface上下载数据集的api的方法。
    相比较下载模型，其实仅多了一个kwarg的指定。
    :param repo_id: huggingface上仓库的id，一般是 "用户名/仓库名" ，可以自动复制的。
    :param local_dir: 本地存储仓库这个文件夹的路径。这里配套的main方法会自动处理一部分。
    """
    try:
        # 下载数据集
        snapshot_download(repo_id=repo_id, local_dir=local_dir, repo_type="dataset")  # 如果是数据集
        print(f"下载完成: {repo_id} 已保存到 {local_dir}")
    except Exception as e:
        print(f"下载失败: {e}")


def batch_download(repo_ids: list, local_dir_str: str):
    """
    批量下载多个模型或多个数据集。
    没有额外写可以单独支持下载一个的情况是因为我懒得去写和维护:(，默认将需要下载的仓库写在list中就好了。
    :param repo_ids: [repo_id1, repo_id2, ...]
    :param local_dir: 本地存储仓库这个文件夹的路径。这里默认下载至同一个文件夹中。
    """
    for repo_id in repo_ids:
        repo_id = repo_id  # 模型或数据集的ID
        local_dir = Path(local_dir_str) / repo_id  # 本地文件夹路径
        download_model_from_huggingface(repo_id, local_dir)


def main(repo_ids, local_dir_str):
    # 这里更改使用一个镜像。
    os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
    # os.environ['HF_HOME'] = "~/.cache/huggingface"

    # 批量下载多个模型或数据集。
    batch_download(repo_ids, local_dir_str)


if __name__ == "__main__":
    # 仓库id。
    repo_ids = [r"Qwen/Qwen2.5-0.5B-Instruct", r"Qwen/Qwen2-VL-2B-Instruct"]
    # 本地路径，服务器上需要选择一下类型。
    local_dir_str = r"D:/model/"

    main(repo_ids, local_dir_str)
