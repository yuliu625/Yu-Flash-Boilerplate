#旧的我使用命令行在huggingface上下载模型和数据集的脚本。
conda activate dl_env
#这个是下载模型的。
huggingface-cli download --resume-download ${} --local-dir /home/models/hf/${}/${}
#这个是下载数据集的。
huggingface-cli download --repo-type dataset --resume-download ${} --local-dir ${}
