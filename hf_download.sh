conda activate dl_env
huggingface-cli download --resume-download ${} --local-dir /home/liuyu/liuyu_data/models/hf/${}/${}

huggingface-cli download --repo-type dataset --resume-download wikitext --local-dir wikitext
