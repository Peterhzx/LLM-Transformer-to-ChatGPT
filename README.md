## Project is in progress.
### Transformer and BERT(pre-train) have been reproduced.
### TODO: write comments
- all_chars formate: string of all allowed character with no separation or list of unicode integers
- data formate: .csv .xlsx .json .parquet **mind that:** there should be only two columns. First is the source and second be the target. In nsp task, the second should be the next sentence of the first.
- regex: example: en_tokenizer_regex = r"""\d+(?:[\.,]\d+)* | \w+(?:[-']\w+)* | \S\S+ | \S""" The last two expression will be removed in training tokenizer
- special_tokens in config of BPE should not include <EOW> which stands for End Of Word. it was built in the tokenizer
- when using sagemaker mode, the only channel should be named as "train" for mapping training dataset. And also data_file in dataloader config shoulde be the file name without parent dir.
- max_num_ckpt in Trainer config is the max num of ckpt to keep. -1 means no limit, 0 means do not save ckpt anymore.