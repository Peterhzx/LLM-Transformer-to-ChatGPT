## Project is in progress.
### Transformer and BERT(pre-train) have been reproduced.
### TODO: write comments
- all_chars formate: string of all allowed character with no separation or list of unicode integers
- data formate: .csv .xlsx .json .parquet **mind that:** there should be only two columns. First is the source language and second be the target. In nsp task, the second should be the next sentence of the first.
- regex: example: en_tokenizer_regex = r"""\d+(?:[\.,]\d+)* | \w+(?:[-']\w+)* | \S\S+ | \S""" The last two expression will be excluded in training tokenizer
- special_tokens in config of BPE should not include <EOW> which stands for End Of Word. it was built in the tokenizer