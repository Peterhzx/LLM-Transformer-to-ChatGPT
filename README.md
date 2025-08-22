### Project is in progress.
#### Transformer has been reproduced.
- all_chars formate: string of all allowed character with no separation or list of unicode integers
- data formate: .csv .xlsx .json **mind that:** there should be only two columns. First is the source language and second be the target
- regex: example: en_tokenizer_regex = r"""\d+(?:[\.,]\d+)* | \w+(?:[-']\w+)* | \S\S+ | \S""" The last two expression will be excluded in training tokenizer
- special_tokens should not include <EOW> which stands for End Of Word. it was built in the tokenizer
- when using Wordpiece tokenizer, mind that the number of Wordpiece is always even, since every merged token has a sub-word version. so be careful about the number of params like num_tokens in models and random_token_end in dataloader