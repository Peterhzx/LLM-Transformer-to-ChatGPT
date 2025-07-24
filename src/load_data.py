from tqdm import tqdm


def clean_data(df, allowed_chars):
    df_clean = df.drop_duplicates().dropna().astype(str).reset_index(drop=True)
    invalid_sentence_index = set()
    column_list = df_clean.columns.tolist()

    for col_name in column_list:
        clean_col = []
        for index, sentence in tqdm(enumerate(df_clean[col_name]), desc=f"Cleaning column {col_name}",
                                    total=len(df_clean)):
            filtered_chars = [char for char in sentence if char in allowed_chars]
            clean_sentence = ''.join(filtered_chars)
            if len(clean_sentence) == 0:  # avoid nan in training
                clean_sentence = " "
                invalid_sentence_index.add(index)
            clean_col.append(clean_sentence)
        df_clean[col_name] = clean_col

    valid_indices = [i for i in df_clean.index if i not in invalid_sentence_index]
    df_clean = df_clean.loc[valid_indices].reset_index(drop=True)
    return df_clean
