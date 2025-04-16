from transformers import AutoTokenizer

def hf_tokenizer(
    path: str,
    pad_token: str | None = None,
    max_seq_len: int | None = None,
):
    tokenizer = AutoTokenizer.from_pretrained(
        path,
        pad_token = pad_token,
        model_max_length = max_seq_len
    )
    
    tokenizer.pad_id = tokenizer.pad_token_id
    tokenizer.eos_id = tokenizer.eos_token_id
    
    return tokenizer
