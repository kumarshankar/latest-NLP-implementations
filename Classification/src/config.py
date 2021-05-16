from transformers import BertTokenizer

MAX_LEN = 512
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 10
BERT_PATH = "../input/bert_base_uncased/"
MODEL_PATH = "pytorch_model.bin"
TRAINING_FILE = "../input/IMDB Dataset.csv"
DEVICE = "cuda"
TOKENIZER = BertTokenizer.from_pretrained(
    BERT_PATH, 
    do_lower_case=True
    )
