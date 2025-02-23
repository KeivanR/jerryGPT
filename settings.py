text_inputs = [
    # 'Data/WikiQA-train.txt',
    # 'Data/WikiQA-test.txt',
    # 'Data/example.txt',
    'Data/harry.txt',
    'Data/harry2.txt',
    'Data/harry3.txt',
    'Data/harry4.txt',
    # 'Data/shakespeare.txt',
]
fine_tuning = [
    'Data/WikiQA-train.txt',
    'Data/WikiQA-test.txt',
]
if len(fine_tuning)==0:
    fine_tuning = None
val_split = .2
block_size = 64
batch_size = 32
n_epochs = 4
n_epochs_fine_tuning = 2
model_name = None#'1713851209' #None #'1710872451' #
min_token_count = 10e-7

delim0 = ['" ', ' "', "'", "-", "/", "newline", "_", "—", "‘", " “", "“ "]
delimL = ["(", "£", "`", "{", "~"]
delimR = [",", ".", "!", "?", ":", ")", "$", "*", "}"]
delimRL = [";", "|", "+", "&"]
delimMaj = [' "', ".", "!", "?", "newline"]
all_delims = delim0 + delimL + delimR + delimRL
