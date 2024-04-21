text_inputs = [
    'Data/WikiQA-train.txt',
    'Data/WikiQA-test.txt',
    'Data/example.txt',
    'Data/harry.txt',
    'Data/harry2.txt',
    'Data/harry3.txt',
    'Data/harry4.txt',
    'Data/shakespeare.txt',
]
val_split = .2
block_size = 64
batch_size = 32
epochs = 4
model_name = '1711182129' #None #'1710872451' #
min_token_count = 10e-7
