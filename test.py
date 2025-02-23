import utils



text_inputs = [
    # 'Data/WikiQA-train.txt',
    # 'Data/WikiQA-test.txt',
    # 'Data/example.txt',
    'Data/harry.txt',
    # 'Data/harry2.txt',
    # 'Data/harry3.txt',
    # 'Data/harry4.txt',
    # 'Data/shakespeare.txt',
]
train_text, val_text = utils.train_val_split(text_inputs)  
   
text = train_text 
vocab_size = 2000

print(utils.mytokenizer(text, vocab_size))