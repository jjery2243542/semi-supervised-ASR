import sys
# usage: python3 bpe2vocab.py [bpe_file_path] [vocab_file_path] [index_file_path] 
if len(sys.argv) < 4:
    exit(0)

bpe_path = sys.argv[1]
vocab_path = sys.argv[2]
index_path = sys.argv[3]

# read vocab 
vocab_dict = {}
with open(vocab_path, 'r') as f_in:
    for line in f_in:
        word, _ = line.strip().split(maxsplit=1)
        vocab_dict[word] = len(vocab_dict)

print(f'vocab_size={len(vocab_dict)}')

with open(bpe_path, 'r') as f_bpe, open(index_path, 'w') as f_ind:
    for line in f_bpe:
        indexes = [str(vocab_dict[word]) for word in line.strip().split()]
        indexes_string = ' '.join(indexes)
        f_ind.write(f'{indexes_string}\n')

