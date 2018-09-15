"""
code for processing wsj data
turn character to bpe tokens
"""
import sys
import glob 
import os 
import subprocess
import json
import kaldi_io
import pickle

def load_data(directory):
    feature = {key: mat for key, mat in kaldi_io.read_mat_scp(os.path.join(directory, 'feats.scp'))}
    with open(os.path.join(directory, 'data.json')) as f:
        data = json.load(f)
    return feature, data

def load_dict(dict_path):
    vocab_dict = {}
    with open(dict_path) as f:
        for i, line in enumerate(f):
            # no UNK in character-based 
            if i == 0:
                continue
            sym, ind = line.strip().split(maxsplit=1)
            new_ind = int(ind) - 2
            vocab_dict[sym] = new_ind
    return vocab_dict

def store_data(output_dict, feature, data_dict):
    for utt_id in data_dict['utts']:
        # 2 is <BLANK>, <UNK>
        token_ids = [int(token_id) - 2 for token_id in data_dict['utts'][utt_id]['output'][0]['tokenid'].split()]
        features = feature[utt_id]
        output_dict[utt_id] = {}
        output_dict[utt_id]['feature'] = features
        output_dict[utt_id]['token_ids'] = token_ids
    return 

#def collect_text(data_dict):
#    sents = []
#    for utt_id in data_dict['utts']:
#        text = data_dict['utts'][utt_id]['output'][0]['token_id']
#        sents.append(text)
#
#    return sents

if __name__ == '__main__':
    if len(sys.argv) < 5:
        print('usage: python3 preprocess.py [root_dir] [dsets (ex. train_si84,train_si284...)] [dict_path] '
                '[output_dir]')

    root_dir = sys.argv[1]
    dsets = sys.argv[2].strip().split(',')
    dict_path = sys.argv[3]
    output_dir = sys.argv[4]

    # dump dict
    vocab_dict = load_dict(dict_path)
    dict_output_path = os.path.join(output_dir, 'vocab_dict.pkl')
    with open(dict_output_path, 'wb') as f:
        pickle.dump(vocab_dict, f)
    
    # process data
    in_dir = 'deltafalse'
    for i, dset in enumerate(dsets):
        data = {}
        print(f'processing {dset}...')
        directory = os.path.join(root_dir, f'{dset}/{in_dir}')
        print('load data...')
        feature, data_dict = load_data(directory)
        store_data(data, feature, data_dict)
        del feature, data_dict
        data_output_path = os.path.join(output_dir, f'{dset}.pkl')
        with open(data_output_path, 'wb') as f:
            pickle.dump(data, f)
        del data

        '''
        deprecate
        '''
        ## write data to all_text to generate bpe
        #sents = collect_text(data_dict)
        #text_to_write = '\n'.join(sents)
        #all_text = os.path.join(bpe_output_dir, f'{dset}.txt')
        #with open(all_text, 'w') as f:
        #    f.write(text_to_write)

        #    # only for label_dset
        #    if i == 0:
        #        # learn bpe
        #        learn_bpe_path = os.path.join(bpe_root_dir, 'learn_bpe.py')
        #        bpe_code_path = os.path.join(bpe_output_dir, 'bpe_code.txt')
        #        cmd = f'python3 {learn_bpe_path} -t -s {n_bpe_tokens} -i {all_text} -o {bpe_code_path}'
        #        subprocess.run(cmd.split())

        #    # apply
        #    apply_bpe_path = os.path.join(bpe_root_dir, 'apply_bpe.py')
        #    output_bpe_path = os.path.join(bpe_output_dir, f'{dset}_bpe.txt')
        #    cmd = f'python3 {apply_bpe_path} -i {all_text} -c {bpe_code_path} -o {output_bpe_path} -s __'
        #    subprocess.run(cmd.split())

