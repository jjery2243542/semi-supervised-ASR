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
import glob

def load_data(directory):
    feature = {}
    for ark_file in sorted(glob.glob(os.path.join(directory, '*.ark'))):
        print(f'loading {ark_file}...')
        for key, mat in kaldi_io.read_mat_ark(os.path.join(directory, ark_file)):
            feature[key] = mat

    with open(os.path.join(directory, 'data.json')) as f:
        data = json.load(f)
    return feature, data

def load_dict(dict_path, non_char_syms):
    vocab_dict = {'<PAD>':0, '<BOS>':1, '<EOS>':2}
    with open(dict_path) as f:
        for i, line in enumerate(f):
            # no UNK in character-based 
            if i == 0:
                continue
            sym, ind = line.strip().split(maxsplit=1)
            # only add characters and some syms 
            if sym in non_char_syms or sym.isalpha():
                vocab_dict[sym] = len(vocab_dict)
    return vocab_dict

def get_token_ids(data_dict, vocab_dict):
    data = {}
    for utt_id in data_dict['utts']:
        tokens = data_dict['utts'][utt_id]['output'][0]['token'].split()
        token_ids = [vocab_dict[token] for token in tokens if token in vocab_dict]
        data[utt_id] = token_ids
    return data

def merge_data(feature, token_ids):
    data = {}
    for utt_id in feature:
        data[utt_id] = {'feature': feature[utt_id], 'token_ids': token_ids[utt_id]}
    return data

#def collect_text(data_dict):
#    sents = []
#    for utt_id in data_dict['utts']:
#        text = data_dict['utts'][utt_id]['output'][0]['token_id']
#        sents.append(text)
#
#    return sents

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('usage: python3 preprocess.py [root_dir] [dict_path] [output_dir]')

    root_dir = sys.argv[1]
    dict_path = sys.argv[2]
    output_dir = sys.argv[3]

    labeled = 'train_si84'
    unlabeled = 'train_si284'
    dev = 'test_dev93'
    test = 'test_eval92'

    non_char_syms = ['\'', '.', '-', '<space>', '<NOISE>']
    # dump dict
    vocab_dict = load_dict(dict_path, non_char_syms)
    dict_output_path = os.path.join(output_dir, 'vocab_dict.pkl')
    with open(dict_output_path, 'wb') as f:
        pickle.dump(vocab_dict, f)

    # load non-lang sym
    non_lang_syms = ['<NOISE>', '<PAD>', '<BOS>', '<EOS>']
    non_lang_syms_output_path = os.path.join(output_dir, 'non_lang_syms.pkl')
    with open(non_lang_syms_output_path, 'wb') as f:
        pickle.dump(non_lang_syms, f)
    
    # process data
    in_dir = 'deltatrue'
    dsets = [labeled, unlabeled, dev, test]
    utterances = {}
    for i, dset in enumerate(dsets):
        print(f'processing {dset}...')
        directory = os.path.join(root_dir, f'{dset}/{in_dir}')
        print('load data...')
        feature, data_dict = load_data(directory)
        token_ids = get_token_ids(data_dict, vocab_dict)
        data = merge_data(feature, token_ids)
        print(f'total utterance={len(data)}')
        print('dump data...')
        data_output_path = os.path.join(output_dir, f'{dset}.pkl')
        with open(data_output_path, 'wb') as f:
            pickle.dump(data, f)
        # recording utterance ids
        utterances[dset] = list(data.keys())
        # remove the utterances in si84, and dump si200
        if dset == unlabeled:
            print(f'remove {len(utterances[labeled])} utterances in {unlabeled} set')
            for utt_to_remove in utterances[labeled]:
                data.pop(utt_to_remove)
            data_output_path = os.path.join(output_dir, 'train_si200.pkl')
            with open(data_output_path, 'wb') as f:
                pickle.dump(data, f)
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

