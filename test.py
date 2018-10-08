import sys
import json 
import editdistance

def cer(hyps, refs):
    total_dis, total_len = 0., 0.
    for hyp, ref in zip(hyps, refs):
        dis = editdistance.eval(hyp, ref)
        total_dis += dis
        total_len += len(ref)
    return total_dis / total_len

hyp_path = sys.argv[1]
ref_path = sys.argv[2]

with open(ref_path, 'r') as f:
    refs = []
    data_dict = json.load(f)
    for utt_id in data_dict['utts']:
        tokens = data_dict['utts'][utt_id]['output'][0]['token'].split()
        tokens = list(map(lambda x: ' ' if x == '<space>' else x, tokens))
        tokens = list(filter(lambda x: x != '<NOISE>', tokens))
        text = ''.join(tokens)
        refs.append(text)

with open(hyp_path, 'r') as f:
    hyps = [line.strip() for line in f.readlines()]

print(cer(hyps, refs))


