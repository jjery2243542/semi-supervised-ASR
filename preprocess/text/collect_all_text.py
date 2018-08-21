import sys
import os
import glob 

if __name__ == '__main__':
    n_args = 3
    if len(sys.argv) < n_args + 1:
        print('usage: python3 collect_all_text.py [LibriSpeech root] [dset1,dset2,dset3...] [output file path]')
        exit(0)

    root_dir = sys.argv[1]
    dsets = sys.argv[2].split(',')
    output_path = sys.argv[3]
    with open(output_path, 'w') as f_out:
        for dset in dsets:
            print(f'processing {dset}')
            for text_file_path in sorted(glob.glob(os.path.join(root_dir, f'{dset}/*/*/*.trans.txt'))):
                #print(f'{text_file_path}')
                with open(text_file_path, 'r') as f_in:
                    lines = [line.strip() for line in f_in.readlines()]
                    lines = [line.split(maxsplit=1)[1] for line in lines]
                    for line in lines:
                        f_out.write(f'{line}\n')
