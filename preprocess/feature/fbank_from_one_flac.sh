# usage: fbank.sh [src_flac_path]
. fbank_config
# step 1: file conversion & cut utt_id
filename=`echo $1 | rev | cut -d '/' -f 1 | rev`
utt_id=`echo ${filename::-5}`
dataset=`echo $1 | rev | cut -d '/' -f 4 | rev`
echo shell:$1
python3 convert.py $1 $tmp_dir/$utt_id.wav $sample_rate

# step 2: generate tmp scp
echo $utt_id $tmp_dir/$utt_id.wav > $tmp_dir/$utt_id.scp 

# step 3: compute fbank
compute-fbank-feats --sample-frequency=$sample_rate --num-mel-bins=$num_mel_bins scp:$tmp_dir/$utt_id.scp ark,t:$tmp_dir/$utt_id.ark

# step 4: convert to numpy array
python3 ark2npy.py $tmp_dir/$utt_id.ark $npy_dir $dataset $utt_id

# step 5: rm files in tmp
rm -rf $tmp_dir/$utt_id.wav
rm -rf $tmp_dir/$utt_id.scp
rm -rf $tmp_dir/$utt_id.ark
