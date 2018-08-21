# usage: ./text2bpe_ind.sh [XXXX.trans.txt] [directory] 
. label_config
# step 1: cut the first col to tmp file 
filename=`echo $1 | rev | cut -d '/' -f 1 | rev`
chapter_id=`echo $filename | cut -d '.' -f 1`
cat $1 | cut -d ' ' -f 1 > $tmp_dir/$chapter_id.utt_id.txt
cat $1 | cut -d ' ' -f 2- > $tmp_dir/$chapter_id.txt

# step 2: apply bpe to text file 
../../../subword-nmt/subword_nmt/apply_bpe.py -c $bpe_code_path -i $tmp_dir/$chapter_id.txt -o $tmp_dir/$chapter_id.subword.txt

# step 3: translate bpe to vocab index
python3 bpe2vocab.py $tmp_dir/$chapter_id.subword.txt $vocab_path $tmp_dir/$chapter_id.index.txt

# step 4: merge file horizontally
echo $2
paste -d ',' $tmp_dir/$chapter_id.utt_id.txt $tmp_dir/$chapter_id.index.txt > $2/$chapter_id.label.txt
