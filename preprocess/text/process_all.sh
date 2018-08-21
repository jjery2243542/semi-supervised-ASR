. label_config
for dataset_dir in $librispeech_dir/*/; do 
    dataset=`echo $dataset_dir | rev | cut -d '/' -f 2 | rev`
    echo processing $dataset...
    for filename in $dataset_dir*/*/*.trans.txt; do
        speaker_id=`echo $filename | rev | cut -d '/' -f 3 | rev`
        if [ ! -d $label_dir/$speaker_id ]; then 
            mkdir -p $label_dir/$dataset/$speaker_id
        fi
        bash text2bpe_ind.sh $filename $label_dir/$dataset/$speaker_id
    done
done
