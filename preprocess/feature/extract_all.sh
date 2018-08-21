. fbank_config
#for filename in $librispeech_dir/*/*/*/*; do
for dataset_dir in $librispeech_dir/*/; do 
    dataset=`echo $dataset_dir | rev | cut -d '/' -f 2 | rev`
    echo processing $dataset...
    for filename in $dataset_dir*/*/*.flac; do
        bash fbank_from_one_flac.sh $filename
    done
done
