python3 collect_all_text.py /storage/datasets/LibriSpeech/LibriSpeech/ train-clean-100 ../../processed_data/train-text.txt
python3 collect_all_text.py /storage/datasets/LibriSpeech/LibriSpeech/ train-clean-100,train-clean-360,train-other-500,dev-clean,dev-other,test-clean,test-other  ../../processed_data/all-text.txt
