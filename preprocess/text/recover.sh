# $1: file produced by apply-bpe, $2: output file
sed -r 's/(@@ )|(@@ ?$)//g' < $1 > $2
