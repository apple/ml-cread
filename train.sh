#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2021 Apple Inc. All Rights Reserved.
#

# general info
model_name=$1
task=qr_coref
batch_size=15
lr=5e-05 # learning rate
seed=1

# data path
domain='all' # one of the following: all, calling, messaging, music, news, reminders, weather
train_file='proc_data/'$domain'/train.json'
dev_file='proc_data/'$domain'/dev.json'
test_file='proc_data/'$domain'/test.json'
dev_conll='proc_data/'$domain'/dev.conll'
test_conll='proc_data/'$domain'/test.conll'

# coreference resolution
coref_layer_idx=10,11
n_coref_head=3

# coref2qr attention
use_coref_attn=true
coref_attn_layer=7

# binary classification
use_binary_cls=true

checkpoint='checkpoint/'$model_name
log='log/'$model_name'.log'
mkdir -p temp/
python main.py  --mode="training" --seed=$seed --task=$task \
				--model_name=$model_name --checkpoint=$checkpoint \
				--train_file=$train_file --dev_file=$dev_file --test_file=$test_file \
				--dev_conll=$dev_conll --test_conll=$test_conll \
				--train_batch_size=$batch_size --learning_rate=$lr \
				--coref_layer_idx=$coref_layer_idx --n_coref_head=$n_coref_head \
				--use_coref_attn=$use_coref_attn --coref_attn_layer=$coref_attn_layer \
				--use_binary_cls=$use_binary_cls > $log
