#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2021 Apple Inc. All Rights Reserved.
#

import re
import sys
import time
import argparse

def str2bool(v):
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:   
		raise argparse.ArgumentTypeError('Boolean value expected.')

def verify_args(args):
	assert args.mode in ['training', 'testing']
	assert args.eval_batch_size == 1
	if args.mode == 'testing': args.decode_file != ''
	assert args.task in ['qr', 'coref', 'qr_coref']
	if 'coref' in args.task:
		assert args.dev_conll != ''
		assert args.test_conll != ''
	assert args.n_coref_head >= 1 and args.n_coref_head <= 12
	assert args.coref_attn_layer <= 12
	assert args.class0_loss_w > 0
	assert args.class1_loss_w > 0


def str2list(v):
	''' convert str into list, e.g., "1,2,3" -> [1,2,3] '''
	# format check
	res = re.findall(r"[^0-9,]", v) # only allow , and digits
	assert len(res) == 0
	res = re.findall(r",,+", v)
	assert len(res) == 0
	
	l = v.split(',')
	l = sorted([int(x) for x in l])
	assert len(l) > 0
	assert len(l) == len(set(l)) # repeat element
	assert min(l) >= 0 and max(l) <= 11
	return l


def get_args():
	parser = argparse.ArgumentParser(description='')

	# general info
	parser.add_argument('--mode', type=str, required=True, help='')
	parser.add_argument('--task', type=str, required=True, help='which task [qr, coref, qr_coref] to perform? \
		`qr` for `qr-only` model; `coref` for `coref-only` model; `both` for `joint learning` model')
	parser.add_argument('--seed', type=int, default=1122)
	parser.add_argument('--model_name', type=str, required=True, help='model name, can be random but has to be provided')
	parser.add_argument('--model_name_or_path', type=str, default='gpt2')
	parser.add_argument('--checkpoint', type=str, default='', help='path of your model to save/load')
	parser.add_argument('--disable_display', type=str2bool, default=False, help='display progress bar or not')

	# data path
	parser.add_argument('--train_file', type=str, default='')
	parser.add_argument('--dev_file', type=str, default='')
	parser.add_argument('--test_file', type=str, default='')
	parser.add_argument('--dev_conll', type=str, default='')
	parser.add_argument('--test_conll', type=str, default='')

	# training
	parser.add_argument('--load_pretrained_weight', type=str2bool, default=True, \
		help='whether to load pretrained gpt2 weight or train from scratch')
	parser.add_argument('--train_batch_size', type=int, default=15, help='batch size of training per gpu')
	parser.add_argument('--eval_batch_size', type=int, default=1, help='batch size of evaluation per gpu')
	parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help="steps for accumulating gradients")
	parser.add_argument('--learning_rate', type=float, default=6.25e-5)
	parser.add_argument('--adam_epsilon', type=float, default=1e-12)
	parser.add_argument('--max_grad_norm', type=float, default=1.0)
	parser.add_argument('--max_epoch', type=int, default=16)
	parser.add_argument('--use_scheduler', type=str2bool, default=True, help='whether to use lr scheduler')
	parser.add_argument('--warmup_steps', type=int, default=0)
	parser.add_argument('--train_size', type=int, default=-1, help='examples used for training. -1 means all data')
	parser.add_argument('--eval_interval', type=int, default=16000, help='how frequent (in steps) to evaluate the model during training')
	parser.add_argument('--no_improve_max', type=int, default=5, help='The max tolerance for model not improving')
	parser.add_argument('--eps', type=float, default=1e-12)
#	parser.add_argument('--fp16', type=str2bool, default=False, help='Whether to use float16')
#	parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training (-1: not distributed)')

	# coreference resolution
	parser.add_argument('--coref_layer_idx', type=str2list, default=[10,11], \
		help='which layer to use for coref prediction, e.g., "1,5,11". 0<=n<=11')
	parser.add_argument('--n_coref_head', type=int, default=1, \
		help='How many heads to be used for coref prediction in each layer. 1<=n<=12')

	# coref2qr attention
	parser.add_argument('--use_coref_attn', type=str2bool, default=True, help="whether to use coref2qr attention")
	parser.add_argument('--coref_attn_layer', type=int, default=1, help="how many layer involved in coref2qr attention")
	parser.add_argument('--coref_attn_mention', type=str2bool, default=False, \
		help="whether to consider mentions' hidden states for coref2qr")
	parser.add_argument('--coref_attn_share_between_layer', type=str2bool, default=True, \
		help="whether to share parameters in coref2qr attention across layers")
	
	# binary classification
	parser.add_argument('--use_binary_cls', type=str2bool, default=True, help="whether to use binary classification")
	parser.add_argument('--filter_not_rewrite_loss', type=str2bool, default=True, \
		help="if True, lm loss of examples not requiring rewrite won't be considered")
	parser.add_argument('--copy_not_rewrite', type=str2bool, default=True, \
		help="if True, the model copies the input query as output when it predicts `no-rewrite`")
	parser.add_argument('--class0_loss_w', type=float, default=1., help="loss weight for `no-rewrite` class")
	parser.add_argument('--class1_loss_w', type=float, default=1.5, help="loss weight for `rewrite` class")
	
	# decoding
	parser.add_argument('--dec_max_len', type=int, default=100)
	parser.add_argument('--num_beams', type=int, default=1)
	parser.add_argument('--temperature', type=float, default=1.0)
	parser.add_argument('--decode_file', type=str, default='')

	args = parser.parse_args()
	print(args)
	verify_args(args)
	return args


def print_loss(epoch, data_type, LOSS, t0):
	print('Epoch: {} | {} total loss: {:.3f} (binary: {:.2f}, rewrite: {:.3f}, mention: {:.3f}, reference: {:.3f}) | time: {:.1f}'.format(epoch, data_type, LOSS['total'], LOSS['bi'], LOSS['lm'], LOSS['mention'], LOSS['reference'], time.time()-t0))
	print('Epoch: {} | {} total loss: {:.3f} (binary: {:.2f}, rewrite: {:.3f}, mention: {:.3f}, reference: {:.3f}) | time: {:.1f}'.format(epoch, data_type, LOSS['total'], LOSS['bi'], LOSS['lm'], LOSS['mention'], LOSS['reference'], time.time()-t0), file=sys.stderr)


def print_score(args, epoch, data_type, res, t0):
	if 'qr' in args.task:
		qr_p, qr_r, qr_f1, qr_f1_2 = res['qr']['Macro P'], res['qr']['Macro R'], res['qr']['Macro F1'], res['qr']['Micro F1']
		qr_bi_f1, qr_bi_m = res['qr']['Binary F1'], res['qr']['Binary M']
		if qr_bi_f1 is None: qr_bi_f1 = 0 
		if qr_bi_m is None: qr_bi_m = 0
	else:
		qr_p, qr_r, qr_f1, qr_f1_2 = 0, 0, 0, 0
		qr_bi_f1, qr_bi_m = 0, 0
	if 'coref' in args.task:
		coref_p, coref_r, coref_f1, lea_f1 = res['coref']['avg_precision'], res['coref']['avg_recall'], res['coref']['avg_f1'], res['coref']['lea']['f1']
	else:
		coref_p, coref_r, coref_f1, lea_f1 = 0, 0, 0, 0
	print('Epoch: {} | {}: QR Binary F1: {:.2f} ({:.2f}), LM: P: {:.2f} R: {:.2f} F1: {:.2f} ({:.2f}) | COREF P: {:.2f} R: {:.2f} F1: {:.2f} ({:.2f}) | time: {:.1f}'.format(epoch, data_type, qr_bi_f1, qr_bi_m, qr_p, qr_r, qr_f1, qr_f1_2, coref_p, coref_r, coref_f1, lea_f1, time.time()-t0))
	print('Epoch: {} | {}: QR Binary F1: {:.2f} ({:.2f}), LM: P: {:.2f} R: {:.2f} F1: {:.2f} ({:.2f}) | COREF P: {:.2f} R: {:.2f} F1: {:.2f} ({:.2f}) | time: {:.1f}'.format(epoch, data_type, qr_bi_f1, qr_bi_m, qr_p, qr_r, qr_f1, qr_f1_2, coref_p, coref_r, coref_f1, lea_f1, time.time()-t0), file=sys.stderr)

def print_qr_result(args, res, split):
	if 'qr' in args.task:
		print("\n{} QR Result on {} {}".format('***'*8, split, '***'*8))
		print("\tMacro P R F1 | Micro P R F1 | BLEU | ROUGE-1 -2 L F1\n\t{:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}"\
		.format(res['Macro P'], res['Macro R'], res['Macro F1'], res['Micro P'], res['Micro R'], res['Micro F1'], res['BLEU'], res['ROUGE1 F1'], res['ROUGE2 F1'], res['ROUGEL F1']))
		print("\tMacro P R F1 | Micro P R F1 | BLEU | ROUGE-1 -2 L F1\n\t{:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}"\
		.format(res['Macro P'], res['Macro R'], res['Macro F1'], res['Micro P'], res['Micro R'], res['Micro F1'], res['BLEU'], res['ROUGE1 F1'], res['ROUGE2 F1'], res['ROUGEL F1']), file=sys.stderr)
		print('***'*35, '\n\n')
	else:
		print('qr is not in task')
		print('qr is not in task', file=sys.stderr)
