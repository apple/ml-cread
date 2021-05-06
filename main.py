#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2021 Apple Inc. All Rights Reserved.
#

import os
import sys
import json
import time
import random
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from utils.utils import get_args, print_loss, print_score, print_qr_result
from utils.evaluate import score_fn, coref_evaluate, get_coref_summary, get_binary_res
from qr_eval.qr.training.metrics import MetricCollection

from dataset import (
	Dataset,
	SPECIAL_TOKENS
)

from transformers import (
	AdamW,
	AutoConfig,
	AutoTokenizer,
	GPT2LMHeadModel,
	get_linear_schedule_with_warmup,
)

from model import (
	JointModel,
	decode
)


def run_one_epoch(data_type, dataloader, trainer, epoch, run_type, collector=None):
	t0 = time.time()
	assert data_type in ['dev', 'test']
	assert run_type in ['teacher_force', 'generation']
	model, optimizer, scheduler, tokenizer = trainer

	LOSS, match, bi_match = {'bi': 0, 'lm': 0, 'mention': 0, 'reference': 0, 'total': 0}, [], [] # result container
	coref_lines = []
	iterator = enumerate(tqdm(dataloader, desc="Epoch {} {}".format(epoch, run_type), disable=args.disable_display))

	if args.disable_display:
		print('Evaluation progress is not showing')

	for step, batch in iterator:
		if run_type == 'teacher_force':
			loss, _, _, _, _, _, _ = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], \
											token_type_ids=batch['token_type_ids'], labels=batch['label_ids'], \
											mention_labels=batch['mention_label_ids'], batch=batch, coref_links=batch['coref_label'])
			for k, v in loss.items():
				LOSS[k] += v.item()

		else:
			decode_output = decode(args, batch, model, tokenizer)
			score_fn(args, decode_output, batch, match, collector, qr_metric, coref_lines, bi_match)

	# log
	if run_type == 'teacher_force':
		for k, v in LOSS.items():
			LOSS[k] /= (step+1)
		print_loss(epoch, data_type, LOSS, t0)
		return LOSS
	else: # record decoding result
		res = {}
		if 'qr' in args.task:
			qr_res = qr_metric.get_metric(reset=True)
			qr_res['Exact match'] = sum(match) / len(match) * 100
			get_binary_res(bi_match, qr_res, args)
			res['qr'] = qr_res
		else:
			res['qr'] = {}

		if 'coref' in args.task:
			# prepare conll files
			key_path = args.dev_conll if data_type == 'dev' else args.test_conll
			response_path = 'temp/{}.response'.format(args.model_name) # a temp file for calculating coref score
			with open(response_path, 'w') as f:
				f.writelines(coref_lines)
			res['coref'] = coref_evaluate(key_path, response_path, args)
		else:
			res['coref'] = {}

		print_score(args, epoch, data_type, res, t0)
		return res


def set_dataloader(args, tokenizer, data_type, run_type, data_size=-1):
	dataset = Dataset(args, tokenizer, data_type, run_type=='generation', data_size)

	if data_type == 'train':
		sampler = RandomSampler(dataset)
	else:
		sampler = SequentialSampler(dataset)

	dataloader = DataLoader(
		dataset,
		sampler=sampler,
		batch_size=args.train_batch_size if data_type == 'train' else args.eval_batch_size,
		collate_fn=dataset.collate_fn
	)
	return dataloader


def train(args, tokenizer, model):
	# set dataloader
	train_dataloader = set_dataloader(args, tokenizer, 'train', 'teacher_force', data_size=args.train_size)
	dev_dataloader = set_dataloader(args, tokenizer, 'dev', 'teacher_force')
	dev_gen_dataloader = set_dataloader(args, tokenizer, 'dev', 'generation')

	# set optimizer, lr scheduler
	optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
	if args.use_scheduler:
		t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.max_epoch
		scheduler = get_linear_schedule_with_warmup(
			optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
		)
	else:
		scheduler = None
	trainer = (model, optimizer, scheduler, tokenizer)

	print('Test before training!')
	model.eval()
	with torch.no_grad():
		_ = run_one_epoch('dev', dev_dataloader, trainer, -1, 'teacher_force')

	print('Start training!\n{}'.format('***'*30))
	eval_step = args.eval_interval // args.train_batch_size

	# score of query rewrite, corerference resolution, and joint learning (average of two)
	best_score = {'best-QR': -10000, 'best-COREF': -10000, 'best-JOINT': -10000}
	global_step = 0
	no_improve_count = 0
	for epoch in range(args.max_epoch):
		t0 = time.time()
		model.train()
		model.zero_grad()
		LOSS, match = {'bi': 0, 'lm': 0, 'mention': 0, 'reference': 0, 'total': 0}, []
		iterator = enumerate(tqdm(train_dataloader, desc="Epoch {}".format(epoch), disable=args.disable_display))

		if args.disable_display:
			print('Training progress is not showing')

		for local_step, batch in iterator:
			loss, _, _, _, _, _, _ = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], \
											token_type_ids=batch['token_type_ids'], labels=batch['label_ids'], step=None, \
											mention_labels=batch['mention_label_ids'], batch=batch, coref_links=batch['coref_label'])

			for k, v in loss.items():
				LOSS[k] += v.item()
			global_step += 1

			# update model
			if loss['total'].item() != 0:
				loss['total'] = loss['total'] / args.gradient_accumulation_steps
				loss['total'].backward()

			# accumulate gradients
			if global_step % args.gradient_accumulation_steps == 0:
				norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
				optimizer.step()
				if args.use_scheduler:
					scheduler.step()
				optimizer.zero_grad()

			# evaluate model
			if global_step % eval_step == 0 and epoch > 0: 
				model.eval()
				with torch.no_grad():
					loss = run_one_epoch('dev', dev_dataloader, trainer, epoch, 'teacher_force') # get dev loss
					res = run_one_epoch('dev', dev_gen_dataloader, trainer, epoch, 'generation') # get dev result
				model.train()

				# save model
				save_model = save_best_model(args.task, res, best_score)
				if save_model:
					no_improve_count = 0
				else:
					no_improve_count += 1

				# early stop
				if no_improve_count == args.no_improve_max:
					print('Early stop!')
					return

		# get train loss
		for k, v in LOSS.items():
			LOSS[k] /= (local_step+1)
		print_loss(epoch, 'train', LOSS, t0)

		print('***'*30)

	print('Reach max epoch: {}!'.format(args.max_epoch))


def save_best_model(task, res, best_score):
	save_model_flag = False
	if task == "qr": # qr-only model
		score = res['qr']['Macro F1']
		key = "best-QR"
	elif task == "coref": # coref-only model
		score = res['coref']['avg_f1']
		key = "best-COREF"
	else: # joint model, check average of both performance
		score = 0.5*(res['qr']['Macro F1']+res['coref']['avg_f1'])
		key = "best-JOINT"

	if score > best_score[key]:
		save_model_flag = True
		best_score[key] = score
		save_checkpoint(args, tokenizer, model, best_score, key)

	return save_model_flag


def test(args, tokenizer, model):
	# set dataloader
	dev_dataloader = set_dataloader(args, tokenizer, 'dev', 'teacher_force')
	dev_gen_dataloader = set_dataloader(args, tokenizer, 'dev', 'generation')
	test_gen_dataloader = set_dataloader(args, tokenizer, 'test', 'generation')

	trainer = (model, None, None, tokenizer)
	model.eval()
	collector = {'decode-dev': [], 'decode-test': []}
	with torch.no_grad():
		# evaluate on dev
		_ = run_one_epoch('dev', dev_dataloader, trainer, 'Eval', 'teacher_force')

		# generate on dev
		res_dev = run_one_epoch('dev', dev_gen_dataloader, trainer, 'Dev', 'generation', collector=collector['decode-dev'])
		collector['result-dev'] = res_dev
		print_qr_result(args, res_dev['qr'], 'dev')

		# generate on test
		res_test = run_one_epoch('test', test_gen_dataloader, trainer, 'Test', 'generation', collector=collector['decode-test'])
		collector['result-test'] = res_test
		print_qr_result(args, res_test['qr'], 'test')

	out_file = args.decode_file
	with open(out_file, 'w') as f:
		json.dump(collector, f, indent=4, sort_keys=True)
	print('Decode file is saved at {}'.format(out_file))
	print('Done decoding!')


def save_checkpoint(args, tokenizer, model, best_score, best_type):
	save_path = args.checkpoint #+ '_' + best_type
	print('Best score in "{}": {:.2f}. Save model in {}!\n'.format(best_type, best_score[best_type], save_path))
	tokenizer.save_pretrained(save_path)
	model.save_pretrained(save_path)
	

def load_checkpoint(args):
	save_path = args.checkpoint #+ '_' + best_type
	print('Load model, tokenizer from {}'.format(save_path))
	tokenizer = AutoTokenizer.from_pretrained(save_path)
	model = JointModel.from_pretrained(save_path, args=args)
	model.to(args.device)
	return tokenizer, model


def set_model(args):
	''' initiate config, tokenizer and model '''
	config = AutoConfig.from_pretrained(args.model_name_or_path)
	config.attn_pdrop = 0
	tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
	tokenizer.add_special_tokens(SPECIAL_TOKENS)

	model = JointModel(config, args=args)
	if args.load_pretrained_weight:
		model.load_pretrained_weight()

	model.resize_token_embeddings(len(tokenizer))
	model.to(args.device)
	return config, tokenizer, model
	


def set_seed(args):
	''' for reproduction '''
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.enabled = False
	torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
	# load arguments
	args = get_args()

	# set seed, device
	set_seed(args)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	args.device = device

	# set query rewrite evalaution
	qr_metric = MetricCollection()

	if args.mode == 'training':
		config, tokenizer, model = set_model(args)
		train(args, tokenizer, model)

	elif args.mode == 'testing':
		tokenizer, model = load_checkpoint(args)
		test(args, tokenizer, model)
