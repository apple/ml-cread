#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2021 Apple Inc. All Rights Reserved.
#

import os
import sys
import json
import copy
from tqdm import tqdm
from nltk.tokenize import word_tokenize 
from coref_utils import (
	generate_conll_format,
	get_encIdx,
	build_cluster_from_links
)


def filter_links(original_links, turn, ctx, utt):
	'''
		1. filter out improper links
		2. sort links by the mention order in the current utterance
	'''
	links = []
	# iterate all coreference links in annotation
	for link in original_links:
		if len(link) == 1: # incomplete link with only mention or reference
			continue
		reference, mention = link[0]['text'], link[1]['text']
		if reference.lower() == 'i' or mention.lower() == "i": # skip mention or refearence which is "I"
			continue
		if reference.lower() in ["you", "your"] or mention.lower() in ["you", "your"]: # skip link involving "you" which refers to the system
			continue
		if reference.lower() in ["my", "me"] or mention.lower() in ["my", "me"]:
			continue
		if reference.lower() in ["she", "her"] and mention.lower() in ["she", "her"]:
			continue
		if reference.lower() in ["he", "his", "him"] and mention.lower() in ["he", "his", "him"]:
			continue
		if reference.lower() in ['it', 'its', 'he', 'his', 'him', 'she', 'her', 'hers', 'they', 'them', 'their', 'i', 'my', 'mine', 'that', 'you', 'your', 'one']: # remove a link if reference is a pronoun
			continue

		# deal with some rare cases with invalid links
		r_turn_id = link[0]['turn_id']-1
		if r_turn_id > len(ctx): # refer to future turns
			continue
		r_end, m_end = link[0]['span']['end'] + spk_offset, link[1]['span']['end'] + spk_offset
		if m_end >= len(utt): # M out of index of utt
			continue
		if len(ctx) == r_turn_id and r_end >= len(utt):
			continue
		if len(ctx) != r_turn_id and r_end >= len(ctx[r_turn_id]):
			continue
		if mention == 'l the reminder':
			continue
		links.append(link)

	# sort link by the mention start index
	if len(links) == 0:
		return []

	sorted_links = sorted(links, key=lambda x: x[1]['span']['start'])

	# sanity check
	mentions, references = [], []
	prev_end = -1
	for link in sorted_links:
		m_start, m_end = link[1]['span']['start'], link[1]['span']['end']
		assert m_start > prev_end # check no overlapping between mentions
		prev_end = m_end

	return sorted_links


def prepare_coref_signal_idx(ctx, utt, links):
	'''
		get mention / reference span index for coreference modeling
	'''
	ctx = copy.deepcopy(ctx)
	ctx.append(utt)
	coref_span_index = []
	for link in links:
		mention, m_start, m_end = link[1]['text'], link[1]['span']['start']+spk_offset, link[1]['span']['end']+spk_offset
		reference, r_start, r_end, r_turn_id = link[0]['text'], link[0]['span']['start']+spk_offset, link[0]['span']['end']+spk_offset, link[0]['turn_id']-1
		ref_utt = ctx[r_turn_id]
		assert m_end < len(utt)
		assert r_end < len(ref_utt)
		mention_attention = []

		# get index for start word
		m_start_idx, m_start_word = get_encIdx(ctx, utt, len(ctx)-1, m_start, span_end=False)
		r_start_idx, r_start_word = get_encIdx(ctx, ref_utt, r_turn_id, r_start, span_end=False)
		assert m_start_idx not in coref_span_index
		mention_attention.append( {'attention_idx': r_start_idx, 'attention_word': r_start_word, 'mention_type': 'start', 'mention_word': m_start_word, 'mention_idx': m_start_idx} )

		# get index for end word
		m_end_idx, m_end_word = get_encIdx(ctx, utt, len(ctx)-1, m_end, span_end=True)
		r_end_idx, r_end_word = get_encIdx(ctx, ref_utt, r_turn_id, r_end, span_end=True)
		assert m_end_idx not in coref_span_index
		mention_attention.append( {'attention_idx': r_end_idx, 'attention_word': r_end_word, 'mention_type': 'end', 'mention_word': m_end_word, 'mention_idx': m_end_idx} )

		if '_NOT_FOUND_' in [m_start_word, r_start_word, m_end_word, r_end_word]:
			continue
		coref_span_index.append(mention_attention)
	return coref_span_index


def get_rewrite_utterance(turn, turn_idx):
	if turn["graded"] == False: # for turns not graded and not skipped, treated as non-rewrite example
		turn['rewrite_required'] = False
		turn['rewritten_utterance'] = turn['utterance']

	rewrite = proc_utterance(turn['rewritten_utterance'])
	rewrite_happen = turn['rewrite_required']
	return rewrite, rewrite_happen


def proc_utterance(utt):
 	# NOTE: handle discrepency in format between pre-processed utterances in mudoco dataset and our labeled rewritten utterances
	utt = " ".join(word_tokenize(utt.strip()))
	if utt[-1] not in ['.', '!', '?']:
		utt += " ."
	return utt


def include_example(split, spk):
	# dont include sys side for coref evalutaion since 
	# there are coref label is imperfect original dataset and we care more about user side
	if split in ['dev', 'test'] and spk == 'sys':
		return False
	return True




####### STARTS HERE #######
'''
	This script preprocess the dataset into the training format
	It only consider examples (turns) where the speaker is user, namely, 
		we don't rewrite system utterance though they are provided.
'''

# global info
domains = ['calling', 'messaging', 'music', 'news', 'reminders', 'weather']
usr_token, sys_token, cur_token = "<USR>", "<SYS>", "<CUR>"
spk_offset = len(usr_token) + 1 # include a space
total_dial, incomplete_dial = 0, 0 # num of dialogues, total: 7509, incomplete: 168
total_example, used_example = 0, 0 # num of examples

# i/o data path
IN_DATA_PATH = "MuDoCo-QR-dataset"
OUT_DATA_PATH = "proc_data"

content = {data_type: [] for data_type in ['train', 'dev', 'test']}
DATA = {domain: copy.deepcopy(content) for domain in (domains+['all'])}
CONLL_lines = {domain: copy.deepcopy(content) for domain in (domains+['all'])} # for coreference evaluation

for domain in domains:
	with open('{}/mudoco_{}.json'.format(IN_DATA_PATH, domain)) as f:
		data = json.load(f)

	total_dial += len(data['dialogs'])
	for dial_idx, key in enumerate(tqdm(data['dialogs'])):
		dial = data['dialogs'][key]
		split = dial['split']
		if split == 'eval': split = 'dev'
		total_example += len(dial['turns'])
		ctx = []
		skip_turn = False
		ref_indexes = []
		for turn_idx, turn in enumerate(dial['turns']):
			# we filter out bad turn either with imcomplete dialogue context or empty utterance
			if (turn_idx+1) != turn['number']: # missing turns cases, happen in origianl MuDoCo dataset
				skip_turn = True

			if turn['utterance'] == "": # empty utterance
				skip_turn = True

			if skip_turn:
				incomplete_dial += 1
				break

			# start to process a turn
			used_example += 1
			utt = proc_utterance(turn['utterance']) # could be usr or sys
			utt = (cur_token + ' ' + utt)

			# sort link in links
			links = filter_links(turn['links'], turn, ctx, utt)

			# get index of link
			coref_span_index = prepare_coref_signal_idx(ctx, utt, links)

			# get rewrite
			rewrite, rewrite_happen = get_rewrite_utterance(turn, turn_idx)

			if turn_idx % 2 == 0:
				spk = 'usr'
				if len(ctx) > 0:
					assert sys_token in ctx[-1] and usr_token not in ctx[-1]
			else:
				spk = 'sys'
				if len(ctx) > 0:
					assert usr_token in ctx[-1] and sys_token not in ctx[-1]

			if spk == "usr":
				example_idx = "{}-dial{}-turn{}".format(domain, dial_idx, turn_idx)
				proc_dial = {}
				proc_dial['example index'] = example_idx
				proc_dial['dialogue context'] = copy.deepcopy(ctx)
				proc_dial['current utterance'] = utt
				proc_dial['coref happen'] = (len(turn['links']) != 0)
				proc_dial['rewrite utterance'] = rewrite
				proc_dial['rewrite happen'] = rewrite_happen
				proc_dial['link index'] = coref_span_index
				proc_dial['speaker'] = spk
				DATA[domain][split].append(proc_dial)
				DATA['all'][split].append(proc_dial)

			# conll formate data for coreference evaluation
			if spk == "usr" and include_example(split, spk):
				cluster_info = build_cluster_from_links(ctx, utt, coref_span_index)
				lines = generate_conll_format(ctx, utt, cluster_info, example_idx)
				CONLL_lines[domain][split] += lines
				CONLL_lines['all'][split] += lines

			# add current utterance into context
			if turn['number'] % 2 == 0: # sys
				utt = utt.replace(cur_token, sys_token)
			else:
				utt = utt.replace(cur_token, usr_token)
			ctx.append(utt)

	print('Done {} domain!'.format(domain))

print("\nData Info:")
print('# of dialogues -> total: {}, incomplete: {}'.format(total_dial, incomplete_dial))
print('# of examples -> total: {}, used: {}'.format(total_example, used_example))

# write example file and coref label in conll format
print_stats = False # turn this on to print statistics
for domain in domains+['all']:
	os.makedirs("{}/{}".format(OUT_DATA_PATH, domain), exist_ok=True)
	if print_stats: print('In domain: {}'.format(domain))

	for split in ['train', 'dev', 'test']:
		coref_happen, rewrite_happen, rewrite_graded = 0, 0, 0
		for ex in DATA[domain][split]:
			if ex['coref happen']:
				coref_happen += 1
			if ex['rewrite happen']:
				rewrite_happen += 1
			if ex['rewrite utterance'] != "":
				rewrite_graded += 1

		if print_stats: print('# of examples in {} -> {}, coref -> {}, rewrite (graded) -> {}({})'.format(split, len(DATA[domain][split]), coref_happen, rewrite_happen, rewrite_graded))
	
		with open("{}/{}/{}.json".format(OUT_DATA_PATH, domain, split), 'w') as f:
			json.dump(DATA[domain][split], f, indent=4, sort_keys=True)

		if split != 'train':
			with open("{}/{}/{}.conll".format(OUT_DATA_PATH, domain, split), 'w') as f:
				f.writelines(CONLL_lines[domain][split])
