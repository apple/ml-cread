#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2021 Apple Inc. All Rights Reserved.
#

import sys
import copy
from itertools import chain


def get_encIdx(ctx, sentence, sentenceIdx, charIdx, span_end):
	'''
		Given ctx, sentence and charIdx, update its word (token) in the whole ctx
	'''
	wordIdx, word = charIdx2wordIdx(sentence, charIdx, span_end) # word index in its sentence
	if word == '_NOT_FOUND_':
		return -1, word

	ctx = [s.split() for s in ctx]
	encIdx = sum([len(s) for s in ctx[:sentenceIdx]]) + wordIdx
	return encIdx, word


def charIdx2wordIdx(sentence, charIdx, span_end):
	'''
		Given an sentence (str) and a charIdx, return the corresponding word (token) index
	'''
	assert isinstance(sentence, str) 

	# fix inaccurate end_idx
	if span_end and sentence[charIdx] != " ":
		while sentence[charIdx] != " ":
			charIdx += 1
	if span_end:
		assert sentence[charIdx] == " " and sentence[charIdx+1] != " "
		charIdx += 1

	p_idx = 0
	target_idx = -1
	for token_idx, token in enumerate(sentence.split()):
		if p_idx == charIdx:
			target_idx = token_idx
			break
		else:	
			p_idx += (len(token) + 1) # 1 for space

	# in some system turns (very few, only 6 turns in whole data), labels are inaccurate, which cause M/R cannot be found #
	if target_idx == -1:
		return -1, '_NOT_FOUND_'

	target_word = sentence.split()[target_idx]
	return target_idx, target_word


def align_cluster(clusterIdx2spanList, mention, reference):
	'''
		Align a cluster index to a link (mention/reference)
		If its overlapping with any "reference" in previous links, they are the same cluster. Otherwise, a new cluster is created.
	'''
	assert isinstance(mention, str) and isinstance(reference, str)
	for cluster_idx, span_list in clusterIdx2spanList.items():
		if reference in span_list:
			return cluster_idx

	cluster_idx = len(clusterIdx2spanList)
	if cluster_idx not in clusterIdx2spanList:
		clusterIdx2spanList[cluster_idx] = []
	clusterIdx2spanList[cluster_idx].append(reference)
	return cluster_idx


def build_cluster_from_links(context, curr_utt, links):
	'''
		Return the cluster index of each token in the context + current utterance
	'''
	assert isinstance(context, list) and isinstance(curr_utt, str) #and isinstance(coref_utt, str)
	context = copy.deepcopy(context)
	context.append(curr_utt)
	context = [ s.split() for s in context ]
	context_flat = list(chain(*context))
	cluster_info = [['-']*len(s) for s in context]

	all_MR = []
	for link in links:
		m_start_idx, m_end_idx = link[0]['mention_idx'], link[1]['mention_idx']
		r_start_idx, r_end_idx = link[0]['attention_idx'], link[1]['attention_idx']
		mention = " ".join(context_flat[m_start_idx: m_end_idx])
		reference = " ".join(context_flat[r_start_idx: r_end_idx])
		all_MR.append((mention, reference))

	# fill in cluster indexes starting by the short reference since there might be overlapping between references
	clusterIdx2spanList = {}
	all_MR = sorted(all_MR, key= lambda x: len(x[1].split()))
	for mention, reference in all_MR:
		if mention == "" or reference == "":
			continue
		cluster_idx = align_cluster(clusterIdx2spanList, mention, reference)

		# align cluster index as long as the span can be found in context or current utterance
		fill_in_cluster_info(cluster_idx, cluster_info[-1], context[-1], mention.split())
		for sent_idx, sent in enumerate(context): # consider reference in current utterance as well
			fill_in_cluster_info(cluster_idx, cluster_info[sent_idx], sent, reference.split())

	return cluster_info
			

def build_cluster(context, curr_utt, coref_utt, ground_truth=True):
	'''
		return the cluster index of each token in the context + current utterance
	'''
	assert isinstance(context, list) and isinstance(curr_utt, str) and isinstance(coref_utt, str)
	context = copy.deepcopy(context)
	context.append(curr_utt)
	context = [ s.split() for s in context ]
	coref_utt = coref_utt.split()

	cluster_info = [['-']*len(s) for s in context]
	if ground_truth:
		assert coref_utt.count('<M>') == coref_utt.count('</M>')
		assert coref_utt.count('<R>') == coref_utt.count('</R>')
		assert coref_utt.count('<M>') == coref_utt.count('<R>')

	is_mention, is_reference = False, False
	all_MR = []
	for token in coref_utt:
		if token == '<M>':
			is_mention = True
			mention = []

		elif token == '</M>':
			is_mention = False

		elif token == '<R>':
			is_reference = True
			reference = []

		elif token == '</R>':
			is_reference = False
			mention = ' '.join(mention)
			reference = ' '.join(reference)
			all_MR.append((mention, reference))
		else:
			if is_mention:
				mention.append(token)
			elif is_reference:
				reference.append(token)

	# fill in cluster indexes starting by the short reference since there might be overlapping between references
	clusterIdx2spanList = {}
	all_MR = sorted(all_MR, key= lambda x: len(x[1].split()))
	for mention, reference in all_MR:
		if mention == "" or reference == "":
			continue
		cluster_idx = align_cluster(clusterIdx2spanList, mention, reference)

		# align cluster index as long as the span can be found in context or current utterance
		fill_in_cluster_info(cluster_idx, cluster_info[-1], context[-1], mention.split())
		for sent_idx, sent in enumerate(context): # consider reference in current utterance as well
			fill_in_cluster_info(cluster_idx, cluster_info[sent_idx], sent, reference.split())

	return cluster_info


def fill_in_cluster_info(idx, info, sentence, span):
	indexes = find_span_index(sentence, span)
	for start_idx, end_idx in indexes:
		# NOTE: might happen when generated coref_utt is bad or substring overlapping between references
		if start_idx == end_idx:
			info[start_idx] = '({})'.format(idx)
		else:
			info[start_idx] = '({}'.format(idx)
			info[end_idx] = '{})'.format(idx)


def find_span_index(sentence, span):
	'''
		return all matched span (start, end) indexes in a sentence
	'''
	assert isinstance(sentence, list)
	assert isinstance(span, list)
	indexes = []
	for w_idx, w in enumerate(sentence):
		t_offset = 0
		find = True
		while t_offset < len(span):
			if (w_idx+t_offset) >= len(sentence) or span[t_offset] != sentence[w_idx+t_offset]:
				find = False
				break
			t_offset += 1
	
		if find:
			start_idx = w_idx
			end_idx = w_idx + len(span) - 1
			indexes.append((w_idx, end_idx))

	return indexes	


def generate_conll_format(context, curr_utt, cluster_info, example_idx):
	'''
		Generate CONLL format for coreference resolution evaluation
	'''
	context = copy.deepcopy(context)
	context.append(curr_utt)
	context = [ s.split() for s in context ]
	assert len(context) == len(cluster_info)
	lines = []
	lines.append('#begin document (example-{});'.format(example_idx))
	for sent_idx, (tokens, cluster_indexes) in enumerate(zip(context, cluster_info)):
		lines.append('\n')
		assert len(tokens) == len(cluster_indexes)
		spk = 'USR' if sent_idx % 2 == 0 else 'SYS'
		for token_idx, token in enumerate(tokens):
			line = "{}\t{}\t{}\t{}\t{}\n".format(spk, example_idx, token_idx, token, cluster_indexes[token_idx])
			lines.append(line)

	lines.append('#end document\n\n')
	return lines


if __name__ == '__main__':
	pass
