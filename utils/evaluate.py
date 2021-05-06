#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2021 Apple Inc. All Rights Reserved.
#

''' the evaluation code for coreference resolution is adapted from https://github.com/ns-moosavi/coval '''

import sys
import os
sys.path.append(os.getcwd())
from nltk.tokenize import word_tokenize 

from utils.coref_utils import generate_conll_format, build_cluster, build_cluster_from_links
from coval.conll import reader
from coval.conll import util
from coval.eval import evaluator

def _post_for_qr_eval(utt):
	if utt == "" or utt == None:
		return utt

	if utt[-1] in [',', '.', '?', '!']:
		utt = utt[:-1]

	utt = utt.replace('<CUR>', '')
	utt = word_tokenize(utt) # a list
	return utt

def score_fn(args, decode, batch, match, collector, qr_metric, coref_lines, bi_match=None):
	batch_size = len(batch['curr_utt'])
	for bs_idx in range(batch_size):
		example_id = batch['example_ids'][bs_idx]
		spk = batch['spk'][bs_idx]
		whole_input = batch['whole_input'][bs_idx]
		ctx = batch['context'][bs_idx]
		usr = batch['curr_utt'][bs_idx]
		qr_ref = batch['rewt_utt'][bs_idx]
		binary_class = batch['binary_rewrite'][bs_idx]

		qr_hyp = decode['qr'][bs_idx]
		coref_hyp = decode['coref'][bs_idx]
		binary_class_pred = None

		# compute qr score
		if 'qr' in args.task and qr_ref != "":
			proc_usr = _post_for_qr_eval(usr)
			proc_qr_ref = _post_for_qr_eval(qr_ref)
			proc_qr_hyp = _post_for_qr_eval(qr_hyp)

			# store in metric
			qr_metric([proc_qr_ref], [proc_qr_hyp], [proc_usr])

			# exact match rate, not used 
			match.append(float((' '.join(proc_qr_hyp)).lower()==(' '.join(proc_qr_ref)).lower()))

			# binary rewrite score
			if args.use_binary_cls:
				binary_class_pred = decode['bi'][bs_idx] # 0/1
				assert isinstance(binary_class_pred, int) and binary_class_pred in [0, 1]
				assert isinstance(binary_class, bool)
				bi_match.append((binary_class_pred, int(binary_class)))

		# compute coref score
		if 'coref' in args.task and spk == 'usr':
			# collect lines for conll file
			cluster_info = build_cluster_from_links(ctx, usr, coref_hyp)
			coref_lines += generate_conll_format(ctx, usr, cluster_info, example_id)

		# collect output in testing
		if collector is not None:
			ex = {'CTX': ctx, 'USR': usr, 'QR_REF': qr_ref, 'QR_HYP': qr_hyp, 'COREF_HYP': coref_hyp, \
					'COREF_SUMMARY': get_coref_summary(coref_hyp, whole_input), 'example_id': example_id, \
					'Bi_REF': binary_class, 'Bi_HYP': binary_class_pred}
			collector.append(ex)


def get_coref_summary(coref_hyp, whole_input):
	'''
		formatting the coref links into a strings
	'''
	if coref_hyp == None: # for qr only
		return coref_hyp

	assert isinstance(whole_input, str)
	summary = []
	seq = whole_input.split()
	for link in coref_hyp: # (start, end)
		m_start_idx, m_end_idx = link[0]['mention_idx'], link[1]['mention_idx']
		m_start_word, m_end_word = link[0]['mention_word'], link[1]['mention_word']
		assert seq[m_start_idx] == m_start_word and seq[m_end_idx] == m_end_word

		r_start_idx, r_end_idx = link[0]['attention_idx'], link[1]['attention_idx']
		r_start_word, r_end_word = link[0]['attention_word'], link[1]['attention_word']
		assert seq[r_start_idx] == r_start_word and seq[r_end_idx] == r_end_word

		mention = " ".join(seq[m_start_idx: m_end_idx])
		reference = " ".join(seq[r_start_idx: r_end_idx])
		m2r = "{} ({}) <- {} ({})".format(reference, r_start_idx, mention, m_start_idx)
		summary.append(m2r)

	return summary	


def coref_evaluate(key_file, sys_file, args):
	metrics = [('mentions', evaluator.mentions), ('muc', evaluator.muc),
				('bcub', evaluator.b_cubed), ('ceafe', evaluator.ceafe),
				('lea', evaluator.lea)]
	NP_only, remove_nested, keep_singletons, min_span = False, False, True, False

	doc_coref_infos = reader.get_coref_infos(key_file, sys_file, NP_only, remove_nested, keep_singletons, min_span, mode=args.mode)

	conll = 0
	conll_subparts_num = 0
	results = {}
	for name, metric in metrics:
		try:
			recall, precision, f1 = evaluator.evaluate_documents(doc_coref_infos, metric, beta=1)
		except:
			recall = precision = f1 = -10

		results[name] = {'recall': recall*100, 'precision': precision*100, 'f1': f1*100}
		if args.mode == 'testing':
			print(name.ljust(10), 'Recall: %.2f' % (recall * 100), ' Precision: %.2f' % (precision * 100), ' F1: %.2f' % (f1 * 100))

	for key in ['recall', 'precision', 'f1']:
		results['avg_{}'.format(key)] = (results["muc"][key] + results["bcub"][key] + results["ceafe"][key])/3
	return results

def get_binary_res(bi_match, qr_res, args):
	if not args.use_binary_cls:
		qr_res['Binary M'] = None
		qr_res['Binary P'] = None
		qr_res['Binary R'] = None
		qr_res['Binary F1'] = None
		return

	Match = sum( [binary_class_pred == binary_class for binary_class_pred, binary_class in bi_match] )
	match_rate = Match / len(bi_match)
	qr_res['Binary M'] = match_rate * 100

	TP = sum( [binary_class_pred == binary_class and binary_class == 1 for binary_class_pred, binary_class in bi_match] )
	Pred = sum([binary_class_pred for binary_class_pred, _ in bi_match])
	GT = sum([binary_class for _, binary_class in bi_match])
	Precision = TP / Pred
	Recall = TP / GT
	F1 = 2* Precision*Recall / (Precision+Recall)
	qr_res['Binary P'] = Precision * 100
	qr_res['Binary R'] = Recall * 100
	qr_res['Binary F1'] = F1 * 100

if __name__ == '__main__':
	''' usage '''
	key_file = 'temp/mudoco-trial.response'
	sys_file = 'proc_data/all/dev.conll'
	res = coref_evaluate(key_file, sys_file)
