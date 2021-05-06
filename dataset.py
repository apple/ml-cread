#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2021 Apple Inc. All Rights Reserved.
#

import sys
import time
import json
import copy
from itertools import chain
from tqdm import tqdm, trange

import torch
from torch.utils.data import DataLoader, RandomSampler

SPECIAL_TOKENS = {
	"bos_token": "<BOS>",
	"eos_token": "<EOS>",
	"pad_token": "<PAD>",
	"sep_token": "<SEP>",
	"additional_special_tokens": ["<USR>", "<SYS>", "<M>", "</M>", "<R>", "</R>", "<CUR>"]
}
SPECIAL_TOKENS_VALUES = ["<BOS>", "<EOS>", "<PAD>", "<SEP>", "<USR>", "<SYS>", "<M>", "</M>", "<R>", "</R>", "<CUR>"]


class Dataset(torch.utils.data.Dataset):
	def __init__(self, args, tokenizer, data_type, generation, data_size):
		assert data_type in ['train', 'dev', 'test']
		self.args = args
		self.data_size = data_size
		self.tokenizer = tokenizer
		self.data_type = data_type
		self.generation = generation

		self._get_special_token_ids()
		self._create_examples()


	def _get_special_token_ids(self):
		self.SPECIAL_TOKENS = SPECIAL_TOKENS
		self.SPECIAL_TOKENS_VALUES = SPECIAL_TOKENS_VALUES
		self.bos_id = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS["bos_token"])
		self.eos_id = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS["eos_token"])
		self.pad_id = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS["pad_token"])
		self.sep_id = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS["sep_token"])

		# mention detection vocab
		self.mc_cl2idx = {'<N>': 0, '<M>': 1, '</M>': 2} # <N>: none, <M>: start of mention, "</M>": end of mention
		self.mc_idx2cl = {v: k for k, v in self.mc_cl2idx.items()} 


	def prepare_reference_label(self, word_label_index, wordId2tokenId, input_ids):
		'''
			record the index of start/end of mention and refernece in the input otterance
			this info will be used as attention signal in reference resolution step
		'''
		reconstruct_sentence = self.tokenizer.convert_ids_to_tokens(input_ids)
		reconstruct_sentence = [token.replace('Ġ', '') for token in reconstruct_sentence]
		token_label_index = []
		for start_end_link in word_label_index:
			for link_meta in start_end_link:
				attention_word_idx, mention_word_idx = link_meta['attention_idx'], link_meta['mention_idx']

				if link_meta['mention_type'] == 'start':
					attention_token_idx = wordId2tokenId[attention_word_idx][0]
				else: # end
					attention_token_idx = wordId2tokenId[attention_word_idx][-1]

				for mention_token_idx in wordId2tokenId[mention_word_idx]:
					link = {}
					link['mention_token_idx'] = mention_token_idx
					link['attention_token_idx'] = attention_token_idx
					assert reconstruct_sentence[mention_token_idx] in link_meta['mention_word']
					assert reconstruct_sentence[attention_token_idx] in link_meta['attention_word']
				token_label_index.append(link)
		return token_label_index


	def prepare_binary_label(self, input_ids, wordId2tokenId, binary_rewrite, curr_end_token_idx):
		''' only the start of rewriting token receives binary signal '''
		binary_label = [-100] * len(input_ids)
		assert isinstance(binary_rewrite, bool)
		if binary_rewrite == True:
			binary_label[curr_end_token_idx] = 1 # rewrite
		else:
			binary_label[curr_end_token_idx] = 0 # not rewrite
		return binary_label


	def prepare_mention_label(self, input_ids, word_label_index, wordId2tokenId, curr_start_idx, curr_end_idx):
		'''
			get label index for mention detection
			only the parts of current utterance receive signal, everwhere else will get -100
		'''
		mention_label = [-100] * len(input_ids)
		curr_start_idx = wordId2tokenId[curr_start_idx][0]
		curr_end_idx   = wordId2tokenId[curr_end_idx-1][-1] + 1

		# align class <N> (none) to everywehere in current utterance first
		mention_label[curr_start_idx: curr_end_idx] = [ self.mc_cl2idx['<N>'] ] * (curr_end_idx-curr_start_idx)

		for start_end_link in word_label_index: # iterate over links in one example
			for link_meta in start_end_link: # iterate over start and end of a link
				idx = link_meta['mention_idx']
				if link_meta['mention_type'] == 'start': # align class <M> (start of mention)
					for idx in wordId2tokenId[idx]:
						mention_label[idx] = self.mc_cl2idx['<M>']
				else: # # align class </M> (end of mention)
					idx = wordId2tokenId[idx][-1]
					mention_label[idx] = self.mc_cl2idx['</M>']

		return mention_label, curr_start_idx, curr_end_idx


	def _check_label_index(self, whole_input, links):
		''' sanity check for index correctness '''
		seq = whole_input.split()
		for link in links:
			for start_or_end in link:
				for word_type in ['mention', 'attention']:
					assert seq[start_or_end['{}_idx'.format(word_type)]] == start_or_end['{}_word'.format(word_type)]


	def _create_examples(self):
		if self.data_type == 'train':
			data_file = self.args.train_file
		elif self.data_type == 'dev':
			data_file = self.args.dev_file
		else:
			data_file = self.args.test_file

		with open(data_file) as f:
			data = json.load(f)

		self.examples = []
		for example_num, example in enumerate(tqdm(data, disable=self.args.disable_display)):
			if self.data_size != -1 and example_num == self.data_size:
				break

			# get data
			context = example['dialogue context'] # context, list of str
			curr_utt = example['current utterance'] # current utterance, str
			rewt_utt = example['rewrite utterance'] # rewrite utterance, str
			word_label_index = example['link index'] # index of mention/reference span
			binary_rewrite = example['rewrite happen'] # binary label for rewrite or not, bool

			# prepare input sequence to model
			whole_input = copy.deepcopy(context)
			whole_input.append(curr_utt)
			curr_start_idx = sum([len(s.split()) for s in context]) # the (word) start idx of current utt
			curr_end_idx = curr_start_idx + len(curr_utt.split())

			whole_input = " ".join(whole_input)
			self._check_label_index(whole_input, word_label_index)
			input_ids, wordId2tokenId, tokenId2wordId = self.tokenize_with_map(whole_input)

			if rewt_utt == "":
				rewt_utt_ids = []
			else:
				rewt_utt_ids = self.tokenizer(rewt_utt)['input_ids'] # list
			target_utt_ids = rewt_utt_ids
			target_utt_len = len(target_utt_ids)

			if not self.generation:
				# input seq: CTX <CUR> current utterance <SEP> rewritten utterance <EOS>
				input_ids = input_ids + [self.sep_id] + target_utt_ids + [self.eos_id]

				# mention detection signal
				mention_label, curr_start_token_idx, curr_end_token_idx = \
					self.prepare_mention_label(input_ids, word_label_index, wordId2tokenId, curr_start_idx, curr_end_idx)

				# reference resolution signal
				reference_label_index = self.prepare_reference_label(word_label_index, wordId2tokenId, input_ids)

				# binary classification of rewriting signal
				binary_label = self.prepare_binary_label(input_ids, wordId2tokenId, binary_rewrite, curr_end_token_idx)

				# rewriting singal
				ignore_len = len(input_ids) - target_utt_len - 1 # eos_id
				label_ids = [-100] * ignore_len + target_utt_ids + [self.eos_id]
				assert len(input_ids) == len(label_ids)

			else: # generation
				# <sep> is given at first step during decoding
				input_ids = input_ids
				label_ids = None
				mention_label, curr_start_token_idx, curr_end_token_idx = \
						self.prepare_mention_label(input_ids, word_label_index, wordId2tokenId, curr_start_idx, curr_end_idx)
				reference_label_index = self.prepare_reference_label(word_label_index, wordId2tokenId, input_ids)
				binary_label = None

			self.examples.append({
				'input_ids': input_ids, # list of ids
				'label_ids': label_ids, # list of ids
				'mention_label_ids': mention_label,
				'curr_start_token_idx': curr_start_token_idx,
				'curr_end_token_idx': curr_end_token_idx,
				'reference_label': reference_label_index,
				'wordId2tokenId': wordId2tokenId,
				'tokenId2wordId': tokenId2wordId,
				'context': context,
				'curr_utt': curr_utt,
				'whole_input': whole_input,
				'rewt_utt': rewt_utt,
				'example_id': example['example index'],
				'spk': example['speaker'],
				'coref_label': word_label_index,
				'binary_label_ids': binary_label,
				'binary_rewrite': binary_rewrite
			})

		print('Data Statistics: {} -> {} examples'.format(self.data_type, len(self.examples)))


	def _pad(self, sentences, pad_id):
		'''
			sentences: a list of list with ids
		'''
		max_len = max((map(len, sentences)))
		attention_mask = []
		sentences_pad = []
		for sent in sentences:
			pad_len = max_len - len(sent)
			sentences_pad.append( sent + [pad_id]*pad_len )
			attention_mask.append( [1]*len(sent) + [0]*pad_len)
		return sentences_pad, attention_mask


	def __len__(self):
		return len(self.examples)


	def __getitem__(self, index):
		return self.examples[index]


	def collate_fn(self, batch):
		input_ids = [example['input_ids'] for example in batch]
		input_ids, attention_mask = self._pad(input_ids, self.pad_id)
		input_ids, attention_mask = torch.tensor(input_ids).long().to(self.args.device), torch.tensor(attention_mask).long().to(self.args.device)

		if not self.generation:
			label_ids = [example['label_ids'] for example in batch]
			label_ids, _ = self._pad(label_ids, -100)
			label_ids = torch.tensor(label_ids).long().to(self.args.device)
			mention_label_ids = [example['mention_label_ids'] for example in batch]
			mention_label_ids, _ = self._pad(mention_label_ids, -100)
			mention_label_ids = torch.tensor(mention_label_ids).long().to(self.args.device)
			binary_label_ids = [example['binary_label_ids'] for example in batch]
			binary_label_ids, _ = self._pad(binary_label_ids, -100)
			binary_label_ids = torch.tensor(binary_label_ids).long().to(self.args.device)
		else:
			label_ids = None
			mention_label_ids = [example['mention_label_ids'] for example in batch]
			mention_label_ids, _ = self._pad(mention_label_ids, -100)
			mention_label_ids = torch.tensor(mention_label_ids).long().to(self.args.device)
			binary_label_ids = None
		token_type_ids = None # TODO: not sure if this makes any effect to gpt2

		# record info
		context = [example['context'] for example in batch]
		curr_utt = [example['curr_utt'] for example in batch]
		rewt_utt = [example['rewt_utt'] for example in batch]
		example_ids = [example['example_id'] for example in batch] # record the example idx in batch
		curr_start_token_idx = [example['curr_start_token_idx'] for example in batch]
		curr_end_token_idx = [example['curr_end_token_idx'] for example in batch]
		reference_label = [example['reference_label'] for example in batch]
		wordId2tokenId = [example['wordId2tokenId'] for example in batch]
		tokenId2wordId = [example['tokenId2wordId'] for example in batch]
		whole_input = [example['whole_input'] for example in batch]
		spk = [example['spk'] for example in batch]
		coref_label = [example['coref_label'] for example in batch]
		binary_rewrite = [example['binary_rewrite'] for example in batch]

		return {'input_ids': input_ids, 'attention_mask': attention_mask, \
				'token_type_ids': token_type_ids, 'label_ids': label_ids, \
				'context': context, 'curr_utt': curr_utt, 'rewt_utt': rewt_utt, \
				'example_ids': example_ids, 'spk': spk, 'mention_label_ids': mention_label_ids, \
				'curr_start_token_idx': curr_start_token_idx, 'curr_end_token_idx': curr_end_token_idx, \
				'reference_label': reference_label, 'wordId2tokenId': wordId2tokenId, \
				'tokenId2wordId': tokenId2wordId, 'whole_input': whole_input, \
				'coref_label': coref_label, 'binary_label_ids': binary_label_ids, \
				'binary_rewrite': binary_rewrite}


	def tokenize_with_map(self, sentence):
		'''
			Build the mapping of indexes before/after tokenizer to handel BPE

			Input:
				sentence: a natural sentence, str
			Returns:
				wordId2tokenId, a 1-to-many map
				tokenId2wordId, a many-to-1 map
		'''
		assert isinstance(sentence, str)
		token_ids = self.tokenizer(sentence)['input_ids']
		reconstruct_sentence = self.tokenizer.convert_ids_to_tokens(token_ids)
		reconstruct_sentence = [token.replace('Ġ', '') for token in reconstruct_sentence]
		sentence = sentence.split()
	
		wordId2tokenId = {}
		tokenId = 0
		for wordId, word in enumerate(sentence):
			wordId2tokenId[wordId] = []
			token = ""
			while word != token:
				wordId2tokenId[wordId].append(tokenId)
				token += reconstruct_sentence[tokenId]
				tokenId += 1
	
		tokenId2wordId = {}
		for wordId, tokenIds in wordId2tokenId.items():
			for tokenId in tokenIds:
				assert tokenId not in tokenId2wordId
				tokenId2wordId[tokenId] = wordId
	
		assert len(wordId2tokenId) == len(sentence)
		assert len(tokenId2wordId) == len(reconstruct_sentence)
		return token_ids, wordId2tokenId, tokenId2wordId


if __name__ == '__main__':
	pass
