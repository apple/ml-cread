#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2021 Apple Inc. All Rights Reserved.
#

import sys
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, NLLLoss

from transformers import (
	GPT2Model,
	GPT2LMHeadModel,
	GPT2PreTrainedModel
)

class JointModel(GPT2PreTrainedModel):
	def __init__(self, config, **kwargs):
		super().__init__(config)
		self.args = kwargs['args']
		self.config = config

		# core gpt2 and lm head
		self.transformer = GPT2Model(config)
		self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

		# mention detection output index
		self.mc_cl2idx = {'<N>': 0, '<M>': 1, '</M>': 2}
		self.mc_idx2cl = {v: k for k, v in self.mc_cl2idx.items()}
		self.cl_head = nn.Linear(config.n_embd, 3) # head for 3 classes in mention dection

		# attention parameters in coref2qr mechanism
		if self.args.coref_attn_share_between_layer:
			self.c_attn = Conv1D(3 * config.n_embd, config.n_embd)
		else:
			self.c_attn = nn.ModuleList([Conv1D(3 * config.n_embd, config.n_embd) for _ in range(self.config.n_layer+1)])

		# binary classification for rewriting or not
		if self.args.use_binary_cls:
			self.binary_cls1 = nn.Linear(config.n_embd, config.n_embd)
			self.binary_cls2 = nn.Linear(config.n_embd, 2, bias=False) # output layer for rewrite or not

		self.init_weights()


	def load_pretrained_weight(self):
		print('Load pretrained GPT')
		pretrained_gpt = GPT2LMHeadModel.from_pretrained('gpt2', config=self.config)
		self.transformer.load_state_dict(pretrained_gpt.transformer.state_dict())
		self.lm_head.load_state_dict(pretrained_gpt.lm_head.state_dict())


	def get_output_embeddings(self):
		return self.lm_head


	def prepare_inputs_for_generation(self, input_ids, past, **kwargs):
		# only last token for inputs_ids if past is defined in kwargs
		if past:
			input_ids = input_ids[:, -1].unsqueeze(-1)

		return {"input_ids": input_ids, "past": past, "use_cache": kwargs["use_cache"]}


	def _compute_reference_loss(self, batch, attentions):
		'''
			attentions: all attnetion heads tuple of (B, n_heads, T, T)
		'''
		LOSS, c = 0, 0
		for layer_idx in range(self.config.n_layer): # iterate layers
			if layer_idx not in self.args.coref_layer_idx:
				continue

			for head_idx in range(self.args.n_coref_head): # iterate heads
				head_loss = self._get_one_head_loss(batch, attentions[layer_idx][:, head_idx, :, :].contiguous())
				LOSS += head_loss
				c += 1

		zero_loss = torch.tensor(0).to(self.args.device)
		if LOSS == zero_loss: # all heads no loss
			return zero_loss
		else:
			return LOSS / c


	def _get_one_head_loss(self, batch, reference_head):
		'''
			Compute the reference resolution loss per attention head
			reference_head: an attention head in a layer (B, T, T)
		'''
		dists, labels = [], []
		batch_links = batch['reference_label']
		for b_idx, links in enumerate(batch_links):
			for link in links:
				m_idx = link['mention_token_idx']
				a_idx = link['attention_token_idx']
				dist = reference_head[b_idx, m_idx, :]
				dists.append(dist)
				labels.append(a_idx)

		if len(dists) == 0: # might have no link in a batch
			return torch.tensor(0).to(self.args.device)

		dists = torch.stack(dists, dim=0) # (B', T)
		labels = torch.tensor(labels).long().to(self.args.device) # (B',)

		nll_fct = NLLLoss()
		loss = nll_fct(torch.log(dists+self.args.eps), labels)
		return loss


	def _filter_not_rewrite(self, logits, labels, batch):
		'''
			Filter out examples in a batch that do not require rewirting for lm loss

			logits: (B, T, C)
			labels: (B, T)
		'''
		assert logits.size(0) == labels.size(0)
		assert logits.size(1) == labels.size(1)
		new_logits, new_labels = [], []
		pruned_bs_idx = -1
		for bs_idx, rewrite_bool in enumerate(batch['binary_rewrite']): # a list of bool
			pruned_bs_idx += 1
			if rewrite_bool == False:
				continue

			new_logits.append(logits[pruned_bs_idx, :, :])
			new_labels.append(labels[pruned_bs_idx, :])

		if len(new_logits) == 0 and len(new_labels) == 0:
			return None, None

		new_logits = torch.stack(new_logits, dim=0)
		new_labels = torch.stack(new_labels, dim=0)
		return new_logits, new_labels
		

	def _compute_lm_loss(self, lm_logits, labels, batch):
		# filter out examples if `filter_not_rewrite_loss` switch is on
		if self.args.filter_not_rewrite_loss:
			lm_logits, labels = self._filter_not_rewrite(lm_logits, labels, batch)

		if labels == None: # no examples left in a batch
			return torch.tensor(0).to(self.args.device)

		# Shift so that tokens < n predict n
		shift_logits = lm_logits[..., :-1, :].contiguous() # (B, T, V)
		shift_labels = labels[..., 1:].contiguous() # (B, T)

		# Flatten the tokens
		loss_fct = CrossEntropyLoss()
		loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
		return loss

	def _compute_binary_loss(self, bi_logits, batch):
		label = batch['binary_label_ids']
		loss_weight = torch.tensor([self.args.class0_loss_w, self.args.class1_loss_w]).float().to(self.args.device) # tune weighted loss
		loss_fct = CrossEntropyLoss(weight=loss_weight)
		loss = loss_fct(bi_logits.view(-1, bi_logits.size(-1)), label.view(-1))
		return loss


	def _compute_mention_loss(self, cl_logits, mention_labels):
		loss_fct = CrossEntropyLoss()
		loss_mention = loss_fct(cl_logits.view(-1, cl_logits.size(-1)), mention_labels.view(-1))
		return loss_mention


	def collect_coref_hiddens(self, meta_link_batch, all_hiddens, batch):
		'''
			Collect coreference hiddent states pool for coref2qr mechanism
			all_hiddens: tuple of (B, T, H) with len = 1 + n_layer, 1 for embedding
		'''
		all_coref_attn = []
		for layer_idx in range(self.config.n_layer + 1):
			if layer_idx <= (self.config.n_layer - self.args.coref_attn_layer):
				all_coref_attn.append(None)
			else:
				coref_attn = self.collect_coref_hiddens_layer(meta_link_batch, all_hiddens[layer_idx], batch)
				all_coref_attn.append(coref_attn)
		return all_coref_attn


	def collect_coref_hiddens_layer(self, meta_link_batch, hiddens, batch):
		'''
		collect coref hiddens from one layer hiddens (B, T, H)
		'''
		B = hiddens.size(0)
		assert len(meta_link_batch) == B
		coref_hiddens_batch = []
		for b_idx, meta_link_example in enumerate(meta_link_batch):
			wordId2tokenId = batch['wordId2tokenId'][b_idx]
			start_end_list = []
			for start_end_link in meta_link_example:
				assert start_end_link[0]['mention_type'] == 'start'
				assert start_end_link[1]['mention_type'] == 'end'
				m_word_idx_start = start_end_link[0]['mention_idx']
				m_word_idx_end   = start_end_link[1]['mention_idx']
				r_word_idx_start = start_end_link[0]['attention_idx']
				r_word_idx_end   = start_end_link[1]['attention_idx']

				m_token_idx_start =  wordId2tokenId[m_word_idx_start][0]
				m_token_idx_end   =  wordId2tokenId[m_word_idx_end][-1]
				r_token_idx_start =  wordId2tokenId[r_word_idx_start][0]
				r_token_idx_end   =  wordId2tokenId[r_word_idx_end][-1]

				# mention/reference_start/end_token_idx
				if self.args.coref_attn_mention and m_token_idx_start < m_token_idx_end: # only consider reasonable reasonable predictions
					start_end_list.append((m_token_idx_start, m_token_idx_end))
				if r_token_idx_start < r_token_idx_end:
					start_end_list.append((r_token_idx_start, r_token_idx_end))

			if len(start_end_list) > 0: # has at least one coref link
				start_end_list = sorted(start_end_list, key=lambda x: x[0]) # sort by start_idx
				coref_hiddens_example = []
#				if self.args.coref_attn_zeros:
				coref_hiddens_example.append( torch.zeros(1, 1, self.config.n_embd).to(self.args.device) )
				for start_idx, end_idx in start_end_list:
					coref_hiddens_example.append( hiddens[b_idx, start_idx: end_idx, :].unsqueeze(0) ) # (1, T'', H)
				coref_hiddens_example = torch.cat(coref_hiddens_example, dim=1) # (1, T', H)
			else:
				coref_hiddens_example = torch.zeros(1, 1, self.config.n_embd).to(self.args.device) # (1, 1, H)
			coref_hiddens_batch.append(coref_hiddens_example)

		assert len(coref_hiddens_batch) == B
		# padding
		coref_len_batch = [ x.size(1) for x in coref_hiddens_batch]
		max_coref_len = max(coref_len_batch)
		mask = []
		for b_idx in range(B):
			coref_len = coref_len_batch[b_idx]
			pad_len = max_coref_len - coref_len
			mask.append( [1]*coref_len + [0]*pad_len )
			coref_hiddens_batch[b_idx] = torch.cat([coref_hiddens_batch[b_idx], torch.zeros(1, pad_len, self.config.n_embd).to(self.args.device)], dim=1)

		coref_hiddens_batch = torch.cat(coref_hiddens_batch, dim=0) # (B, T', H)
		mask = torch.tensor(mask).float().to(self.args.device) # (B, T')
		assert coref_hiddens_batch.size() == (B, max_coref_len, self.config.n_embd)
		assert mask.size() == (B, max_coref_len)
		coref_attn = {'hiddens': coref_hiddens_batch, 'mask': mask}
		return coref_attn


	def merge_heads(self, x):
		x = x.permute(0, 2, 1, 3).contiguous() # (B, T, n_head, F)
		new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
		return x.view(*new_x_shape)  # (B, T, n_head*F)


	def split_heads(self, x, k=False):
		new_x_shape = x.size()[:-1] + (self.config.n_head, x.size(-1) // self.config.n_head)
		x = x.view(*new_x_shape)  # (B, T, n_head, F)
		if k:
			return x.permute(0, 2, 3, 1)  # (B, n_head, F, T)
		else:
			return x.permute(0, 2, 1, 3)  # (B, n_head, T, F)


	def attn_on_coref(self, coref_attn, all_hiddens, last_hiddens):
		'''
			Perform coref2qr across layers in gpt2, return the feature vector f
		'''
		ctx_vec = 0
		count = 0
		for layer_idx in range(self.config.n_layer + 1):
			if layer_idx <= (self.config.n_layer - self.args.coref_attn_layer):
				continue
			ctx_vec_layer = self.attn_on_coref_layer(coref_attn[layer_idx], all_hiddens[layer_idx], layer_idx)
			ctx_vec += ctx_vec_layer
			count += 1
		assert count == self.args.coref_attn_layer
		ctx_vec = ctx_vec / self.args.coref_attn_layer # avg all ctx_vec across layers

		mix_hiddens = 0.5 * last_hiddens + 0.5 * ctx_vec
		return mix_hiddens
		

	def attn_on_coref_layer(self, coref_attn, hidden_states, layer_idx):
		'''
			Perform coref2qr mechanism in one layer

			hidden_states: (B, T, H), where T = sequence length
			coref_hiddens: (B, T', H), where T' = coref pool length
			coref_mask   : (B, T')
		'''
		assert coref_attn is not None
		coref_hiddens, coref_mask = coref_attn['hiddens'], coref_attn['mask']

		if self.args.coref_attn_share_between_layer:
			query, _, _   = self.c_attn(hidden_states).split(self.config.n_embd, dim=2) # (B, T, H)
			_, key, value = self.c_attn(coref_hiddens).split(self.config.n_embd, dim=2) # (B, T', H)
		else:
			query, _, _   = (self.c_attn[layer_idx](hidden_states)).split(self.config.n_embd, dim=2) # (B, T, H)
			_, key, value = (self.c_attn[layer_idx](coref_hiddens)).split(self.config.n_embd, dim=2) # (B, T', H)

		query = self.split_heads(query)
		key = self.split_heads(key, k=True)
		value = self.split_heads(value)

		coref_mask = coref_mask[:, None, None, :] # (B, T') -> (B, 1, 1, T')
		coref_mask = (1.0 - coref_mask) * -10000.0 # convert 1. -> 0, 0. -> -10000

		w = torch.matmul(query, key) # (B, n_head, T, T')
		w = w + coref_mask
		w = torch.softmax(w, dim=-1)
		ctx_vec = torch.matmul(w, value) # (B, n_head, T, F)
		ctx_vec = self.merge_heads(ctx_vec) # (B, T, H)

		assert ctx_vec.size() == hidden_states.size()
		return ctx_vec


	def forward(self, input_ids=None, past=None, attention_mask=None, token_type_ids=None, position_ids=None,
				head_mask=None, inputs_embeds=None, labels=None, use_cache=None, output_attentions=None,
				output_hidden_states=None, step=None, mention_labels=None, predict_mention=True, predict_lm=True,
				coref_attn=None, batch=None, coref_links=None):

		# run gpt2
		# last hidden state, (presents), (all hidden_states), (attentions)
		transformer_outputs = self.transformer(input_ids, past=past, attention_mask=attention_mask,
												token_type_ids=token_type_ids, position_ids=position_ids,
												head_mask=head_mask, inputs_embeds=inputs_embeds, use_cache=use_cache,
												output_attentions=True, output_hidden_states=True)

		hidden_states = transformer_outputs[0] # (B, T, H)
		all_hidden_states = transformer_outputs[2] # tuple of (B, T, H) with len = 1 + n_layer, 1 for embedding
		attentions = transformer_outputs[3] # tuple of (B, n_heads, T, T), e.g., attentions[-1][b,n,i,:]

		# get lm logits
		if predict_lm:
			if self.args.task == 'qr_coref' and self.args.use_coref_attn:
				if coref_attn is None:
					coref_attn = self.collect_coref_hiddens(coref_links, all_hidden_states, batch)

				hidden_states_lm = self.attn_on_coref(coref_attn, all_hidden_states, hidden_states)
				lm_logits = self.lm_head(hidden_states_lm)

			else:
				lm_logits = self.lm_head(hidden_states)

			# get binary logits
			if self.args.use_binary_cls and (step is None or step == 0): # step=None for training, step=0 for first decoding step
				bi_logits = self.binary_cls2(self.binary_cls1(hidden_states)) # (B, T, 2)
			else:
				bi_logits = None
		else:
			lm_logits, bi_logits = None, None

		# get mention detection logits
		if predict_mention:
			cl_logits = self.cl_head(hidden_states) # (B, T, C)
		else:
			cl_logits = None

		# prepare output
		transformer_outputs = transformer_outputs[:-2] # for output consistency, dont return H and A
		outputs = (bi_logits, lm_logits, cl_logits, attentions,) + transformer_outputs[1:] # return all attentions
		outputs = outputs + (coref_attn,)

		# compute loss
		if labels is not None:
			# qr loss: binary loss and lm loss
			if 'qr' in self.args.task:
				loss_lm = self._compute_lm_loss(lm_logits, labels, batch)

				if self.args.use_binary_cls:
					loss_bi = self._compute_binary_loss(bi_logits, batch)
				else:
					loss_bi = torch.tensor(0).to(self.args.device)
			else:
				loss_lm = torch.tensor(0).to(self.args.device)
				loss_bi = torch.tensor(0).to(self.args.device)

			# coref loss: mention loss and reference loss
			if 'coref' in self.args.task:
				loss_mention = self._compute_mention_loss(cl_logits, mention_labels)
				loss_reference = self._compute_reference_loss(batch, attentions)
			else:
				loss_mention, loss_reference = torch.tensor(0).to(self.args.device), torch.tensor(0).to(self.args.device)

			# final loss
			loss_total = loss_bi + loss_lm + loss_mention + loss_reference
			loss_dict = {'bi': loss_bi, 'lm': loss_lm, 'mention': loss_mention, 'reference': loss_reference, 'total': loss_total}
			outputs = (loss_dict,) + outputs

		return outputs  # (loss), lm_logits, presents, (all hidden_states), (attentions)


class Conv1D(nn.Module):
	"""
	From HuggingFace library
	1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).

	Basically works like a linear layer but the weights are transposed.

	Args:
		nf (:obj:`int`): The number of output features.
		nx (:obj:`int`): The number of input features.
	"""

	def __init__(self, nf, nx):
		super().__init__()
		self.nf = nf
		w = torch.empty(nx, nf)
		nn.init.normal_(w, std=0.02)
		self.weight = nn.Parameter(w)
		self.bias = nn.Parameter(torch.zeros(nf))

	def forward(self, x):
		size_out = x.size()[:-1] + (self.nf,)
		x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
		x = x.view(*size_out)
		return x


def _extend_mask(mask):
	mask = torch.cat([mask, mask.new_ones((mask.shape[0], 1))], dim=-1)
	return mask


def _post_proc(gen):
	''' gen: a list of tokens '''
	if '<SEP>' in gen:
		gen = gen[gen.index('<SEP>')+1: ]
	if '<EOS>' in gen:
		gen = gen[: gen.index('<EOS>')]
	return gen


def decode(args, batch, model, tokenizer):
	'''
		decode query rewriting and coreference resolution
	'''

	if 'coref' in args.task:
		corefs = predict_coref(args, batch, model, tokenizer)
	else:
		corefs = [None]

	if 'qr' in args.task and batch['rewt_utt'][0] != '':
		binary_class, sentences = generate(args, batch, model, tokenizer, corefs)
	else:
		sentences = [None]
		binary_class = [None]
	return {'coref': corefs, 'qr': sentences, 'bi': binary_class}


def get_ref_word_idx(men_token_idx, attentions, tokenId2wordId, args, config):
	attn_dist = 0
	for layer_idx in range(config.n_layer):
		if layer_idx not in args.coref_layer_idx:
			continue
		for head_idx in range(args.n_coref_head):
			attn_dist += attentions[layer_idx][0, head_idx, men_token_idx, :]

	attn_dist = attn_dist / args.n_coref_head
	ref_token_idx = torch.argmax(attn_dist).item()
	ref_word_idx = tokenId2wordId[ref_token_idx]
	return ref_word_idx, ref_token_idx, attn_dist


def get_valid_ref(start_meta, end_meta, ref_start_dist, ref_end_dist, tokenId2wordId, whole_input):
	'''
		re-compute reference start and end idx based on bi-gram probability and make sure the indexes are valid
	'''
	assert ref_start_dist.size() == ref_end_dist.size()
	T = ref_start_dist.size(0)
	prod = torch.matmul(ref_start_dist.view(-1, 1), ref_end_dist.view(1, -1)) # (T, T)
	prod = prod.cpu().numpy()

	collect = []
	for i in range(T):
		for j in range(i+1, T):
			collect.append( (prod[i, j], i, j) )

	sort_collect = sorted(collect, key=lambda x: x[0], reverse=True)
	start_word_idx, end_word_idx = None, None
	for value, start_token_idx, end_token_idx in sort_collect:
		start_word_idx = tokenId2wordId[start_token_idx]
		end_word_idx   = tokenId2wordId[end_token_idx]
		if start_word_idx < end_word_idx:
			break
	assert start_word_idx < end_word_idx
	start_meta['attention_idx'], start_meta['attention_word'] = start_word_idx, whole_input[start_word_idx]
	end_meta['attention_idx'], end_meta['attention_word'] = end_word_idx, whole_input[end_word_idx]


def proc_coref_output(batch, token_pred, attentions, token_ids, tokenizer, args, config):
	'''
		process the model output, extract the start/end index and the corresponding words in coreference links
	'''
	assert isinstance(attentions, tuple)
	tokenId2wordId = batch['tokenId2wordId'][0]

	# token index of current utterance
	curr_start_token_idx = batch['curr_start_token_idx'][0]
	curr_end_token_idx = batch['curr_end_token_idx'][0]
	curr_utt_token_len = curr_end_token_idx - curr_start_token_idx

	# work index of current utterance
	curr_utt_word = batch['curr_utt'][0]
	curr_utt_word_len = len(curr_utt_word.split())
	curr_start_word_idx = tokenId2wordId[curr_start_token_idx]
	
	token_pred = token_pred[0][curr_start_token_idx:].tolist()
	assert len(token_pred) == curr_utt_token_len

	whole_input = batch['whole_input'][0].split()
	recon_input = tokenizer.convert_ids_to_tokens(token_ids[0].tolist())
	recon_input = [token.replace('Ġ', '') for token in recon_input]

	mention = False
	word_pred = [-1] * curr_utt_word_len
	links = []
	for local_token_idx, step_pred in enumerate(token_pred):
		global_token_idx = local_token_idx + curr_start_token_idx # token index in the whole input sequence

		# map mention prediction back to word sequence
		global_word_idx = tokenId2wordId[global_token_idx]
		local_word_idx = global_word_idx-curr_start_word_idx
		word_pred[local_word_idx] = step_pred

		# formulate the same format as input data
		if not mention and step_pred == 1.:
			mention = True
			ref_start_word_idx, ref_start_token_idx, ref_start_dist = get_ref_word_idx(global_token_idx, attentions, tokenId2wordId, args, config)
			_start = {'mention_type': 'start', 'mention_idx': global_word_idx, 'mention_word': whole_input[global_word_idx], \
						'attention_idx': ref_start_word_idx, 'attention_word': whole_input[ref_start_word_idx]}
		if mention and step_pred == 2.:
			mention = False
			ref_end_word_idx, ref_end_token_idx, ref_end_dist = get_ref_word_idx(global_token_idx, attentions, tokenId2wordId, args, config)
			_end = {'mention_type': 'end', 'mention_idx': global_word_idx, 'mention_word': whole_input[global_word_idx], \
						'attention_idx': ref_end_word_idx, 'attention_word': whole_input[ref_end_word_idx]}

			get_valid_ref(_start, _end, ref_start_dist, ref_end_dist, tokenId2wordId, whole_input)
			links.append([_start, _end])

	assert -1 not in word_pred
	return [links]


def predict_coref(args, batch, model, tokenizer):
	input_ids, attention_mask, token_type_ids = batch['input_ids'], batch['attention_mask'], batch['token_type_ids']
	batch_size = input_ids.size(0)
	assert batch_size == 1

	curr_start = batch['curr_start_token_idx'][0]
	curr_end = batch['curr_end_token_idx'][0]
	curr_utt_len = curr_end - curr_start
	mention_label_ids = batch['mention_label_ids']

	# model forward
	_, _, cl_logits, attentions, _, _ = \
		model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, past=None, predict_lm=False, batch=batch)
	cl_pred = torch.argmax(cl_logits, dim=-1) # (B, T')

	out = proc_coref_output(batch, cl_pred, attentions, input_ids, tokenizer, model.args, model.config)
	return out


def generate(args, batch, model, tokenizer, coref_pred):
	'''
		Generation of query rewriting
	'''
	# basic info
	input_ids, attention_mask, token_type_ids = batch['input_ids'], batch['attention_mask'], batch['token_type_ids']
	batch_size = input_ids.size(0)
	ctx_len = input_ids.size(1)
	bos_id, eos_id, pad_id, sep_id = tokenizer.convert_tokens_to_ids(['<BOS>', '<EOS>', '<PAD>', '<SEP>'])
	assert batch['curr_end_token_idx'][0] == ctx_len
	assert batch_size == 1 # don't support batch_size larger thatn 1, when batch_size > 1, the padded input is not straightforward for decoding.

	# add <SEP> token to start decoding
	tokens_to_add = input_ids.new(batch_size, 1).fill_(sep_id)
	input_ids = torch.cat([input_ids, tokens_to_add], dim=-1)
	attention_mask = _extend_mask(attention_mask)
	assert 0 not in attention_mask # since batch_size == 1, no padding happens

	past = None
	coref_attn = None
	finish_sent = [False for _ in range(batch_size)]
	binary_class, copy_not_rewrite, binary_class_pred = None, False, None
	for i in range(args.dec_max_len):
		if past: # with past, the model only needs current input
			input_ids_step = input_ids[:, -1].unsqueeze(-1)
			if args.task == 'qr_coref' and args.use_coref_attn:
				assert coref_attn is not None

		else: # only the first step enters here
			input_ids_step = input_ids

		bi_logits, logits, _, _, past, coref_attn = model(input_ids=input_ids_step, attention_mask=attention_mask, \
														token_type_ids=token_type_ids, past=past, predict_mention=False, \
														coref_attn=coref_attn, batch=batch, coref_links=coref_pred, step=i)

		if args.use_binary_cls and i == 0: # check if to run the rest geenration based on the binary classification result
			# bi_logits: (B, T, 2)
			binary_class_pred = torch.argmax(bi_logits[:, -1, :], dim=-1)
			binary_class_pred = binary_class_pred.tolist()
			assert len(binary_class_pred) == 1
			if binary_class_pred[0] == 0 and args.copy_not_rewrite: # not rewrite
				copy_not_rewrite = True
				break

		# logits: (B, T, V), T=1 when past is passed
		next_token_logits = logits[:, -1, :]
		next_token = torch.argmax(next_token_logits, dim=-1)
		input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
		attention_mask = _extend_mask(attention_mask)

		for bs_idx, token_id in enumerate(next_token):
			if finish_sent[bs_idx] is False and token_id.item() == eos_id: # first produce <eos>
				finish_sent[bs_idx] = True

		if sum(finish_sent) == batch_size:
			break

	if copy_not_rewrite: # return the input current utterance as rewrite if predicts `not-rewrite`
		return binary_class_pred, batch['curr_utt']

	# post-process output sentence
	sentences = []
	for bs_idx in range(batch_size):
		gen = tokenizer.decode(input_ids[bs_idx, :]).split()
		gen = _post_proc(gen)
		sentences.append(' '.join(gen))
	assert len(sentences) == 1
	return binary_class_pred, sentences
