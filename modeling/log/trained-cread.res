Namespace(adam_epsilon=1e-12, checkpoint='checkpoint/trained-cread', class0_loss_w=1.0, class1_loss_w=1.5, copy_not_rewrite=True, coref_attn_layer=7, coref_attn_mention=False, coref_attn_share_between_layer=True, coref_layer_idx=[10, 11], dec_max_len=100, decode_file='decode/trained-cread.json', dev_conll='proc_data/all/dev.conll', dev_file='proc_data/all/dev.json', disable_display=False, eps=1e-12, eval_batch_size=1, eval_interval=16000, filter_not_rewrite_loss=True, gradient_accumulation_steps=4, learning_rate=6.25e-05, load_pretrained_weight=True, max_epoch=16, max_grad_norm=1.0, mode='testing', model_name='trained-cread', model_name_or_path='gpt2', n_coref_head=3, no_improve_max=5, num_beams=1, seed=1122, task='qr_coref', temperature=1.0, test_conll='proc_data/all/test.conll', test_file='proc_data/all/test.json', train_batch_size=15, train_file='proc_data/all/train.json', train_size=-1, use_binary_cls=True, use_coref_attn=True, use_scheduler=True, warmup_steps=0)
Load model, tokenizer from checkpoint/trained-cread
Data Statistics: dev -> 1901 examples
Data Statistics: dev -> 1901 examples
Data Statistics: test -> 1988 examples
Epoch: Eval | dev total loss: 1.179 (binary: 0.42, rewrite: 0.122, mention: 0.099, reference: 0.537) | time: 42.5
Problem in the coreference annotation:
 USR	calling-dial216-turn6	3	Billy	0)

Problem in the coreference annotation:
 USR	calling-dial1910-turn2	14	'	0)

Problem in the coreference annotation:
 USR	calling-dial3464-turn2	4	call	0)

Problem in the coreference annotation:
 USR	calling-dial3904-turn2	2	Anderson	0)

Nested coreferring mentions.
USR	music-dial329-turn0	6	Bills	(0

mentions   Recall: 75.33  Precision: 83.02  F1: 78.99
muc        Recall: 69.57  Precision: 78.27  F1: 73.67
bcub       Recall: 70.99  Precision: 79.15  F1: 74.85
ceafe      Recall: 74.46  Precision: 79.45  F1: 76.87
lea        Recall: 68.01  Precision: 75.14  F1: 71.40
Epoch: Dev | dev: QR Binary F1: 86.97 (93.27), LM: P: 88.68 R: 88.33 F1: 88.00 (62.78) | COREF P: 78.96 R: 71.67 F1: 75.13 (71.40) | time: 166.4

************************ QR Result on dev ************************
	Macro P R F1 | Micro P R F1 | BLEU | ROUGE-1 -2 L F1
	88.68 88.33 88.00 65.51 60.26 62.78 91.19 95.91 89.62 95.78
********************************************************************************************************* 


Problem in the coreference annotation:
 USR	calling-dial2012-turn2	12	call	0)

Problem in the coreference annotation:
 USR	calling-dial2018-turn2	4	Kyle	0)

Problem in the coreference annotation:
 USR	messaging-dial1153-turn2	5	Mary	1)

Problem in the coreference annotation:
 USR	messaging-dial1356-turn4	3	BC	0)

Problem in the coreference annotation:
 USR	music-dial274-turn0	4	night	0)

Problem in the coreference annotation:
 USR	reminders-dial386-turn2	9	morning	1)

Problem in the coreference annotation:
 USR	weather-dial96-turn2	7	today	0)

mentions   Recall: 76.45  Precision: 82.22  F1: 79.23
muc        Recall: 70.41  Precision: 77.14  F1: 73.62
bcub       Recall: 72.36  Precision: 78.56  F1: 75.33
ceafe      Recall: 76.69  Precision: 80.18  F1: 78.40
lea        Recall: 69.57  Precision: 74.88  F1: 72.13
Epoch: Test | test: QR Binary F1: 89.01 (93.91), LM: P: 88.63 R: 88.40 F1: 87.99 (63.76) | COREF P: 78.63 R: 73.15 F1: 75.78 (72.13) | time: 187.3

************************ QR Result on test ************************
	Macro P R F1 | Micro P R F1 | BLEU | ROUGE-1 -2 L F1
	88.63 88.40 87.99 67.88 60.12 63.76 90.99 96.27 89.83 96.20
********************************************************************************************************* 


Decode file is saved at decode/trained-cread.json
Done decoding!
