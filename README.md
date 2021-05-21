# CREAD: Combined Resolution of Ellipses and Anaphora in Dialogues 

[**Paper**](https://arxiv.org/abs/2105.09914) |
[**Tasks**](#Task) | [**Dataset**](#Dataset) | [**Evaluation**](#Run-the-code) |
[**Citation**](#Citation) | [**License**](#License)

This is the source code of the paper [CREAD: Combined Resolution of Ellipses and Anaphora in Dialogues](https://arxiv.org/abs/2105.09914).
In this work, we propose a novel joint learning framework of modeling coreference resolution and query rewriting for complex, multi-turn dialogue understanding.
The coreference resolution [MuDoCo](https://github.com/facebookresearch/mudoco) dataset augmented with our query rewrite annotation is released as well.

## Task Description
Given an ongoing dialogue between a user and a dialogue assistant, for the user query, the model is required to predict both coreference links between the query and the dialogue context, and the self\-contained rewritten user query that is independent to the dialogue context.

## Dataset
The MuDoCo dataset is a public dataset that contains 7.5k task\-oriented multi\-turn dialogues across 6 domains (calling, messaging, music, news, reminders, weather). Each dialogue turn is annotated with coreference links (`links` field). Please refer to [MuDoCo](https://github.com/facebookresearch/mudoco) for more details.

In the **MuDoCo\-QR\-dataset** used in work, we annotate the query rewrite for each utterance, including both user and system turn. On top of the MudoCo data format, we add three fields `graded`, `rewrite_required` and `rewritten_utterance`. Most of the turns are with annotated with query rewrite (`graded` is true). Only 1.4% dialogue turns with incomplete dialogue context (e.g., missing turns) in MuDoCo are filtered out (`graded` is false). `rewrite_required` records whether the input utterance should be rewritten or not. `rewritten_utterance` is the rewritten query, same as the utterance if `rewrite_required` is false.

The resulting dataset is provided in the folder `MuDoCo-QR-dataset`.

```json
{
    "number": 3,
    "utterance": "Show me a live version that he moonwalks on .",
    "links": [
        [
            {
                "turn_id": 1,
                "text": "Michael Jackson",
                "span": {
                    "start": 5,
                    "end": 20
                }
            },
            {
                "turn_id": 3,
                "text": "he",
                "span": {
                    "start": 28,
                    "end": 30
                }
            }
        ]
    ],
    "graded": true,
    "rewritten_utterance": "Show me a live version that Michael Jackson moonwalks on",
    "rewrite_required": true
}
```

## Requirements
python3.6 and the packages in `requirements.txt`, install them by running
```console
>>> pip install -r requirements.txt
```

## Run the Code
Enter the `modeling` folder and follow the instruction below.

```console
>>> cd modeling
```

## Data Pre-processing
First run the following command to prepare the data for training.

The processed data will be stored in the `proc_data/` directory.

```console
>>> python utils/process_data.py
```


## Training
Run `train.sh` to train the model, which calls `main.py` with default hyper-parameters.

```console
>>> bash train.sh [job_name]
```

The model checkpoint will be stored at `checkpoint/$job_name`, and training log file is at `log/$job_name.log`

A reference training log (`log/trained-cread.log`) is provided.


## Evaluation
Run `decode.sh` to decode using a trained model. `job_name` is the same as specified in training.

```console
>>> bash decode.sh [job_name]
```

Evaluation result, with both generated rewritten utterances and model performance, is recorded in `deocde/$job_name.json`.

A reference decoding file (`decode/trained-cread.json`) is provided.



## Key Hyper-parameters in Main.py
- task: which task to perform. The default value `qr-coref` specifies our complete joint learning model. Set to `qr` for the model variant `qr-only` model or `coref` for the model variant `coref-only` model.

- coref\_layer\_idx: which gpt2 layers to use for coreference resolution, e.g., "1,5,11" uses three layers. n is between 0 to 11, if default gpt2\-small is used.
- n\_coref\_head: how many attention heads to use in each layer for coreference resolution. n is between 1 to 12.
- use_coref\_attn: whether to use coref2qr attention mechanism.
- use\_binary\_cls: whether to use binary rewriting classifier.

More detailed explanation of other arguments can be found in `utils/utils.py`.

## Citation
```bibtex
@misc{tseng2021cread,
      title={CREAD: Combined Resolution of Ellipses and Anaphora in Dialogues}, 
      author={Bo-Hsiang Tseng and Shruti Bhargava and Jiarui Lu and Joel Ruben Antony Moniz and Dhivya Piraviperumal and Lin Li and Hong Yu},
      year={2021},
      eprint={2105.09914},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
```

## License
The code in this repository is licensed according to the [LICENSE](LICENSE) file.

## Contact Us
Please contact bht26@cam.ac.uk or hong\_yu@apple.com, or raise an issue in this repository.
