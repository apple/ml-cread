import bleu
import logging
from typing import List, Dict
from collections import Counter
from qr_eval.qr.common.constants import indicator_tokens
from qr_eval.qr.rouge import rouge_scorer
from qr_eval.qr.rouge.scoring import Score

LOGGER = logging.getLogger(__name__)

'''
Metrics for query rewrite
'''


# Contains a collection of metrics
# Removing Bleu for now because of efficiency issue
class MetricCollection:
    def __init__(self):
        self.f1_metric = F1()
        self.utterance_metric = UtteranceScore()

    # Call each different metric
    def __call__(
        self, ref_list: List[List[str]], hypo_list: List[List[str]], input_list: List[List[str]]
    ):
        self.f1_metric(ref_list, hypo_list, input_list)
        self.utterance_metric(ref_list, hypo_list, input_list)

    # Get and aggregate metric
    def get_metric(self, reset: bool = False) -> Dict[str, float]:
        metrics = {**self.f1_metric.get_metric(), **self.utterance_metric.get_metric()}
        if reset:
            self.reset()
        return metrics

    def get_result(self):
        return self.utterance_metric.get_result()

    # Call each reset
    def reset(self):
        self.f1_metric.reset()
        self.utterance_metric.reset()

    # Pretty Print
    def __str__(self):
        # Set left align 12 spaces for each column
        metric = self.get_metric()
        keys = sorted(metric.keys())
        header_format = "{:<12}" * len(keys)
        value_format = "{:<12.2f}" * len(keys)
        header = header_format.format(*keys)
        value = value_format.format(*map(self.get_metric().get, keys))
        return header + "\n" + value


class F1:
    def __init__(self):
        # Macro F1 stats
        self.macro_precision_list = []
        self.macro_recall_list = []
        self.macro_f1_list = []
        # Micro F1 stats
        self.micro_true_positive = 0.0
        self.micro_false_negative = 0.0
        self.micro_false_positive = 0.0

    # Macro stats increment, even though append is 'atomic', there is no guarantee that the internal order
    # of all three lists will be in sync, so we return these result, enforce their order using map
    def _single_call(self, ref: List[str], hypo: List[str], implicit: List[str]):
        # Construct counts, + removes remove zero and negative
        ground_truth = +(Counter(ref) - Counter(implicit))
        prediction = +(Counter(hypo) - Counter(implicit))
        true_positive = sum((prediction & ground_truth).values())
        # Micro stats increment, += is 'atomic' in python so we can do this
        self.micro_true_positive += true_positive
        self.micro_false_negative += sum((+(ground_truth - prediction)).values())
        self.micro_false_positive += sum((+(prediction - ground_truth)).values())
        # if Ref doesn't contain more info than implicit, consider it correct
        if sum(ground_truth.values()) == 0 and sum(prediction.values()) == 0:
            precision, recall, f1 = 1.0, 1.0, 1.0
        elif true_positive == 0.0:
            precision, recall, f1 = 0.0, 0.0, 0.0
        else:
            precision = true_positive / sum(prediction.values())
            recall = true_positive / sum(ground_truth.values())
            f1 = 2 * precision * recall / (precision + recall)
        return precision, recall, f1

    def __call__(
        self, ref_list: List[List[str]], hypo_list: List[List[str]], implicit_list: List[List[str]]
    ):
        precision, recall, f1 = zip(*map(self._single_call, ref_list, hypo_list, implicit_list))
        self.macro_precision_list += precision
        self.macro_recall_list += recall
        self.macro_f1_list += f1

    # Return metrics * 100
    def get_metric(self, reset: bool = False) -> Dict[str, float]:
        metrics = {}
        # Compute Micro Stats
        precision = self.micro_true_positive / (
#            self.micro_true_positive + self.micro_false_negative
            self.micro_true_positive + self.micro_false_positive
        )
#        recall = self.micro_true_positive / (self.micro_true_positive + self.micro_false_positive)
        recall = self.micro_true_positive / (self.micro_true_positive + self.micro_false_negative)
        if self.micro_true_positive == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        metrics["Micro P"] = precision * 100
        metrics["Micro R"] = recall * 100
        metrics["Micro F1"] = f1 * 100
        # Compute Macro Stats
        metrics["Macro P"] = sum(self.macro_precision_list) / len(self.macro_precision_list) * 100
        metrics["Macro R"] = sum(self.macro_recall_list) / len(self.macro_recall_list) * 100
        metrics["Macro F1"] = sum(self.macro_f1_list) / len(self.macro_f1_list) * 100
        if reset:
            self.reset()
        return metrics

    def reset(self):
        # Macro F1 stats
        self.macro_precision_list = []
        self.macro_recall_list = []
        self.macro_f1_list = []
        # Micro F1 stats
        self.micro_true_positive = 0.0
        self.micro_false_negative = 0.0
        self.micro_false_positive = 0.0


# Combining BLEU and ROUGE
class UtteranceScore:
    def __init__(self):
        # Store results to compute bleu / rouge score afterwards
        self.ref_list: List[str] = []
        self.hypo_list: List[str] = []
        self.implicit_list: List[str] = []
        self.rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        self.metric_mapping = {"precision": "P", "recall": "R", "fmeasure": "F1"}

    @staticmethod
    # Get rid of indicator tokens, turn indices into string
    def _filter_detokenize(indices: List[str]):
        return " ".join([x for x in indices if x not in indicator_tokens])

    def __call__(
        self, ref_list: List[List[str]], hypo_list: List[List[str]], implicit_list: List[List[str]]
    ):
        # Get rid of indicator tokens, turn indices into string
        ref, hypo, implicit = zip(
            *map(
                lambda r, h, i: (
                    self._filter_detokenize(r),
                    self._filter_detokenize(h),
                    self._filter_detokenize(i),
                ),
                ref_list,
                hypo_list,
                implicit_list,
            )
        )
        self.ref_list += ref
        self.hypo_list += hypo
        self.implicit_list += implicit

    def _unpack_rouge(self, r, h):
        rouge_metrics = self.rouge.score(r, h)
        return [rouge_metrics[k] for k in self.rouge.rouge_types]

    def get_metric(self, reset: bool = False):
        metrics = {"BLEU": bleu.list_bleu([self.ref_list], self.hypo_list)}
        # Compute score for each pair and rouge type
        rouge_list: List[List[Score]] = list(map(self._unpack_rouge, self.ref_list, self.hypo_list))
        for i, current_rouge in enumerate(zip(*rouge_list)):
            for j, metric_name in enumerate(self.metric_mapping.keys()):
                metrics[
                    f"{self.rouge.rouge_types[i].upper()} {self.metric_mapping[metric_name]}"
                ] = (sum(score[j] for score in current_rouge) / len(current_rouge) * 100)
        if reset:
            self.reset()
        return metrics

    def reset(self):
        self.ref_list = []
        self.hypo_list = []
        self.implicit_list = []

    def get_result(self):
        return list(zip(self.ref_list, self.hypo_list, self.implicit_list))
