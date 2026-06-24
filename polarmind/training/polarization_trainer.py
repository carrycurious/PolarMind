from __future__ import annotations

import torch
from sklearn.metrics import f1_score
from transformers import DataCollatorWithPadding, TrainerCallback
from trl import SFTTrainer

from polarmind.constants import MULTILABEL_CATEGORIES


class LabelPreservingCollator(DataCollatorWithPadding):
    """Keep multi-label columns in the batch for custom loss computation."""

    def __call__(self, features):
        batch = super().__call__(features)
        for category in MULTILABEL_CATEGORIES:
            batch[category] = torch.tensor([row[category] for row in features], dtype=torch.long)
        return batch


class PolarizationTrainer(SFTTrainer):
    """SFT trainer with token-level binary loss at category delimiter positions."""

    def __init__(self, *args, pos_weight: torch.Tensor, delimiter_token: str = " ::", **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight
        self.delimiter_token = delimiter_token
        self._id_0 = self.processing_class.encode("0", add_special_tokens=False)[-1]
        self._id_1 = self.processing_class.encode("1", add_special_tokens=False)[-1]
        self._delimiter_id = self.processing_class.encode(delimiter_token, add_special_tokens=False)[-1]

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        logits = outputs["logits"]
        input_ids = inputs["input_ids"]
        target_labels = torch.stack([inputs[cat] for cat in MULTILABEL_CATEGORIES], dim=1)

        batch_total_loss = 0.0
        batch_size = input_ids.shape[0]

        for batch_idx in range(batch_size):
            colon_indices = (input_ids[batch_idx] == self._delimiter_id).nonzero(as_tuple=True)[0]
            target_indices = colon_indices[-5:]
            sample_loss = 0.0

            for category_idx in range(5):
                if category_idx >= len(target_indices):
                    continue
                position = target_indices[category_idx]
                logit_0 = logits[batch_idx, position, self._id_0]
                logit_1 = logits[batch_idx, position, self._id_1]
                log_denom = torch.logsumexp(torch.stack([logit_0, logit_1]), dim=0)
                label = target_labels[batch_idx, category_idx]

                if label == 1:
                    category_loss = self.pos_weight[category_idx] * (log_denom - logit_1)
                else:
                    category_loss = log_denom - logit_0
                sample_loss += category_loss

            batch_total_loss += sample_loss

        final_loss = batch_total_loss / max(batch_size, 1)
        return (final_loss, outputs) if return_outputs else final_loss


class F1EvaluationCallback(TrainerCallback):
    def __init__(self, eval_dataset, tokenizer, delimiter_token: str = " ::"):
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.id_0 = tokenizer.encode("0", add_special_tokens=False)[-1]
        self.id_1 = tokenizer.encode("1", add_special_tokens=False)[-1]
        self.delimiter_id = tokenizer.encode(delimiter_token, add_special_tokens=False)[-1]

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % args.eval_steps == 0 and state.global_step > 0:
            self._run_eval(model, state.global_step)

    def _run_eval(self, model, step: int):
        model.eval()
        results = {cat: {"preds": [], "labels": []} for cat in MULTILABEL_CATEGORIES}
        device = next(model.parameters()).device

        for item in self.eval_dataset:
            input_ids = torch.tensor([item["input_ids"]]).to(device)
            with torch.no_grad():
                logits = model(input_ids).logits
            delimiter_indices = (input_ids[0] == self.delimiter_id).nonzero(as_tuple=True)[0][-5:]
            if len(delimiter_indices) != 5:
                continue

            for idx, position in enumerate(delimiter_indices):
                category = MULTILABEL_CATEGORIES[idx]
                pred = int(logits[0, position, self.id_1] > logits[0, position, self.id_0])
                results[category]["preds"].append(pred)
                results[category]["labels"].append(item[category])

        f1_scores = [
            f1_score(results[cat]["labels"], results[cat]["preds"], average="binary", zero_division=0)
            for cat in MULTILABEL_CATEGORIES
            if results[cat]["labels"]
        ]
        if f1_scores:
            macro = sum(f1_scores) / len(f1_scores)
            print(f"[Step {step}] category-averaged macro F1: {macro:.4f}")
        model.train()
