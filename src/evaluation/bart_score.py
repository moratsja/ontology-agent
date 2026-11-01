import torch
import torch.nn as nn
from transformers import BartForConditionalGeneration, BartTokenizer

CHECKPOINT = "facebook/bart-large-cnn"

class BARTScorer:
    def __init__(self, device=None, max_length=1024, checkpoint=CHECKPOINT):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.max_length = max_length
        self.tokenizer = BartTokenizer.from_pretrained(checkpoint)
        self.model = BartForConditionalGeneration.from_pretrained(checkpoint)
        self.model.eval()
        self.model.to(device)
        self.loss_fct = nn.NLLLoss(reduction="none", ignore_index=self.model.config.pad_token_id)
        self.lsm = nn.LogSoftmax(dim=1)

    def score(self, srcs, tgts, batch_size=4):
        # Log-likelihood of generating each target from its source. Closer to zero is better.
        score_list = []
        for start in range(0, len(srcs), batch_size):
            src_list = srcs[start:start + batch_size]
            tgt_list = tgts[start:start + batch_size]
            with torch.no_grad():
                encoded_src = self.tokenizer(
                    src_list,
                    max_length=self.max_length,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                )
                encoded_tgt = self.tokenizer(
                    tgt_list,
                    max_length=self.max_length,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                )
                src_tokens = encoded_src["input_ids"].to(self.device)
                src_mask = encoded_src["attention_mask"].to(self.device)
                tgt_tokens = encoded_tgt["input_ids"].to(self.device)
                tgt_len = encoded_tgt["attention_mask"].sum(dim=1).to(self.device)
                output = self.model(
                    input_ids=src_tokens,
                    attention_mask=src_mask,
                    labels=tgt_tokens,
                )
                logits = output.logits.view(-1, self.model.config.vocab_size)
                loss = self.loss_fct(self.lsm(logits), tgt_tokens.view(-1))
                loss = loss.view(tgt_tokens.shape[0], -1)
                loss = loss.sum(dim=1) / tgt_len
                score_list.extend(-value.item() for value in loss)
        return score_list

    def score_summary(self, posts, summary):
        chunks = self._chunks(posts)
        if not chunks or not summary.strip():
            return None
        scores = self.score(chunks, [summary] * len(chunks), batch_size=1)
        return sum(scores) / len(scores)

    def _chunks(self, posts):
        limit = self.max_length - 2
        chunks = []
        current = []
        current_len = 0
        for post in posts:
            text = str(post).strip()
            if not text:
                continue
            length = len(self.tokenizer(text, add_special_tokens=False)["input_ids"])
            if length > limit:
                length = limit
                text = self.tokenizer.decode(
                    self.tokenizer(text, add_special_tokens=False)["input_ids"][:limit]
                )
            if current and current_len + length > limit:
                chunks.append("\n".join(current))
                current = []
                current_len = 0
            current.append(text)
            current_len += length
        if current:
            chunks.append("\n".join(current))
        return chunks
