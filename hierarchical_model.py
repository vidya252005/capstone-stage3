"""The Stage-1 model, shared by the Stage-2 ablation and the Stage-3 audit.

This used to live inside ``opennyai_ablation.py``. It is defined once here so the
architecture the audit explains cannot drift from the architecture that was trained.
Parameter names are unchanged (``bert.``, ``gru.``, ``classifier.``), so existing
checkpoints load as before.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel


class HierarchicalLegalBERT(nn.Module):
    """InLegalBERT chunk encoder -> per-chunk [CLS] -> 2-layer Bi-GRU -> classifier."""

    def __init__(
        self,
        model_name: str,
        gru_hidden: int = 256,
        gru_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 2,
        unfreeze_last_n: int = 3,
    ) -> None:
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        for param in self.bert.parameters():
            param.requires_grad = False
        num_layers = len(self.bert.encoder.layer)
        for i in range(num_layers - unfreeze_last_n, num_layers):
            for param in self.bert.encoder.layer[i].parameters():
                param.requires_grad = True
        if self.bert.pooler is not None:
            for param in self.bert.pooler.parameters():
                param.requires_grad = True

        bert_dim = self.bert.config.hidden_size
        self.gru = nn.GRU(
            input_size=bert_dim,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(gru_hidden * 2, num_classes)

    def forward(self, input_ids, attention_mask, chunk_counts):
        batch, max_chunks, seq_len = input_ids.shape
        ids = input_ids.view(batch * max_chunks, seq_len)
        masks = attention_mask.view(batch * max_chunks, seq_len)
        bert_out = self.bert(input_ids=ids, attention_mask=masks)
        cls_emb = bert_out.last_hidden_state[:, 0, :]
        cls_emb = cls_emb.view(batch, max_chunks, -1)
        cls_emb = self.dropout(cls_emb)
        packed = nn.utils.rnn.pack_padded_sequence(
            cls_emb, chunk_counts.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.gru(packed)
        final = torch.cat([hidden[-2], hidden[-1]], dim=1)
        final = self.dropout(final)
        return self.classifier(final)

    def forward_from_embeddings(self, word_embeddings, attention_mask):
        """Same computation as :meth:`forward`, entered at the word-embedding layer.

        Integrated Gradients needs a differentiable path from a continuous input, and
        token ids are discrete, so attribution is taken with respect to the word
        embeddings. ``word_embeddings`` is ``[batch, chunks, seq_len, hidden]``; the
        BERT embedding module still adds position and token-type embeddings on top.

        Every document is passed with all of its chunks present, so the length-packing
        used in :meth:`forward` would be a no-op and is skipped here. ``test_stage3.py``
        asserts the two paths agree numerically.
        """
        batch, n_chunks, seq_len, hidden_size = word_embeddings.shape
        flat_embeds = word_embeddings.reshape(batch * n_chunks, seq_len, hidden_size)
        flat_mask = attention_mask.reshape(batch * n_chunks, seq_len)
        bert_out = self.bert(inputs_embeds=flat_embeds, attention_mask=flat_mask)
        cls_emb = bert_out.last_hidden_state[:, 0, :].view(batch, n_chunks, -1)
        cls_emb = self.dropout(cls_emb)
        _, hidden = self.gru(cls_emb)
        final = torch.cat([hidden[-2], hidden[-1]], dim=1)
        final = self.dropout(final)
        return self.classifier(final)

    @property
    def word_embedding_layer(self) -> nn.Module:
        return self.bert.embeddings.word_embeddings


def load_checkpoint(
    model: nn.Module, checkpoint_path: str, device, strict: bool = False
) -> dict:
    """Load a checkpoint and return a report of what did and did not match.

    Stage 1/2 previously monkey-patched ``nn.Module.load_state_dict`` globally to force
    ``strict=False`` for every module in the process. That hides genuine mismatches, so
    the audit loads non-strictly but returns the missing/unexpected keys for inspection
    instead: an audit has to be able to state that the model it explained is the model
    that was trained.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    result = model.load_state_dict(state_dict, strict=strict)
    missing = list(getattr(result, "missing_keys", []))
    unexpected = list(getattr(result, "unexpected_keys", []))
    return {
        "epoch": checkpoint.get("epoch"),
        "best_f1": checkpoint.get("best_f1"),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "n_missing": len(missing),
        "n_unexpected": len(unexpected),
    }
