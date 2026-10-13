"""Model adapters: the only part of the audit that knows about a specific architecture.

Everything downstream (alignment, metrics, reporting) consumes the adapter's output
and never touches InLegalBERT, 512-token windows or a Bi-GRU. Auditing a different
legal-judgment-prediction model means writing a new subclass of :class:`ModelAdapter`,
not modifying the audit.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from .alignment import DocumentSpace, aggregate_chunk_scores_to_tokens, build_document_space
from .config import (
    DROPOUT,
    GRU_HIDDEN,
    GRU_LAYERS,
    IG_INTERNAL_BATCH,
    IG_MAX_RELATIVE_DELTA,
    IG_STEPS,
    IG_STEPS_RETRY,
    NUM_CLASSES,
    UNFREEZE_LAYERS,
)


@dataclass
class Prediction:
    """A single model decision, with the raw logits kept for delta-based metrics."""

    label: int
    confidence: float
    probabilities: np.ndarray
    logits: np.ndarray

    def logit(self, target: int) -> float:
        return float(self.logits[target])

    def probability(self, target: int) -> float:
        return float(self.probabilities[target])

    def margin(self, target: int) -> float:
        """Target logit minus the other class's logit: the log-odds of the decision."""
        return float(self.logits[target] - self.logits[1 - target])

    def score(self, target: int, mode: str = "logit") -> float:
        return self.margin(target) if mode == "margin" else self.logit(target)


@dataclass
class Attribution:
    """Word-level attribution over the shared document space."""

    word_scores: np.ndarray          # magnitude of contribution, one per word
    word_scores_signed: np.ndarray   # signed: positive pushes toward the target class
    token_scores: np.ndarray         # global sub-word scores, before word folding
    target: int
    method: str
    convergence_delta: Optional[float] = None
    target_mode: str = "logit"
    relative_delta: Optional[float] = None
    steps_used: Optional[int] = None

    def to_dict(self) -> Dict[str, object]:
        def _num(value):
            return None if value is None else float(value)

        return {
            "attribution_method": self.method,
            "attribution_target": self.target_mode,
            "attribution_target_class": int(self.target),
            "ig_convergence_delta": _num(self.convergence_delta),
            "ig_relative_delta": _num(self.relative_delta),
            "ig_steps_used": self.steps_used,
        }


_WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")
HF_ENCODER_ID = "law-ai/InLegalBERT"


def is_local_model_path(name: str) -> bool:
    """True only for filesystem paths, not Hugging Face repo ids like ``org/model``.

    A naive ``'/' in name`` check treats Hub ids as local folders, which is why
    ``law-ai/InLegalBERT`` previously failed with a repo-id validation error.
    """
    if not name:
        return False
    expanded = os.path.expanduser(name)
    if os.path.isdir(expanded):
        return True
    if name.startswith(".") or name.startswith("~") or os.path.isabs(name):
        return True
    if len(name) >= 2 and name[1] == ":":
        return True
    return False


def _has_encoder_weights(directory: str) -> bool:
    return any(os.path.exists(os.path.join(directory, f)) for f in _WEIGHT_FILES)


def resolve_encoder(model_name: str) -> str:
    """Pick a loadable encoder: a complete local folder, otherwise the Hub id.

    ``./InLegalBERT`` with only tokenizer files is not enough for an audit, so we
    fall through to ``law-ai/InLegalBERT`` rather than dying inside transformers.
    """
    if is_local_model_path(model_name):
        if _has_encoder_weights(model_name):
            return model_name
        print(
            "Local encoder '{}' has no weights; loading {} from Hugging Face.".format(
                model_name, HF_ENCODER_ID
            )
        )
        return HF_ENCODER_ID
    if model_name in (HF_ENCODER_ID, "InLegalBERT") and _has_encoder_weights("./InLegalBERT"):
        return "./InLegalBERT"
    return model_name or HF_ENCODER_ID


def check_audit_inputs(model_name: str, checkpoint_path: Optional[str]) -> None:
    """Fail early when a *local* encoder folder is incomplete.

    Hub ids are not checked here: ``from_pretrained`` downloads them. A missing
    Stage-1 checkpoint is handled by :meth:`HierarchicalLegalBERTAdapter.from_checkpoint`.
    """
    if is_local_model_path(model_name) and not _has_encoder_weights(model_name):
        raise FileNotFoundError(
            "'{}' has no encoder weights ({}).\n"
            "Pass --model law-ai/InLegalBERT to load the public encoder from the Hub.".format(
                model_name, " or ".join(_WEIGHT_FILES)
            )
        )
    if checkpoint_path and not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            "Trained checkpoint '{}' not found (looked in {}).\n"
            "This is your Stage-1 HierarchicalLegalBERT, not public InLegalBERT. "
            "Either copy best_hierarchical_model.pt here, or run with "
            "--allow-untrained to load InLegalBERT from Hugging Face and keep a "
            "random GRU/classifier (pipeline test only, not the paper result).\n"
            "To pull the real checkpoint:\n"
            "    hf auth login\n"
            "    hf download Capstone-LawAI/capstone-legal --local-dir . "
            "--repo-type model".format(
                checkpoint_path, os.path.abspath(checkpoint_path)
            )
        )


def load_tokenizer(model_name: str):
    """Load the fast tokenizer from a local folder or a Hugging Face repo id."""
    encoder = resolve_encoder(model_name)
    if is_local_model_path(encoder) and not os.path.isdir(encoder):
        raise FileNotFoundError(
            "Model directory '{}' does not exist (looked in {}).".format(
                encoder, os.path.abspath(encoder)
            )
        )
    tokenizer = AutoTokenizer.from_pretrained(encoder, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError(
            "Stage 3 needs a fast tokenizer for character offsets, but "
            "'{}' produced a slow one.".format(encoder)
        )
    return tokenizer


class ModelAdapter(ABC):
    """Interface the audit relies on. Implement these four members for a new model."""

    @abstractmethod
    def document_space(self, document: str) -> DocumentSpace:
        """Tokenise a document into the shared word/token/chunk coordinate system."""

    @abstractmethod
    def predict(self, document: str, space: Optional[DocumentSpace] = None) -> Prediction:
        """Predict on a document."""

    @abstractmethod
    def attribute(
        self,
        document: str,
        space: Optional[DocumentSpace] = None,
        target: Optional[int] = None,
        method: str = "integrated_gradients",
    ) -> Attribution:
        """Attribute the target-class logit back onto the document's words."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Identifier recorded in the audit report."""


class HierarchicalLegalBERTAdapter(ModelAdapter):
    """Adapter for the Stage-1 InLegalBERT + Bi-GRU model.

    By default (``target_mode="margin"``) attribution explains the **logit margin**,
    the chosen class's logit minus the other class's. For two classes the softmax
    probability is sigmoid(margin), so the margin is exactly the log-odds the
    decision rests on, without the saturation of the probability. A single class
    logit (``target_mode="logit"``, the setting of the first Stage-3 run) also credits
    text that raises both logits equally, which cannot change the decision.
    """

    def __init__(
        self,
        model,
        tokenizer,
        device,
        ig_steps: int = IG_STEPS,
        ig_internal_batch: int = IG_INTERNAL_BATCH,
        baseline: str = "pad",
        chunk_aggregation: str = "mean",
        word_aggregation: str = "sum",
        model_name: str = "HierarchicalLegalBERT",
        gradient_checkpointing: bool = False,
        target_mode: str = "margin",
        ig_steps_retry: int = IG_STEPS_RETRY,
        ig_max_relative_delta: float = IG_MAX_RELATIVE_DELTA,
    ) -> None:
        if target_mode not in ("margin", "logit"):
            raise ValueError("target_mode must be 'margin' or 'logit'")
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.ig_steps = ig_steps
        self.ig_internal_batch = ig_internal_batch
        self.baseline = baseline
        self.chunk_aggregation = chunk_aggregation
        self.word_aggregation = word_aggregation
        self.target_mode = target_mode
        self.ig_steps_retry = ig_steps_retry
        self.ig_max_relative_delta = ig_max_relative_delta
        self._name = model_name
        self.model.eval()
        if gradient_checkpointing:
            # A document is 16 chunks of 512 tokens and the Bi-GRU needs all of them
            # in one pass, so an IG step is a batch-16 forward *and* backward through
            # BERT. Checkpointing trades roughly 30% throughput for a large drop in
            # activation memory, which is what makes this fit on a smaller GPU.
            self.model.bert.gradient_checkpointing_enable()

    # ── construction ────────────────────────────────────────────────────────
    @classmethod
    def from_checkpoint(
        cls,
        model_name: str,
        checkpoint_path: Optional[str] = None,
        device=None,
        allow_untrained: bool = False,
        **kwargs,
    ) -> "HierarchicalLegalBERTAdapter":
        from hierarchical_model import HierarchicalLegalBERT, load_checkpoint

        encoder = resolve_encoder(model_name)
        has_checkpoint = bool(checkpoint_path) and os.path.exists(checkpoint_path)
        if checkpoint_path and not has_checkpoint and not allow_untrained:
            check_audit_inputs(encoder, checkpoint_path)

        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = load_tokenizer(encoder)
        print("Loading encoder from {}".format(encoder))
        model = HierarchicalLegalBERT(
            encoder,
            gru_hidden=GRU_HIDDEN,
            gru_layers=GRU_LAYERS,
            dropout=DROPOUT,
            num_classes=NUM_CLASSES,
            unfreeze_last_n=UNFREEZE_LAYERS,
        ).to(device)

        if has_checkpoint:
            report = load_checkpoint(model, checkpoint_path, device)
        else:
            torch.manual_seed(0)
            print(
                "WARNING: no Stage-1 checkpoint loaded. InLegalBERT weights are "
                "pretrained; the Bi-GRU and classifier are randomly initialised. "
                "This validates the audit pipeline, not the trained LJP model."
            )
            report = {
                "epoch": None,
                "best_f1": None,
                "missing_keys": ["gru.*", "classifier.*"],
                "unexpected_keys": [],
                "n_missing": 2,
                "n_unexpected": 0,
                "untrained_aggregator": True,
            }

        model.eval()
        adapter = cls(model, tokenizer, device, model_name=encoder, **kwargs)
        adapter.checkpoint_report = report
        return adapter

    @property
    def name(self) -> str:
        return self._name

    # ── core interface ──────────────────────────────────────────────────────
    def document_space(self, document: str) -> DocumentSpace:
        return build_document_space(document, self.tokenizer)

    def _batch(self, space: DocumentSpace):
        ids = torch.tensor([c[0] for c in space.chunks], dtype=torch.long, device=self.device)
        mask = torch.tensor([c[1] for c in space.chunks], dtype=torch.long, device=self.device)
        return ids.unsqueeze(0), mask.unsqueeze(0)

    def predict(self, document: str, space: Optional[DocumentSpace] = None) -> Prediction:
        space = space or self.document_space(document)
        ids, mask = self._batch(space)
        counts = torch.tensor([space.n_chunks], dtype=torch.long)
        with torch.no_grad():
            logits = self.model(ids, mask, counts)
        probabilities = F.softmax(logits, dim=-1)[0].detach().cpu().numpy()
        logits_np = logits[0].detach().cpu().numpy()
        label = int(probabilities.argmax())
        return Prediction(
            label=label,
            confidence=float(probabilities.max()),
            probabilities=probabilities,
            logits=logits_np,
        )

    def predict_many(self, documents: Sequence[str]) -> List[Prediction]:
        return [self.predict(d) for d in documents]

    def _attribution_mode(self):
        """cuDNN will not backprop through a GRU that is in eval() mode.

        Integrated Gradients needs that backward pass, so the module is set to
        train() for the duration of attribution. ``model.train()`` would also
        re-enable the 2-layer GRU's *internal* dropout (0.3 between layers), which
        the outer ``self.dropout.eval()`` does not cover. We zero ``gru.dropout``
        while leaving the GRU in train() so the heatmap matches eval-time
        computation as closely as cuDNN allows.
        """
        from contextlib import contextmanager

        @contextmanager
        def _ctx():
            was_training = self.model.training
            bert_training = self.model.bert.training
            dropout_training = self.model.dropout.training
            gru_dropout = float(getattr(self.model.gru, "dropout", 0.0) or 0.0)
            self.model.train()
            self.model.bert.eval()
            self.model.dropout.eval()
            self.model.gru.dropout = 0.0
            try:
                yield
            finally:
                self.model.gru.dropout = gru_dropout
                self.model.train(was_training)
                self.model.bert.train(bert_training)
                self.model.dropout.train(dropout_training)

        return _ctx()

    def attribute(
        self,
        document: str,
        space: Optional[DocumentSpace] = None,
        target: Optional[int] = None,
        method: str = "integrated_gradients",
    ) -> Attribution:
        space = space or self.document_space(document)
        if target is None:
            target = self.predict(document, space).label

        relative_delta, steps_used = None, None
        with self._attribution_mode():
            if method == "integrated_gradients":
                chunk_signed, delta, relative_delta = self._integrated_gradients(
                    space, target, self.ig_steps
                )
                steps_used = self.ig_steps
                if (
                    relative_delta > self.ig_max_relative_delta
                    and self.ig_steps_retry > self.ig_steps
                ):
                    retry = self._integrated_gradients(space, target, self.ig_steps_retry)
                    if retry[2] < relative_delta:
                        chunk_signed, delta, relative_delta = retry
                        steps_used = self.ig_steps_retry
            elif method == "gradient_x_input":
                chunk_signed, delta = self._gradient_x_input(space, target)
            else:
                raise ValueError(
                    "Unknown attribution method '{}'. This adapter implements "
                    "'integrated_gradients' and 'gradient_x_input' only -- occlusion "
                    "is not implemented anywhere in this codebase yet, despite an "
                    "earlier version of this message implying otherwise.".format(method)
                )

        token_signed = aggregate_chunk_scores_to_tokens(
            chunk_signed, space, mode=self.chunk_aggregation
        )
        token_magnitude = aggregate_chunk_scores_to_tokens(
            np.abs(chunk_signed), space, mode=self.chunk_aggregation
        )
        word_scores = space.aggregate_tokens_to_words(
            token_magnitude, mode=self.word_aggregation
        )
        word_signed = space.aggregate_tokens_to_words(
            token_signed, mode=self.word_aggregation
        )
        return Attribution(
            word_scores=word_scores,
            word_scores_signed=word_signed,
            token_scores=token_magnitude,
            target=int(target),
            method=method,
            convergence_delta=delta,
            target_mode=self.target_mode,
            relative_delta=relative_delta,
            steps_used=steps_used,
        )

    # ── attribution back-ends ───────────────────────────────────────────────
    def _embed(self, ids: torch.Tensor) -> torch.Tensor:
        return self.model.word_embedding_layer(ids)

    def _baseline_ids(self, ids: torch.Tensor) -> torch.Tensor:
        """Baseline sequence: real tokens replaced by [PAD], [CLS]/[SEP] preserved.

        Keeping the special tokens means the baseline is a structurally valid but
        content-free input, so the attribution measures the contribution of the
        judgment text rather than the contribution of having a sequence at all.
        """
        if self.baseline == "zero":
            return None
        pad_id = self.tokenizer.pad_token_id
        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        keep = (ids == cls_id) | (ids == sep_id)
        return torch.where(keep, ids, torch.full_like(ids, pad_id))

    def _forward_for_captum(self, embeds, attention_mask):
        logits = self.model.forward_from_embeddings(embeds, attention_mask)
        if self.target_mode == "margin":
            # Column c becomes logit_c - logit_(other); selecting the target column
            # then gives the decision margin. Two classes only (NUM_CLASSES == 2).
            return logits - logits.flip(-1)
        return logits

    def _integrated_gradients(self, space: DocumentSpace, target: int, n_steps: int):
        from captum.attr import IntegratedGradients

        ids, mask = self._batch(space)
        embeds = self._embed(ids)
        baseline_ids = self._baseline_ids(ids)
        baselines = (
            torch.zeros_like(embeds) if baseline_ids is None else self._embed(baseline_ids)
        )

        integrated_gradients = IntegratedGradients(self._forward_for_captum)
        attributions, delta = integrated_gradients.attribute(
            inputs=embeds,
            baselines=baselines,
            additional_forward_args=(mask,),
            target=int(target),
            n_steps=n_steps,
            internal_batch_size=self.ig_internal_batch,
            return_convergence_delta=True,
        )
        # Completeness: attributions should sum to F(input) - F(baseline). The raw
        # delta is in logit units, so it is judged relative to that difference, with
        # a floor of 1 so a near-zero difference does not inflate a tiny error.
        with torch.no_grad():
            gap = (
                self._forward_for_captum(embeds, mask)[0, int(target)]
                - self._forward_for_captum(baselines, mask)[0, int(target)]
            ).item()
        delta_abs = float(delta.abs().max().item())
        relative = delta_abs / max(abs(gap), 1.0)
        # [1, chunks, seq, hidden] -> [chunks, seq]; the embedding dimension is summed
        # rather than abs-summed so the sign of the contribution survives.
        chunk_scores = attributions.sum(dim=-1)[0].detach().cpu().numpy()
        return chunk_scores, delta_abs, float(relative)

    def _gradient_x_input(self, space: DocumentSpace, target: int):
        ids, mask = self._batch(space)
        embeds = self._embed(ids).detach().requires_grad_(True)
        logits = self._forward_for_captum(embeds, mask)  # margin or logit, per target_mode
        self.model.zero_grad(set_to_none=True)
        logits[0, int(target)].backward()
        grad = embeds.grad
        chunk_scores = (grad * embeds).sum(dim=-1)[0].detach().cpu().numpy()
        return chunk_scores, None

    # ── perturbation helper used by faithfulness tests ──────────────────────
    def predict_with_words_masked(
        self, space: DocumentSpace, word_mask: np.ndarray
    ) -> Prediction:
        """Re-predict with the selected words replaced by the mask token."""
        replacement = self.tokenizer.mask_token or "[MASK]"
        pieces: List[str] = []
        cursor = 0
        for index, (start, end) in enumerate(space.word_spans):
            if not word_mask[index]:
                continue
            pieces.append(space.document[cursor:start])
            pieces.append(replacement)
            cursor = end
        pieces.append(space.document[cursor:])
        return self.predict("".join(pieces))
