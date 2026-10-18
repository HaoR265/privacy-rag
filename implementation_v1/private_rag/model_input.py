"""Owner-local plaintext input preparation (D/U only), MOD-004--MOD-008.

This module does not run models, reconstruct shares, or send plaintext to S.
Prompt assembly over secret evidence is implemented separately by the R2 core.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import unicodedata

from tokenizers import Tokenizer


class InputRejected(ValueError):
    """Local structured rejection; messages never echo supplied text."""


@dataclass(frozen=True, repr=False)
class LocalTokenSlots:
    tokens: tuple[int, ...]
    validity: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.tokens) != len(self.validity) or any(v not in (0, 1) for v in self.validity):
            raise InputRejected("INVALID_LOCAL_SLOT_SHAPE")


@dataclass(frozen=True, repr=False)
class QuestionInput:
    body: LocalTokenSlots
    embedding: LocalTokenSlots


@dataclass(frozen=True, repr=False)
class DocumentChunk:
    text: str  # remains at D and belongs in encrypted retained state only
    character_start: int
    character_end: int
    authorization_binding: str
    body: LocalTokenSlots
    embedding: LocalTokenSlots


def normalize_text(raw: bytes | str) -> str:
    try:
        text = raw.decode("utf-8", errors="strict") if isinstance(raw, bytes) else raw
        text.encode("utf-8", errors="strict")
    except (UnicodeDecodeError, UnicodeEncodeError):
        raise InputRejected("INVALID_UTF8") from None
    if text.startswith("\ufeff"):
        text = text[1:]
    return unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))


class InputPreparer:
    def __init__(self, models: Path, template: Path):
        self.generation = Tokenizer.from_file(str(models / "generation/checkpoint/tokenizer.json"))
        self.embedding = Tokenizer.from_file(str(models / "embedding/checkpoint/tokenizer.json"))
        self.template = json.loads(template.read_text(encoding="utf-8"))
        self.reserved = tuple(self.template["reserved_control_strings"])
        self.dummy = self.template["dummy_token_id"]

    def _text(self, raw: bytes | str) -> str:
        text = normalize_text(raw)
        if any(control in text for control in self.reserved):
            raise InputRejected("RESERVED_CONTROL_STRING")
        return text

    def _body_ids(self, text: str) -> list[int]:
        ids = self.generation.encode(text, add_special_tokens=False).ids
        if any(t < 0 or t > 151642 for t in ids):
            raise InputRejected("RESERVED_CONTROL_TOKEN")
        return ids

    def _slots(self, ids: list[int], capacity: int) -> LocalTokenSlots:
        if len(ids) > capacity:
            raise InputRejected("TOKEN_BUDGET_EXCEEDED")
        return LocalTokenSlots(tuple(ids + [self.dummy] * (capacity - len(ids))),
                               tuple([1] * len(ids) + [0] * (capacity - len(ids))))

    def _embedding(self, text: str, capacity: int) -> LocalTokenSlots:
        # Official Embedding tokenizer postprocessor appends exactly one EOS.
        plain = self.embedding.encode(text, add_special_tokens=False).ids
        ids = self.embedding.encode(text, add_special_tokens=True).ids
        if ids != plain + [151643]:
            raise InputRejected("EMBEDDING_POSTPROCESSOR_MISMATCH")
        return self._slots(ids, capacity)

    def question(self, raw: bytes | str) -> QuestionInput:
        text = self._text(raw)
        body = self._slots(self._body_ids(text), 128)
        formatted = "Instruct: Given a question, retrieve relevant passages that provide evidence for answering it.\nQuery:" + text
        return QuestionInput(body, self._embedding(formatted, 160))

    def document_chunks(self, raw: bytes | str, authorization_binding: str) -> tuple[DocumentChunk, ...]:
        text = self._text(raw)
        if not authorization_binding:
            raise InputRejected("MISSING_AUTHORIZATION_BINDING")
        chunks: list[DocumentChunk] = []
        start = 0
        while start < len(text):
            encoded = self.generation.encode(text[start:], add_special_tokens=False)
            end = len(text) if len(encoded.ids) <= 256 else start + encoded.offsets[255][1]
            # Offsets are character boundaries, not byte/token decode fragments.
            while end > start:
                candidate = text[start:end]
                body_ids = self._body_ids(candidate)
                emb_ids = self.embedding.encode(candidate, add_special_tokens=True).ids
                if len(body_ids) <= 256 and len(emb_ids) <= 257:
                    break
                end -= 1
            if end == start:
                raise InputRejected("UNREPRESENTABLE_CHARACTER_BUDGET")
            candidate = text[start:end]
            chunks.append(DocumentChunk(candidate, start, end, authorization_binding,
                                        self._slots(body_ids, 256), self._embedding(candidate, 257)))
            if end == len(text):
                break
            offsets = self.generation.encode(candidate, add_special_tokens=False).offsets
            overlap_start = start + offsets[max(0, len(offsets) - 64)][0]
            overlap_start = max(start + 1, overlap_start)
            while overlap_start < end and len(self._body_ids(text[overlap_start:end])) > 64:
                overlap_start += 1
            start = overlap_start
        return tuple(chunks)


def public_prompt_layout(template: dict) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Public structure only. Private body slots stay zero-valid until R2 fills them."""
    tokens = [template["dummy_token_id"]] * 1024
    valid = [0] * 1024
    for part in template["segments"]:
        if part["kind"] == "public":
            offset = part["offset"]
            ids = part["token_ids"]
            tokens[offset:offset + len(ids)] = ids
            valid[offset:offset + len(ids)] = [1] * len(ids)
    return tuple(tokens), tuple(valid)
