from __future__ import annotations

import hashlib
import random
import re
import time

import torch
import torch.nn.functional as F

from .geometry import clone_state
from .utils import seed_all


class TinyTokenizer:
    """Offline test tokenizer; never used for research results."""
    pad_token_id, eos_token_id = 0, 1

    def encode(self, text, add_special_tokens=False):
        labels = {"negative": 2, "positive": 3}
        return [labels[word] if word in labels else 4 + int(hashlib.sha256(word.encode()).hexdigest()[:8], 16) % 124
                for word in text.split()]


class Backend:
    def __init__(self, cfg, resolved_revision=None):
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

        self.cfg = cfg
        runtime, model_cfg = cfg["runtime"], cfg["model"]
        torch.set_num_threads(runtime["threads"])
        seed_all(cfg["seed"])
        if runtime["deterministic"]:
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
        device = runtime["device"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        if model_cfg["backend"] == "tiny_qwen":
            self.tokenizer = TinyTokenizer()
            self.revision = "random-tiny-qwen2-v1"
            config = Qwen2Config(vocab_size=128, hidden_size=32, intermediate_size=64,
                                num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                                max_position_embeddings=512, attention_dropout=0.0,
                                bos_token_id=1, eos_token_id=1, pad_token_id=0, tie_word_embeddings=True)
            config._attn_implementation = "eager"
            base = Qwen2ForCausalLM(config)
        else:
            from huggingface_hub import HfApi
            requested = model_cfg["revision"]
            self.revision = resolved_revision or (requested if re.fullmatch(r"[0-9a-f]{40}", requested)
                                                  else HfApi().model_info(model_cfg["name"], revision=requested).sha)
            self.tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"], revision=self.revision)
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            dtype = getattr(torch, model_cfg["dtype"]) if self.device.type == "cuda" else torch.float32
            base = AutoModelForCausalLM.from_pretrained(model_cfg["name"], revision=self.revision,
                                                       torch_dtype=dtype, attn_implementation="eager")
        base.config.use_cache = False
        self.model = get_peft_model(base, LoraConfig(r=model_cfg["rank"], lora_alpha=model_cfg["alpha"],
                                                   lora_dropout=model_cfg["dropout"], target_modules=model_cfg["targets"],
                                                   bias="none", task_type="CAUSAL_LM")).to(self.device)
        self.scale = model_cfg["alpha"] / model_cfg["rank"]
        self.answers = [self.tokenizer.encode(" negative", add_special_tokens=False) + [self.tokenizer.eos_token_id],
                        self.tokenizer.encode(" positive", add_special_tokens=False) + [self.tokenizer.eos_token_id]]
        if self.answers[0] == self.answers[1]:
            raise RuntimeError("Label tokenization is not distinct")
        self.header = self.tokenizer.encode("Review: ", add_special_tokens=False)
        self.footer = self.tokenizer.encode("\nSentiment:", add_special_tokens=False)

    def state(self):
        from peft import get_peft_model_state_dict
        # The vocabulary/base weights are frozen. Explicitly excluding embeddings
        # also avoids PEFT's automatic remote base-config lookup on every snapshot.
        return clone_state(get_peft_model_state_dict(self.model, save_embedding_layers=False))

    def load(self, state):
        from peft import set_peft_model_state_dict
        current = self.state()
        if set(current) != set(state) or any(current[k].shape != state[k].shape for k in current):
            raise ValueError("Adapter keys or shapes do not match the model")
        if not all(torch.isfinite(t).all() for t in state.values()):
            raise ValueError("Non-finite adapter weights")
        set_peft_model_state_dict(self.model, state)
        self.model.zero_grad(set_to_none=True)

    def encode(self, row, label=None):
        answer = self.answers[row["label"] if label is None else label]
        room = self.cfg["model"]["max_length"] - len(self.header) - len(self.footer) - max(map(len, self.answers))
        if room < 1:
            raise ValueError("Sequence length leaves no room for review tokens")
        review = self.tokenizer.encode(row["text"], add_special_tokens=False)[:room]
        prefix = self.header + review + self.footer
        return prefix + answer, [-100] * len(prefix) + answer

    def batch(self, rows, label=None):
        encoded = [self.encode(row, label) for row in rows]
        length = max(len(x[0]) for x in encoded)
        tokens = torch.full((len(rows), length), self.tokenizer.pad_token_id, dtype=torch.long, device=self.device)
        labels = torch.full_like(tokens, -100)
        attention = torch.zeros_like(tokens)
        for i, (ids, targets) in enumerate(encoded):
            tokens[i, :len(ids)] = torch.tensor(ids, device=self.device)
            labels[i, :len(ids)] = torch.tensor(targets, device=self.device)
            attention[i, :len(ids)] = 1
        return {"input_ids": tokens, "attention_mask": attention}, labels

    def _losses(self, rows, label=None):
        inputs, labels = self.batch(rows, label)
        targets = labels[:, 1:]
        # Qwen still attends to the full sequence. Only the vocabulary projection
        # for positions with supervised next tokens is needed for this objective.
        positions = torch.nonzero((targets != -100).any(dim=0), as_tuple=True)[0]
        targets = targets.index_select(1, positions)
        logits = self.model(**inputs, logits_to_keep=positions).logits.float()
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
                               reduction="none", ignore_index=-100).reshape(targets.shape)
        return loss.sum(dim=1), (targets != -100).sum(dim=1)

    def train(self, state, rows, seed, max_steps=None):
        self.load(state)
        seed_all(seed)
        self.model.train()
        train = self.cfg["train"]
        optimizer = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad],
                                      lr=train["learning_rate"], weight_decay=train["weight_decay"])
        steps = 0
        total_loss = total_tokens = 0.0
        start = time.perf_counter()
        limit = max_steps if max_steps is not None else train["max_steps"]
        for epoch in range(train["local_epochs"]):
            shuffled = list(rows)
            random.Random(seed + epoch).shuffle(shuffled)
            window_size = train["batch_size"] * train["gradient_accumulation"]
            for position in range(0, len(shuffled), window_size):
                window = shuffled[position:position + window_size]
                token_count = sum(len(self.answers[row["label"]]) for row in window)
                optimizer.zero_grad(set_to_none=True)
                for j in range(0, len(window), train["batch_size"]):
                    losses, counts = self._losses(window[j:j + train["batch_size"]])
                    if not torch.isfinite(losses).all():
                        raise FloatingPointError("Non-finite training loss")
                    (losses.sum() / token_count).backward()
                    total_loss += float(losses.detach().sum())
                    total_tokens += int(counts.sum())
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), train["max_grad_norm"], error_if_nonfinite=True)
                optimizer.step()
                steps += 1
                if limit is not None and steps >= limit:
                    break
            if limit is not None and steps >= limit:
                break
        result = self.state()
        self.model.zero_grad(set_to_none=True)
        return result, {"loss": total_loss / max(total_tokens, 1), "optimizer_steps": steps,
                        "answer_tokens": int(total_tokens), "seconds": time.perf_counter() - start}

    @torch.no_grad()
    def evaluate(self, state, rows):
        if not rows:
            return {"n": 0, "accuracy": None, "loss": None}
        self.load(state)
        self.model.eval()
        correct = count = 0
        loss_sum = token_sum = 0.0
        for position in range(0, len(rows), self.cfg["train"]["batch_size"]):
            batch = rows[position:position + self.cfg["train"]["batch_size"]]
            scores = []
            for label in (0, 1):
                losses, _ = self._losses(batch, label)
                scores.append(losses)
            scores = torch.stack(scores, dim=1)
            prediction = scores.argmin(dim=1).cpu().tolist()
            correct += sum(p == row["label"] for p, row in zip(prediction, batch))
            truth = torch.tensor([row["label"] for row in batch], device=self.device)
            loss_sum += float(scores.gather(1, truth[:, None]).sum())
            token_sum += sum(len(self.answers[row["label"]]) for row in batch)
            count += len(batch)
        return {"n": count, "accuracy": correct / count, "loss": loss_sum / token_sum}

    def probe(self, state, validation):
        saved = self.state()
        try:
            return [self.evaluate(state, validation[domain])["loss"] for domain in sorted(validation)]
        finally:
            self.load(saved)

    @torch.no_grad()
    def attack_loss(self, state, rows):
        """Attacker-owned clean-label NLL; never accesses validation or test."""
        self.load(state)
        self.model.eval()
        numerator = denominator = 0.0
        for start in range(0, len(rows), self.cfg['train']['batch_size']):
            losses, counts = self._losses(rows[start:start + self.cfg['train']['batch_size']])
            numerator += float(losses.sum())
            denominator += float(counts.sum())
        value = numerator / max(denominator, 1)
        if not torch.isfinite(torch.tensor(value)):
            raise FloatingPointError('Non-finite attacker probe loss')
        return value
