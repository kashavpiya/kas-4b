import json
import os
import string

from decision_index.engines.base import Engine, Unsupported

SYSTEM = (
    "You are a decision engine. Read the JSON object carefully, "
    "then output exactly one option key from the options field. "
    "Output only the key, nothing else."
)


def _build_payload(state, q_type, q_instructions, q_criteria, labels):
    """Build the JSON payload dict and return (payload, original_keys).

    For choice questions options maps label → description.
    For noul the two labels map to "No" and "Yes".
    Returns the original option keys in the same order as labels so the
    caller can map probabilities back.
    """
    if q_type == "choice":
        keys = list(q_criteria)
        options = {}
        for label, key in zip(labels, keys):
            desc = q_criteria[key]
            options[label] = desc if desc is not None else key
    else:
        keys = ["false", "true"]
        options = {labels[0]: "No", labels[1]: "Yes"}

    payload = {}
    if state not in ("", None, {}, []):
        payload["state"] = state
    payload["question"] = q_instructions
    payload["options"] = options
    return payload, keys


class KasEngine(Engine):
    """Decision engine for kas0/kas-4b.

    Differences from the stock TransformersEngine:
    - Options are mapped to single-token uppercase labels (A–Z then two-letter
      combos) so the answer is always a single token.
    - The JSON prompt is optionally sent twice separated by "Let me repeat that:"
      — Scion measured +1.3 pts at 4B from this.
    - Probabilities come from the label-token logits at the generation position
      (one forward pass), divided by a temperature before softmax. Full logits
      over all label tokens, not a top-k approximation.
    - Thinking mode is disabled for Qwen3.5.
    """

    name = "kas"
    latency = (
        "Device-synchronized in-process request wall time including prompt "
        "construction and state-prefix KV cache reuse; excludes model loading."
    )

    def __init__(
        self,
        model="kpiya/kas-4b",  # Qwen3-4B base, LoRA r64
        revision=None,
        device=None,
        dtype=None,
        temperature=1.0,
        repeat_prompt=True,
        attn="sdpa",
        max_tokens=None,
        **options,
    ):
        super().__init__(**options)
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.temperature = float(temperature)
        self.repeat_prompt = str(repeat_prompt).lower() not in ("false", "0", "no")
        self.model_id = model

        self.device = device or (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        if dtype is None:
            dtype = "bfloat16" if self.device == "cuda" else "float32"
        self.dtype = getattr(torch, dtype)

        self.tok = AutoTokenizer.from_pretrained(model, revision=revision)
        self.mdl = (
            AutoModelForCausalLM.from_pretrained(
                model, revision=revision, torch_dtype=self.dtype,
                attn_implementation=attn,
            )
            .to(self.device)
            .eval()
        )

        self._has_template = bool(getattr(self.tok, "chat_template", None))
        raw_limit = (
            getattr(self.mdl.config, "max_position_embeddings", None)
            or getattr(self.tok, "model_max_length", None)
        )
        self.limit = min(raw_limit, int(max_tokens)) if max_tokens else raw_limit

        self._labels = self._build_labels()
        # token id for each label string — used to read logits at generation pos
        self._label_tok_ids = [
            self.tok(lbl, add_special_tokens=False)["input_ids"][0]
            for lbl in self._labels
        ]

        self._kv = None        # DynamicCache; None until first forward
        self._kv_ids: list = []  # full prompt ids whose KV is currently cached

        self.provenance = {
            "kind": "kas-engine",
            "repo": model,
            "revision": revision or getattr(self.mdl.config, "_commit_hash", None),
            "device": self.device,
            "dtype": dtype,
            "temperature": self.temperature,
            "repeat_prompt": self.repeat_prompt,
            "n_labels": len(self._labels),
            "context_limit": self.limit,
            "policy": (
                "Options mapped to single-token uppercase labels (A–Z then "
                "two-letter). JSON prompt optionally repeated once. Label logits "
                "at the generation position divided by temperature, softmaxed to "
                "probabilities. Over-context prompts refused, never truncated."
            ),
        }

    # ------------------------------------------------------------------
    # Label vocabulary
    # ------------------------------------------------------------------

    def _build_labels(self) -> list:
        """Return the first 255 single-token uppercase labels for this tokenizer."""
        tok = self.tok
        labels = []
        for c in string.ascii_uppercase:
            if len(tok(c, add_special_tokens=False)["input_ids"]) == 1:
                labels.append(c)
        for a in string.ascii_uppercase:
            for b in string.ascii_uppercase:
                if len(labels) >= 255:
                    break
                combo = a + b
                if len(tok(combo, add_special_tokens=False)["input_ids"]) == 1:
                    labels.append(combo)
            if len(labels) >= 255:
                break
        if len(labels) < 255:
            raise RuntimeError(
                f"Tokenizer yields only {len(labels)} single-token labels; "
                "need 255. Extend _build_labels for this tokenizer."
            )
        return labels[:255]

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _prompt_text(self, state, q_type, q_instructions, q_criteria, n_opts) -> str:
        labels = self._labels[:n_opts]
        payload, _ = _build_payload(state, q_type, q_instructions, q_criteria, labels)
        body = json.dumps(payload, ensure_ascii=False, separators=(", ", ": "))
        if self.repeat_prompt:
            body = body + "\n\nLet me repeat that:\n\n" + body
        if self._has_template:
            msgs = [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": body},
            ]
            try:
                return self.tok.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                # tokenizer doesn't accept enable_thinking
                return self.tok.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True,
                )
        return SYSTEM + "\n\n" + body + "\nAnswer:"

    def _tokenize(self, state, question) -> tuple:
        """Return (token_ids, n_options) for one question."""
        n = len(question["criteria"]) if question["type"] == "choice" else 2
        text = self._prompt_text(
            state,
            question["type"],
            question.get("instructions", ""),
            question.get("criteria", {}),
            n,
        )
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        return ids, n

    # ------------------------------------------------------------------
    # KV-cache prefix reuse
    # ------------------------------------------------------------------

    def _prefix_len(self, ids: list) -> int:
        """Length of shared prefix between ids and the currently cached sequence.
        Always leaves at least 1 token in the new portion.
        """
        cached = self._kv_ids
        limit = min(len(ids) - 1, len(cached))
        p = 0
        while p < limit and ids[p] == cached[p]:
            p += 1
        return p

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def _forward(self, ids: list, p: int):
        """Run one forward pass starting from cached prefix at length p.
        Returns float32 logits tensor of shape [vocab_size] at the last position.
        Updates self._kv in-place.
        """
        torch = self.torch
        from transformers import DynamicCache

        if p == 0 or self._kv is None:
            self._kv = DynamicCache()
        else:
            self._kv.crop(p)

        new_ids = ids[p:]
        input_ids = torch.tensor([new_ids], device=self.device)
        pos = torch.arange(p, p + len(new_ids), device=self.device).unsqueeze(0)

        with torch.inference_mode():
            out = self.mdl(
                input_ids=input_ids,
                position_ids=pos,
                past_key_values=self._kv,
                use_cache=True,
            )
        return out.logits[0, -1].float()  # [vocab_size]

    # ------------------------------------------------------------------
    # Main call
    # ------------------------------------------------------------------

    def __call__(self, state, questions):
        torch = self.torch
        answers, raw = {}, {}

        for k, q in questions.items():
            if q["type"] not in ("choice", "noul"):
                raise Unsupported("unsupported question type: " + str(q["type"]))

            ids, n_opts = self._tokenize(state, q)

            if len(ids) > self.limit:
                raise Unsupported(
                    f"prompt ({len(ids)} tokens) exceeds context limit ({self.limit})"
                )

            keys = list(q["criteria"]) if q["type"] == "choice" else ["false", "true"]
            p = self._prefix_len(ids) if self._kv is not None else 0

            logits_full = self._forward(ids, p)
            self._kv_ids = ids  # update cached prefix tracking

            # Extract the logit for each label token, apply temperature, softmax
            label_tok_ids = self._label_tok_ids[:n_opts]
            raw_logits = torch.tensor(
                [logits_full[tid].item() for tid in label_tok_ids],
                dtype=torch.float32,
            )
            if self.temperature != 1.0:
                raw_logits = raw_logits / self.temperature

            probs_t = torch.softmax(raw_logits, dim=0)
            probs = {key: probs_t[i].item() for i, key in enumerate(keys)}
            choice = max(keys, key=lambda x: probs[x])

            raw[k] = {
                "label_logits": dict(zip(keys, raw_logits.tolist())),
                "prompt_tokens": len(ids),
                "cached_prefix_tokens": p,
            }

            if q["type"] == "choice":
                answers[k] = {"type": "choice", "choice": choice, "probabilities": probs}
            else:
                answers[k] = {"type": "noul", "noul": probs["true"]}

        return {
            "model": self.model_id,
            "answers": answers,
            "usage": {"input_tokens": sum(v["prompt_tokens"] for v in raw.values())},
        }, raw

    # ------------------------------------------------------------------
    # Engine protocol helpers
    # ------------------------------------------------------------------

    def synchronize(self):
        torch = self.torch
        if self.device == "cuda":
            torch.cuda.synchronize()
        elif self.device == "mps":
            torch.mps.synchronize()

    def runtime(self):
        torch = self.torch
        info = {"torch": torch.__version__, "device": self.device}
        if self.device == "cuda":
            info.update(cuda=torch.version.cuda, gpu=torch.cuda.get_device_name())
        try:
            import transformers
            info["transformers"] = transformers.__version__
        except Exception:
            pass
        return info

    def close(self):
        if hasattr(self, "mdl"):
            del self.mdl
        if hasattr(self, "tok"):
            del self.tok
