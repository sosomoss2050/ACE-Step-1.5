"""LLM inference engine: public API for text generation with CFG support."""

import os
import atexit
import threading
from dataclasses import dataclass
from time import perf_counter

import torch
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoTokenizer

from acestep.customized_vllm.sampling import SamplingParams
from acestep.customized_vllm.sequence import Sequence, SequenceStatus, BlockAllocator
from acestep.customized_vllm.runner import ModelRunner


@dataclass
class EngineConfig:
    """Configuration for the LLM engine."""
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    enforce_eager: bool = False
    kvcache_block_size: int = 256

    def __post_init__(self):
        assert os.path.isdir(self.model)
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len


class LLM:
    """High-level LLM inference engine.

    Provides generate() for text generation with optional classifier-free guidance,
    constrained decoding, and efficient paged KV cache.
    """

    def __init__(self, model, **kwargs):
        cfg = EngineConfig(
            model=model,
            max_num_batched_tokens=kwargs.get("max_num_batched_tokens", 16384),
            max_num_seqs=kwargs.get("max_num_seqs", 512),
            max_model_len=kwargs.get("max_model_len", 4096),
            gpu_memory_utilization=kwargs.get("gpu_memory_utilization", 0.9),
            enforce_eager=kwargs.get("enforce_eager", False),
            kvcache_block_size=kwargs.get("kvcache_block_size", 256),
        )
        self._cfg = cfg
        self._lock = threading.Lock()
        self.runner = ModelRunner(
            hf_config=cfg.hf_config, model_path=cfg.model,
            block_size=cfg.kvcache_block_size, max_num_seqs=cfg.max_num_seqs,
            max_num_batched_tokens=cfg.max_num_batched_tokens,
            max_model_len=cfg.max_model_len,
            gpu_memory_utilization=cfg.gpu_memory_utilization,
            enforce_eager=cfg.enforce_eager,
        )
        tok = kwargs.get("tokenizer", None)
        self.tokenizer = tok if tok is not None else AutoTokenizer.from_pretrained(model, use_fast=True)
        self._eos = self.tokenizer.eos_token_id
        self._allocator = BlockAllocator(self.runner.num_kvcache_blocks, cfg.kvcache_block_size)
        self._running: list[Sequence] = []
        atexit.register(self.exit)

    def exit(self):
        self.runner.exit()

    def reset(self):
        """Release all KV cache blocks (call on error to prevent block leaks)."""
        for seq in self._running:
            if seq.block_table:
                self._allocator.deallocate(seq)
        self._running.clear()

    def generate(self, prompts, sampling_params, use_tqdm=True, unconditional_prompts=None):
        """Generate completions for a batch of prompts.

        Args:
            prompts: List of prompt strings or token-ID lists.
            sampling_params: SamplingParams or list thereof.
            unconditional_prompts: Optional list of unconditional prompts for CFG.

        Returns:
            List of dicts with "text" and "token_ids" keys.
        """
        with self._lock:
            return self._generate(prompts, sampling_params, use_tqdm, unconditional_prompts)

    def _generate(self, prompts, sampling_params, use_tqdm, unconditional_prompts):
        if self._running:
            self.reset()

        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        if unconditional_prompts is None:
            unconditional_prompts = [None] * len(prompts)

        all_seqs = []
        for prompt, sp, uncond in zip(prompts, sampling_params, unconditional_prompts):
            ids = self.tokenizer.encode(prompt) if isinstance(prompt, str) else prompt
            cond = Sequence(ids, sp)
            if sp.cfg_scale > 1.0:
                u_ids = (self.tokenizer.encode(uncond) if isinstance(uncond, str)
                         else (uncond if uncond is not None else ids))
                uncond_seq = Sequence(u_ids, sp, is_unconditional=True)
                cond.paired_seq = uncond_seq
                uncond_seq.paired_seq = cond
                all_seqs.extend([cond, uncond_seq])
            else:
                all_seqs.append(cond)

        # Allocate KV cache blocks
        total_blocks = sum(s.num_blocks for s in all_seqs)
        if not self._allocator.can_allocate(total_blocks):
            raise RuntimeError(
                f"Insufficient KV cache: need {total_blocks} blocks, "
                f"have {len(self._allocator.free_ids)}/{self._allocator.total}"
            )
        for seq in all_seqs:
            self._allocator.allocate(seq)
            seq.status = SequenceStatus.RUNNING
        self._running = list(all_seqs)

        ordered = self._order_cfg_batch(all_seqs)
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True) if use_tqdm else None
        prefill_tps = decode_tps = 0.0
        outputs = {}

        try:
            # Prefill
            t = perf_counter()
            token_ids = self.runner.run(ordered, is_prefill=True)
            ntok = sum(len(s) for s in ordered)
            prefill_tps = ntok / (perf_counter() - t)
            self._postprocess(ordered, token_ids, outputs, pbar)
            if pbar:
                pbar.set_postfix(Prefill=f"{int(prefill_tps)}tok/s")

            # Decode loop
            while self._running:
                for seq in self._running:
                    self._allocator.may_append(seq)
                batch = self._order_cfg_batch(self._running)
                t = perf_counter()
                token_ids = self.runner.run(batch, is_prefill=False)
                n_cond = sum(1 for s in batch if not s.is_unconditional)
                decode_tps = n_cond / max(perf_counter() - t, 1e-9)
                self._postprocess(batch, token_ids, outputs, pbar)
                if pbar:
                    pbar.set_postfix(Prefill=f"{int(prefill_tps)}tok/s",
                                     Decode=f"{int(decode_tps)}tok/s")
        except Exception:
            self.reset()
            raise
        finally:
            if pbar:
                pbar.close()

        result = [outputs[sid] for sid in sorted(outputs)]
        return [{"text": self.tokenizer.decode(tids), "token_ids": tids} for tids in result]

    def _order_cfg_batch(self, seqs):
        """Order sequences: non-CFG, then CFG conditional, then CFG unconditional."""
        normal = [s for s in seqs if s.cfg_scale <= 1.0]
        cond = [s for s in seqs if s.cfg_scale > 1.0 and not s.is_unconditional]
        uncond = [s for s in seqs if s.is_unconditional]
        return normal + cond + uncond

    def _postprocess(self, seqs, token_ids, outputs, pbar):
        """Append tokens, check EOS, collect finished outputs."""
        is_cfg = (len(seqs) > 0 and seqs[0].cfg_scale > 1.0
                  and seqs[0].paired_seq is not None)
        if is_cfg:
            nc = len(seqs) // 2
            for cond, uncond, tid in zip(seqs[:nc], seqs[nc:], token_ids):
                cond.append_token(tid)
                uncond.append_token(tid)
                done = ((not cond.ignore_eos and tid == self._eos) or
                        cond.num_tokens - cond.num_prompt_tokens >= cond.max_tokens)
                if done:
                    for s in (cond, uncond):
                        s.status = SequenceStatus.FINISHED
                        self._allocator.deallocate(s)
                        if s in self._running:
                            self._running.remove(s)
                    outputs[cond.seq_id] = cond.completion_token_ids
                    if pbar:
                        pbar.update(1)
        else:
            for seq, tid in zip(seqs, token_ids):
                seq.append_token(tid)
                done = ((not seq.ignore_eos and tid == self._eos) or
                        seq.num_tokens - seq.num_prompt_tokens >= seq.max_tokens)
                if done:
                    seq.status = SequenceStatus.FINISHED
                    self._allocator.deallocate(seq)
                    if seq in self._running:
                        self._running.remove(seq)
                    outputs[seq.seq_id] = seq.completion_token_ids
                    if pbar:
                        pbar.update(1)
