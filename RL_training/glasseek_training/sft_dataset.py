"""Lossless JSON fields on disk; verl handles chat rendering and assistant masks."""
import json
import os
import numpy as np
import pandas as pd

from verl.utils.tokenizer.chat_template import extract_system_prompt_and_generation
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
from verl.utils.py_functional import convert_nested_value_to_list_recursive


class GlasseekSFTDataset(MultiTurnSFTDataset):
    def _read_files_and_process(self):
        from glasseek_shared.trajectory_release import verify_release_file
        for path in self.parquet_files:
            verify_release_file(path)
        self.dataframe = pd.concat([pd.read_parquet(path) for path in self.parquet_files], ignore_index=True)
        mode = os.environ.get('GLASSEEK_URL_MODE', 'raw')
        modes = set(self.dataframe['url_mode'].fillna('raw')) if 'url_mode' in self.dataframe else {'raw'}
        if modes != {mode}:
            raise ValueError(f'SFT URL protocol mismatch: dataset={modes}, --url-mode={mode}')
        if mode != 'raw':
            from glasseek_shared.hermes.url_registry import VERSION
            versions = set(self.dataframe['url_protocol']) if 'url_protocol' in self.dataframe else set()
            if versions != {VERSION}:
                raise ValueError(f'SFT URL protocol version mismatch: {versions}; regenerate data using {VERSION}')
        if 0 < self.max_samples < len(self.dataframe):
            indices = (np.random.default_rng(self.seed).choice(len(self.dataframe), self.max_samples, replace=False)
                       if self.shuffle else np.arange(self.max_samples))
            self.dataframe = self.dataframe.iloc[indices].reset_index(drop=True)
        for key in (self.messages_key, self.tools_key):
            if key in self.dataframe:
                self.dataframe[key] = self.dataframe[key].map(
                    lambda value: json.loads(value) if isinstance(value, str)
                    else convert_nested_value_to_list_recursive(value))
        self.messages = self.dataframe[self.messages_key].tolist()
        from glasseek_shared.context.config import mode as context_mode
        if context_mode() == 'clm':
            from glasseek_shared.context.official.completion import final_text
            for row in self.messages:
                calls = [c for m in row for c in (m.get('tool_calls') or [])]
                if not any((c.get('function') or c).get('name') == 'bash' for c in calls):
                    raise ValueError('Official CLM SFT requires canonical bash action trajectories; note summaries are not CLM labels')
                if not row or row[-1].get('role')!='assistant' or not final_text(row[-1],row[-1].get('finish_reason')):
                    raise ValueError('CLM SFT requires a final assistant answer without tool calls; legacy file/echo submissions must be regenerated')
        if mode != 'raw':
            from glasseek_shared.hermes.url_prompt import apply_to_messages
            self.messages = [apply_to_messages(messages, mode) for messages in self.messages]
        self.tools = (
            [None if pd.api.types.is_scalar(value) and pd.isna(value) else value
             for value in self.dataframe[self.tools_key]]
            if self.tools_key in self.dataframe else None)
        self.enable_thinking = (
            [self.enable_thinking_default if pd.isna(value) else bool(value)
             for value in self.dataframe[self.enable_thinking_key]]
            if self.enable_thinking_key in self.dataframe else None)
        self.system_prompt, self.generation_prompt = extract_system_prompt_and_generation(
            self.tokenizer, **self.apply_chat_template_kwargs)

    def _process_single_message(self, index, message, full_message, tools=None, enable_thinking=None):
        # Qwen3.5 refuses system-only input; verl's generic fallback prepends a
        # user turn and then rejects the system as no longer first. Append the
        # dummy user instead and remove its exact token suffix.
        if index != 0 or message.get('role') != 'system':
            return super()._process_single_message(index, message, full_message, tools, enable_thinking)
        import torch

        processor = self.processor if self.processor is not None else self.tokenizer
        opts = dict(self.apply_chat_template_kwargs)
        if enable_thinking is not None:
            opts['enable_thinking'] = enable_thinking
        dummy = {'role': 'user', 'content': ''}
        kwargs = dict(add_generation_prompt=False, tokenize=True, return_dict=True, return_tensors='pt', **opts)
        try:
            inputs = dict(processor.apply_chat_template([message], tools=tools, **kwargs))
        except Exception:
            inputs = dict(processor.apply_chat_template([message, dummy], tools=tools, **kwargs))
            suffix = dict(processor.apply_chat_template([dummy], tools=None, **kwargs))['input_ids'][0]
            ids = inputs['input_ids'][0]
            if len(suffix) > len(ids) or not torch.equal(ids[-len(suffix):], suffix):
                raise ValueError('Cannot safely isolate system message tokens in chat template')
            for key in ('input_ids', 'attention_mask', 'mm_token_type_ids'):
                if key in inputs:
                    inputs[key] = inputs[key][..., :-len(suffix)]
        ids = inputs.pop('input_ids')[0]
        attention = inputs.pop('attention_mask')[0]
        return ids, torch.zeros_like(attention), attention, inputs
