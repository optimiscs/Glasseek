"""Lossless JSON fields on disk; verl handles chat rendering and assistant masks."""
import json
import os
import numpy as np
import pandas as pd

from verl.utils.tokenizer.chat_template import extract_system_prompt_and_generation
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
from verl.utils.py_functional import convert_nested_value_to_list_recursive


class GlasseekSFTDataset(MultiTurnSFTDataset):
    def _verify_release(self, path):
        from glasseek_shared.trajectory_release import verify_release_file
        return verify_release_file(path)

    def _read_files_and_process(self):
        for path in self.parquet_files:
            self._verify_release(path)
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
        from glasseek_shared.curation.supervision import validate_plan,message_hash
        self.supervision_by_messages = {}
        for i,messages in enumerate(self.messages):
            plan=self.dataframe.iloc[i].get('supervision')
            if pd.api.types.is_scalar(plan) and pd.isna(plan): plan=None
            if isinstance(plan,str): plan=json.loads(plan)
            if plan is not None:
                validate_plan(plan,messages,require_learn=True)
            key=message_hash(messages)
            if key in self.supervision_by_messages and self.supervision_by_messages[key]!=plan:
                raise ValueError('Conflicting supervision for identical messages')
            self.supervision_by_messages[key]=plan
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
            transformed = [apply_to_messages(messages, mode) for messages in self.messages]
            for before,after in zip(self.messages,transformed):
                plan=self.supervision_by_messages.get(message_hash(before))
                if plan is not None:
                    if len(before)!=len(after) or [m['role'] for m in before]!=[m['role'] for m in after]:
                        raise ValueError('URL rendering changed supervision boundaries')
                    self.supervision_by_messages[message_hash(after)]=plan
            self.messages=transformed
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

    def _build_messages(self, example):
        import copy
        from glasseek_shared.curation.supervision import message_hash
        before=copy.deepcopy(example[self.messages_key])
        after=super()._build_messages(copy.deepcopy(example))
        bindings=getattr(self, 'supervision_by_messages', {})
        key=message_hash(before)
        if bindings:
            if key not in bindings:
                raise ValueError('Input messages lost supervision binding')
            # A text-only processor wraps strings in text blocks. Rebind only
            # that exact representation change; other content changes fail closed.
            comparable=copy.deepcopy(after)
            for message in comparable:
                content=message.get('content')
                if isinstance(content,list) and all(set(part)=={'type','text'} and part['type']=='text' for part in content):
                    message['content']=''.join(part['text'] for part in content)
            if comparable!=before:
                raise ValueError('Processor changed bound message content')
            bindings[message_hash(after)]=bindings[key]
        return after

    def _process_single_message(self, index, message, full_message, tools=None, enable_thinking=None):
        # Qwen3.5 refuses system-only input; verl's generic fallback prepends a
        # user turn and then rejects the system as no longer first. Append the
        # dummy user instead and remove its exact token suffix.
        if index != 0 or message.get('role') != 'system':
            ids,mask,attention,extra=super()._process_single_message(index, message, full_message, tools, enable_thinking)
            from glasseek_shared.curation.supervision import message_hash
            plan=getattr(self,'supervision_by_messages',{}).get(message_hash(full_message))
            if getattr(self,'supervision_by_messages',{}) and message_hash(full_message) not in self.supervision_by_messages:
                raise ValueError('Rendered messages lost supervision binding')
            if plan is not None and plan['messages'][index]['decision']!='learn':
                mask=mask*0
            return ids,mask,attention,extra
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


class ExperimentalPilotSFTDataset(GlasseekSFTDataset):
    """Opt-in pilot data; strict release loading remains the default."""
    def _verify_release(self, path):
        from glasseek_shared.experimental_release import verify
        return verify(path)
