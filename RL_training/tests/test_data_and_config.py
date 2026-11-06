import json
import os
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from hydra import compose, initialize_config_dir

from glasseek_training.prepare_data import convert_record, prepare
from glasseek_training.validation import validate_cm_config


def test_question_split_and_optional_fields(tmp_path):
    source = tmp_path / 'sft.jsonl'
    records = []
    for i in range(60):
        for attempt in range(2):
            row = {'messages': [{'role': 'user', 'content': f'question {i}'},
                                {'role': 'assistant', 'content': str(attempt)}]}
            if i == 4:
                row['enable_thinking'] = True
            records.append(row)
    source.write_text('\n'.join(json.dumps(r) for r in records))
    result = prepare(source, tmp_path / 'out', 'sft', val_ratio=.3)
    assert sum(result['counts'].values()) == 120
    tables = [pq.read_table(tmp_path / 'out' / f'{split}.parquet').to_pylist() for split in ('train', 'val')]
    groups = [{json.loads(r['messages'])[0]['content'] for r in rows} for rows in tables]
    assert not groups[0] & groups[1]
    assert sum(r['enable_thinking'] is True for rows in tables for r in rows) == 2


def test_rl_reference_not_in_policy_prompt():
    result, _ = convert_record({'question': 'Who?', 'answer': 'secret reference'}, 'rl')
    assert 'secret reference' not in json.dumps(result['prompt'])
    assert result['reward_model']['ground_truth'] == ['secret reference']


def test_incomplete_input_rejected(tmp_path):
    with pytest.raises(ValueError, match='part'):
        prepare(tmp_path / 'train.jsonl.part', tmp_path / 'out', 'sft')


@pytest.mark.parametrize('name', ['sft', 'cm_gspo'])
def test_compose_against_pinned_verl(name):
    root = os.environ.get('VERL_SOURCE')
    if not root:
        pytest.skip('Set VERL_SOURCE for upstream configuration compatibility tests')
    configs = Path(__file__).resolve().parents[1] / 'configs'
    with initialize_config_dir(config_dir=str(configs), version_base=None):
        cfg = compose(config_name=name, overrides=[f'hydra.searchpath=[file://{root}/verl/trainer/config]'])
    if name == 'sft':
        assert cfg.trainer.total_epochs == 3
        assert cfg.data.train_batch_size == 64
        assert cfg.data.max_length == 43008
        assert cfg.data.max_token_len_per_gpu == 43008
        assert cfg.model.lora_rank == 16
        assert cfg.engine.ulysses_sequence_parallel_size == 1
        assert cfg.optim.lr * cfg.optim.min_lr_ratio == pytest.approx(2e-6)
    else:
        validate_cm_config(cfg)
        assert cfg.trainer.total_training_steps == 100
        assert cfg.data.train_batch_size == 16
        assert cfg.actor_rollout_ref.rollout.n == 8
        cfg.actor_rollout_ref.actor.ppo_epochs = 2
        with pytest.raises(ValueError, match='one optimizer step'):
            validate_cm_config(cfg)


def test_sft_tool_dicts_roundtrip_without_arrow_null_keys(tmp_path):
    records = []
    for i in range(40):
        records.append({'messages': [
            {'role': 'user', 'content': f'question {i}'},
            {'role': 'assistant', 'content': None, 'tool_calls': [
                {'function': {'name': 'search', 'arguments': {'query': 'q'}}},
                {'function': {'name': 'reset', 'arguments': {}}}]}],
            'tools': [{'type': 'function', 'function': {'name': 'reset', 'parameters': {
                'type': 'object', 'properties': {}}}}]})
    source = tmp_path / 'tools.jsonl'
    source.write_text('\n'.join(json.dumps(r) for r in records))
    prepare(source, tmp_path / 'out', 'sft', val_ratio=.3)
    rows = pq.read_table(tmp_path / 'out/train.parquet').to_pylist()
    messages = json.loads(rows[0]['messages'])
    assert messages[1]['tool_calls'][1]['function']['arguments'] == {}
    assert json.loads(rows[0]['tools'])[0]['function']['parameters']['properties'] == {}


def test_sft_dataset_normalizes_missing_tool_values(tmp_path, monkeypatch):
    import pandas as pd
    from glasseek_training import sft_dataset

    path = tmp_path / 'data.parquet'
    tool = {'type': 'function', 'function': {'name': 'search'}}
    pd.DataFrame({'messages': ['[]', '[]'], 'tools': [None, json.dumps([tool])]}).to_parquet(path)
    ds = object.__new__(sft_dataset.GlasseekSFTDataset)
    ds.parquet_files, ds.messages_key, ds.tools_key = [str(path)], 'messages', 'tools'
    ds.enable_thinking_key, ds.enable_thinking_default = 'enable_thinking', None
    ds.max_samples, ds.seed, ds.shuffle = -1, 42, False
    ds.tokenizer, ds.apply_chat_template_kwargs = None, {}
    monkeypatch.setattr(sft_dataset, 'extract_system_prompt_and_generation', lambda *a, **kw: ([], []))
    ds._read_files_and_process()
    assert ds.tools == [None, [tool]]


def test_system_only_template_fallback_preserves_context_and_masks_loss():
    import ast
    import torch

    source = Path(__file__).resolve().parents[1] / 'glasseek_training/sft_dataset.py'
    tree = ast.parse(source.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'GlasseekSFTDataset')
    node.bases = [ast.Name(id='object', ctx=ast.Load())]
    ast.fix_missing_locations(node)
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            if len(messages) == 1 and messages[0]['role'] == 'system':
                raise ValueError('Qwen requires a user message')
            ids = [1, 2, 3, 4] if messages[0]['role'] == 'system' else [3, 4]
            return {'input_ids': torch.tensor([ids]), 'attention_mask': torch.ones(1, len(ids), dtype=torch.long)}
    ds = namespace['GlasseekSFTDataset']()
    ds.processor, ds.tokenizer, ds.apply_chat_template_kwargs = None, Tokenizer(), {}
    ids, loss_mask, attention, _ = ds._process_single_message(0, {'role':'system','content':'x'}, [])
    assert ids.tolist() == [1, 2]
    assert loss_mask.tolist() == [0, 0]
    assert attention.tolist() == [1, 1]
