import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def notebook_namespace(monkeypatch, tmp_path, fail=False):
    notebook = json.loads(Path('notebooks/rand_seed_test.ipynb').read_text())
    calls = []

    class Engine:
        def __init__(self, **kwargs):
            calls.append(('init', kwargs))

        def generate(self, **kwargs):
            calls.append(('generate', kwargs))
            if fail:
                raise RuntimeError('GPU failure')
            return {'output_ids': [7, 8]}

        def shutdown(self):
            calls.append(('shutdown',))

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            calls.append(('prompt', messages))
            return 'chat:' + messages[0]['content']

        def encode(self, text, **kwargs):
            return [9] if text == '</think>' else [1, 2]

        def decode(self, ids, **kwargs):
            return 'generated answer'

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(path, **kwargs):
            assert kwargs['local_files_only']
            return Tokenizer()

    monkeypatch.setitem(sys.modules, 'sglang', SimpleNamespace(Engine=Engine))
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(AutoTokenizer=AutoTokenizer))
    ns = {'TEMPERATURE': 1.0, 'MAX_NEW_TOKENS': 64}
    exec(''.join(notebook['cells'][3]['source']), ns)
    ns['MODEL_PATH'] = str(tmp_path)
    return ns, calls


def test_seed_reaches_fresh_engine_and_prompt_reaches_tokenizer(monkeypatch, tmp_path):
    ns, calls = notebook_namespace(monkeypatch, tmp_path)
    assert ns['sample_once']('test question', 42) == 'generated answer'
    assert ns['sample_once']('test question', 43) == 'generated answer'
    assert [v[1]['random_seed'] for v in calls if v[0] == 'init'] == [42, 43]
    assert [v for v in calls if v[0] == 'shutdown'] == [('shutdown',), ('shutdown',)]
    assert next(v for v in calls if v[0] == 'prompt')[1][0]['content'] == 'test question'
    assert all('seed' not in v[1]['sampling_params'] for v in calls if v[0] == 'generate')


def test_runtime_error_stops_and_closes_engine(monkeypatch, tmp_path):
    ns, calls = notebook_namespace(monkeypatch, tmp_path, fail=True)
    with pytest.raises(RuntimeError, match='GPU failure'):
        ns['sample_once']('test question', 42)
    assert calls[-1] == ('shutdown',)
