"""Run only the pinned author's pure scoring helpers, without GPU imports."""
import ast
from functools import lru_cache
from pathlib import Path
import re
import subprocess
import warnings

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / 'latent_grpo_minfix/Latent-GRPO-final'
PINNED_COMMIT = '0b7e85f15e9859033653964282348e518f9f6291'
COMMON = ('fix_fracs', 'fix_a_slash_b', 'remove_right_units', 'fix_sqrt', 'strip_string',
          'remove_boxed', 'last_boxed_only_string')
NAMES = {'low': COMMON + ('extract_answer', 'normalize_answer_text', '_as_float', 'check_is_correct'),
         'high': COMMON + ('is_equiv', 'compute_score')}


@lru_cache(maxsize=4)
def _compile(profile, code, path):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', SyntaxWarning)  # pinned author's \\% string literal
        tree = ast.parse(code, filename=path)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    missing = set(NAMES[profile]) - functions.keys()
    if missing:
        raise ValueError(f'missing required author helpers: {sorted(missing)}')
    namespace = {'re': re}
    selected = ast.Module(body=[functions[name] for name in NAMES[profile]], type_ignores=[])
    exec(compile(selected, path, 'exec'), namespace)
    return namespace


def load_author_helpers(profile, path=None):
    if profile not in NAMES:
        raise ValueError('unknown scoring profile')
    if path is None:
        relative = f'Latent-GRPO/eval/eval_{profile}_tasks_sglang.py'
        path = UPSTREAM / relative
        if not path.is_file():
            raise FileNotFoundError(f'missing required reference: {path}')
        head = subprocess.check_output(['git', '-C', str(UPSTREAM), 'rev-parse', 'HEAD'], text=True).strip()
        if head != PINNED_COMMIT:
            raise ValueError('author reference HEAD differs from pinned commit')
        changed = subprocess.run(['git', '-C', str(UPSTREAM), 'diff', '--quiet', 'HEAD', '--', relative])
        if changed.returncode:
            raise ValueError('author scoring reference differs from pinned commit')
    path = Path(path)
    return _compile(profile, path.read_text(encoding='utf-8'), str(path))


def verify_answer(source, text, ground_truth, finish_reason=None):
    profiles = {'gsm8k_aug': 'low', 'gsm8k_aug_test': 'low', 'dapo_math': 'high', 'math500_test': 'high'}
    if source not in profiles or not isinstance(text, str) or not isinstance(ground_truth, str) or not ground_truth.strip():
        raise ValueError('invalid source, generated text or ground truth')
    profile = profiles[source]
    helpers = load_author_helpers(profile)
    answer = ''
    if text.strip():
        if profile == 'low':
            answer = helpers['extract_answer'](text)
        else:
            boxed = helpers['last_boxed_only_string'](text)
            if boxed is not None:
                try:
                    answer = helpers['remove_boxed'](boxed)
                except AssertionError:
                    answer = ''  # malformed model answer, not a scorer execution failure
    invalid = 'empty_output' if not text.strip() else ('answer_unextractable' if not answer.strip() else None)
    reward = 0
    if invalid is None:
        reward = int(helpers['check_is_correct'](answer, ground_truth) if profile == 'low'
                     else helpers['is_equiv'](answer, ground_truth))
    return {'reward': reward, 'invalid_reason': invalid,
            'truncated': finish_reason in ('length', 'max_tokens') if finish_reason is not None else None,
            'finish_reason': finish_reason, 'extracted_answer': answer}
