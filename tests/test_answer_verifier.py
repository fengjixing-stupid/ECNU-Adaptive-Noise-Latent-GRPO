from pathlib import Path
import pytest
from adaptive_noise.answer_verifier import verify_answer, load_author_helpers


@pytest.mark.parametrize('source,text,answer,reward,invalid', [
    ('gsm8k_aug', r'Answer: \boxed{42}', '42', 1, None),
    ('gsm8k_aug', '#### 42', '42', 1, None),
    ('gsm8k_aug', 'The answer is 42', '42', 1, None),
    ('gsm8k_aug', r'\boxed{43}', '42', 0, None),
    ('gsm8k_aug', '', '42', 0, 'empty_output'),
    ('gsm8k_aug', 'No answer', '42', 0, 'answer_unextractable'),
    ('dapo_math', r'\boxed{34}', '34', 1, None),
    ('dapo_math', '34', '34', 0, 'answer_unextractable'),
    ('math500_test', r'\boxed{\frac{1}{2}}', '0.5', 1, None),
    ('math500_test', r'\boxed{}', '42', 0, 'answer_unextractable'),
])
def test_scoring_and_independent_invalid_classification(source, text, answer, reward, invalid):
    result = verify_answer(source, text, answer)
    assert result['reward'] == reward and result['invalid_reason'] == invalid
    helpers = load_author_helpers('low' if source == 'gsm8k_aug' else 'high')
    expected = int(helpers['check_is_correct'](helpers['extract_answer'](text), answer)) if source == 'gsm8k_aug' else int(helpers['compute_score'](text, answer))
    assert result['reward'] == expected


def test_truncated_correct_answer_still_correct():
    result = verify_answer('dapo_math', r'\boxed{34}', '34', finish_reason='length')
    assert result['reward'] == 1 and result['truncated'] is True


def test_scoring_contract_errors_do_not_turn_into_reward_zero(tmp_path):
    with pytest.raises(ValueError): verify_answer('unknown', '42', '42')
    with pytest.raises(ValueError): verify_answer('gsm8k_aug', '42', '')
    with pytest.raises(FileNotFoundError): load_author_helpers('low', tmp_path / 'missing.py')
    broken = tmp_path / 'broken.py'
    broken.write_text('def extract_answer(text):\n    return text\n')
    with pytest.raises(ValueError): load_author_helpers('low', broken)
