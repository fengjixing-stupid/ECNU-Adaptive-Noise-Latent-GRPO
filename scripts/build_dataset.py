"""Construct the approved CPU candidate pool; never load the checkpoint."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/dataset_build.yaml')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    try:
        if not args.config.is_file():
            raise FileNotFoundError(f'missing required config: {args.config}')
        import yaml
        from adaptive_noise.dataset_builder import build_dataset
        config = yaml.safe_load(args.config.read_text(encoding='utf-8'))
        if args.output_dir:
            config['output_dir'] = str(args.output_dir)
        manifest = build_dataset(config, ROOT)
        print(json.dumps({'status': manifest['status'], 'counts': manifest['counts'],
                          'output': config['output_dir']}, ensure_ascii=False))
        return 0
    except (OSError, ValueError, KeyError, TypeError, ImportError) as error:
        print(f'STOP: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
