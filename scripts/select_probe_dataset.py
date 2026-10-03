"""Select completed probing artifacts; does not run a model."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate-dir',type=Path,required=True)
    parser.add_argument('--rollouts',type=Path,required=True)
    parser.add_argument('--trajectory-root',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    try:
        from adaptive_noise.probe_selection import select_probe_dataset
        manifest=select_probe_dataset(args.candidate_dir,args.rollouts,args.output_dir,args.trajectory_root)
        print(json.dumps({'status':manifest['status'],'counts':manifest['counts']}))
        return 0
    except (OSError,ValueError,KeyError,TypeError,ImportError) as error:
        print(f'STOP: {error}',file=sys.stderr)
        return 2


if __name__=='__main__':
    raise SystemExit(main())
