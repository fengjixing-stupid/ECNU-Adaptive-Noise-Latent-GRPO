"""Write missing probing requests from a fixed pool and optional completed history."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate-dir',type=Path,required=True)
    parser.add_argument('--rollouts',type=Path)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        from adaptive_noise.probe_selection import plan_candidate_requests
        from adaptive_noise.dataset_builder import file_sha256
        if args.output_dir.exists():
            raise FileExistsError(f'refusing to overwrite output: {args.output_dir}')
        if args.rollouts and not args.rollouts.is_file():
            raise FileNotFoundError(f'missing required rollouts: {args.rollouts}')
        history=pq.read_table(args.rollouts).to_pylist() if args.rollouts else []
        requests=plan_candidate_requests(args.candidate_dir,history)
        args.output_dir.mkdir(parents=True,exist_ok=False)
        schema=pa.schema([('problem_id',pa.string()),('source',pa.string()),('split',pa.string()),
            ('scale',pa.float64()),('rollout_id',pa.int64()),('seed',pa.int64()),('max_new_tokens',pa.int64()),
            ('temperature',pa.float64()),('top_p',pa.float64()),('max_topk',pa.int64()),
            ('gumbel_softmax_temperature',pa.float64()),('noise_scale',pa.float64()),
            ('add_noise_gumbel_softmax',pa.bool_()),('use_one_sided_gumbel_noise',pa.bool_())])
        pq.write_table(pa.Table.from_pylist(requests,schema=schema),args.output_dir/'requests.parquet')
        manifest={'status':'complete','scope':'request_planning_no_model_execution','requests':len(requests),
            'candidate_manifest_sha256':file_sha256(args.candidate_dir/'build_manifest.json'),
            'history_sha256':file_sha256(args.rollouts) if args.rollouts else None,
            'requests_sha256':file_sha256(args.output_dir/'requests.parquet')}
        temp=args.output_dir/'request_manifest.json.tmp'
        temp.write_text(json.dumps(manifest,indent=2)+'\n')
        temp.rename(args.output_dir/'request_manifest.json')
        print(json.dumps(manifest))
        return 0
    except (OSError,ValueError,KeyError,TypeError,ImportError) as error:
        print(f'STOP: {error}',file=sys.stderr)
        return 2


if __name__=='__main__':
    raise SystemExit(main())
