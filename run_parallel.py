from __future__ import annotations

from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List, Tuple

from esl_project.data_utils import DataLoadConfig
from esl_project.pipelines import ContractRunConfig, build_tasks_from_dir, run_single_contract


def build_tasks(
    data_dir: Path,
    output_root: Path,
    max_contracts: int = 5,
    candidate_C: List[float] = None,
    start_date: str | None = None,
    end_date: str | None = None,
    nrows_per_file: int | None = None,
) -> List[Tuple]:
    if candidate_C is None:
        candidate_C = [0.0001, 0.001, 0.01, 0.1, 1.0]

    run_cfg = ContractRunConfig(candidate_C=candidate_C)
    load_cfg = DataLoadConfig(
        data_dir=data_dir,
        max_files=max_contracts,
        nrows_per_file=nrows_per_file,
        start_date=start_date,
        end_date=end_date,
    )
    return build_tasks_from_dir(data_dir, output_root, run_cfg, load_cfg, max_files=max_contracts)


def run_task(args):
    return run_single_contract(*args)


def main():
    data_dir = Path("2005年__20250905")
    output_root = Path("outputs_parallel") / "run"
    output_root.mkdir(parents=True, exist_ok=True)

    tasks = build_tasks(
        data_dir=data_dir,
        output_root=output_root,
        max_contracts=5,
        candidate_C=[0.0001, 0.001, 0.01, 0.1, 1.0],
        start_date="2014-01-01",
        end_date="2018-12-31",
        nrows_per_file=None,
    )

    n_workers = min(5, cpu_count())
    with Pool(processes=n_workers) as pool:
        results = pool.map(run_task, tasks)

    for res in results:
        print(f"Contract {res.get('contract')} finished; best C={res.get('best_C')}")


if __name__ == "__main__":
    main()
