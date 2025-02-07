# main.py

import argparse
from optuna_tuner import run_optuna_study

def main():
    parser = argparse.ArgumentParser(description="UJIIndoorLoc 室内定位——直接进行超参数调优")
    parser.add_argument('--n_trials', type=int, default=500, help="调参试验次数（默认 500）")
    args = parser.parse_args()

    # 直接进行超参数搜索
    run_optuna_study(n_trials=args.n_trials)

if __name__ == "__main__":
    main()
