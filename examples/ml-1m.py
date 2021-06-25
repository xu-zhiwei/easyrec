import pandas as pd
from pathlib import Path


def main():
    dataset_path = Path(args.dataset_path)
    df = pd.read_csv(dataset_path / 'ratings.dat', sep='::', engine='python', header=None)
    df.loc[df[2] <= 3, 2] = 0
    df.loc[df[2] > 3, 2] = 1
    print(df)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--dataset_path')
    args = parser.parse_args()
    main()
