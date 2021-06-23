import pandas as pd
from pathlib import Path


def main():
    dataset_path = Path(option.dataset_path)
    df = pd.read_csv(dataset_path / 'ratings.dat', sep='::', engine='python')



if __name__ == '__main__':
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option('--dataset_path', dest='dataset_path')
    option, args = parser.parse_args()
    main()
