import numpy as np
import pandas as pd

import Config

if Config.fixed_randomness:
    import Util.Randomness as Randomness

def main():
    # Data Load
    df = pd.read_csv('./Data/train.csv')
    if Config.test_run:
        df = df.sample(frac=1).reset_index(drop=True)
        df = df.head(Config.test_run_data_size)

    

    print()

if __name__ == '__main__':
    main()
