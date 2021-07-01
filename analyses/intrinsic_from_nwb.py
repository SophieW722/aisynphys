import argparse
import pandas as pd
from aisynphys.intrinsic_ephys import process_file_list

def main():
    parser = argparse.ArgumentParser(
            description="Process intrinsic features from multipatch NWB2 files into a flat csv."
        )
    parser.add_argument('files', type=str, nargs='+', help='NWB2 file(s) to process')
    parser.add_argument('--output', default='features.csv', help='path to write output csv')
    args = parser.parse_args()
    records = process_file_list(args.files)
    ephys_df = pd.DataFrame.from_records(records)
    ephys_df.to_csv(args.output)


if __name__ == "__main__":
    main()