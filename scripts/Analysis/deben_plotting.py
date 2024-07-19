import argparse
from datetime import datetime

import pandas as pd
from matplotlib import pyplot as plt


def get_options(the_parser):
    """
    Specify options for this script with ArgParse
    :param the_parser: argument parser in use
    :return parsed arguments
    """
    the_parser.add_argument(
        "-c",
        "--csv",
        help="Specify csv path",
        action="store",
        dest="csv",
        type=str,
    )
    return the_parser


def main(csv_path):
    df = pd.read_csv(csv_path, skiprows=20)
    print(df)
    df["Start Time"] = pd.Timestamp(year=2022, month=10, day=14, hour=15, minute=31, second=1)
    df["Timedelta"] = [pd.Timedelta(seconds=sec) for sec in list(df["Point"])]
    df["Time"] = df["Start Time"] + df["Timedelta"]
    plt.plot(df["Time"], df["Force"])
    plt.xlabel("Date and hour")
    plt.ylabel("Force (N)")
    plt.show()


if __name__ == "__main__":
    # Call get_options function to get arguments:
    parser = argparse.ArgumentParser()
    myparser = get_options(parser)
    options, args = myparser.parse_known_args()
    main(csv_path=options.csv)