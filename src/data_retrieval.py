#!/usr/bin/env python3

"""
Data retrieval utilities.
Fetches the housing data to predict median house values in Californian districts.
"""


from pathlib import Path
from urllib.request import urlretrieve
import tarfile
from pandas import read_csv



def download_data(directory="data", file_name="data"):
    """Downloads the data. Returns the Path"""
    # link to the housing data from the github account of A.Geron
    URL = "https://github.com/ageron/data/raw/main/housing.tgz"
    DIR = Path(directory)
    CSV_FILE = Path(f"{file_name}.csv")
    TGZ_FILE = Path(f"{file_name}.tgz")

    if not DIR.is_dir():
        Path(DIR).mkdir()

    if not (DIR / CSV_FILE).is_file():
        # Download the tgz file
        urlretrieve(URL, DIR / TGZ_FILE)
        # Extract tgz
        with tarfile.open(DIR / TGZ_FILE) as tgz:
            tgz.extractall(path=DIR)
        # Remove the tgz file
        Path(DIR / TGZ_FILE).unlink()
        # Move, rename
        Path(DIR / Path("housing/housing.csv")).rename(DIR / CSV_FILE)
        # Remove the empty folder
        Path(DIR / "housing").rmdir()

    # Return Path
    return Path(DIR / CSV_FILE)


def load_data(path):
    """Loads the csv data into a df"""
    return read_csv(path)


def fetch_data():
    """
    Wrapper function.
    Return a df
    """
    return load_data(download_data())



def add_doctored_features(df):
    """
    Adds some "manufactured" features to practice certain theoretical subjects
    like freature transformation.
    """



# demo / test
if __name__ == '__main__':
    ...
    df = fetch_data()
    print(df.head())