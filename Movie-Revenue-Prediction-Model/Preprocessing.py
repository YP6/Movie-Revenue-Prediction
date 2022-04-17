import pandas as pd
import numpy as np

directors = pd.read_csv('../Datasets/movie-director.csv')
revenues = pd.read_csv('../Datasets/movies-revenue.csv')
voice_actors = pd.read_csv('../Datasets/movie-voice-actors.csv')


def main():
    print("------------------------Preprocessing------------------------")
    HandleMissingValues()
    ParseDate()
    ParseRevenue()
    # HandlingCategoricalData()


def HandleMissingValues():
    """Drop all rows with NaNs in the dataset."""
    print("\nHandling Null Values: ")
    print("-"*25)

    # Checking for missing values.
    print("Revenues #NaNs:\n", revenues.isna().sum())
    print("-"*25)
    print("Directors #NaNs:\n", directors.isna().sum())
    print("-"*25)
    print("Voice Actors #NaNs:\n", voice_actors.isna().sum())
    print("-"*25)

    # Dropping NaNs.
    print("Revenues shape before dropping NaNs: ", revenues.shape)

    revenues.dropna(inplace=True)

    print("Revenues shape after dropping NaNs:  ", revenues.shape)
    print("-"*50)


def ParseDate():
    """Converts release-date data type to datetime instead of string."""
    print("\nParsing Date: ")
    print("-" * 25)

    # Checking date format consistency.
    date_lengths = revenues.release_date.str.len()

    print("Date Lengths :")
    print(date_lengths.value_counts())
    print("-" * 25)

    print("Release-Date datatype before Parsing: ", revenues.release_date.dtype)

    revenues.release_date = pd.to_datetime(revenues.release_date)

    print("Release-Date datatype after Parsing: ", revenues.release_date.dtype)
    print("-"*50)


def ParseRevenue():
    """Converts revenue data type to float instead of string."""
    print("\nParsing Revenue:")
    print("-"*25)

    print("Revenues before Parsing:\n", revenues.revenue.head())
    print("-" * 25)

    revenues.revenue = revenues.revenue.replace('[$,]', '', regex=True).astype(float)

    print("Revenues after Parsing:\n", revenues.revenue.head())
    print("-" * 50)


def HandlingCategoricalData():
    """Converts categorical data to numerical data to be able to apply regression."""
    print("\nHandling Categorical Data: ")
    print("-"*25)

    dummies = pd.get_dummies(revenues.genre)
    unique_genres = pd.DataFrame(dummies)

    del(revenues['genre'])

    revenue_preproccessed = pd.concat([revenues, unique_genres], axis=1)
    print(revenue_preproccessed.shape)


if __name__ == '__main__':
    main()
