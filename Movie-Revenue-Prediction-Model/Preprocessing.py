import pandas as pd
import numpy as np
import re

directors = pd.read_csv('../Datasets/movie-director.csv')
revenue = pd.read_csv('../Datasets/movies-revenue.csv')
voice_actors = pd.read_csv('../Datasets/movie-voice-actors.csv')

def HandleMissingValues():
    print("Handling Null Values:\n")
    #Checking For Missing Values
    revenue.isna().sum()
    print("Before Dropping NULLs Table Shape was ", revenue.shape)
    revenue.dropna(inplace=True)
    print("After Dropping NULLs Table Shape is ", revenue.shape)

def ParseDate():
    print("\n\nParsing Date:\n")
    #Checking Date Format
    date_lengths = revenue.release_date.str.len()
    print("Date Lengths : \n")
    print(date_lengths.value_counts())

    revenue.release_date = pd.to_datetime(revenue.release_date)
    print("Release Date type: ", revenue.release_date.dtype)

def ParseRevenue():
    print("\n\nParsing Revenue:\n")

    print("Revenue Data Type Before:",revenue.revenue.dtype)
    revenue.revenue = revenue.revenue.replace('[\$,]', '', regex=True).astype(float)
    print("Revenue Data Type After:", revenue.revenue.dtype)


def HandlingCategoricalData():
    print("\n\nConverting Categorical Data")

    dummies = pd.get_dummies(revenue.genre)
    unique_genres = pd.DataFrame(dummies)

    del(revenue['genre'])

    revenue_preproccessed = pd.concat([revenue, unique_genres], axis=1)
    print(revenue_preproccessed.shape)

print("------------------------Preprocessing------------------------")
HandleMissingValues()
ParseDate()
ParseRevenue()
HandlingCategoricalData()