import pandas as pd
import numpy as np
import imdb

successLevel = pd.read_csv('../Datasets/Classification Datasets/movies-revenue-classification.csv')
voice_actors = pd.read_csv('../Datasets/Classification Datasets/movie-voice-actors.csv')
directors = pd.read_csv('../Datasets/Classification Datasets/movie-director.csv')

features = ['release_date', 'genre', 'MPAA_rating', 'director', 'character', 'voice-actor']


def main(filepath):
    print("------------------------Preprocessing------------------------")
    ParseDate()

    directors.rename(columns={'name': 'movie_title'}, inplace=True)
    voice_actors.rename(columns={'movie': 'movie_title'}, inplace=True)

    # Join Tables
    print("\n\nJoining Tables")
    print("-" * 25)
    dateset_after_joining = JoinTables(successLevel, directors, voice_actors)

    data_after_filing = fillMissingData(dateset_after_joining)


    print("Saving...")
    data_after_filing.to_csv(filepath, index=False)
    print("Finished Preprocessing")
    print("-" * 50)
    print("Data Sample")
    print(data_after_filing.head())


def JoinTables(successLevels, va, dir):
    MovieSuccessLevels_actors = pd.merge(successLevels, va, on="movie_title", how="outer")
    data = pd.merge(MovieSuccessLevels_actors, dir, on="movie_title", how="outer")
    data = data.dropna(axis=0, subset=['MovieSuccessLevel'])
    print("Shape after Joining :", data.shape)
    return data


def ParseDate():
    """Converts release-date data type to datetime instead of string."""
    print("\nParsing Date: ")
    print("-" * 25)

    # Checking date format consistency.
    date_lengths = successLevel.release_date.str.len()

    print("Date Lengths :")
    print(date_lengths.value_counts())
    print("-" * 25)

    print("Release-Date datatype before Parsing: ", successLevel.release_date.dtype)

    # Fixing Parsing Wrong Dates
    for i in range(successLevel.shape[0]):
        date = successLevel.loc[i, 'release_date']
        if 2 < int(date[-2]) < 7:
            new_date = date[:-2] + "19" + date[-2:]
            successLevel.loc[i, 'release_date'] = new_date
        elif int(date[-2]) == 2 and int(date[-1]) > 2:
            new_date = date[:-2] + "19" + date[-2:]
            successLevel.loc[i, 'release_date'] = new_date

    successLevel.release_date = pd.to_datetime(successLevel.release_date)

    print("Release-Date datatype after Parsing: ", successLevel.release_date.dtype)
    print("-" * 50)


def RemoveDuplicates():
    successLevel.sort_values(by='release_date', ascending=False, inplace=True)
    successLevel.drop_duplicates(subset=['movie_title'], inplace=True)


def fillMissingData(data):
    """ Takes the joined data frame as an input and returns the dataframe after filling missing values."""
    ia = imdb.IMDb()

    date_nans = data[data['release_date'].isna()].movie_title
    print("Filling Dates...")
    i = 0
    for name in date_nans:
        i += 1
        print(i, "/", len(date_nans))
        search = ia.search_movie(name)
        data.loc[data['movie_title'] == name, 'release_date'] = pd.to_datetime("1-Jan-" + search[0].items()[2][1])
    print("Filling Dates Has Finished")

    # Filling Movies' Genre.
    genre_nans = data[data['genre'].isna()].movie_title
    print("Filling Genres...")
    i = 0
    for name in genre_nans:
        i += 1
        print(i, "/", len(genre_nans))
        search = ia.search_movie(name)
        id = search[0].movieID
        movie = ia.get_movie(id)
        genre = movie['genres'][0]
        data.loc[data['movie_title'] == name, 'genre'] = genre

    print("Filling Genres Has Finished")

    #Nulls in Director
    directorNulls=data[data['director'].isnull()].index.tolist() #high
    print("director Nulls",directorNulls)
    for i in range(len(directorNulls)):
            name = data['movie_title'][directorNulls[i]]
            search = ia.search_movie(name)
            if len(search) == 0:
                continue
            id = search[0].movieID
            # getting information
            movie = ia.get_movie(id)
            if 'directors' in movie:
                name = movie.data['directors'][0]
                #adding name to CSV
                print (name)
                print(directorNulls[i])
                if(name):
                    data['director'][directorNulls[i]]=name
                else:
                    data['director'][directorNulls[i]]=data['director'].mode()
    return data


if __name__ == '__main__':
    main("Preprocessed-Dataset/preprocessed_data.csv")

