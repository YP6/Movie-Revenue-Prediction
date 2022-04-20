import pandas as pd
import numpy as np
import imdb


revenues = pd.read_csv('../Datasets/movies-revenue.csv')
voice_actors = pd.read_csv('../Datasets/movie-voice-actors.csv')
directors = pd.read_csv('../Datasets/movie-director.csv')

def main(filepath):
    print("------------------------Preprocessing------------------------")
    #HandleMissingValues() #DONE
    ParseDate()           #DONE
    ParseRevenue()        #DONE

    data_after_filing = fillMissingData(revenues)
    data_after_filing.dropna(inplace=True)

    directors.rename(columns={'name': 'movie_title'}, inplace=True)
    voice_actors.rename(columns={'movie': 'movie_title'}, inplace=True)

    #Join Tables
    print("\n\nJoining Tables")
    print("-"*25)
    dateset_after_joining = JoinTables(data_after_filing, directors, voice_actors)

    #revenues_preprocessed, voice_actors_preprocessed, directors_preprocessed = HandlingCategoricalData()

    print("Saving...")
    dateset_after_joining.to_csv(filepath, index=False)
    print("Finished Preprocessing")
    print("-"*50)
    print("Data Sample")
    print(dateset_after_joining.head())

def JoinTables(revs, va, dir):
    revenues_actors = pd.merge(revs, va, on="movie_title", how="outer")
    data = pd.merge(revenues_actors, dir, on="movie_title", how="outer")
    data = data.dropna(axis=0, subset=['revenue'])
    print("Shape after Joining :", data.shape)
    return data

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

    #Fixing Parsing Wrong Dates
    for i in range(revenues.shape[0]):
        date = revenues.loc[i, 'release_date']
        if 2 < int(date[-2]) < 7:
            new_date = date[:-2] + "19" + date[-2:]
            revenues.loc[i, 'release_date'] = new_date
        elif int(date[-2]) == 2 and int(date[-1]) > 2:
            new_date = date[:-2] + "19" + date[-2:]
            revenues.loc[i, 'release_date'] = new_date

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

    """Revenues."""
    print("\nHandling Categorical Data: ")
    print("-"*25)

    dummies = pd.get_dummies(data=revenues, columns=['genre', 'MPAA_rating'])
    dummiesDF = pd.DataFrame(dummies)

    print("Revenues Shape Before One Hot Encoding:", revenues.shape)
    del(revenues['genre'])
    del(revenues['MPAA_rating'])

    revenues_preprocessed = dummiesDF
    print("Revenues Shape After One Hot Encoding:",revenues_preprocessed.shape)

    """Voice Actors."""
    print("-"*15)
    #dummies = pd.get_dummies(data=voice_actors, columns=['voice-actor'])
    #dummiesDF = pd.DataFrame(dummies)
    #unique = voice_actors['voice-actor'].unique()

    print("Voice Actors Shape Before One Hot Encoding:", voice_actors.shape)
    #del(voice_actors['voice-actor'])

    voice_actors_preprocessed = (voice_actors.pivot_table(index='movie', columns='voice-actor', aggfunc='size', fill_value=0).reset_index().rename_axis(columns=None))
    #TODO: Aggregation
    print("Voice Actors Shape After One Hot Encoding:", voice_actors_preprocessed.shape)

    """Movie Directors."""
    print("-" * 15)
    #dummies = pd.get_dummies(data=directors, columns=['director'])
    #dummiesDF = pd.DataFrame(dummies)

    print("Directors Shape Before One Hot Encoding:", directors.shape)
    #del (directors['director'])

    directors_preprocessed = (directors.pivot_table(index='name',columns='director',aggfunc='size',fill_value=0).reset_index().rename_axis(columns=None))
    #TODO: Aggregation
    print("Directors Shape After One Hot Encoding:", directors_preprocessed.shape)

    return revenues_preprocessed, voice_actors_preprocessed, directors_preprocessed

def RemoveDuplicates():
    revenues.sort_values(by='release_date', ascending=False, inplace=True)
    revenues.drop_duplicates(subset=['movie_title'], inplace=True)



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
        data.loc[data['movie_title'] == name, 'release_date'] = pd.to_datetime("1-Jan-"+search[0].items()[2][1])
    print("Filling Dates Has Finished")

    # Filling Movies' Genre.
    genre_nans = data[data['genre'].isna()].movie_title
    print("Filling Genres...")
    i=0
    for name in genre_nans:
        i+=1
        print(i,"/",len(genre_nans))
        search = ia.search_movie(name)
        id = search[0].movieID
        movie = ia.get_movie(id)
        genre = movie['genres'][0]
        data.loc[data['movie_title'] == name, 'genre'] = genre

    print("Filling Genres Has Finished")
    # Filling Movies' MPAA Ratings.
    rating_nans = data[data['MPAA_rating'].isna()].movie_title
    MPAA_ratings = ['G', 'PG', 'R', 'PG-13', 'Not Rated']

    print("Filling MPAA Rating...")
    i = 0
    for name in rating_nans:
        i += 1
        print(i, "/", len(rating_nans))
        search = ia.search_movie(name)
        id = search[0].movieID
        movie = ia.get_movie(id)
        ratingsLen = len(movie.data['certificates'])
        ratings = movie.data['certificates']

        for i in range(ratingsLen):
            rating = ratings[i]
            if 'United States' in rating:
                rating = rating.split(":", 1)[1]
                if rating in MPAA_ratings:
                    data.loc[data['movie_title'] == name, 'MPAA_rating'] = rating
                else:
                    data.loc[data['movie_title'] == name, 'MPAA_rating'] = None
    print("Filling MPAA Rating Has Finished")
    return data
if __name__ == '__main__':
    main("data_finalizing_test.csv")
