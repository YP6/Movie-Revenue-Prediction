import pandas as pd
import numpy as np

revenues = pd.read_csv('../Datasets/movies-revenue.csv')
voice_actors = pd.read_csv('../Datasets/movie-voice-actors.csv')
directors = pd.read_csv('../Datasets/movie-director.csv')

def main():
    print("------------------------Preprocessing------------------------")
    #HandleMissingValues() #DONE
    ParseDate()           #DONE
    ParseRevenue()        #DONE
    directors.rename(columns={'name': 'movie_title'}, inplace=True)
    voice_actors.rename(columns={'movie': 'movie_title'}, inplace=True)
    print("\n\nJoining Tables")
    print("-" * 25)
    dataset = JoinTables()
    RemoveDuplicates()    #DONE

    #revenues_preprocessed, voice_actors_preprocessed, directors_preprocessed = HandlingCategoricalData()


    #TODO Uncomment this after finishing aggregation
    #final_preprocessed_data = joinTables(revenues_preprocessed, voice_actors_preprocessed, directors_preprocessed)
    print("Finished Preprocessing")
    print("-"*50)
    print("Data Sample")
    #print(final_preprocessed_data.head())

def JoinTables():
    revenues_directors = pd.merge(revenues, directors, on="movie_title", how="outer")
    data = pd.merge(revenues_directors, voice_actors, on="movie_title", how="outer")
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


def joinTables(rev, va):
    data = rev.merge(va, how='left', left_on='movie-title', right_on='movie')
    print(data.head())




if __name__ == '__main__':
    main()
