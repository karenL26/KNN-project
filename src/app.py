##############################
#       KNN Project          #
##############################
### Load libraries and modules ###
# Dataframes, matrices and others --------------------------------------
import pandas as pd 
import numpy as np
import ast
# Machine learning -----------------------------------------------------
from sklearn.feature_extraction.text import CountVectorizer
# Metrics --------------------------------------------------------------
from sklearn.metrics.pairwise import cosine_similarity

#########################################
# Data Preprocessing and Transformation #
#########################################
# Load the datasets
movies = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/k-nearest-neighbors-project-tutorial/main/tmdb_5000_movies.csv')
credits = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/k-nearest-neighbors-project-tutorial/main/tmdb_5000_credits.csv')
# Merge of both datasets on the 'title' column
movies = movies.merge(credits, on='title')
# Extract the important columns
movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]
# Remove the null values
movies.dropna(inplace = True)
# Obtain the genres information #
def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L
movies.dropna(inplace = True)
movies['genres'] = movies['genres'].apply(convert)

# Repeat the process for the 'keywords' column.
movies['keywords'] = movies['keywords'].apply(convert)

# Obtain the cast information #
# For the 'cast' column we will create a new but similar function. This time we will limit the number of items to three.
def convert3(obj):
    L = []
    count = 0
    for i in ast.literal_eval(obj):
        if count < 3:
            L.append(i['name'])
        count +=1  
    return L
movies['cast'] = movies['cast'].apply(convert3)

# Obtain the director information #
def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L
movies['crew'] = movies['crew'].apply(fetch_director)

# Obtain information from overview #
# For the 'overview' column, we will convert it in a list by using 'split()' methode.
movies['overview'] = movies['overview'].apply(lambda x : x.split())

# Remove spaces between words #
def collapse(L):
    L1 = []
    for i in L:
        L1.append(i.replace(" ",""))
    return L1
# Now let's apply our function to the 'genres', 'cast', 'crew' and 'keywords' columns.
movies['cast'] = movies['cast'].apply(collapse)
movies['crew'] = movies['crew'].apply(collapse)
movies['genres'] = movies['genres'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)

# Reduce the number of columns # 
movies['tags'] = movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']

new_df = movies[['movie_id','title','tags']]
new_df['tags'] = new_df['tags'].apply(lambda x :" ".join(x))

###################################
# Build of the Recommender System #
###################################
# Text vectorization
cv = CountVectorizer(max_features=5000 ,stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()

# Find the cosine_similarity among the movies.
similarity = cosine_similarity(vectors)

# Create a recommendation function based on the cosine_similarity
def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0] ##fetching the movie index
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate( distances)),reverse =True , key = lambda x:x[1])[1:6]
    
    for i in movie_list:
        print(new_df.iloc[i[0]].title)