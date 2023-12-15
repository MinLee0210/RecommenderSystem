import sys
import os
import platform
from pathlib import Path

import pandas as pd
import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


genre = ['Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime',
	    'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
	    'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

def get_dataset_path(name):
    file_path = str(ROOT) + f'\dataset\{name}.csv'
    return file_path

class OneHotDataset():
    def __init__(self, movies:pd.DataFrame, ratings:pd.DataFrame, users:pd.DataFrame, genre:list=genre):
        self.movies = movies
        self.ratings = ratings
        self.users = users
        self.genre = genre

        self.genre_index_by_name =  {name:i for i, name in enumerate(genre)}
        self.movie_index_by_id = {id: i for i, id in enumerate(movies["MovieID"])}
        
        self.movie_features = np.zeros((len(self.movies.MovieID), len(self.genre)))
        
    def one_hot_genre_film(self):
        for i, movie_genres in enumerate(self.movies["Genres"]):
            for genre in movie_genres.split("|"):        
                genre_index = self.genre_index_by_name[genre]
                self.movie_features[i, genre_index] = 1
        self.movie_features = pd.DataFrame(data=self.movie_features, columns=self.genre)
        self.movie_features.insert(loc=0, column='MovieID', value=self.movies.MovieID)
        return self.movie_features