import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import yaml
import streamlit as st
import sys
import os
import platform
from pathlib import Path

from engine.content_based_recommender import *

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

n_films = 10
threshhold = 3
tab = '\t'
def get_data_filepath(name):
    file_path = str(ROOT) + f'\dataset\{name}.csv'
    return file_path

movies = pd.read_csv(get_data_filepath('movies'), index_col=None).drop(columns=['Unnamed: 0'])
movie_sample_mask = np.random.randint(0, 3883, n_films)
movie_samples = movies.loc[movie_sample_mask]
movie_samples = movie_samples.iloc[: , 1: 3]

movie_titles = movie_samples.Title
user_prefer_list = {
    'Titles': [],
    'Ratings': []
}
for i in range(n_films):
    user_prefer_list['Titles'].append(movie_titles.iloc[i])

def get_recommendation(user_profile:dict):
    user_profile = pd.DataFrame(pd.DataFrame(user_profile))
    recommender_system = UserProfile(user_profile)
    reccommends = recommender_system.recommend(n_films=n_films, threshhold=threshhold)
    return reccommends

st.title("RECOMMENDER SYSTEM")
st.divider()

st.header('PLEASE RANKING BELOW MOVIES')
with st.form('my_form'):
    film_1 = st.slider(f'{movie_samples.iloc[0, 0]} ===== {movie_samples.iloc[0, 1]}', 0, 5, 1)
    user_prefer_list['Ratings'].append(film_1)

    film_2 = st.slider(f'{movie_samples.iloc[1, 0]} ===== {movie_samples.iloc[1, 1]}', 0, 5, 1)
    user_prefer_list['Ratings'].append(film_2)

    film_3 = st.slider(f'{movie_samples.iloc[2, 0]} ===== {movie_samples.iloc[2, 1]}', 0, 5, 1)
    user_prefer_list['Ratings'].append(film_3)

    film_4 = st.slider(f'{movie_samples.iloc[3, 0]} ===== {movie_samples.iloc[3, 1]}', 0, 5, 1)
    user_prefer_list['Ratings'].append(film_4)

    film_5 = st.slider(f'{movie_samples.iloc[4, 0]} ===== {movie_samples.iloc[4, 1]}', 0, 5, 1)
    user_prefer_list['Ratings'].append(film_5)

    film_6 = st.slider(f'{movie_samples.iloc[5, 0]} ===== {movie_samples.iloc[5, 1]}', 0, 5, 1)
    user_prefer_list['Ratings'].append(film_6)

    film_7 = st.slider(f'{movie_samples.iloc[6, 0]} ===== {movie_samples.iloc[6, 1]}', 0, 5, 1)
    user_prefer_list['Ratings'].append(film_7)

    film_8 = st.slider(f'{movie_samples.iloc[7, 0]} ===== {movie_samples.iloc[7, 1]}', 0, 5, 1)
    user_prefer_list['Ratings'].append(film_8)

    film_9 = st.slider(f'{movie_samples.iloc[8, 0]} ===== {movie_samples.iloc[8, 1]}', 0, 5, 1)
    user_prefer_list['Ratings'].append(film_9)

    film_10 = st.slider(f'{movie_samples.iloc[9, 0]} ===== {movie_samples.iloc[9, 1]}', 0, 5, 1)
    user_prefer_list['Ratings'].append(film_10)

    if st.form_submit_button('Submit'):
        st.divider()
        st.header('WE RECOMMEND YOU USING CONTENT_BASED RECOMMENDATION')
        recommends = get_recommendation(user_prefer_list)
        recommend_movies = recommends.iloc[:, [1, 2]]
        st.write(recommend_movies)
    else:
        st.divider()


