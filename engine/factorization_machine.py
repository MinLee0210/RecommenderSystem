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

import torch
from torch import nn
import torch.multiprocessing
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

GLOBAL_SEED = 42  # number of life
torch.manual_seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def get_dataset_path(name):
    file_path = str(ROOT) + f'\dataset\{name}.csv'
    return file_path


df_dict = pd.read_csv(get_dataset_path('movies'), index_col=None).drop(columns=['Unnamed: 0'])
users, movies, ratings = df_dict["users"], df_dict["movies"], df_dict["ratings"]
ratings["Rating"] = ratings["Rating"] - 3  # rating range (-2, 2)
train_ratings, validation_ratings = train_test_split(
    ratings, test_size=0.1, random_state=GLOBAL_SEED
)

# user_featuers
user_index_by_id = {id: idx for idx, id in enumerate(users["UserID"]) }
gender_index_by_name = {"M":0, "F": 1}
age_index_by_name = {1: 0, 18: 1, 25: 2, 35:3, 45: 4, 50: 5, 56:6}
occupations = [
"other",
"academic/educator",
"artist",
"clerical/admin",
"college/grad student",
"customer service",
"doctor/health care",
"executive/managerial",
"farmer",
"homemaker",
"K-12 student",
"lawyer",
"programmer",
"retired",
"sales/marketing",
"scientist",
"self-employed",
"technician/engineer",
"tradesman/craftsman",
"unemployed",
"writer",
]
occupation_index_by_name = {name: index for index, name in enumerate(occupations)}

num_users = len(users)
gender_offset = num_users
age_offset = gender_offset + len(gender_index_by_name)
occupation_offset = age_offset + len(age_index_by_name)

user_features = []
for index in range(num_users):
    gender_index = gender_index_by_name[users["Gender"][index]] + gender_offset
    age_index = age_index_by_name[users["Age"][index]] + age_offset
    occupation_index = users["Occupation"][index] + occupation_offset
    user_features.append([index, gender_index, age_index, occupation_index])
    
print("Example for the first user: ", user_features[0])


# build moive_features

movie_index_by_id = {id: idx for idx, id in enumerate(movies["MovieID"])}
movie_offset = occupation_offset + len(occupation_index_by_name)

genres = [
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]
genre_index_by_name = {name:i for i, name in enumerate(genres)}
num_movies = len(movies)

movie_features = []
for i, movie_genres in enumerate(movies["Genres"]):
    movie_feature = [movie_offset + i]
    for genre in movie_genres.split("|"):
        genre_index = genre_index_by_name[genre] + num_movies + movie_offset
        movie_feature.append(genre_index)
    movie_features.append(movie_feature)
print("Example for the first movie:", movie_features[0])
total_inputs = movie_offset + num_movies + len(genres)
print(total_inputs)


NUM_MOVIES = len(movies)
NUM_USERS = len(users)
padding_idx = total_inputs


class FactorizationMachineDataset(torch.utils.data.Dataset):
    def __init__(self, rating_df):
        self.rating_df = rating_df
        self.max_size = 5 + len(genres)  # 4 for user feature + movie index + genres

    def __len__(self):
        return len(self.rating_df)

    def __getitem__(self, index):
        user_index = user_index_by_id[self.rating_df["UserID"].iloc[index]]
        movie_index = movie_index_by_id[self.rating_df["MovieID"].iloc[index]]
        rating = self.rating_df["Rating"].iloc[index]
        user_feature = user_features[user_index]
        movie_feature = movie_features[movie_index]
        padding_size = self.max_size - len(user_feature) - len(movie_feature)
        feature = user_feature + movie_feature + [padding_idx] * padding_size
        return torch.IntTensor(feature), rating
    
from pytorch_lightning.loggers import TensorBoardLogger
import jdc

LR = 5e-4
WEIGHT_DECAY = 5e-5


class FactorizationMachine(pl.LightningModule):
    def __init__(self, num_inputs, num_factors):
        super(FactorizationMachine, self).__init__()
        self.embedding = nn.Embedding(num_inputs + 1, num_factors, padding_idx=padding_idx)
        self.embedding.weight.data.uniform_(-.1, .1)
        torch.nn.init.xavier_normal_(self.embedding.weight.data, gain=1e-3)
        self.linear_layer = nn.Embedding(num_inputs+1, 1, padding_idx=padding_idx)
        self.bias = nn.Parameter(data=torch.rand(1))
    

    def forward(self, x):
        emb = self.embedding(x)
        pow_of_sum = emb.sum(dim=1, keepdim=True).pow(2).sum(dim=2)
        sum_of_pow = emb.pow(2).sum(dim=1, keepdim=True).sum(dim=2)
        out_inter = 0.5 * (pow_of_sum - sum_of_pow)
        out_lin = self.linear_layer(x).sum(1)
        out = out_inter + out_lin + self.bias

        return torch.clip(out.squeeze(), min=-2, max=2)
    
# Training and Evaluating
n_factors = 100
logger = TensorBoardLogger(
    "fm_2_tb_logs", name=f"ilr{LR}_wd{WEIGHT_DECAY}_emb{n_factors}_b{batch_size}"
)

model = FactorizationMachine(num_inputs=total_inputs, num_factors=n_factors)
trainer = pl.Trainer(gpus=1, max_epochs=40, logger=logger)

trainer.fit(model, train_dataloader, validation_dataloader)

def eval_model(model, train_dataloader):
    loss = 0
    for feature, rating in train_dataloader:
        pred = model(feature)
        loss += F.mse_loss(pred, rating)
    RMSE = (loss / len(train_dataloader))**.5
    return RMSE
    
print("Train RMSE: {:.3f}".format(eval_model(model, train_dataloader)))
print("Validation RMSE: {:.3f}".format(eval_model(model, validation_dataloader)))

