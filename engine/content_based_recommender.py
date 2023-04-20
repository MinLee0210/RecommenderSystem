from ..lib import *

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def get_dataset_path(name):
    file_path = str(ROOT) + f'\dataset\{name}.csv'
    return file_path

class UserProfile():
    def __init__(self, user_prefer_list:pd.DataFrame):
        if type(user_prefer_list) != pd.DataFrame:
            raise "THERE IS A PROBLEM FROM THE SYSTEM"
        self.movies = pd.read_csv(get_dataset_path('movies')).drop(columns=['Unnamed: 0'])
        self.ratings = pd.read_csv(get_dataset_path('ratings')).drop(columns=['Unnamed: 0'])
        self.users = pd.read_csv(get_dataset_path('users')).drop(columns=['Unnamed: 0'])
        self.genre = ['Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime',
	                    'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
	                    'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
        self.user_prefer_list = user_prefer_list
        self.movie_one_hot = pd.read_csv(get_dataset_path('movie_one_hot')).drop(columns=['Unnamed: 0'])

    def analyzed_user(self, user_prefer_list):
        # User prefer list consists of <movie_name, ratings> => <{movie_genres}, ratings> -> LinearRegression/RidgeRegression
        movie_mask = self.movies.Title.isin(user_prefer_list.Titles.values)
        user_prefer_movie_id = self.movies.loc[movie_mask]['MovieID']
        genre_mask = self.movie_one_hot.MovieID.isin(user_prefer_movie_id)
        movie_genres = self.movie_one_hot.loc[genre_mask]
        user_profile = LinearRegression().fit(movie_genres, user_prefer_list.Ratings)
        return user_profile
    
    def predict(self, user, vector):
        output = user(vector)
        output = max(min(output, 1), 5)
        return output
        
    def get_user_taste(self, analyzed_user):
        user_taste = {}
        for genre, coef in zip(self.genre, analyzed_user.coef_):
            user_taste[genre] = coef
        return user_taste   

    def get_prefer_genres(self, analysed_user, num_genre=5):
        prefer_genres = []
        user_taste = self.get_user_taste(analysed_user)
        key_list = list(user_taste.keys())
        val_list = list(user_taste.values())

        if num_genre > len(self.genre):
            return "No.Genre is way too much!!!"
        else:
            sorted_coef = sorted(val_list.copy())[::-1]
            genres_coef = sorted_coef[:num_genre]
            for genre_coef in genres_coef:
                position = val_list.index(genre_coef)
                prefer_genres.append(key_list[position])
            return prefer_genres
        
    def recommend(self, n_films=20, threshhold=3):
        """
            1. Chọn riêng rẽ (theo tổ hợp) các phim có thể loại dựa trên thông tin của người dùng đã được xử lý.
            2. Với mỗi tổ hợp, chọn ra n phim để đề xuất cho người dùng. 
        """
        recommend_idx = []
        if threshhold > 5:
            raise "INVALID THRESH-HOLD"
        user_prefer_genres = self.get_prefer_genres(self.analyzed_user(self.user_prefer_list))
        chosen_films = self.movie_one_hot[user_prefer_genres]
        for idx in range(len(chosen_films)):
            film_info = chosen_films.iloc[idx, :]
            if film_info.sum() < threshhold:
                continue
            else:
                recommend_idx.append(idx + 1)
        recommend_idx = np.random.choice(recommend_idx, n_films)
        mask = self.movies.MovieID.isin(recommend_idx)
        recommend_movies = self.movies.loc[mask]
        return recommend_movies