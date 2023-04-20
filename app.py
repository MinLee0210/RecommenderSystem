from lib import *

'''
    CONFIGURATION
'''
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

n_films = 10
def get_data_filepath(name):
    file_path = str(ROOT) + f'\dataset\{name}.csv'
    return file_path
movies = pd.read_csv(get_data_filepath('movies'), index_col=None).drop(columns=['Unnamed: 0']).sample(frac=1)
movie_samples = movies.iloc[: n_films, 1: 3]
movie_titles = movie_samples.Title
user_prefer_list = {
    'Titles': [],
    'Ratings': []
}

'''
    APPLICATION
'''
st.title("RECOMMENDER SYSTEM")
st.divider()

st.header('PLEASE RANGING BELOW MOVIES')


# st.divider()
# st.header('WE RECOMMEND YOU')