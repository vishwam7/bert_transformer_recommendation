import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import json
import ast
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import faiss
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
import numpy as np
import faiss

df_meta = pd.read_csv("movies_metadata.csv")
df_meta = df_meta[['id', 'genres', 'original_language', 'production_countries', 'tagline', 'original_title', 'adult', 'release_date', 'status']]

df_meta.status.unique()

for i in range(len(df_meta)):
    tmp = df_meta['status'][i]
    if (tmp == "Released"):
        df_meta['status'][i] = "Released"
    else:
        df_meta['status'][i] = ''

df_meta['genres'][0]

for i in range(len(df_meta['genres'])):
    str_tmp = ""
    genre = df_meta['genres'][i]
    genre = genre.replace("\'", "\"")
    json_genre = json.loads(genre)
    for j in range(len(json_genre)):
        str_tmp += (json_genre[j]['name']) + " "
    df_meta['genres'][i] = str_tmp

df_meta['production_countries'][0]

df_meta['production_countries'].replace(np.nan, '', inplace=True)

for i in range(len(df_meta['production_countries'])):
    str_tmp = ""
    country = df_meta['production_countries'][i]
    if (country != ''):
        country = json.dumps(ast.literal_eval(country))
        json_country = json.loads(country)
        try:
            for j in range(len(json_country)):
                str_tmp += (json_country[j]['name'])
            df_meta['production_countries'][i] = str_tmp
        except:
            pass

df_keyword = pd.read_csv('keywords.csv')

for i in range(len(df_keyword['keywords'])):
    str_tmp = ""
    keyword = df_keyword['keywords'][i]
    keyword = json.dumps(ast.literal_eval(keyword))
    json_keyword = json.loads(keyword)
    for j in range(len(json_keyword)):
        str_tmp += (json_keyword[j]['name']) + " "
    df_keyword['keywords'][i] = str_tmp

df_credit = pd.read_csv('credits.csv')
df_credit = df_credit.rename(columns=({'crew': 'director'}))

for i in range(len(df_credit['cast'])):
    str_tmp = ""
    credit = df_credit['cast'][i]
    names = credit.split()
    str_tmp = " ".join(names)
    df_credit['cast'][i] = str_tmp

for i in range(len(df_credit['director'])):
    str_tmp = ""
    director = df_credit['director'][i]
    director = json.dumps(ast.literal_eval(director))
    json_director = json.loads(director)
    for j in range(len(json_director)):
        if json_director[j]['job'] == 'Director':
            str_tmp += (json_director[j]['name']) + " "
    df_credit['director'][i] = str_tmp

df_meta['id'] = df_meta['id'].astype(str)
df_keyword['id'] = df_keyword['id'].astype(str)

df_merge = pd.merge(df_keyword, df_meta, on='id', how='inner')[['id', 'genres', 'original_language', 'production_countries', 'tagline',
       'original_title', 'keywords', 'adult', 'release_date', 'status']]

df_credit['id'] = df_credit['id'].astype(str)

df_merge_whole = pd.merge(df_merge, df_credit, on='id', how='inner')[['id', 'genres', 'original_language', 'production_countries', 'tagline',
       'original_title', 'keywords', 'cast', 'director', 'adult', 'release_date', 'status']]

df_merge_whole['keywords'].replace('', np.nan, inplace=True)

df_merge_whole['genres'].replace('', np.nan, inplace=True)
df_merge_whole['original_title'].replace('', np.nan, inplace=True)
df_merge_whole['cast'].replace('', np.nan, inplace=True)
df_merge_whole['director'].replace('', np.nan, inplace=True)
df_merge_whole['release_date'].replace('', np.nan, inplace=True)
df_merge_whole['status'].replace('', np.nan, inplace=True)
df_merge_whole['production_countries'].replace(np.nan, '', inplace=True)
df_merge_whole['adult'].replace(np.nan, '', inplace=True)
df_merge_whole['tagline'].replace(np.nan, '', inplace=True)
df_merge_whole = df_merge_whole.dropna()

df_merge_whole.to_csv('filter_data.csv')
df = pd.read_csv('filter_data.csv')

def combine_features(row):
    return row['original_title']+' '+row['genres']+' '+ row['original_language']+' '+row['director']+' '+row['keywords']+' '+row['cast']+' '+row['tagline']+' '+row['production_countries']

df['combined_value'] = df.apply(combine_features, axis = 1)

df['index'] = [i for i in range(0, len(df))]

!pip install sentence_transformers

from sentence_transformers import SentenceTransformer
bert = SentenceTransformer('bert-base-nli-mean-tokens')

sentence_embeddings = bert.encode(df['combined_value'].tolist())

from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(sentence_embeddings)

movie_name = 'The Internship'
movie_recommendation = sorted(list(enumerate(similarity[df['index'][df['original_title'] == movie_name].values[0]])), key = lambda x:x[1], reverse = True)

print(title(movie_recommendation[1][0]), title(movie_recommendation[2][0]), title(movie_recommendation[3][0]), title(movie_recommendation[4][0]), title(movie_recommendation[5][0]), sep = "\n")

from gensim.models import Word2Vec

tokenized_text = [text.split() for text in df['combined_value'].tolist()]
word2vec_model = Word2Vec(sentences=tokenized_text, vector_size=100, window=5, min_count=1, workers=4)
word2vec_model.save('word2vec_model.model')

def get_word2vec_embedding(row):
    words = row['combined_value'].split()
    embedding = np.zeros(word2vec_model.vector_size)
    count = 0
    for word in words:
        if word in word2vec_model.wv:
            embedding += word2vec_model.wv[word]
            count += 1
    if count != 0:
        embedding /= count
    return embedding

df['word2vec_embedding'] = df.apply(get_word2vec_embedding, axis=1)

similarity_word2vec = cosine_similarity(df['word2vec_embedding'].tolist())

movie_name = 'Inception'
idx_movie_recommendation_word2vec = sorted(list(enumerate(similarity_word2vec[df['index'][df['original_title'] == movie_name].values[0]])), key=lambda x: x[1], reverse=True)

print(title(idx_movie_recommendation_word2vec[1][0]), title(idx_movie_recommendation_word2vec[2][0]),
      title(idx_movie_recommendation_word2vec[3][0]), title(idx_movie_recommendation_word2vec[4][0]),
      title(idx_movie_recommendation_word2vec[5][0]), sep="\n")

from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import faiss

word2vec_model = Word2Vec.load('word2vec_model.model')
word2vec_embeddings = np.array([get_word2vec_embedding(row) for _, row in df.iterrows()])
index_word2vec = faiss.IndexFlatL2(word2vec_embeddings.shape[1])
index_word2vec.add(word2vec_embeddings)

def visualize_nearby_movies_word2vec(movie_name, num_neighbors=25):
    movie_index = df[df['original_title'] == movie_name].index[0]
    query_embedding_word2vec = word2vec_embeddings[movie_index].reshape(1, -1)
    _, neighbor_indices_word2vec = index_word2vec.search(query_embedding_word2vec, num_neighbors + 1)

    nearby_movies_word2vec = df.loc[neighbor_indices_word2vec[0], ['original_title', 'combined_value']]

    pca_word2vec = PCA(n_components=2)
    reduced_embeddings_word2vec = pca_word2vec.fit_transform(word2vec_embeddings[neighbor_indices_word2vec[0]])

    plt.figure(figsize=(10, 8))
    plt.scatter(reduced_embeddings_word2vec[:, 0], reduced_embeddings_word2vec[:, 1], color='blue', label='Nearby Movies')
    plt.scatter(reduced_embeddings_word2vec[0, 0], reduced_embeddings_word2vec[0, 1], color='red', marker='x', s=100, label='Input Movie')
    plt.title(f'PCA Visualization of Movies Nearby {movie_name} (Word2Vec)')

    for i, txt in enumerate(nearby_movies_word2vec['original_title']):
        plt.annotate(txt, (reduced_embeddings_word2vec[i, 0], reduced_embeddings_word2vec[i, 1]),
                     fontsize=8, ha='right', va='bottom' if i % 2 == 0 else 'top')

    plt.legend()
    plt.show()

movie_name_input_word2vec = "Inception"
visualize_nearby_movies_word2vec(movie_name_input_word2vec)