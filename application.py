from flask import Flask, request, render_template
import requests
import pandas as pd
import pickle
from bs4 import BeautifulSoup
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk

app = Flask(__name__)
model = pickle.load(open('nb_model.pkl', 'rb'))
cv = pickle.load(open('vectorizer.pkl', 'rb'))
ps = PorterStemmer()
movies = pd.read_csv('IMDb movies.csv')
names = movies[['imdb_title_id', 'title', 'year']]
nltk.download('stopwords')
stopwords.words('english')


# Text pre-processing functions
def convert_lower(text):
    return text.lower()


def remove_special(text):
    x = ''

    for i in text:
        if i.isalnum():
            x = x + i
        else:
            x = x + ' '
    return x


def remove_stopwords(text):
    x = []
    for i in text.split():

        if i not in stopwords.words('english'):
            x.append(i)
    y = x[:]
    x.clear()
    return y


def stem_words(text):
    y = []
    for i in text:
        y.append(ps.stem(i))
    z = y[:]
    y.clear()
    return z


def join_back(list_input):
    return " ".join(list_input)


def predict_reviews(reviews):
    predicted_result = []
    for review in reviews:
        # Processing the review to our desired format
        processed_review = join_back(stem_words(remove_stopwords(remove_special(convert_lower(review.text)))))

        # Predicting the review
        predicted_result.append([review.text, model.predict(cv.transform([processed_review]))[0]])

    predicted_result = np.array(predicted_result)

    # Generating the ratio of positive reviews to all the reviews and multiplying it by 10
    positive = (predicted_result[:, 1] == '1').sum() / len(predicted_result) * 10
    return predicted_result, positive


def search_movie(search):
    # Extracting all the words
    search = search.strip().split()

    # Creating random substrings
    searchsub = []

    for word in search:
        if len(word) > 4:
            for i in range(int(0.8 * len(word))):
                low = np.random.randint(low=0, high=len(word))
                high = np.random.randint(low=low, high=len(word))
                if high - low <= 0.8 * len(word):
                    searchsub.append(word[low:high + 1].lower())
        else:
            low = 0
            high = len(word) - 1
            searchsub.append(word[low:high + 1].lower())

    # Sorting the substrings based on their lengths: Lowest length first
    lengths = []
    searchsub = np.array(searchsub)
    for sub in searchsub:
        lengths.append(len(sub))

    searchsub = searchsub[np.array(lengths).argsort()]

    # Finding all the rows that match the substrings
    results = pd.DataFrame(names[names['title'].str.contains(searchsub[0], case=False)], columns=names.columns)

    for subs in searchsub:
        new = names[names['title'].str.contains(subs, case=False)]
        if new.shape[0] != 0:
            results = pd.merge(results, new, how='inner')
    return results.sort_values('year', ascending=False)


def retrive_reviews(movie_id):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.162 Safari/537.36'}
    webpage = requests.get('https://www.imdb.com/title/{}/reviews'.format(movie_id), headers=headers).text
    soup = BeautifulSoup(webpage, 'lxml')

    # Extracting the reviews
    reviews = soup.find_all('div', class_='text show-more__control')

    # Extracting the poster url
    poster_url = soup.find_all('img', class_='poster')[0]['src']
    return reviews, poster_url


@app.route('/')
def index():
    return render_template('index.html', search=0, search_results=[])


@app.route('/movie', methods=['GET'])
def show_movie():
    movie_id = request.args.get('movie_id')

    # Retrives the reviews from the IMDB page with the movie_id
    reviews, poster_url = retrive_reviews(movie_id)

    # The problem with the poster url that was extracted was: it was low quality. Accidentally found out a new url which is a modification of this one to give better resolution poster
    poster_url.replace('m.media-', 'www.')
    poster_url = poster_url[:-4]

    # Getting Predictions
    reviews, positive = predict_reviews(reviews)

    # Getting out the movie details
    movie_details = \
    movies[movies['imdb_title_id'] == movie_id][['title', 'avg_vote', 'duration', 'genre', 'date_published']].values[0]

    return render_template('show_movie.html', generated_rating=positive, movie_details=movie_details, reviews=reviews,
                           poster=poster_url)


@app.route('/search', methods=['POST'])
def search_route():
    search = request.form.get('search')

    search_results = search_movie(search).values

    return render_template('index.html', search=1, search_results=search_results)


if __name__ == '__main__':
    app.run()
