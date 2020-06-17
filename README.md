
# Movie Sentiment Analyzer

<img src="https://github.com/rajtilakls2510/MovieSentimentAnalyzer/blob/master/demo_image.png">

Video Demostration: 

Project Link : https://movie-sentiment-analyzer.herokuapp.com

## Aim

This project aims to analyze the sentiments of the reviews a movie and tell the user how good the movie is.

### How to use?

1. Clone the repository
2. Install the required packages in "requirements.txt" file.
 Some packages are:

numpy
pandas
scikit-learn

3. Run the "application.py" file

And you are good to go.


## Description:

#### What this project does?

1. When you open the website, it asks for you to search for a movie.

<img src="https://github.com/rajtilakls2510/MovieSentimentAnalyzer/blob/master/search.png">

2. It presents you with a list of movies in the descending order of their release years.

<img src="https://github.com/rajtilakls2510/MovieSentimentAnalyzer/blob/master/search_result.png">

3. You select one of them and it gives you back a page where there are different reviews from the IMDB website.

<img src="https://github.com/rajtilakls2510/MovieSentimentAnalyzer/blob/master/search.png">
4. Those reviews are classified into positive and negative sentiment. Positive reviews are placed in Green Card and the negative ones are placed in Red Card.
5. It also gives you a rating in 10 on the basis of the number of positive reviews.

#### How it works?

1. First of all a Naive Bayes model was trained on the dataset whose link is given below
https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

2. The Project has the data for 80000 movies with their IMDB data.

Link for data:  https://www.kaggle.com/stefanoleone992/imdb-extensive-dataset

3. When you search for a movie, it generates random search strings and tries to find the movie with the best match.

4. When you select one of the movies, it goes to the IMDB user reviews page of movie and retrieves all the reviews that are present on the page through web scraping.

5. It then classifies all the reviews to be either positive or negative and presents it to us.

6. It also gives us a rating in 10 which is the ratio of positive reviews to total reviews scaled to 10. 
 
