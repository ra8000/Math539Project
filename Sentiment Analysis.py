# MAT 539 Chipotle Consulting Group

# Import Libraries -----------------------------------------------------------------------------------------------------
import pandas as pd
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from textblob import TextBlob  # Note: May need to pip install textblob
import seaborn as sns
import plotly.express as px
import plotly.io as pio

pio.renderers.default = "svg"

# Switches -------------------------------------------------------------------------------------------------------------
first_run = False  # If running for the first time, set True. Else, set False.

# Preliminary Steps ----------------------------------------------------------------------------------------------------
# WARNING: MAY TAKE A WHILE!!!!
if first_run:
    nltk.download("all")  # downloads nltk packages

# Import Data ----------------------------------------------------------------------------------------------------------
data_path = r"C:\Users\Paolo\Documents\School\CSUF\MAT 539\Data"  # file path to data
data_year = "SinceJan" + "2018"  # year to go back to when importing

# restaurant data
restaurant_data = pd.read_csv(data_path + r"\RestaurantData.csv", encoding="unicode_escape")

# review data
foursquare_data = pd.read_csv(data_path + r"\FoursquareReviews" + data_year + ".csv", encoding="unicode_escape")
google_data = pd.read_csv(data_path + r"\GoogleMyBusiness" + data_year + ".csv", encoding="unicode_escape")
tripadvisor_data = pd.read_csv(data_path + r"\TripAdvisorReviews" + data_year + ".csv", encoding="unicode_escape")
yelp_data = pd.read_csv(data_path + r"\YelpReviews" + data_year + ".csv", encoding="unicode_escape")
facebook_data = pd.read_csv(data_path + r"\FacebookReviews" + data_year + ".csv", encoding="unicode_escape")

# Initial Cleaning of Data ---------------------------------------------------------------------------------------------
# restaurant data cleaning
restaurant_data.drop_duplicates(subset="RestaurantNumber", keep="first", inplace=True)  # drop duplicates
restaurant_data["RestaurantNumber"] = restaurant_data["RestaurantNumber"].apply(str)  # convert restaurant # to string
restaurant_data.fillna({'Chipotlane': 'False'}, inplace=True)  # replace NaN Chipotlane with False

# review data cleaning
review_dfs = [foursquare_data, google_data, tripadvisor_data, yelp_data, facebook_data]  # review data
review_data = pd.concat(review_dfs, sort=True)  # consolidate review data
review_data = review_data[review_data["Name"] != "Pizzeria Locale"]  # remove Pizzeria Locale
review_data = review_data[(review_data["Language"] == "English") | (pd.isnull(review_data["Language"]))]  # eng only
review_data = review_data[pd.notnull(review_data["Review"])]  # remove NaN reviews
review_data.reset_index(drop=True, inplace=True)
review_data["ID"] = review_data["ID"].apply(str)  # convert review ID # to string

# Join Restaurant Data with Review Data --------------------------------------------------------------------------------
review_data.rename(columns={'Store ID': 'RestaurantNumber'}, inplace=True)  # rename column for joining
complete_data = restaurant_data.merge(review_data, on="RestaurantNumber", how="left")  # join data

# Text Processing ------------------------------------------------------------------------------------------------------
text_data = complete_data.copy()  # make a copy of original data for text processing
text_data = text_data[["ID", "Review"]]
text_data["Review"].fillna("", inplace=True)  # Fill NaN reviews with blanks

exclude = stopwords.words('english')  # stopwords to exclude

text_data["Review"] = text_data["Review"].str.lower()  # make all text lowercase
text_data["Review"] = text_data["Review"].str.replace(r'[^\w\s\.]+', '')  # remove punctuation
text_data["Review"] = text_data["Review"].apply(
    lambda x: " ".join([item for item in x.split() if item not in exclude]))  # remove stopwords and other exclusions
text_data["Review"] = text_data["Review"].apply(lambda x: sent_tokenize(x))  # tokenize text data
text_data = text_data.explode("Review")  # explode review column
text_data["Review"] = text_data["Review"].str.replace(".", "")  # now that we have already tokenized, remove periods

# Things to consider: Removal of common/rare words, spelling correction, stemming and lemmatization

# Sentiment Analysis ---------------------------------------------------------------------------------------------------
text_data["Review"].fillna("", inplace=True)
test_sample = text_data.sample(n=10, random_state=2)  # sample to test sentiment
test_sample["Sentiment"] = test_sample["Review"].apply(lambda x: TextBlob(x).sentiment[0])  # detect sentiment
print(test_sample)

text_data["Polarity"] = text_data["Review"].apply(lambda x: TextBlob(x).sentiment[0])  # sentiment: polarity
# text_data["Subjectivity"] = text_data["Review"].apply(lambda x: TextBlob(x).sentiment[1])  # sentiment: subjectivity
sns.distplot(text_data["Polarity"])  # histogram of polarity
# sns.distplot(text_data["Subjectivity"])

print(text_data[text_data["Polarity"] == 1])  # examine reviews with polarity score = 1
print(text_data[text_data["Polarity"] == -1])  # examine reviews with polarity score = 1
print(text_data[text_data["Polarity"] == 0])  # examine reviews with polarity score = 0


# Plotting -------------------------------------------------------------------------------------------------------------
sns.set_style("ticks")
sns.distplot(text_data["Polarity"])
