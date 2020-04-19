# MAT 539 Chipotle Consulting Group

# Import Libraries -----------------------------------------------------------------------------------------------------
import pandas as pd
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from textblob import TextBlob  # Note: May need to pip install textblob
import matplotlib.pyplot as plt
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
# exploded_data = pd.read_csv(data_path + r"\exploded_review_data.csv", encoding="unicode_escape")
exploded_data = pd.read_csv(data_path + r"\exploded_categorized_polarity2.csv", encoding="unicode_escape")

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
text_data = text_data[text_data["Polarity"] != 0]
sns.distplot(text_data["Polarity"])  # histogram of polarity

# Plotting -------------------------------------------------------------------------------------------------------------
sns.set_style("ticks")
sns.distplot(text_data["Polarity"])

# class example
test = review_data[["ID", "Review"]][review_data["ID"] == "1031030542"]
test["Review"] = 'The food is bomb. The staff are cool. This place is very clean.'
test["Change"] = test["Review"]
test["Change"] = test["Change"].str.lower()
test["Change"] = test["Change"].str.replace(r'[^\w\s\.]+', '')  # remove punctuation
test["Change"] = test["Change"].apply(
    lambda x: " ".join([item for item in x.split() if item not in exclude]))  # remove stopwords and other exclusions
test["Change"] = test["Change"].apply(lambda x: sent_tokenize(x))  # tokenize text data
test = test.explode("Change")  # explode review column
test["Change"] = test["Change"].str.replace(".", "")  # now that we have already tokenized, remove periods

test["Polarity"] = test["Change"].apply(lambda x: TextBlob(x).sentiment[0])  # sentiment: polarity

# exploded_data.to_csv(data_path + r"\exploded_review_data_w_polarity3.csv", index=False, header=True)

# Visualizations -------------------------------------------------------------------------------------------------------
# sentiment for exploded data set
exploded_data["StructuredReview"] = exploded_data["Review"].str.lower()
exploded_data["StructuredReview"] = exploded_data["StructuredReview"].str.replace(r'[^\w\s]+', '')
exploded_data["StructuredReview"] = exploded_data["StructuredReview"].apply(
    lambda x: " ".join([item for item in x.split() if item not in exclude]))  # remove stopwords and other exclusions
exploded_data["StructuredReview"] = exploded_data["StructuredReview"].str.replace(".", "")  # remove periods
exploded_data["Polarity"] = exploded_data["StructuredReview"].apply(lambda x: TextBlob(x).sentiment[0])
neutral_band = 0
exploded_data["Sentiment"] = np.where(exploded_data["Polarity"] > neutral_band, "Positive",
                                      np.where(exploded_data["Polarity"] < -neutral_band, "Negative", "Neutral"))

x, y, hue = "HighestTopic", "Proportion", "Sentiment"
x_order = ["Topic" + str(i) for i in range(5)]
hue_order = ["Negative", "Neutral", "Positive"]
palette = {'Negative': 'r', 'Neutral': 'y', 'Positive': 'g'}

polarity_by_topic = (exploded_data[hue]
 .groupby(exploded_data[x])
 .value_counts(normalize=True)
 .rename(y)
 .reset_index()
 .sort_values(x, ascending=True)
 .reset_index()
 .drop("index", axis=1))

polarity_by_topic[y] = polarity_by_topic[y].apply(lambda x: x*100)

plt.figure()
ax = sns.barplot(x=x, y=y, hue=hue, palette=palette, data=polarity_by_topic, hue_order=hue_order)
ax.set_title("Proportion of Sentiment Scores by Topic", fontsize=40)
ax.set_ylabel("")
ax.set_xlabel("")
ax.tick_params(labelsize=30)
for p in ax.patches:
    ax.annotate("%.2f%%" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 20), textcoords='offset points', size=30)
plt.legend(fontsize='30')
plt.show()

i = 0
n_terms = 10
for topic in x_order:
    topic_subset = exploded_data[exploded_data["HighestTopic"] == topic].copy()
    for sign, color in [("Positive", "g"), ("Negative", "r")]:
        sign_subset = topic_subset["StructuredReview"][exploded_data["Sentiment"] == sign].copy()
        term_proportions = sign_subset.value_counts(normalize=True).rename(y).reset_index()
        term_proportions.sort_values(y, ascending=False).reset_index().drop("index", axis=1)
        term_proportions.rename({"index": "Review"}, axis=1, inplace=True)
        plt.figure()
        ax = sns.barplot(x=y, y="Review", data=term_proportions[1:n_terms], color=color)
        sns.despine(left=True, bottom=True)
        ax.set_title(f"Proportion of Top {n_terms} Terms From All {sign} Reviews For Topic {i}", fontsize=40)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(labelsize=30)
        plt.show()
    i = i + 1
