import pandas as pd
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk import FreqDist


# Importing data
foursquare_data_df= pd.read_csv(filepath_or_buffer= os.path.join('Data','FoursquareReviewsSinceJan2019.csv'), encoding= 'unicode_escape')
google_data_df= pd.read_csv(filepath_or_buffer= os.path.join('Data','GoogleMyBusinessSinceJan2019.csv'), encoding= 'unicode_escape')
tripadvisor_data_df= pd.read_csv(filepath_or_buffer= os.path.join('Data','TripAdvisorReviewsSinceJan2019.csv'), encoding= 'unicode_escape')
yelp_data_df= pd.read_csv(filepath_or_buffer= os.path.join('Data','YelpReviewsSinceJan2019.csv'), encoding= 'unicode_escape')
restaurant_data_df= pd.read_csv(filepath_or_buffer= os.path.join('Data','RestaurantData.csv'), encoding= 'unicode_escape')
# Remove duplicates from Chipotle restaurant file:
restaurant_data_df.drop_duplicates(subset= "RestaurantNumber", keep='first', inplace= True)

# Consolidate reviews data
total_review_data_df = pd.concat([foursquare_data_df, google_data_df, tripadvisor_data_df, yelp_data_df], ignore_index= False)

# Subset of review data: Only language of English and "blanks" with actual review text
complete_review_data_df = total_review_data_df[
    ((total_review_data_df['Language']=="English") | (pd.isnull(total_review_data_df['Language'])))
    & (pd.notnull(total_review_data_df['Review'])) & (total_review_data_df['Name']!="Pizzeria Locale")].reset_index(drop=True)

# Append review counts to Restaurant data
# Attain counts of reviews by store id
review_counts_df= complete_review_data_df.groupby('Store ID', as_index= False).agg('count').loc[:,['Store ID','Review']].astype({'Store ID': 'int64'})

restaurant_data_df= pd.merge(restaurant_data_df, review_counts_df, left_on="RestaurantNumber", right_on="Store ID", how='left')\
    .drop(columns='Store ID').rename(columns= {"Review": "Review_Count"})\
    .fillna({"Review_Count": 0})\
    .sort_values(by=['Review_Count'], ascending=True, na_position='first')


# count of binned reviews
review_count_summary_df= restaurant_data_df.groupby(pd.cut(restaurant_data_df.Review_Count,\
    [-1, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,50,100, 150, 200, 300]), as_index= True)\
    .count().loc[:, ['Review_Count']].rename(columns= {"Review_Count" : "Interval_Counts"})

print(review_count_summary_df)

# Find restaurants in Review data but not in Restaurant data:
missing_restaurants= pd.merge(restaurant_data_df, review_counts_df, left_on="RestaurantNumber",\
    right_on="Store ID", how='right', indicator=True)\
    .drop(columns='RestaurantNumber').rename(columns= {"Review": "Review_Count2"})\
    .fillna({"Review_Count": 0})\
    .sort_values(by=['Review_Count'], ascending=True, na_position='first')
missing_restaurants= missing_restaurants[missing_restaurants._merge == "right_only"].loc[:,['Store ID', 'Review_Count2']]
print(missing_restaurants)

#complete_review_data_df.to_csv(r'C:\Users\estev\OneDrive - Cal State Fullerton\Documents\Work-School Docs\CSUF\Math 539 Statistical Consulting\Project\Data\Python Exports\complete_review_data.csv', index=False)


#------- Text parsing

# subset of data, top restaurant with most reviews
sub_data = complete_review_data_df[complete_review_data_df['Store ID'] == restaurant_data_df.tail(1)['RestaurantNumber'].to_string(index=False).strip()]['Review']

# regex for removing punctuation
REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

# function to remove punctuation
def preprocess_reviews(reviews):
    reviews = [REPLACE_NO_SPACE.sub("", line.lower()) for line in reviews]
    reviews = [REPLACE_WITH_SPACE.sub(" ", line) for line in reviews]

    return reviews

# Data with removed punctuation
reviews_train_clean = preprocess_reviews(sub_data)


# variable of stopwords per nlk stopwords and function to remove the stopwords from our data
english_stop_words = stopwords.words('english')
def remove_stop_words(corpus):
    removed_stop_words = []
    for review in corpus:
        removed_stop_words.append(
            ' '.join([word for word in review.split()
                      if word not in english_stop_words])
        )
    return removed_stop_words

# Data without punctuation or stopwords
no_stop_words = remove_stop_words(reviews_train_clean)


# Function to lemmatize the data
def get_lemmatized_text(corpus):
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    return [' '.join([lemmatizer.lemmatize(word) for word in review.split()]) for review in corpus]

# Data lemmatized without punctuation and stopwords
lemmatized_reviews = get_lemmatized_text(no_stop_words)


# Function not working, but supposed to separate all words in reviews for counting
def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token

# Not working, aggregating all words for counting
all_words = get_all_words(lemmatized_reviews)





freq_dist_pos = FreqDist(all_words)
print(freq_dist_pos.most_common(100))


from nltk.probability import ConditionalFreqDist
from nltk.tokenize import word_tokenize
cfdist = ConditionalFreqDist((len(word), word)
                             for text in lemmatized_reviews
                                for word in word_tokenize(text))
print(cfdist)