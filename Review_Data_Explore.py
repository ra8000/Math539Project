import pandas as pd
import numpy as np
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk import FreqDist
import pickle


# Importing data
foursquare_data_df= pd.read_csv(filepath_or_buffer= os.path.join('Data','FoursquareReviewsSinceJan2018.csv'), encoding= 'unicode_escape')
google_data_df= pd.read_csv(filepath_or_buffer= os.path.join('Data','GoogleMyBusinessSinceJan2018.csv'), encoding= 'unicode_escape')
tripadvisor_data_df= pd.read_csv(filepath_or_buffer= os.path.join('Data','TripAdvisorReviewsSinceJan2018.csv'), encoding= 'unicode_escape')
yelp_data_df= pd.read_csv(filepath_or_buffer= os.path.join('Data','YelpReviewsSinceJan2018.csv'), encoding= 'unicode_escape')
facebook_data_df= pd.read_csv(filepath_or_buffer= os.path.join('Data','FacebookReviewsSinceJan2018.csv'), encoding= 'unicode_escape')
restaurant_data_df= pd.read_csv(filepath_or_buffer= os.path.join('Data','RestaurantData.csv'), encoding= 'unicode_escape')
# Remove duplicates from Chipotle restaurant file:
restaurant_data_df.drop_duplicates(subset= "RestaurantNumber", keep='first', inplace= True)

# Consolidate reviews data
total_review_data_df = pd.concat([foursquare_data_df, google_data_df, tripadvisor_data_df, yelp_data_df, facebook_data_df], ignore_index= False, sort=True)

# Subset of review data: Only language of English and "blanks" with actual review text
complete_review_data_df = total_review_data_df[
    ((total_review_data_df['Language']=="English") | (pd.isnull(total_review_data_df['Language'])))
    & (pd.notnull(total_review_data_df['Review'])) & (total_review_data_df['Name']!="Pizzeria Locale")].reset_index(drop=True)

# Append review counts to Restaurant data
# Attain counts of reviews by store id
review_counts_df= complete_review_data_df.astype({'Store ID': 'int64'}).groupby('Store ID', as_index= False).agg('count').loc[:,['Store ID','Review']]

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
# test_sub_data = complete_review_data_df[complete_review_data_df['Store ID'] == restaurant_data_df.tail(1)['RestaurantNumber'].to_string(index=False).strip()][['Review', 'Store ID']]


# regex for removing punctuation
REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

# function to remove punctuation
def preprocess_reviews(reviews):
    reviews = [REPLACE_NO_SPACE.sub("", line.lower()) for line in reviews]
    reviews = [REPLACE_WITH_SPACE.sub(" ", line) for line in reviews]

    return reviews



# Data with removed punctuation
reviews_train_clean = preprocess_reviews(complete_review_data_df["Review"])



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


# Function to remove numbers
def digit_removal(reviews):
    reviews = [re.sub(r"\d+","", line) for line in reviews]
    return reviews

# Removed digits from lemmatized data
digits_removed_data = digit_removal(lemmatized_reviews)


# Separates all words in reviews for counting
def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens.split():
            yield token

# Aggregating all words for counting
all_words = get_all_words(digits_removed_data)




freq_dist_pos = FreqDist(all_words)
print(freq_dist_pos.most_common(100))






#------- TF-IDF feature extraction

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(digits_removed_data)


# Running kmeans using saved model (need to also have .pkl file modelkmeans_k4

# Saved the trained model as pickle file
from sklearn.externals import joblib
# Save the model as a pickle in a file
##joblib.dump(modelkmeans, 'modelkmeans_k4.pkl')

# Load the model from the file, this is k-means with k=4
model_kmeans_k4_joblib = joblib.load('modelkmeans_k4.pkl')

assumed_k = 4
#Only run if needed for kmeans other than saved model of k=4 above
##modelkmeans = KMeans(n_clusters= assumed_k, init='k-means++', max_iter=200, n_init=50)
##modelkmeans.fit(X)


order_centroids = model_kmeans_k4_joblib.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()



# Showing cluster and word buckets
for i in range(assumed_k):
 print("Cluster %d:" % i),
 for ind in order_centroids[i, :10]:
    print(' %s' % terms[ind])



#kmeans prediction (for testing only)
test_predict = vectorizer.transform(["Food portion getting littt price, not good."])
predicted = model_kmeans_k4_joblib.predict(test_predict)
print(predicted)


# Only use below chunk for finding optimum k, takes a long time
# check for optimum k in k-means
##sum_of_squared_distances = []
##K = range(1 , 10)
##for k in K:
##    km = KMeans(n_clusters=k)
##    km = km.fit(X)
##    print(k)
##    sum_of_squared_distances.append(km.inertia_)

# plot optimal k using elbow method
##import matplotlib.pyplot as plt
##plt.plot(K, sum_of_squared_distances, 'bx-')
##plt.xlabel('k')
##plt.ylabel('Sum_of_squared_distances')
##plt.title('Elbow Method For Optimal k')
##plt.show()



#--- Predictsion of k-means
sub_complete_reviews_df = complete_review_data_df[['Review', 'Store ID']]

# Explode/Split column into multiple rows
# split into sentences
def sentence_split_data(corpus):
    sentences = []
    for review in corpus:
        from nltk import sent_tokenize
        sentences.append(sent_tokenize(review))
    return sentences


# exploding columns into multiple rows by sentences in each review
new_df = pd.DataFrame(sentence_split_data(sub_complete_reviews_df.Review), index= sub_complete_reviews_df['Store ID']).stack()
new_df = new_df.reset_index([0, 'Store ID'])
new_df.columns = ['Store ID', 'Review']

#kmeans prediction on full dataset by sentence
test_predict = vectorizer.transform(new_df['Review'])
predicted = model_kmeans_k4_joblib.predict(test_predict)
print(predicted)
new_df["pred_lin_regr"] = model_kmeans_k4_joblib.predict(test_predict)

# kmeans prediction frequencies
unique_elements, counts_elements = np.unique(predicted, return_counts=True)
print("Frequency of unique values of the said array:")
print(np.asarray((unique_elements, counts_elements)))



tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 3), stop_words='english')
features = tfidf.fit_transform(new_df.Review)
labels = new_df['pred_lin_regr']
features.shape



from sklearn.feature_selection import chi2

N = 3
for category_id in sorted(new_df.pred_lin_regr.unique()):
  features_chi2 = chi2(features, labels == category_id)
  indices = np.argsort(features_chi2[0])
  feature_names = np.array(tfidf.get_feature_names())[indices]
  unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
  bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
  trigrams = [v for v in feature_names if len(v.split(' ')) == 3]
  print("# '{}':".format(category_id))
  print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))
  print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-N:])))
  print("  . Most correlated trigrams:\n. {}".format('\n. '.join(trigrams[-N:])))
