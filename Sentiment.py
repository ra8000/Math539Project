# Sentiment Tagging
# Packages -------------------------------------------------------------------------------------------------------------
import pandas as pd

import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from textblob import TextBlob  # Note: May need to pip install textblob

# Switches -------------------------------------------------------------------------------------------------------------
first_run = False  # If running for the first time, set True. Else, set False.
show_example = True  # use if you want to see example

# Preliminary Steps ----------------------------------------------------------------------------------------------------
# WARNING: May take some time.
if first_run:
    nltk.download("all")  # downloads nltk packages


# Functions ------------------------------------------------------------------------------------------------------------
def processing_example(review='The food is bombay. The staff are cool. This place is very clean.'):
    """
    Construct a DataFrame showcasing the step-by-step process for cleaning a SINGLE review.
    Please note that this function handles processing slightly differently from the main processing function as this
    function is intended to look at raw reviews that has NOT been through the tokenization and explosion steps.
    This function is intended for demonstration purposes and previews.

    :param review: string
        Review that you would like to see the processing steps for.
        Default value is the example shown in the presentation.

    :return: DataFrame
        First column is the step, second column is the result after applying the step.
    """

    exclude = stopwords.words('english')  # stopwords to exclude; add to this list to add more exclusions

    # step 0: original review
    df = pd.DataFrame({'Step': ['0: Original'], 'Review': [review]})

    # step 1: convert review to lowercase
    review = review.lower()
    step_1 = pd.Series(['1: Lowercase', review], index=df.columns)

    # step 2: remove punctuation
    review = re.sub(r'[^\w\s.!?]+', '', review)
    step_2 = pd.Series(['2: Remove Punctuation', review], index=df.columns)

    # step 3: remove stopwords
    review = " ".join([item for item in review.split() if item not in exclude])
    step_3 = pd.Series(['3: Remove Stopwords', review], index=df.columns)

    # step 4: tokenization
    review = sent_tokenize(review)
    step_4 = pd.Series(['4: Sentence Tokenization', review], index=df.columns)

    # step 5: explosion
    steps = [step_1, step_2, step_3, step_4]
    for exploded in step_4["Review"]:
        steps.append(pd.Series(["5: Explosion", exploded], index=df.columns))

    # step 6: remove sentence dividers
    for subList in steps[4:]:
        removed = re.sub(r'[!?.]', '', subList[1])
        steps.append(pd.Series(["6: Remove Sentence Dividers", removed], index=df.columns))

    # bring all steps into single DataFrame
    df = df.append(steps, ignore_index=True)
    print(df)


def sentiment_processing(exploded_df):
    """
    This function processes the DataFrame imported from the tokenization/explosion and categorization step. It adds an
    additional column to the original DataFrame called "StructuredReview" that has the processed review in it.

    Then, the function will assign a sentiment score to the structured reviews in a column called "Polarity".

    :param exploded_df: DataFrame
        From sentence tokenization/explosion step.

    :return: DataFrame
        Processed DataFrame with structured review and tagged sentiment.
    """
    exclude = stopwords.words('english')  # stopwords to exclude; add to this list to add more exclusions
    df = exploded_df.copy()

    # make all reviews lowercase
    df["StructuredReview"] = df["Review"].str.lower()

    # remove punctuation (since reviews are already tokenized/exploded, remove sentence dividers at this step as well)
    df["StructuredReview"] =  df["StructuredReview"].apply(lambda x: re.sub(r'[^\w\s]+', '', x))

    # remove stopwords in exclude
    df["StructuredReview"] = df["StructuredReview"].apply(lambda x: " ".join(
        [item for item in x.split() if item not in exclude]))

    # tag sentiment scores
    df["Polarity"] = df["StructuredReview"].apply(lambda x: TextBlob(x).sentiment[0])
    return df


# Data Import ----------------------------------------------------------------------------------------------------------
data_path = "FILEPATH TO DATA DIRECTORY HERE"
exploded_data = pd.read_csv(data_path + r"\exploded_review_data.csv", encoding="unicode_escape")

# Tag Sentiment --------------------------------------------------------------------------------------------------------
if show_example:
    processing_example()

sentiment_data = sentiment_processing(exploded_data)
sentiment_data.to_csv(data_path + r"\exploded_review_data_w_polarity.csv", index=False, header=True)
