# Sentiment Plots
# Packages -------------------------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import seaborn as sns


# Functions ------------------------------------------------------------------------------------------------------------
def post_cleaning(df, neutral_band=0):
    """
    This function cleans the DataFrame by adding a column called "Sentiment" which is an indication of positive,
    neutral or negative sentiment based on the argument passed for neutral_band.

    It will also map the topics from their imported names to their proper names in a column called "ProperTopic".
    These data cleaning steps are to assist in plotting and interpretability.

    :param df: DataFrame
        From tokenization/explosion and categorization step.

    :param neutral_band: Positive Float
        Indicates what the neutral band for sentiment should be. For example, setting this parameter to 0.2 would cause
        all sentiment above 0.2 to be positive and all sentiment below -0.2 to be negative. Everything between -0.2 and
        0.2 would be considered neutral. Note that this interval includes endpoints for neutral sentiment.
        Default value is 0.

    :return: DataFrame
        Cleaned DataFrame
    """
    # tag sentiment type
    df["Sentiment"] = np.where(df["Polarity"] > neutral_band, "Positive",
                               np.where(df["Polarity"] < -neutral_band, "Negative", "Neutral"))

    # clean topics
    topic_dic = {"Topic0": "Food", "Topic1": "Order", "Topic2": "Staff", "Topic3": "Location", "Topic4": "Service"}
    df["ProperTopic"] = df["HighestTopic"].map(topic_dic)
    return df


def proportion_of_sentiment_by_topic(df, pal=None, fontsizes=None):
    """
    Generates a plot of the proportion of each type of sentiment for each topic, with the ability to customize a few
    parameters. Tweaking other aspects of the figure requires tweaking the code in the function below.

    :param df: DataFrame
        Cleaned DataFrame with labeled categories and tagged sentiment scores.
    :param pal: Dictionary
        Contains custom palette for sentiment types. Please refer to below for structure.
        Set to None by default and uses a "default-custom" palette.
    :param fontsizes: Dictionary
        Contains fontsizes for specific elements of the graph. Please refer below for structure.
        Set to None by default and uses "default-custom" fontsizes.
    :return: Figure
        Generates plot based on description.
    """
    # handling of default values from arguments; use appropriate structure below if custom parameters are desired
    if pal is None:
        pal = {'Negative': 'r', 'Neutral': 'y', 'Positive': 'g'}
    if fontsizes is None:
        fontsizes = {'title': 40, 'ticks': 30, 'bar labels': 30, 'legend': 30}

    # obtain proportion of sentiment by topic (sbt)
    sbt = (df["Sentiment"]
           .groupby(df["ProperTopic"])
           .value_counts(normalize=True)
           .rename("Proportion")
           .reset_index()
           .sort_values("ProperTopic", ascending=True)
           .reset_index()
           .drop("index", axis=1))

    # multiples proportions by 100 so we can display percentages
    sbt["Proportion"] = sbt["Proportion"].apply(lambda x: x * 100)

    # plots the figure
    plt.figure()
    ax = sns.barplot(x="ProperTopic", y="Proportion", hue="Sentiment", palette=pal, data=sbt,
                     hue_order=["Negative", "Neutral", "Positive"],
                     order=["Food", "Order", "Staff", "Location", "Service"])
    ax.set_title("Proportion of Sentiment Scores by Topic", fontsize=fontsizes['title'])
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.tick_params(labelsize=fontsizes['ticks'])
    for p in ax.patches:
        ax.annotate("%.2f%%" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 20),
                    textcoords='offset points', size=fontsizes['bar labels'])
    plt.legend(fontsize=fontsizes['labels'])
    plt.show()


def proportion_of_top_n_terms(df, n_terms=10, topics=None, sentiment=None, pal=None, fontsizes=None):
    """
    Generates plot(s) of the proportion of the top n terms per topic, with the ability to customize a few
    parameters. Tweaking other aspects of the figure requires tweaking the code in the function below.

    :param df: DataFrame
        Cleaned DataFrame with labeled categories and tagged sentiment scores.
    :param n_terms: int
        Number of terms to display in plot.
    :param topics: list
        List of topics you would like to see plots for.
        Set to None by default (in this function, this means every topic is shown by default).
    :param sentiment: list
        List of sentiments you would like to see plots for.
        Set to None by default (in this function, positive and negative sentiments are shown).
    :param pal: Dictionary
        Contains custom palette for sentiment types. Please refer to below for structure.
        Set to None by default and uses a "default-custom" palette.
    :param fontsizes: Dictionary
        Contains fontsizes for specific elements of the graph. Please refer below for structure.
        Set to None by default and uses "default-custom" fontsizes.
    :return: Figure(s)
        Generates plot(s) based on parameters given.
    """
    # handling of default values from arguments; use appropriate structure below if custom parameters are desired
    if topics is None:
        topics = ['Staff', 'Location', 'Service', 'Food', 'Order']
    if sentiment is None:
        sentiment = ["Positive", "Negative"]
    if pal is None:
        pal = {'Positive': 'g', 'Negative': 'r'}
    if fontsizes is None:
        fontsizes = {'title': 40, 'ticks': 20, 'x label': 20}

    # constructs list of signs to iterate over
    signs = []
    for s in sentiment:
        signs.append((s, pal[s]))

    # generates plot(s)
    for topic in topics:
        # keep subset of df for topic of current iteration
        topic_subset = df[df["ProperTopic"] == topic].copy()

        for sign, color in signs:
            # keep subset of df for sentiment of current iteration
            sign_subset = topic_subset["StructuredReview"][cleaned_data["Sentiment"] == sign].copy()

            # get proportions of all terms
            term_proportions = sign_subset.value_counts(normalize=True).rename("Proportion").reset_index()

            # sort proportions in descending order
            term_proportions.sort_values("Proportion", ascending=False).reset_index().drop("index", axis=1)
            term_proportions.rename({"index": "Review"}, axis=1, inplace=True)

            # multiply proportions by 100 so we can display percentages in plot
            term_proportions["Proportion"] = term_proportions["Proportion"].apply(lambda x: x * 100)

            # generate top n term plot for topic, sentiment of current iteration
            plt.figure()
            ax = sns.barplot(x="Proportion", y="Review", data=term_proportions[1:n_terms], color=color)
            sns.despine(left=True, bottom=True)
            ax.set_title(f"Proportion of Top {n_terms} Terms From All {sign} Reviews For Topic: {topic}",
                         fontsize=fontsizes['title'])
            ax.set_xlabel("Percentage of Term in Topic", fontsize=fontsizes['x label'])
            ax.set_ylabel("")
            ax.xaxis.set_major_formatter(PercentFormatter())
            ax.tick_params(labelsize=fontsizes['ticks'])
            plt.show()


# Data Import ----------------------------------------------------------------------------------------------------------
data_path = "FILEPATH TO DATA DIRECTORY HERE"
exploded_data = pd.read_csv(data_path + r"\exploded_categorized_polarity2.csv", encoding="unicode_escape")

# Plotting -------------------------------------------------------------------------------------------------------------
# Clean data in preparation for plotting. This step is required.
cleaned_data = post_cleaning(exploded_data)

# Plots
proportion_of_sentiment_by_topic(cleaned_data)
proportion_of_top_n_terms(cleaned_data)
