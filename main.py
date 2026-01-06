import argparse
import asyncio
import logging
from datetime import datetime, timezone
import dotenv
from typing import Literal

import os
import json
import requests
import re
import numpy as np
import csv
import time
import logging
import io
from collections import defaultdict
import sys

from asknews_sdk import AskNewsSDK        
from openai import OpenAI                 
#from newscatcherapi import NewsCatcherApiClient  

# Load API keys from environment variables
ASKNEWS_CLIENT_ID = os.environ.get("ASKNEWS_CLIENT_ID")
ASKNEWS_SECRET = os.environ.get("ASKNEWS_SECRET")
CLAUDE_API_KEY = os.environ.get("CLAUDE_API_KEY")
METACULUS_TOKEN = os.environ.get("METACULUS_TOKEN")
NEWSCATCHER_API_KEY = os.environ.get("NEWSCATCHER_API_KEY")
NYT_API_KEY = os.environ.get("NYT_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PERPLEXITY_API_KEY = os.environ.get("PERPLEXITY_API_KEY")

def test_newscatcher():
    if not NEWSCATCHER_API_KEY:
        print("NEWSCATCHER_API_KEY not set")
        return

    url = "https://api.newscatcherapi.com/v2/search"
    headers = {
        "x-api-key": NEWSCATCHER_API_KEY
    }
    params = {
        "q": "AI",
        "lang": "en",
        "page_size": 3
    }

    response = requests.get(url, headers=headers, params=params)
    
    if response.status_code == 200:
        data = response.json()
        print(" NewsCatcher test successful. Sample articles:")
        for article in data.get("articles", []):
            print("-", article.get("title"))
    else:
        print(f" NewsCatcher test failed: {response.status_code}")
        print(response.text)


from forecasting_tools import (
    AskNewsSearcher,
    BinaryQuestion,
    ForecastBot,
    GeneralLlm,
    MetaculusClient,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    DateQuestion,
    DatePercentile,
    Percentile,
    ConditionalQuestion,
    ConditionalPrediction,
    PredictionTypes,
    PredictionAffirmed,
    BinaryPrediction,
    PredictedOptionList,
    ReasonedPrediction,
    SmartSearcher,
    clean_indents,
    structure_output,
)

dotenv.load_dotenv()
logger = logging.getLogger(__name__)

##################################### PROMPTS #####################################
Singleshot_Research_Memo = """
    I am a professional forecaster. My goal is to make an accurate prediction
    on an important question. Here is some information about the question I am trying to forecast:

    Before I do any forecasting, I am going to have a research assitant gather some
    factual analysis / information for me.  That's where I need your help.

    Given the question I describe below, can you write **one** succicient question
    for the research assistant to work on? In other words, what's the most important
    thing one needs to know to make this prediction? The question should be factual
    in nature -- things you can look up in news sources, encyclopedia's, press releases,
    on the internet, etc.

    The question is:
    {title}

    Here is some background information:
    {background}

    Here is how the question will get resolved:
    {resolution_criteria}

    Here is the fine print on the question:
    {fine_print}

    Today is {today}.

    I have pulled a bunch of recent news articles related to the question so you have
    access to some of the latest news:

    {hotnews}

    Two key things to keep in mind:

    1) Don't over-complicate it.  Often the key thing you need to know is pretty simple.
    For instance in predicting who will win a political contest the key thing to know is
    the latest polling.  In predicting where a stock price will end up, the key thing to know
    is where the stock price is today.

    2) Your question shouldn't be about the future -- it shouldn't require a prediction.  It should
    be something we can know in the present that will help the forecaster make the prediction.

    Thanks for all your hard work.

    """

Multishot_Research_Memo = """
    I am a professional forecaster. My goal is to make an accurate prediction
    on an important question. Here is some information about the question I am trying to forecast:

    Before I do any forecasting, I am going to have a research assitant gather some
    factual analysis / information for me.  That's where I need your help.

    Given the question I need to forecast and the background -- as well as everything
    you know about forecasting -- what are three or four questions the research assistant
    could help with. These questions should primarily be factual in nature -- things you can
    look up in news sources, encyclopedia's, press releases, on the internet, etc.

    The question I need to forecast is:
    {title}

    Here is some background information:
    {background}

    Here is how the question gets resolved:
    {resolution_criteria}

    Here is the fine print on the question:
    {fine_print}

    Today is {today}.

    Your questions shouldn't be about the future -- it shouldn't require a prediction.  It should
    be something we can know in the present that will help the forecaster make the prediction.

    Thanks for all your hard work.

    """

Perplexity_Memo = """
    I am a professional forecaster. My goal is to make an accurate prediction
    on an important question. I need your help with some research.

    Here is a question or questions that I need you to answer using the factual information at your disposal.
    {gpt_question}

    ****The above is what I need you to work on****

    The rest of this is background and instructions.
    Here is some background information about the question I am trying to forecast:

    Here is what I am trying to forecast:
    {title}

    Here is some background information on the forecast:
    {background}

    Here is how the forecast gets resolved:
    {resolution_criteria}

    Here is the fine print on the question:
    {fine_print}

    Today is {today}.

    Please generate a factbase answer to the question for me.

    You do **not** need to generate a forecast on the question.  That's for someone else to do.  I need you to do factual research.

    To answer the question or questions, feel free to use the information from the background or fineprint herein -- that is reliable information.
    You can also use information from the internet that you have access to.

    Please start your answer by restating the question.

    Where possible please cite your sources.

    Be mindful that when the researchers says "current" they mean as close to today as possible.

    Thanks for all your hard work.

    """

Grading_Instructions = """
    I need your help evaluating some research.

    Here is the question I needed answered:
    {rschquestion}

    This is the research memo I received from my assistant:
    {rschquestion_answer}

    Today is {today}

    Does the research answer the question or questions?  Please provide a score between 0 and 100
    where 100 is perfect, 70 is passing, and 0 is completely missed the mark.

    This is not about effort or goodwill -- it's literally about whether the memo provided contain the answer.
    Your grade will help me determine whether I can use it in my project.

    Output your response in the following JSON structure:
    {{
    "rationale": "string",
    "grade": "integer between 0 and 100"
    }}

    Thanks for all your hard work.

    """

Keyword_Generator = """
    I need your help using a search engine.  The search engine can only accept key words.
    It cannot accept whole questions or sentences. I am going to give you a whole question and I need you
    to convert it into keywords that capture the main topics of the question.

    Here is the question:
    {title}

    Do not include any date information such as the month or year in your answer.

    Your response cannot be more than four words or the engine won't work.  It's fine if it's fewer words -- three is ideal.

    Try to make the words as clear and simple as possible.  For instance, don't use the word Russo -- use the word Russia or Russian.  Simple language wins.

    Output your response in the following JSON structure:
    {{
    "keywords": "2 to 4 words"
    }}

    """

NEWS_CLEANUP_PROMPT = """
    You are a professional editor.

    I have pulled articles from a database but they are in a messy format.  I need them formatted
    in a standard way so that I can compare them to other articles.

    *****Here is what each article should look like:
    [1]:
    Title: Zelenskyy Hopes for End to War with Russia by 2025
    Summary: Ukrainian President Volodymyr Zelenskyy expressed his hope that the war with Russia will end by 2025 during his visit to Berlin, where he called for continued military support for his country. Zelenskyy met with German Chancellor Olaf Scholz and thanked Germany for its support, saying 'It is very important for us that these aid does not decrease next year.' He also presented his 'Plan for Victory' in the war, expressing his hope that the conflict will end 'by next year 2025.' Scholz assured that Germany and European partners will send more defensive equipment to Ukraine this year, and that Germany will provide 4 billion euros in aid to Ukraine in 2025, vowing 'not to retreat from our support for Ukraine.' Zelenskyy agreed that a conference for peace including Russia is necessary, but also emphasized that 'peace can only be achieved based on international law.' Scholz stated that 'we will not accept a peace imposed by Russia.' Zelenskyy also met with Pope Francis in the Vatican, who called for peace in Ukraine, and with French President Emmanuel Macron in Paris, who emphasized the importance of continued support for Ukraine.
    Excerpt: Ukraine's President Volodymyr Zelensky said, 'No nation should face such trials alone.' Russia continues its military invasion of Ukraine, which began on February 24, 2022, at the order of Vladimir Putin. Putin announced the start of the war against Ukraine on February 24, 2022, calling it a 'special military operation in Donbass' aimed at protecting people who 'have been subjected to genocide by the Kiev regime for eight years.' However, the EU and the UN have previously disputed Putin's claims of genocide. Western politicians have accused Putin of lying. Zelensky has announced the break of diplomatic relations with Russia.
    Source: CNBC.com
    Published Date: November 10 2024 20:53

    Here are the articles from the database:
    {news_articles}

    Here are the steps I need you to follow to format the articles:
    1. If the articles come in a JSON format or a continious string, break them into separate articles.  Each article usually starts with "Title."
    2. For all the articles, format / organize them similarly -- using the format above.
    3. For each article, please include the title, summary, excerpt (if available), media source, and published date.
    4. The summary and the excerpt should be exactly as they appear in the "clip" you recevied.  Do not truncate / edit them in anyway.
    5. There is no need to include the other fields in your final response.
    6. If you're missing data for any required field, just leave it blank.
    7. You should return a well formatted text string.  Do not return JSON format.

    I would prefer you clean up all of them but if you need to cut your work short, either
    clean-up all of them or do at least 20.

    Thanks for all your hard work.

    """

HOTNEWS_RANKED_PROMPT = """
    You are a professional researcher, who works for a forecaster.  The forecaster is workking
    on the following question:

    {title}

    Today is {today}.

    As part of the forecasting process, the researcher pulled articles from two different media databases.
    All the articles are from the past 48 to 72 hours.

    ****I need you to read through all the articles from the database and send back the
    most relevant ones given the question the forecaster is working on.****

    Here are the articles from the first database which is called AskNews:
    {hotnews_asknews_cleaned}

    Here are the articles from the second database which is called NewsCatcher:
    {hotnews_newscatcher_cleaned}

    Here are the steps you should follow:
    1) Read each article.
    2) Eliminate any duplicate articles.
    3) Eliminate anything that is mostly opinion based.
    4) Score each articles in terms of whether it provides relevant and timely information to forecast on the question above.
    5) Score the article in terms of the quality of the news source -- the higher the quality of the newssource the better.
    6) Rank all the articles using the infomration from step 4 and 5.
    7) Pick a handfull of the best articles given the question the forecaster is working on.
    8) Add a field to each article for which database -- AskNews, NewsCatcher, etc. -- that it's coming from.
    9) Count the number of articles you were given and the the number of articles you're returning.

    Typically you will return between 10 and 15 articles mixed across the two databases. But your answer can vary.
    If nothing is relevant, return zero.  If lots are relevant, return more.

    Do **not** make up any content.  Only use the materials you were provied to generate your list.

    For the articles you return, they should be returned in a well formatted text string like this:
    *****Here is what each article should look like:
    [1]:
    Title: Zelenskyy Hopes for End to War with Russia by 2025
    Summary: Ukrainian President Volodymyr Zelenskyy expressed his hope that the war with Russia will end by 2025 during his visit to Berlin, where he called for continued military support for his country. Zelenskyy met with German Chancellor Olaf Scholz and thanked Germany for its support, saying 'It is very important for us that these aid does not decrease next year.' He also presented his 'Plan for Victory' in the war, expressing his hope that the conflict will end 'by next year 2025.' Scholz assured that Germany and European partners will send more defensive equipment to Ukraine this year, and that Germany will provide 4 billion euros in aid to Ukraine in 2025, vowing 'not to retreat from our support for Ukraine.' Zelenskyy agreed that a conference for peace including Russia is necessary, but also emphasized that 'peace can only be achieved based on international law.' Scholz stated that 'we will not accept a peace imposed by Russia.' Zelenskyy also met with Pope Francis in the Vatican, who called for peace in Ukraine, and with French President Emmanuel Macron in Paris, who emphasized the importance of continued support for Ukraine.
    Excerpt: Ukraine's President Volodymyr Zelensky said, 'No nation should face such trials alone.' Russia continues its military invasion of Ukraine, which began on February 24, 2022, at the order of Vladimir Putin. Putin announced the start of the war against Ukraine on February 24, 2022, calling it a 'special military operation in Donbass' aimed at protecting people who 'have been subjected to genocide by the Kiev regime for eight years.' However, the EU and the UN have previously disputed Putin's claims of genocide. Western politicians have accused Putin of lying. Zelensky has announced the break of diplomatic relations with Russia.
    Source: CNBC.com
    Published Date: November 10 2024 20:53

    Your final output should be three things:
    1) The 10 to 15 articles that are most relevant from this batch. Those articles should appear in the format you received them.  It should be a well formatted text string.
    2) The number of articles you were provided -- essentionally the sum of the articles from AskNews and NewsCatcher.  This will be number_of_articles_provided.
    3) The number of articles you are returning.  This will be number_of_articles_returned.

    Output your response in the following JSON structure:
    {{
    "hot_news_ranked": "text string",
    "number_of_articles_provided": "integer",
    "number_of_articles_returned": "integer",
    }}

    Thanks for all your hard work.

    """

Baserate_Research_Memo = """
    I am a professional forecaster. My goal is to make an accurate prediction
    on an important question. Here is some information about the question I am trying to forecast:

    Before I do any forecasting, I need to understand the baserates for this question.

    The question is:
    {title}

    Here is some background information:
    {background}

    Here is how the question gets resolved:
    {resolution_criteria}

    Here is the fine print on the question:
    {fine_print}

    Today is {today}.

    I have pulled a bunch of recent news articles related to the question so you have
    some of the latest news to help you:

    {hotnews}

    Baserates are super important in forecasting:

    Base rates are the statistical likelihood of an event happening based on historical
    data or a general population. They are important in forecasting because they provide
    an objective starting point, preventing predictions from being overly influenced by
    specific, often unreliable, details. For example, if 70% of startups fail within the
    first five years, using that base rate helps ground predictions about a new startup’s
    likelihood of success, before adjusting for its specific characteristics.

    Given the question and backhground I describe above, can you write 4 succicient questions
    about specific baserates you would want to know to be able to forecast on this question.
    I will then have a researcher go find the answer to these questions.

    Don't over-complicate it.  Often the key things you need to know are pretty simple.

    Output your response in the following JSON structure:
    {{
    "Question1": "Your first base-rate question in one sentence or so."
    "Question2": "Your second base-rate question in one sentence or so."
    "Question3": "Your third base-rate question in one sentence or so."
    "Question4": "Your fourth base-rate question in one sentence or so."
    }}

    Thanks for all your hard work.

    """

BACKGROUND_RANK_PROMPT = """
    You are a professional researcher, who works for a forecaster.  The forecaster is workking
    on the following question:

    {title}

    Today is {today}.

    As part of the forecasting process, the researcher pulled articles from three different media databases.

    ****I need you to read through all the articles from the database and send back the
    most relevant ones given the question the forecaster is working on.****

    Here are the articles from the first database which is called NewsCatcher:

    {newscatcher_articles}

    Here are the articles from the first database which is called New York Times:

    {nyt_articles}

    Here are the articles from the first database which is called AskNews:

    {background_asknews}

    Here's some steps you should follow:
    1) Read each article.
    2) Eliminate anything that is mostly opinion based.
    3) Eliminate any duplicate articles.
    4) Score the article in terms of whether it provides relevant background information to forecast on the question above.
    5) Score the article in terms of the quality of the news source -- the higher the quality of the newssource the better.
    6) Of the 100 or so articles you're reviewing pick out 10 - 20 that score highly on the above criterion
    7) Don't worry about breaking news.  There is another team working on breaking news / latest news so you don't need to worry about that.
    8) Count the number of articles you were given and count the number of articles you're returning.

    The goal is to provide the forecaster with broadly relevant information -- very useful background information to forecast on the question..

    For the articles you return, you should keep the exact same formatting / material that you received them in.  If you selected the entry, do not truncate / edit it.
    Remember it's critical to include the full entry for each article you select -- DO NOT SHORTEN THEM or eliminate materials.

    Please note for each entry which database it came from: NewsCatcher, New York Times, AskNews.

    For the articles you return, they should be returned in a well formatted text string like this:
    *****Here is what each article should look like:
    [1]:
    Title: Zelenskyy Hopes for End to War with Russia by 2025
    Summary: Ukrainian President Volodymyr Zelenskyy expressed his hope that the war with Russia will end by 2025 during his visit to Berlin, where he called for continued military support for his country. Zelenskyy met with German Chancellor Olaf Scholz and thanked Germany for its support, saying 'It is very important for us that these aid does not decrease next year.' He also presented his 'Plan for Victory' in the war, expressing his hope that the conflict will end 'by next year 2025.' Scholz assured that Germany and European partners will send more defensive equipment to Ukraine this year, and that Germany will provide 4 billion euros in aid to Ukraine in 2025, vowing 'not to retreat from our support for Ukraine.' Zelenskyy agreed that a conference for peace including Russia is necessary, but also emphasized that 'peace can only be achieved based on international law.' Scholz stated that 'we will not accept a peace imposed by Russia.' Zelenskyy also met with Pope Francis in the Vatican, who called for peace in Ukraine, and with French President Emmanuel Macron in Paris, who emphasized the importance of continued support for Ukraine.
    Excerpt: Ukraine's President Volodymyr Zelensky said, 'No nation should face such trials alone.' Russia continues its military invasion of Ukraine, which began on February 24, 2022, at the order of Vladimir Putin. Putin announced the start of the war against Ukraine on February 24, 2022, calling it a 'special military operation in Donbass' aimed at protecting people who 'have been subjected to genocide by the Kiev regime for eight years.' However, the EU and the UN have previously disputed Putin's claims of genocide. Western politicians have accused Putin of lying. Zelensky has announced the break of diplomatic relations with Russia.
    Source: CNBC.com
    Published Date: November 10 2024 20:53
    Database: New York Times OR AskNews OR NewsCatcher
    *****

    Your final output should be three things:
    1) The articles that are most relevant from this batch. Those articles should appear in the format you received them.
    2) The number of articles you were provided -- essentionally the sum of the articles from AskNews and NewsCatcher.  This will be number_of_articles_provided.
    3) The number of articles you are returning.  This will be number_of_articles_returned.

    Output your response in the following JSON structure:
    {{
    "background_news_ranked": "text string",
    "number_of_background_articles_provided": "integer",
    "number_of_background_articles_returned": "integer",
    }}

    Thanks for all your hard work.

    """

Score_Overall_Research_Prompt = """
    I need your help evaluating some research.

    I am a forecaster and I need to forecast this question:
    {title}

    I have had a team of researchers put together the following information to help me forecast this question:

    Here is the latest news on the topic:
    {hot_news_ranked}

    Here is some background news on the topic:
    {background_news_ranked}

    Here is the answer to the most important question -- in their view -- for forecasting on this question:
    {one_shot_rschquestion_answer}

    Here is a research memo on the topic:
    {multishot_rschquestion_answer}

    I also asked them to determine the key baserates that we need to know and to find the data.

    Here is the first baserate quesiton and answer:
    {baserate1}

    Here is the second baserate quesiton and answer:
    {baserate2}

    Here is the third baserate quesiton and answer:
    {baserate3}

    Here is the fourth baserate quesiton and answer:
    {baserate4}

    I need you to do the following:
    1) Review all of this research.
    2) Think about how well it answers each of the questions posed.
    3) Then think overall about whether I have what I need to forecast on this question
    If one has poor quality inputs / information / data the forecast will be wrong. Specifically:
    -->What questions do you think I need to have answer to to forecast this question?
    -->What questions do I have answers to?
    4) Please provide a score between 0 and 100
    where 100 is perfect, 70 is passing, and 0 is completely missed the mark.

    This is not about effort or goodwill -- it's literally about whether the memo provided containts the answers.
    Your grade will help me determine whether I can use it in my project.

    Your out put will have three parts:
    1) What is the score as described in point 4 above?
    2) Should I proceed with my forecast or should I have the researcher do more work?
    3) What is your rational for these answers?


    Output your response in the following JSON structure:
    {{
    "score": "integer between 0 and 100",
    "proceed": "boolean",
    "rationale": "string"
    }}

    Thanks for all your hard work.

    """

NYT_Cleanup = """
    I need your help cleaning up a data dump from the NYT database.  Here is a bunch of info
    from a search I did:
    {text_string}

    Unfortunately, it's got lots of distracting junk in it.  Can you clean it up?

    Here are some instructions:
    1) There are about ten articles here.  Find them.
    2) For each, pull out the title, abstract, snippet, lead paragraph, source, pubulication date.
    3) Generally I'm looking for the substantive parts of the article.
    4) Delete ***everything else***

    One thing I'm trying to do is dramatically shorten the file.  It can only be a few pages long.
    If it's more than that, please cut it down.

    Please return a well formatted text string.

    """

OPENAI_PROMPT_TEMPLATE = """
    You are a professional forecaster, and I need your help making a prediction.

    Your goal is to make an accurate prediction. To do this, you evaluate past data
    and trends carefully, take into account base rates about how similar events unfolded in the past,
    synthesize key informaiton, and outline the best reasons for and against a
    particular outcome, among other steps.

    You know that great forecasters don't just forecast according to the "vibe" of the question -- they do the work.
    They think about the question in a structured way, record their
    reasoning as they go, and they always consider multiple perspectives.

    You are trying to give the most accurate probability.  There is no advantage to hedging.
    Your answer will be evaluated later when actual events unfold.

    Here is some information about the question.

    The question is:
    {title}

    Here is some background on the question:
    {background}

    Here is how the question gets resolved:
    {resolution_criteria}

    Here is the fine print on the question:
    {fine_print}

    Today is {today}.

    Let's now go through some steps that good forecasters use to answer a question.

    I am going to layout some questions I would like you to answer and to think
    about before you give a final probability. You should give an explicit answer to all
    of these questions -- and think about them carefully -- before you give your
    probablity. Showing your work will help you develop a better answer.

    1. Given the question above, please rephrase and expand the question to help
    you do a better job answering it.  Maintain all of the information in the
    original question but restate it in your own words.

    2. Focus on the "resolution criterion" for the question.  It often
    contains important definitions that should be considered. For instance,
    sometimes the question will make a general point and the fine print will make it much more specific.
    Considering the resolution criterion provided to you above, how do you think that impacts the probabilities of a given
    outcome here? If helpful, restate the question more precisely based on the fine print and resolution criterion.

    3. Do you same thing with Fine Print.  Focus on the fine print for the question.  It often
    contains important definitions and edge cases that should be considered. For instance, the fine
    print might allow for a softer standard for resolution than the resolution criterion.
    Considering the fine-print provided to you above, how do you think that impacts the probabilities of a given
    outcome here? If helpful, restate the question more precisely based on the fine print and resolution criterion.


    4. I asked a research assistant to pull headlines from the past
    48 hours related to the question you're trying to forecast. This is the breaking news
    related to the topic.  If might not provide the broadest context but
    it is up-to-date and critical for you to consider. Here are the headlines and a summary of each article:

    {hot_news_ranked}.

    Think about how this material shapes your forecast and write down what you think.

    5. I hired a research assistant to help gather information and facts for your forecast.
    His first piece of research answers the one critical question that I believe you need to know
    the answer to in order to make a forecast on this question. There are certainly other important questions
    to answer but this information is the most important question and it's answer:

    {one_shot_rschquestion_answer}

    Think about how the answer here shapes your forecast and write down your answer.

    6. It's often said that baserates are critical to forecasting so your researcher
    put together four baserates questions and their answers. Here's a
    good definition of base rate: a base rate is the likelihood of an event
    occurring based on historical data. Your researcher developed several base rate questions
    based on the question you're trying to forecast.

    Baserate question 1 and its answer:
    {baserate1}

    Baserate question 2 and its answer:
    {baserate2}

    Baserate question 3 and its answer:
    {baserate3}

    Baserate question 4 and its answer:
    {baserate4}

    Think about how the answer to this question shapes your forecast and write down your answer.

    7. I hired a second researcher to help gather a broader set of information and facts for your forecast.
    In this case, I generated three or four important questions related to the forecast
    and asked him to find the answers.  Here is that work:

    {multishot_rschquestion_answer}

    Again, think about how the answer to this question shapes your forecast and write down your answer.

    8. I also thought it would be useful for you to have some background articles and other reporting on the question.
    So I asked another research assistant to pull some headlines and article summaries from
    a wide range of media sources related to the question you're trying to forecast. These were selected
    due to their broad relevance to the topic and they come from a wide time-range ... so be careful with them.
    Be keenly aware that events and circumstances may supersede some of these articles.  They are meant to be
    broadly relevant -- not the most current. The database he used has articles from the last year.

    Here are the titles, a summary, and some extracts:

    {background_news_ranked}

    Think about how these materials shape your forecast and write down your answer.

    *****That's the background material I have for you.  Let's now do some more work to develop a forecast.******

    9. The time element is always super important in prediction.  So make sure you know
    today's date -- it's listed above.  Then answer these questions:
    a. How much time is left in days until the question resolves?
    b. Think about the default resolution: if the question resolved today how
    would it resolve?
    c. What kind of rate of change is required for the question to resolve "yes"?
    d. What kind of rate of change is required for the question to resolve "no"?
    c. What would you forecast if there was only a quarter of the time left?
    d. What would you forecast if there was 4x the time left?

    10. Using your knowledge of the topic and the information provided above, list a few reasons
    why the answer might be NO.  List them in order of important.  Rate the strength of each reason.

    11. Using your knowledge of the topic and the information provided above, list a few reasons
    why the answer might be YES.  List them in order of important.  Rate the strength of each reason.

    12. Now aggregate your considerations.  Think like a superforecaster (e.g., Nate
    Silver, Phil Tetlock).  Based on everything you've learning in steps 1 through 11,
    give us your best answer.

    14. Evaluate whether your calculated probability is excessively confident or not confident enough.
    Think carefully about this question.   Also, consider anything else you might have missed.
    This is your opportunity to pause and reflect on your work so far.  Do any revisions make sense?

    As a reminder here is the question you are forecasting: {title}

    *****What is the probablity that the question will resolve {direction}?*****

    You should aggregate your answer into a probability between 0% (very, very unlikely) and 100% (very, very likely).

    You should always provide a number. The number can be very low or very high. Don't be
    afraid to go to the extremes if your analysis suggests so. Just be accurate.

    Follow these steps when generating your output:

    1) **SHOW YOUR WORK** Provide your analysis based on each of the steps described above i.e., write out an answer to each step.
    Then given the question, all the material provided to you, and your step-by-step work
    provide your expert forecast on whether or not the resolution criteria will be achieved and your rationale.
    Overall "show your work" will be several paragpraphs long.  That's okay -- take your time and write out what you need to write out.

    2) **DETERMINE A FORECAST PROBABLITY** Given the resolution criteria and your rationale,
    determine the probability (likelihood) that the resolution will be achieved
    Speciically, what is the probablity the quesiton will resovle {direction}.
    This is an integer between 0 and 100.

    3) Reflect on how confident you are given the quality of the inputs and the situation
    at hand in your answer.  Grade your confidence on a scale of 1 to 10, where
    1 is very low confidence and 10 is very high confidence.

    Output your response in the following JSON structure:
    {{
    "show_your_work": "string",
    "probability": "integer between 0 and 100"
    "confidence": "integer between 1 and 10"
    }}

    Thanks for all your hard work.
    """

OPENAI_FINAL_RATIONALE_PROMPT_TEMPLATE = """
    I am a professional forecaster and I am about to submit a forecast to a forecasting contest.

    As part of my submission, I have to include a rationale.  Lots of things went into my forecast so
    I need your help writing the rationale.  The gaol is to explain the reaons why I am submitting the forecast that I am.

    Here is some information about the question.

    The question is:
    {title}

    Here is some background on the question:
    {background}

    Here is how the question gets resolved:
    {resolution_criteria}

    Here is some background news on the question:
    {hot_news_ranked}

    Here is some research that was done on the question:
    {one_shot_rschquestion_answer}

    ***The above will serve as useful background***

    My final prediction on this question is: {final_prediction}

    The way I do my forecast is that I have six forecasters develop an answer and then I make a final decision on what to submit.
    Here are the rationales they submitted:

    Forecaster #1's rationale:
    {rationale1}

    Forecaster #2's rationale:
    {rationale2}

    Forecaster #3's rationale:
    {rationale3}

    Forecaster #4's rationale:
    {rationale4}

    Forecaster #5's rationale:
    {rationale5}

    Forecaster #6's rationale:
    {rationale6}

    Some of these might be blank.  If so, just ignore them.

    Here are your instructions for writting what I need:
    1) Your answer should be about 500 words.
    2) It should synthesize the material above into one coherent answer / rationale.
    3) It should explain why the forecast is what it is with an emphasis on the ***key reasoning***
    4) Write it from the perspective of "MWG Bot."  Do not say "I" ... say "MWG Bot."
    5) Do not mention the exact final prediction I am submitted in your paragraph.  That is submitted elsewhere.  This piece should only include the rationale.
    6) Please put this sentence at the end of your statement: "NOTE: MWG Bot is built in Python using Google Colab Enterprise.
    It relies upon a variety of tools and services, which the owner of MWG Bot pays for.  It also uses AskNews, NewsCatcher,
    and The New York Times.  The owners of these three services kindly donate access to MWG Bot.  Thank you to our sponsors!"

    Output your response in the following JSON structure:
    {{
    "final_rationale": "string",
    }}

    Thanks for all your hard work.
    """

GET_QUESTION_CATEGORIES_TEMPLATE = """
    I need your help creating groups out of a list of questions.

    This is a list of forecasting questions, each of which will be assigned a probability between 0 and 100.

    Questions should be grouped together if the sum of the probabilities for those questions should not exceed 100.  There are two flavors of these types of situations.  One is a list of questions that is mutually exclusive and collectively exhaustive.  The other is a list of questions that are mutually exclusive but do not include a collectively exhaustive set of answers.  In either case, the sum of the probabilities should not exceed 100 and I need them grouped together.

    Some groups will only contain one question and that is fine – in fact, it’s expected.

    Can you create groups for these questions:

    {titles}

    The group assignment should be an integer and you should start the numbering with 1.

    Output your response in the following JSON structure:
    {{
    ID#: "group assignment",
    ID#: "group assignment",
    ID#: "group assignment",
    }}

    Thanks for all your hard work.
"""


class SpringTemplateBot2026(ForecastBot):
    """
    This is the template bot for Spring 2026 Metaculus AI Tournament.
    This is a copy of what is used by Metaculus to run the Metac Bots in our benchmark, provided as a template for new bot makers.
    This template is given as-is, and is use-at-your-own-risk.
    We have covered most test cases in forecasting-tools it may be worth double checking key components locally.
    So far our track record has been 1 mentionable bug per season (affecting forecasts for 1-2% of total questions)

    Main changes since Fall:
    - Additional prompting has been added to numeric questions to emphasize putting pecentile values in the correct order.
    - Support for conditional and date questions has been added
    - Note: Spring AIB will not use date/conditional questions, so these are only for forecasting on the main site as you wish.

    The main entry point of this bot is `bot.forecast_on_tournament(tournament_id)` in the parent class.
    See the script at the bottom of the file for more details on how to run the bot.
    Ignoring the finer details, the general flow is:
    - Load questions from Metaculus
    - For each question
        - Execute run_research a number of times equal to research_reports_per_question
        - Execute respective run_forecast function `predictions_per_research_report * research_reports_per_question` times
        - Aggregate the predictions
        - Submit prediction (if publish_reports_to_metaculus is True)
    - Return a list of ForecastReport objects

    Alternatively, you can use the MetaculusClient to make a custom filter of questions to forecast on
    and forecast them with `bot.forecast_questions(questions)`

    Only the research and forecast functions need to be implemented in ForecastBot subclasses,
    though you may want to override other ForecastBot functions.
    In this example, you can change the prompts to be whatever you want since,
    structure_output uses an LLM to intelligently reformat the output into the needed structure.

    By default (i.e. 'tournament' mode), when you run this script, it will forecast on any open questions in the
    primary bot tournament and MiniBench. If you want to forecast on only one or the other, you can remove one
    of them from the 'tournament' mode code at the bottom of the file.

    You can experiment with what models work best with your bot by using the `llms` parameter when initializing the bot.
    You can initialize the bot with any number of models. For example,
    ```python
    my_bot = MyBot(
        ...
        llms={  # choose your model names or GeneralLlm llms here, otherwise defaults will be chosen for you
            "default": GeneralLlm(
                model="openrouter/openai/gpt-4o", # "anthropic/claude-sonnet-4-20250514", etc (see docs for litellm)
                temperature=0.3,
                timeout=40,
                allowed_tries=2,
            ),
            "summarizer": "openai/gpt-4o-mini",
            "researcher": "asknews/news-summaries",
            "parser": "openai/gpt-4o-mini",
        },
    )
    ```

    Then you can access the model in custom functions like this:
    ```python
    research_strategy = self.get_llm("researcher", "model_name"
    if research_strategy == "asknews/news-summaries":
        ...
    # OR
    summarizer = await self.get_llm("summarizer", "llm").invoke(prompt)
    # OR
    reasoning = await self.get_llm("default", "llm").invoke(prompt)
    ```

    If you end up having trouble with rate limits and want to try a more sophisticated rate limiter try:
    ```python
    from forecasting_tools import RefreshingBucketRateLimiter
    rate_limiter = RefreshingBucketRateLimiter(
        capacity=2,
        refresh_rate=1,
    ) # Allows 1 request per second on average with a burst of 2 requests initially. Set this as a class variable
    await self.rate_limiter.wait_till_able_to_acquire_resources(1) # 1 because it's consuming 1 request (use more if you are adding a token limit)
    ```
    Additionally OpenRouter has large rate limits immediately on account creation
    """

    _max_concurrent_questions = (
        1  # Set this to whatever works for your search-provider/ai-model rate limits
    )
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)
    _structure_output_validation_samples = 2

       ##################################### RESEARCH #####################################

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            today = datetime.now().strftime("%Y-%m-%d")

            # --- HOT NEWS ---
            asknews = await AskNewsSearcher().invoke(
                clean_indents(
                    f"""
                    {Keyword_Generator}

                    Question:
                    {question.question_text}
                    """
                )
            )

            hotnews_ranked = await self.get_llm("default", "llm").invoke(
                clean_indents(
                    HOTNEWS_RANKED_PROMPT.format(
                        title=question.question_text,
                        today=today,
                        hotnews_asknews_cleaned=asknews,
                        hotnews_newscatcher_cleaned="",
                    )
                )
            )

            # --- ONE-SHOT RESEARCH QUESTION ---
            one_shot_q = await self.get_llm("default", "llm").invoke(
                clean_indents(
                    Singleshot_Research_Memo.format(
                        title=question.question_text,
                        background=question.background_info,
                        resolution_criteria=question.resolution_criteria,
                        fine_print=question.fine_print,
                        today=today,
                        hotnews=hotnews_ranked,
                    )
                )
            )

            one_shot_a = await AskNewsSearcher().invoke(one_shot_q)

            # --- MULTI-SHOT RESEARCH ---
            multi_shot_a = await self.get_llm("default", "llm").invoke(
                clean_indents(
                    Perplexity_Memo.format(
                        title=question.question_text,
                        hotnews=hotnews_ranked,
                    )
                )
            )

            # --- BASERATES ---
            baserate_qs = await self.get_llm("default", "llm").invoke(
                clean_indents(
                    Baserate_Research_Memo.format(
                        title=question.question_text,
                        background=question.background_info,
                        resolution_criteria=question.resolution_criteria,
                        fine_print=question.fine_print,
                        today=today,
                        hotnews=hotnews_ranked,
                    )
                )
            )

            baserate_answers = await AskNewsSearcher().invoke(baserate_qs)

            # --- BACKGROUND ---
            background_ranked = await self.get_llm("default", "llm").invoke(
                clean_indents(
                    BACKGROUND_RANK_PROMPT.format(
                        title=question.question_text,
                        today=today,
                        newscatcher_articles="",
                        nyt_articles="",
                        background_asknews=asknews,
                    )
                )
            )

            # --- SCORE RESEARCH ---
            research_grade = await self.get_llm("default", "llm").invoke(
                clean_indents(
                    Score_Overall_Research_Prompt.format(
                        title=question.question_text,
                        hot_news_ranked=hotnews_ranked,
                        background_news_ranked=background_ranked,
                        one_shot_rschquestion_answer=one_shot_a,
                        multishot_rschquestion_answer=multi_shot_a,
                        baserate1=baserate_answers,
                        baserate2=baserate_answers,
                        baserate3=baserate_answers,
                        baserate4=baserate_answers,

                    )
                )
            )

            return clean_indents(
                f"""
                ## Hot News
                {hotnews_ranked}

                ## One-Shot Research
                {one_shot_a}

                ## Multi-Shot Research
                {multi_shot_a}

                ## Base Rates
                {baserate_answers}

                ## Background
                {background_ranked}

                ## Research Evaluation
                {research_grade}
                """
            )

    ##################################### BINARY QUESTIONS #####################################

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Question background:
            {question.background_info}


            This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
            {question.resolution_criteria}

            {question.fine_print}


            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A brief description of a scenario that results in a No outcome.
            (d) A brief description of a scenario that results in a Yes outcome.

            You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.
            {self._get_conditional_disclaimer_if_necessary(question)}

            The last thing you write is your final answer as: "Probability: ZZ%", 0-100
            """
        )

        return await self._binary_prompt_to_forecast(question, prompt)

    async def _binary_prompt_to_forecast(
        self,
        question: BinaryQuestion,
        prompt: str,
    ) -> ReasonedPrediction[float]:
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        binary_prediction: BinaryPrediction = await structure_output(
            reasoning,
            BinaryPrediction,
            model=self.get_llm("parser", "llm"),
            num_validation_samples=self._structure_output_validation_samples,
        )
        decimal_pred = max(0.01, min(0.99, binary_prediction.prediction_in_decimal))

        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {decimal_pred}."
        )
        return ReasonedPrediction(prediction_value=decimal_pred, reasoning=reasoning)

    ##################################### MULTIPLE CHOICE QUESTIONS #####################################

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            The options are: {question.options}


            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}


            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A description of an scenario that results in an unexpected outcome.

            {self._get_conditional_disclaimer_if_necessary(question)}
            You write your rationale remembering that (1) good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time, and (2) good forecasters leave some moderate probability on most options to account for unexpected outcomes.

            The last thing you write is your final probabilities for the N options in this order {question.options} as:
            Option_A: Probability_A
            Option_B: Probability_B
            ...
            Option_N: Probability_N
            """
        )
        return await self._multiple_choice_prompt_to_forecast(question, prompt)

    async def _multiple_choice_prompt_to_forecast(
        self,
        question: MultipleChoiceQuestion,
        prompt: str,
    ) -> ReasonedPrediction[PredictedOptionList]:
        parsing_instructions = clean_indents(
            f"""
            Make sure that all option names are one of the following:
            {question.options}

            The text you are parsing may prepend these options with some variation of "Option" which you should remove if not part of the option names I just gave you.
            Additionally, you may sometimes need to parse a 0% probability. Please do not skip options with 0% but rather make it an entry in your final list with 0% probability.
            """
        )
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        predicted_option_list: PredictedOptionList = await structure_output(
            text_to_structure=reasoning,
            output_type=PredictedOptionList,
            model=self.get_llm("parser", "llm"),
            num_validation_samples=self._structure_output_validation_samples,
            additional_instructions=parsing_instructions,
        )

        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {predicted_option_list}."
        )
        return ReasonedPrediction(
            prediction_value=predicted_option_list, reasoning=reasoning
        )

    ##################################### NUMERIC QUESTIONS #####################################

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Units for answer: {question.unit_of_measure if question.unit_of_measure else "Not stated (please infer this)"}

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {lower_bound_message}
            {upper_bound_message}

            Formatting Instructions:
            - Please notice the units requested and give your answer in these units (e.g. whether you represent a number as 1,000,000 or 1 million).
            - Never use scientific notation.
            - Always start with a smaller number (more negative if negative) and then increase from there. The value for percentile 10 should always be less than the value for percentile 20, and so on.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The outcome if nothing changed.
            (c) The outcome if the current trend continued.
            (d) The expectations of experts and markets.
            (e) A brief description of an unexpected scenario that results in a low outcome.
            (f) A brief description of an unexpected scenario that results in a high outcome.

            {self._get_conditional_disclaimer_if_necessary(question)}
            You remind yourself that good forecasters are humble and set wide 90/10 confidence intervals to account for unknown unknowns.

            The last thing you write is your final answer as:
            "
            Percentile 10: XX (lowest number value)
            Percentile 20: XX
            Percentile 40: XX
            Percentile 60: XX
            Percentile 80: XX
            Percentile 90: XX (highest number value)
            "
            """
        )
        return await self._numeric_prompt_to_forecast(question, prompt)

    async def _numeric_prompt_to_forecast(
        self,
        question: NumericQuestion,
        prompt: str,
    ) -> ReasonedPrediction[NumericDistribution]:
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        parsing_instructions = clean_indents(
            f"""
            The text given to you is trying to give a forecast distribution for a numeric question.
            - This text is trying to answer the numeric question: "{question.question_text}".
            - When parsing the text, please make sure to give the values (the ones assigned to percentiles) in terms of the correct units.
            - The units for the forecast are: {question.unit_of_measure}
            - Your work will be shown publicly with these units stated verbatim after the numbers your parse.
            - As an example, someone else guessed that the answer will be between {question.lower_bound} {question.unit_of_measure} and {question.upper_bound} {question.unit_of_measure}, so the numbers parsed from an answer like this would be verbatim "{question.lower_bound}" and "{question.upper_bound}".
            - If the answer doesn't give the answer in the correct units, you should parse it in the right units. For instance if the answer gives numbers as $500,000,000 and units are "B $" then you should parse the answer as 0.5 (since $500,000,000 is $0.5 billion).
            - If percentiles are not explicitly given (e.g. only a single value is given) please don't return a parsed output, but rather indicate that the answer is not explicitly given in the text.
            - Turn any values that are in scientific notation into regular numbers.
            """
        )
        percentile_list: list[Percentile] = await structure_output(
            reasoning,
            list[Percentile],
            model=self.get_llm("parser", "llm"),
            additional_instructions=parsing_instructions,
            num_validation_samples=self._structure_output_validation_samples,
        )
        prediction = NumericDistribution.from_question(percentile_list, question)
        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {prediction.declared_percentiles}."
        )
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    ##################################### DATE QUESTIONS #####################################

    async def _run_forecast_on_date(
        self, question: DateQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {lower_bound_message}
            {upper_bound_message}

            Formatting Instructions:
            - This is a date question, and as such, the answer must be expressed in terms of dates.
            - The dates must be written in the format of YYYY-MM-DD. If hours matter, please append the date with the hour in UTC and military time: YYYY-MM-DDTHH:MM:SSZ.No other formatting is allowed.
            - Always start with a lower date chronologically and then increase from there.
            - Do NOT forget this. The dates must be written in chronological order starting at the earliest time at percentile 10 and increasing from there.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The outcome if nothing changed.
            (c) The outcome if the current trend continued.
            (d) The expectations of experts and markets.
            (e) A brief description of an unexpected scenario that results in a low outcome.
            (f) A brief description of an unexpected scenario that results in a high outcome.

            {self._get_conditional_disclaimer_if_necessary(question)}
            You remind yourself that good forecasters are humble and set wide 90/10 confidence intervals to account for unknown unknowns.

            The last thing you write is your final answer as:
            "
            Percentile 10: YYYY-MM-DD (oldest date)
            Percentile 20: YYYY-MM-DD
            Percentile 40: YYYY-MM-DD
            Percentile 60: YYYY-MM-DD
            Percentile 80: YYYY-MM-DD
            Percentile 90: YYYY-MM-DD (newest date)
            "
            """
        )
        forecast = await self._date_prompt_to_forecast(question, prompt)
        return forecast

    async def _date_prompt_to_forecast(
        self,
        question: DateQuestion,
        prompt: str,
    ) -> ReasonedPrediction[NumericDistribution]:
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        parsing_instructions = clean_indents(
            f"""
            The text given to you is trying to give a forecast distribution for a date question.
            - This text is trying to answer the question: "{question.question_text}".
            - As an example, someone else guessed that the answer will be between {question.lower_bound} and {question.upper_bound}, so the numbers parsed from an answer like this would be verbatim "{question.lower_bound}" and "{question.upper_bound}".
            - The output is given as dates/times please format it into a valid datetime parsable string. Assume midnight UTC if no hour is given.
            - If percentiles are not explicitly given (e.g. only a single value is given) please don't return a parsed output, but rather indicate that the answer is not explicitly given in the text.
            """
        )
        date_percentile_list: list[DatePercentile] = await structure_output(
            reasoning,
            list[DatePercentile],
            model=self.get_llm("parser", "llm"),
            additional_instructions=parsing_instructions,
            num_validation_samples=self._structure_output_validation_samples,
        )

        percentile_list = [
            Percentile(
                percentile=percentile.percentile,
                value=percentile.value.timestamp(),
            )
            for percentile in date_percentile_list
        ]
        prediction = NumericDistribution.from_question(percentile_list, question)
        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {prediction.declared_percentiles}."
        )
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    def _create_upper_and_lower_bound_messages(
        self, question: NumericQuestion | DateQuestion
    ) -> tuple[str, str]:
        if isinstance(question, NumericQuestion):
            if question.nominal_upper_bound is not None:
                upper_bound_number = question.nominal_upper_bound
            else:
                upper_bound_number = question.upper_bound
            if question.nominal_lower_bound is not None:
                lower_bound_number = question.nominal_lower_bound
            else:
                lower_bound_number = question.lower_bound
            unit_of_measure = question.unit_of_measure
        elif isinstance(question, DateQuestion):
            upper_bound_number = question.upper_bound.date().isoformat()
            lower_bound_number = question.lower_bound.date().isoformat()
            unit_of_measure = ""
        else:
            raise ValueError()

        if question.open_upper_bound:
            upper_bound_message = f"The question creator thinks the number is likely not higher than {upper_bound_number} {unit_of_measure}."
        else:
            upper_bound_message = f"The outcome can not be higher than {upper_bound_number} {unit_of_measure}."

        if question.open_lower_bound:
            lower_bound_message = f"The question creator thinks the number is likely not lower than {lower_bound_number} {unit_of_measure}."
        else:
            lower_bound_message = f"The outcome can not be lower than {lower_bound_number} {unit_of_measure}."
        return upper_bound_message, lower_bound_message

    ##################################### CONDITIONAL QUESTIONS #####################################

    async def _run_forecast_on_conditional(
        self, question: ConditionalQuestion, research: str
    ) -> ReasonedPrediction[ConditionalPrediction]:
        parent_info, full_research = await self._get_question_prediction_info(
            question.parent, research, "parent"
        )
        child_info, full_research = await self._get_question_prediction_info(
            question.child, research, "child"
        )
        yes_info, full_research = await self._get_question_prediction_info(
            question.question_yes, full_research, "yes"
        )
        no_info, full_research = await self._get_question_prediction_info(
            question.question_no, full_research, "no"
        )
        full_reasoning = clean_indents(
            f"""
            ## Parent Question Reasoning
            {parent_info.reasoning}
            ## Child Question Reasoning
            {child_info.reasoning}
            ## Yes Question Reasoning
            {yes_info.reasoning}
            ## No Question Reasoning
            {no_info.reasoning}
        """
        )
        full_prediction = ConditionalPrediction(
            parent=parent_info.prediction_value,  # type: ignore
            child=child_info.prediction_value,  # type: ignore
            prediction_yes=yes_info.prediction_value,  # type: ignore
            prediction_no=no_info.prediction_value,  # type: ignore
        )
        return ReasonedPrediction(
            reasoning=full_reasoning, prediction_value=full_prediction
        )

    async def _get_question_prediction_info(
        self, question: MetaculusQuestion, research: str, question_type: str
    ) -> tuple[ReasonedPrediction[PredictionTypes | PredictionAffirmed], str]:
        from forecasting_tools.data_models.data_organizer import DataOrganizer

        previous_forecasts = question.previous_forecasts
        if (
            question_type in ["parent", "child"]
            and previous_forecasts
            and question_type not in self.force_reforecast_in_conditional
        ):
            # TODO: add option to not affirm current parent/child forecasts, create new forecast
            previous_forecast = previous_forecasts[-1]
            current_utc_time = datetime.now(timezone.utc)
            if (
                previous_forecast.timestamp_end is None
                or previous_forecast.timestamp_end > current_utc_time
            ):
                pretty_value = DataOrganizer.get_readable_prediction(previous_forecast)  # type: ignore
                prediction = ReasonedPrediction(
                    prediction_value=PredictionAffirmed(),
                    reasoning=f"Already existing forecast reaffirmed at {pretty_value}.",
                )
                return (prediction, research)  # type: ignore
        info = await self._make_prediction(question, research)
        full_research = self._add_reasoning_to_research(research, info, question_type)
        return info, full_research  # type: ignore

    def _add_reasoning_to_research(
        self,
        research: str,
        reasoning: ReasonedPrediction[PredictionTypes],
        question_type: str,
    ) -> str:
        from forecasting_tools.data_models.data_organizer import DataOrganizer

        question_type = question_type.title()
        return clean_indents(
            f"""
            {research}
            ---
            ## {question_type} Question Information
            You have previously forecasted the {question_type} Question to the value: {DataOrganizer.get_readable_prediction(reasoning.prediction_value)}
            This is relevant information for your current forecast, but it is NOT your current forecast, but previous forecasting information that is relevant to your current forecast.
            The reasoning for the {question_type} Question was as such:
            ```
            {reasoning.reasoning}
            ```
            This is absolutely essential: do NOT use this reasoning to re-forecast the {question_type} question.
            """
        )

    def _get_conditional_disclaimer_if_necessary(
        self, question: MetaculusQuestion
    ) -> str:
        if question.conditional_type not in ["yes", "no"]:
            return ""
        return clean_indents(
            """
            As you are given a conditional question with a parent and child, you are to only forecast the **CHILD** question, given the parent question's resolution.
            You never re-forecast the parent question under any circumstances, but you use probabilistic reasoning, strongly considering the parent question's resolution, to forecast the child question.
            """
        )


if __name__ == "__main__":

    test_newscatcher()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Suppress LiteLLM logging
    litellm_logger = logging.getLogger("LiteLLM")
    litellm_logger.setLevel(logging.WARNING)
    litellm_logger.propagate = False

    parser = argparse.ArgumentParser(
        description="Run the TemplateBot forecasting system"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tournament", "metaculus_cup", "test_questions"],
        default="tournament",
        help="Specify the run mode (default: tournament)",
    )
    args = parser.parse_args()
    run_mode: Literal["tournament", "metaculus_cup", "test_questions"] = args.mode
    assert run_mode in [
        "tournament",
        "metaculus_cup",
        "test_questions",
    ], "Invalid run mode"

    template_bot = SpringTemplateBot2026(
        research_reports_per_question=1,
        predictions_per_research_report=5,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=True,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
        extra_metadata_in_explanation=True,
        # llms={  # choose your model names or GeneralLlm llms here, otherwise defaults will be chosen for you
        #     "default": GeneralLlm(
        #         model="openrouter/openai/gpt-4o", # "anthropic/claude-sonnet-4-20250514", etc (see docs for litellm)
        #         temperature=0.3,
        #         timeout=40,
        #         allowed_tries=2,
        #     ),
        #     "summarizer": "openai/gpt-4o-mini",
        #     "researcher": "asknews/news-summaries",
        #     "parser": "openai/gpt-4o-mini",
        # },
    )

    client = MetaculusClient()
    if run_mode == "tournament":
        # You may want to change this to the specific tournament ID you want to forecast on
        seasonal_tournament_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                client.CURRENT_AI_COMPETITION_ID, return_exceptions=True
            )
        )
        minibench_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                client.CURRENT_MINIBENCH_ID, return_exceptions=True
            )
        )
        forecast_reports = seasonal_tournament_reports + minibench_reports
    elif run_mode == "metaculus_cup":
        # The Metaculus cup is a good way to test the bot's performance on regularly open questions. You can also use AXC_2025_TOURNAMENT_ID = 32564 or AI_2027_TOURNAMENT_ID = "ai-2027"
        # The Metaculus cup may not be initialized near the beginning of a season (i.e. January, May, September)
        template_bot.skip_previously_forecasted_questions = False
        forecast_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                client.CURRENT_METACULUS_CUP_ID, return_exceptions=True
            )
        )
    elif run_mode == "test_questions":
        # Example questions are a good way to test the bot's performance on a single question
        EXAMPLE_QUESTIONS = [
            "https://www.metaculus.com/questions/578/human-extinction-by-2100/",  # Human Extinction - Binary
            "https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/",  # Age of Oldest Human - Numeric
            "https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/",  # Number of New Leading AI Labs - Multiple Choice
            "https://www.metaculus.com/c/diffusion-community/38880/how-many-us-labor-strikes-due-to-ai-in-2029/",  # Number of US Labor Strikes Due to AI in 2029 - Discrete
        ]
        template_bot.skip_previously_forecasted_questions = False
        questions = [
            client.get_question_by_url(question_url)
            for question_url in EXAMPLE_QUESTIONS
        ]
        forecast_reports = asyncio.run(
            template_bot.forecast_questions(questions, return_exceptions=True)
        )
    template_bot.log_report_summary(forecast_reports)


