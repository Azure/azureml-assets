# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains prompts for action analyzer."""

BERTOPIC_DEFAULT_PROMPT = """
    This is a list of texts where each collection of texts describe a topic. After each collection of texts, \
    the name of the topic they represent is mentioned as a short-highly-descriptive title
    ---
    Topic:
    Sample texts from this topic:
    - Traditional diets in most cultures were primarily plant-based with a little meat on top, \
    but with the rise of industrial style meat production and factory farming, meat has become a staple food.
    - Meat, but especially beef, is the worst food in terms of emissions.
    - Eating meat doesn't make you a bad person, not eating meat doesn't make you a good one.

    Keywords: meat beef eat eating emissions steak food health processed chicken
    Topic name: Environmental impacts of eating meat
    ---
    Topic:
    Sample texts from this topic:
    - I have ordered the product weeks ago but it still has not arrived!
    - The website mentions that it only takes a couple of days to deliver but I still have not received mine.
    - I got a message stating that I received the monitor but that is not true!
    - It took a month longer to deliver than was advised...

    Keywords: deliver weeks product shipping long delivery received arrived arrive week
    Topic name: Shipping and delivery issues
    ---
    Topic:
    Sample texts from this topic:
    [DOCUMENTS]
    Keywords: [KEYWORDS]
    Topic name:"""


RELEVANCE_TEMPLATE = """
    A chat history between user and bot is shown below
    A list of documents is shown below in json format, and each document has one unique id.
    These listed documents are used as context to answer the given question.
    The task is to score the relevance between the documents and the potential answer \
    to the given question in the range of 1 to 5.
    1 means none of the documents is relevant to the question at all. \
    5 means either one of the document or combination of a few documents is ideal for answering the given question.
    Think through step by step:
    - Summarize each given document first
    - Determine the underlying intent of the given question, when the question is ambiguous, \
    refer to the given chat history
    - Measure how suitable each document to the given question, \
    list the document id and the corresponding relevance score.
    - Summarize the overall relevance of given list of documents to the given question after # Overall Reason, \
    note that the answer to the question can soley from single document or a combination of multiple documents.
    - Finally, output "# Result" followed by a score from 1 to 5.

    # Question
    {{ query }}
    # Chat History
    {{ history }}
    # Documents
    ---BEGIN RETRIEVED DOCUMENTS---
    {{ FullBody }}
    ---END RETRIEVED DOCUMENTS---
"""


RELEVANCE_METRIC_TEMPLATE = """
    system:
    You are an AI assistant. You will be given the definition of an evaluation metric for assessing the quality of \
    an answer in a question-answering task. Your job is to compute an accurate evaluation score using the \
    provided evaluation metric.
    user:
    Relevance measures how well the answer addresses the main aspects of the question, based on the context. \
    Consider whether all and only the important aspects are contained in the answer when evaluating relevance. \
    Given the context and question, score the relevance of the answer between one to five stars using \
    the following rating scale:
    One star: the answer completely lacks relevance
    Two stars: the answer mostly lacks relevance
    Three stars: the answer is partially relevant
    Four stars: the answer is mostly relevant
    Five stars: the answer has perfect relevance

    This rating value should always be an integer between 1 and 5. \
    So the rating produced should be 1 or 2 or 3 or 4 or 5.

    context: Marie Curie was a Polish-born physicist and chemist who pioneered research on radioactivity and \
    was the first woman to win a Nobel Prize.
    question: What field did Marie Curie excel in?
    answer: Marie Curie was a renowned painter who focused mainly on impressionist styles and techniques.
    stars: 1

    context: The Beatles were an English rock band formed in Liverpool in 1960, and they are widely regarded \
    as the most influential music band in history.
    question: Where were The Beatles formed?
    answer: The band The Beatles began their journey in London, England, and they changed the history of music.
    stars: 2

    context: The recent Mars rover, Perseverance, was launched in 2020 with the main goal of searching for \
    signs of ancient life on Mars. The rover also carries an experiment called MOXIE, \
    which aims to generate oxygen from the Martian atmosphere.
    question: What are the main goals of Perseverance Mars rover mission?
    answer: The Perseverance Mars rover mission focuses on searching for signs of ancient life on Mars.
    stars: 3

    context: The Mediterranean diet is a commonly recommended dietary plan that emphasizes fruits, vegetables, \
    whole grains, legumes, lean proteins, and healthy fats. Studies have shown that it offers numerous health \
    benefits, including a reduced risk of heart disease and improved cognitive health.
    question: What are the main components of the Mediterranean diet?
    answer: The Mediterranean diet primarily consists of fruits, vegetables, whole grains, and legumes.
    stars: 4

    context: The Queen's Royal Castle is a well-known tourist attraction in the United Kingdom. It spans over \
    500 acres and contains extensive gardens and parks. The castle was built in the 15th century and has been \
    home to generations of royalty.
    question: What are the main attractions of the Queen's Royal Castle?
    answer: The main attractions of the Queen's Royal Castle are its expansive 500-acre grounds, \
    extensive gardens, parks, and the historical castle itself, which dates back to the 15th century \
    and has housed generations of royalty.
    stars: 5

    context: {{context}}
    question: {{question}}
    answer: {{answer}}
    stars:
"""