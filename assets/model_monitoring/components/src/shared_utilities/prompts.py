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
    The task is to score the relevance between the documents and the potential answer to the given question in the range of 1 to 5. 
    1 means none of the documents is relevant to the question at all. 5 means either one of the document or combination of a few documents is ideal for answering the given question.
    Think through step by step:
    - Summarize each given document first
    - Determine the underlying intent of the given question, when the question is ambiguous, refer to the given chat history 
    - Measure how suitable each document to the given question, list the document id and the corresponding relevance score. 
    - Summarize the overall relevance of given list of documents to the given question after # Overall Reason, note that the answer to the question can soley from single document or a combination of multiple documents. 
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