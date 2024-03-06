# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains prompts for action analyzer."""

import json

RATING = "rating"
INDEX = "index"
PROMPT = "prompt"
COMPLETION = "completion"
CONTEXT = "context"
MIN_RATING = 1
MAX_RATING = 5

RETRIEVAL_DOCUMENT_RELEVANCE_TEMPLATE = "\n\n".join(
    [
        "System:",
        f"You are an AI assistant. You will be given the definition of an evaluation metric for assessing the \
            quality of an {CONTEXT} in a question-answering task. Your job is to compute an accurate evaluation \
            score using the provided evaluation metric.",
        f"Index quality measures how well the retrieved document {CONTEXT} provides relevant background information \
            or knowledge to answer this question {PROMPT}. Consider whether \
            all and only the important aspects are contained in the {CONTEXT} when \
            evaluating index quality. Given the {PROMPT} and {COMPLETION}, \
            score the index quality of the {CONTEXT} \
            between {MIN_RATING} to {MAX_RATING} using the following {RATING} scale:",
        f"{RATING} 1: the document completely lacks information or knowledge of the question",
        f"{RATING} 2: the document mostly lacks information or knowledge of the question",
        f"{RATING} 3: the document is partially information or knowledge of the question",
        f"{RATING} 4: the document is mostly information or knowledge of the question",
        f"{RATING} 5: the document has perfect information or knowledge of the question",
        f"The score should be integer only, between {MIN_RATING} and {MAX_RATING}.",
        "## Example Task #0",
        json.dumps({
            CONTEXT: "Python is a popular general-purpose programming language that can be used for a wide variety of applications. It includes high-level data structures, dynamic typing, dynamic binding, and many more features that make it as useful for complex application development as it is for scripting or glue code that connects components together.",  # noqa: E501
            PROMPT: "How can I use the python tool in the langchain frame",
            COMPLETION: "Sorry, the provided context does not include information about question.",
        }),
        "A good example response would be:",
        "## Example Task #0:",
        json.dumps({
            RATING: 1,
        }),
        "## Example Task #1",
        json.dumps({
            CONTEXT: "Marie Curie was a Polish-born physicist and chemist who pioneered research on radioactivity \
                and was the first woman to win a Nobel Prize.",
            PROMPT: "What field did Thomas Edison excel in?",
            COMPLETION: "This context does not include what field Thomas Edison exceled in. It is the introduction of Marie Curie.", # noqa: E501
        }),
        "A good example response would be:",
        "## Example Task #1:",
        json.dumps({
            RATING: 1,
        }),
        "## Example Task #2",
        json.dumps({
            CONTEXT: "SQL stands for Structured Query Language and is a computer language that we use to interact with a relational database. SQL is a tool for organizing, managing, and retrieving archived data from a computer database.",  # noqa: E501
            PROMPT: "When was SQL developed and what is SQL used for?",
            COMPLETION: "The developed date for SQL is not explicitly mentioned in the provided content. However\
            for SQL usage, SQL is a powerful programming language used to query and manipulate data stored in relational databases.", # noqa: E501
        }),
        "A good example response would be:",
        "## Example Task #2:",
        json.dumps({
            RATING: 2,
        }),
        "## Example Task #3",
        json.dumps({
            CONTEXT: "The Beatles were an English rock band formed in Liverpool in 1960, and they are widely \
                regarded as the most influential music band in history.",
            PROMPT: "Where were The Beatles formed?",
            COMPLETION: "The band The Beatles began their journey in London, England, and they changed the \
                history of music.",
        }),
        "A good example response would be:",
        "## Example Task #3:",
        json.dumps({
            RATING: 2,
        }),
        "## Example Task #4",
        json.dumps({
            CONTEXT: "The recent Mars rover, Perseverance, was launched in 2020 with the main goal of searching \
                for signs of ancient life on Mars. The rover also carries an experiment called MOXIE, which aims \
                    to generate oxygen from the Martian atmosphere.",
            PROMPT: "What are the main goals of Perseverance Mars rover mission?",
            COMPLETION: "The Perseverance Mars rover mission focuses on searching for signs of ancient life on Mars.",
        }),
        "A good example response would be:",
        "## Example Task #4",
        json.dumps({
            RATING: 3,
        }),
        "## Example Task #5",
        json.dumps({
            CONTEXT: "The Mediterranean diet is a commonly recommended dietary plan that emphasizes fruits, \
                vegetables, whole grains, legumes, lean proteins, and healthy fats. Studies have shown that it \
                    offers numerous health benefits, including a reduced risk of heart disease and improved \
                        cognitive health.",
            PROMPT: "What are the main components of the Mediterranean diet?",
            COMPLETION: "The Mediterranean diet primarily consists of fruits, vegetables, whole grains, and legumes.",
        }),
        "A good example response would be:",
        "## Example Task #5:",
        json.dumps({
            RATING: 4,
        }),
        "## Example Task #6",
        json.dumps({
            CONTEXT: "The Queen's Royal Castle is a well-known tourist attraction in the United Kingdom. It spans \
                over 500 acres and contains extensive gardens and parks. The castle was built in the 15th century \
                    and has been home to generations of royalty.",
            PROMPT: "What are the main attractions of the Queen's Royal Castle?",
            COMPLETION: "The main attractions of the Queen's Royal Castle are its expansive 500-acre grounds, \
                extensive gardens, parks, and the historical castle itself, which dates back to the 15th century \
                    and has housed generations of royalty.",
        }),
        "A good example response would be:",
        "## Example Task #6:",
        json.dumps({
            RATING: 5,
        }),
        "User:",
        "{input_samples}"
    ]
)

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
