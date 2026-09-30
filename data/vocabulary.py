"""Target vocabularies for the presentation / Q&A prototype, one per dataset.

Keys are the labels the classifier will predict; values are the dataset's own
name for that sign. Edit these to change the vocabulary, then re-run
data/prepare_dataset.py (clips that were already extracted are reused).

ISL and ASL are different languages with different signs, so each dataset has
its own vocabulary and its own processed output. They are never merged.
"""

# ASL Citizen glosses (ASL). Where the dataset has numbered variants of a sign
# (different forms of the same word, e.g. WHAT1 / WHAT2), the variant with the
# most training clips is used so each class is a single form. RIGHT is left out
# because its three variants likely carry different meanings.
# Not in ASL Citizen at all: QUESTION, REPEAT, EXPLAIN, AGREE, FIVE, TEN.
ASL_CITIZEN = {
    "yes": "YES",
    "no": "NO",
    "help": "HELP",
    "please": "PLEASE",
    "sorry": "SORRY",
    "thank you": "THANKYOU",
    "hello": "HELLO",
    "welcome": "WELCOME1",
    "what": "WHAT1",
    "why": "WHY",
    "how": "HOW1",
    "who": "WHO",
    "where": "WHERE",
    "when": "WHEN",
    "understand": "UNDERSTAND",
    "again": "AGAIN",
    "more": "MORE",
    "finish": "FINISH",
    "know": "KNOW",
    "think": "THINK",
    "want": "WANT1",
    "need": "NEED",
    "good": "GOOD",
    "bad": "BAD",
    "wrong": "WRONG",
    "learn": "LEARN",
    "example": "EXAMPLE",
    "time": "TIME",
    "name": "NAME",
    "deaf": "DEAF1",
    "hearing": "HEARING",
    "sign": "SIGN",
    "language": "LANGUAGE",
    "idea": "IDEA",
    "problem": "PROBLEM2",
    "answer": "ANSWER",
    "ask": "ASK",
    "easy": "EASY",
    "hard": "HARD",
    "important": "IMPORTANT",
    "next": "NEXT",
    "discuss": "DISCUSS",
    "one": "ONE",
    "two": "TWO",
    "three": "THREE",
    "four": "FOUR",
    "six": "SIX",
    "seven": "SEVEN",
    "eight": "EIGHT",
    "nine": "NINE",
}

# INCLUDE signs (ISL), as (category, word) from the dataset's folder names.
# INCLUDE has no signs for yes, no, question, help, please, sorry, question
# words (what/why/how/...), or numbers, so the ISL vocabulary is weaker for Q&A.
INCLUDE = {
    "hello": ("Greetings", "Hello"),
    "how are you": ("Greetings", "How are you"),
    "alright": ("Greetings", "Alright"),
    "good morning": ("Greetings", "Good Morning"),
    "good afternoon": ("Greetings", "Good afternoon"),
    "good evening": ("Greetings", "Good evening"),
    "good night": ("Greetings", "Good night"),
    "thank you": ("Greetings", "Thank you"),
    "pleased": ("Greetings", "Pleased"),
    "i": ("Pronouns", "I"),
    "you": ("Pronouns", "you"),
    "he": ("Pronouns", "he"),
    "she": ("Pronouns", "she"),
    "it": ("Pronouns", "it"),
    "we": ("Pronouns", "we"),
    "you (plural)": ("Pronouns", "you (plural)"),
    "they": ("Pronouns", "they"),
    "good": ("Adjectives", "good"),
    "bad": ("Adjectives", "bad"),
    "new": ("Adjectives", "new"),
    "old": ("Adjectives", "old"),
    "happy": ("Adjectives", "happy"),
    "sad": ("Adjectives", "sad"),
    "slow": ("Adjectives", "slow"),
    "fast": ("Adjectives", "fast"),
    "deaf": ("Adjectives", "Deaf"),
    "today": ("Days_and_Time", "Today"),
    "tomorrow": ("Days_and_Time", "Tomorrow"),
    "yesterday": ("Days_and_Time", "Yesterday"),
    "time": ("Days_and_Time", "Time"),
    "minute": ("Days_and_Time", "Minute"),
    "hour": ("Days_and_Time", "Hour"),
    "sign": ("Society", "Sign"),
    "science": ("Society", "Science"),
    "technology": ("Society", "Technology"),
    "teacher": ("Jobs", "Teacher"),
    "student": ("Jobs", "Student"),
    "university": ("Places", "University"),
    "school": ("Places", "School"),
    "computer": ("Electronics", "Computer"),
}
