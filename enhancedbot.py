import random 
from textblob import TextBlob 
from textblob.np_extractors import ConllExtractor 
extractor = ConllExtractor()

# Responses
negative_responses = [
    "Oh no, that sounds really tough.",
    "I’m sorry to hear that. What happened?",
    "That sounds frustrating. Do tell me more.",
    "Ouch, that must have been hard. Can you elaborate?",
    "I understand, that seems difficult. Please share more.",
    "Oh dear, that doesn’t sound nice. How did it make you feel?",
    "Yikes, that’s rough. Can you tell me what happened?",
    "I hear you, that must have been upsetting.",
    "That sounds really challenging. Tell me more.",
    "I get it, that must have been stressful."
]

neutral_responses = [
    "I see. Can you tell me more about that?",
    "Hmm, okay. What happened next?",
    "Alright. Can you explain a bit more?",
    "Interesting. Could you elaborate?",
    "Got it. How did that go?",
    "Okay, please continue.",
    "I understand. Can you tell me more?",
    "Alright, tell me more about it.",
    "Noted. Can you expand a little?",
    "Hmm, I see. Go on."
]

positive_responses = [
    "That's wonderful! Can you tell me more?",
    "I’m glad to hear that! How did that make you feel?",
    "Yay! That sounds amazing!",
    "Awesome! What else happened?",
    "That’s really good news. Please share more!",
    "Wow, that must have felt great!",
    "I’m happy for you! Tell me more about it.",
    "Fantastic! What made it so enjoyable?",
    "Lovely! How did that go?",
    "That sounds delightful!"
]

# Follow-up templates
neutral_followups = [
    "Can you tell me more about {}?", 
    "What happened with {} exactly?", 
    "How did you feel about {}?", 
    "Could you describe {} a bit more?", 
    "Why do you think {} happened?"
]

positive_followups = [
    "That sounds great! What made {} so enjoyable?", 
    "I’m happy for you! Can you tell me more about {}?", 
    "How did it feel when you {}?", 
    "That’s wonderful! What else happened with {}?", 
    "I’m curious, what led to {}?"
]

negative_followups = [
    "That sounds tough. Can you explain more about {}?", 
    "I’m sorry that {} happened. How did it make you feel?", 
    "That must have been hard. What did you do about {}?", 
    "Can you describe {} a bit more?", 
    "I understand. How did you cope with {}?"
]

# Training data
texts = [
    # negative
    "I feel terrible", "I am sad", "This is awful", "I hate how this turned out", 
    "I feel miserable today", "Nothing is going right", "I am very disappointed", 
    "This makes me angry", "I feel frustrated", "I am stressed out", 
    "I feel hopeless", "Today has been exhausting", "I regret everything", 
    "This ruined my mood", "I am feeling low", "I am upset about this", 
    "This is really bad", "I feel anxious", "I am not okay", "I feel overwhelmed",
    # neutral
    "I am okay", "It is fine", "Nothing special today", "I feel normal", 
    "This is acceptable", "It is an average day", "I have no strong feelings", 
    "Things are stable", "I am just going through the day", 
    "This is neither good nor bad", "I feel calm", "Everything is usual", 
    "I am managing", "It is what it is", "I am indifferent", "No major changes today", 
    "I feel balanced", "Just another day", "I am neutral about it", 
    "Nothing remarkable happened",
    # positive
    "I am happy", "This is amazing", "I feel great", "I am excited", "This made my day", 
    "I feel fantastic", "Everything is going well", "I am really satisfied", 
    "This is wonderful", "I feel optimistic", "I am enjoying this", "I feel confident", 
    "Today is a good day", "I am proud of myself", "This makes me smile", 
    "I feel motivated", "I am grateful", "I feel joyful", "Things are looking good", 
    "I am very pleased"
]

labels = [
    # negative (20)
    "negative","negative","negative","negative","negative","negative","negative","negative",
    "negative","negative","negative","negative","negative","negative","negative","negative",
    "negative","negative","negative","negative",
    # neutral (20)
    "neutral","neutral","neutral","neutral","neutral","neutral","neutral","neutral","neutral",
    "neutral","neutral","neutral","neutral","neutral","neutral","neutral","neutral","neutral","neutral","neutral",
    # positive (20)
    "positive","positive","positive","positive","positive","positive","positive","positive",
    "positive","positive","positive","positive","positive","positive","positive","positive",
    "positive","positive","positive","positive"
]

# Train Naive Bayes model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)
model = MultinomialNB()
model.fit(X, labels)

def predict_sentiment(text):
    vec = vectorizer.transform([text])
    return model.predict(vec)[0]

# Main bot loop
def main():
    user_sentiments = []
    print('Hello, my name is Alex, a friendly bot!')
    print('You can end the conversation at any time by typing bye')
    print('After typing each answer, press \'enter\'')
    print('How are you today?')

    while True:
        user_input = input('> ')
        if user_input.lower() == 'bye':
            break

        # Predict sentiment
        sentiment_label = predict_sentiment(user_input)
        if sentiment_label == 'positive':
            sentiment_value = 1
            followups = positive_followups
            response_list = positive_responses
        elif sentiment_label == 'neutral':
            sentiment_value = 0
            followups = neutral_followups
            response_list = neutral_responses
        else:
            sentiment_value = -1
            followups = negative_followups
            response_list = negative_responses

        user_sentiments.append(sentiment_value)
        average_sentiment = sum(user_sentiments) / len(user_sentiments)

        # Prepare base response
        response = random.choice(response_list)

        # Extract phrases
        user_input_blob = TextBlob(user_input, np_extractor=extractor)
        np = user_input_blob.noun_phrases
        verbs = [word for (word, tag) in user_input_blob.tags if tag.startswith('VB')]

        # Append a random follow-up using noun or verb if available
        if np:
            response += ' ' + random.choice(followups).format(np[0].pluralize())
        elif verbs:
            response += ' ' + random.choice(followups).format(verbs[0])

        print(response)

    print('It was nice talking to you, goodbye!')

main()
