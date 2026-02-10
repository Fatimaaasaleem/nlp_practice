import random 
from textblob import TextBlob 
from textblob.np_extractors import ConllExtractor 
extractor=ConllExtractor()
texts = [
    # negative
    "I feel terrible",
    "I am sad",
    "This is awful",
    "I hate how this turned out",
    "I feel miserable today",
    "Nothing is going right",
    "I am very disappointed",
    "This makes me angry",
    "I feel frustrated",
    "I am stressed out",
    "I feel hopeless",
    "Today has been exhausting",
    "I regret everything",
    "This ruined my mood",
    "I am feeling low",
    "I am upset about this",
    "This is really bad",
    "I feel anxious",
    "I am not okay",
    "I feel overwhelmed",

    # neutral
    "I am okay",
    "It is fine",
    "Nothing special today",
    "I feel normal",
    "This is acceptable",
    "It is an average day",
    "I have no strong feelings",
    "Things are stable",
    "I am just going through the day",
    "This is neither good nor bad",
    "I feel calm",
    "Everything is usual",
    "I am managing",
    "It is what it is",
    "I am indifferent",
    "No major changes today",
    "I feel balanced",
    "Just another day",
    "I am neutral about it",
    "Nothing remarkable happened",

    # positive
    "I am happy",
    "This is amazing",
    "I feel great",
    "I am excited",
    "This made my day",
    "I feel fantastic",
    "Everything is going well",
    "I am really satisfied",
    "This is wonderful",
    "I feel optimistic",
    "I am enjoying this",
    "I feel confident",
    "Today is a good day",
    "I am proud of myself",
    "This makes me smile",
    "I feel motivated",
    "I am grateful",
    "I feel joyful",
    "Things are looking good",
    "I am very pleased"
]
labels = [
    # negative (20)
    "negative","negative","negative","negative","negative",
    "negative","negative","negative","negative","negative",
    "negative","negative","negative","negative","negative",
    "negative","negative","negative","negative","negative",

    # neutral (20)
    "neutral","neutral","neutral","neutral","neutral",
    "neutral","neutral","neutral","neutral","neutral",
    "neutral","neutral","neutral","neutral","neutral",
    "neutral","neutral","neutral","neutral","neutral",

    # positive (20)
    "positive","positive","positive","positive","positive",
    "positive","positive","positive","positive","positive",
    "positive","positive","positive","positive","positive",
    "positive","positive","positive","positive","positive"
]
random_responses = [
        "That is quite interesting, please tell me more.",
        "I see. Do go on.",
        "Why do you say that?",
        "Let's change the subject.",
        "Interesting thought!"
    ] 
from sklearn.feature_extraction import TfidfVectorizer
vectorizer=TfidfVectorizer()
X=vectorizer.fit_transform(texts)

from sklearn.naive_bayes import MultinomialNB
model=MultinomialNB()
model.fit(X,labels)
def predict_sentiment(text):
    vec=vectorizer.transform([text])
    return model.predict(vec)[0]

def main():
    print('Hello, my name is Alex, a friendly bot!')
    print('You can end the conversation at any time by typing bye')
    print('After typing each answer, press \'enter\'')
    print('How are you today?')
    while True:
        user_input=input('> ')
        sentiment=predict_sentiment(user_input)
        if user_input.lower()=='bye':
            break
        else:
            user_input_blob=TextBlob(user_input,np_extractor=extractor)
            np=user_input_blob.noun_phrases
            response=" "
            if sentiment=='negative':
                response="Oh dear, that sounds bad. "
            elif sentiment=='positive':
                response="Well, that sounds positive."
            elif sentiment=='neutral':
                         response=random.choice(random_responses)
                         
                         
            if len(np)!=0:
                response=response+'Can you tell me more about '+  np[0].pluralize()+"?"
            else:
                response=response + 'Can you tell me more?'  
            print(response)
            
    print('It was nice talking to you, goodbye!')
main()                            
