#okay so this is an example conversational bot, i am gonna make my own bot lter on , this bot is just for understanding, hence minimal, lets go 
# Example dataset: sentences + labels
import random
from textblob import TextBlob
from textblob.np_extractors import ConllExtractor
extractor = ConllExtractor()
texts = [
    "I feel terrible",
    "I am sad",
    "I am okay",
    "I am happy",
    "This is amazing"
]

labels = [
    "negative",
    "negative",
    "neutral",
    "positive",
    "positive"
]
#here we are going turn the words into numbers 
from sklearn.feature_extraction import TfidfVectorizer
vectorizer=TfidfVectorizer()
X=vectorizer.fit_transform(texts)
#now we have X, texts , in numbers

from sklearn.naive_bayes import MultinomialNB
#training a small ml model on texts 
model=MultinomialNB()
model.fit(X,labels)


def main():   
    print("Hello, I am Marvin, the friendly robot.")
    print("You can end this conversation at any time by typing 'bye'")    
    print("After typing each answer, press 'enter'")
    print("How are you today?")

    while True:
        # wait for the user to enter some text
        user_input = input("> ")
        user_vec=vectorizer.transform([user_input])
        prediction=model.predict(user_vec)[0]
        print("Predicted sentiment:", prediction)

        if user_input.lower() == "bye":            
            # if they typed in 'bye' (or even BYE, ByE, byE etc.), break out of the loop
            break
        else:
            # Create a TextBlob based on the user input. Then extract the noun phrases
            user_input_blob = TextBlob(user_input, np_extractor=extractor)                        
            np = user_input_blob.noun_phrases                                    
            response = ""
            if prediction=='negative':
                response = "Oh dear, that sounds bad. "
            elif prediction=='positive':
                response = "Well, that sounds positive. "
            elif prediction=='neutral':
                response="interesting"

            if len(np) != 0:
                # There was at least one noun phrase detected, so ask about that and pluralise it
                # e.g. cat -> cats or mouse -> mice
                response = response + "Can you tell me more about " + np[0].pluralize() + "?"
            else:
                response = response + "Can you tell me more?"
            print(response)
    
    print("It was nice talking to you, goodbye!")

# Start the program
main()