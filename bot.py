import random

def main():
    print("Hello, my name is Alex. I am a conversational bot!")
    print("You can end this conversation anytime by typing 'bye'.")
    print("After typing each answer, press Enter.\n")

    random_responses = [
        "That is quite interesting, please tell me more.",
        "I see. Do go on.",
        "Why do you say that?",
        "Let's change the subject.",
        "Interesting thought!"
    ]

    user_input = input("How are you today? ")

    while user_input.lower() != "bye":
        print(random.choice(random_responses))
        user_input = input("> ")

    print("Goodbye! Have a nice day.")

if __name__ == "__main__":
    main()
