# chatbot_v1.py
# Rule-Based AI Chatbot - Version 1 (Basic Terminal)
# DecodeLabs Industrial Training | Project 1

def get_response(user_input):
    """Match user input against known intents and return a reply."""

    # Sanitize: remove extra spaces, convert to lowercase
    text = user_input.lower().strip()

    # --- Greetings ---
    if text in ['hello', 'hi', 'hey', 'hii', 'helo']:
        return 'Hello! How can I help you today?'

    elif text in ['good morning', 'morning']:
        return 'Good morning! Hope you have a great day.'

    elif text in ['good evening', 'evening']:
        return 'Good evening! What can I do for you?'

    # --- Identity ---
    elif text in ['what is your name', 'who are you', 'your name']:
        return 'I am DBot, a simple rule-based chatbot built during my DecodeLabs internship.'

    elif text in ['who made you', 'who created you', 'who built you']:
        return 'I was built by an intern at DecodeLabs as part of the AI training program.'

    # --- How are you ---
    elif text in ['how are you', 'how r u', 'how are you doing']:
        return 'I am just code, but I am running fine! What about you?'

    # --- Capabilities ---
    elif text in ['what can you do', 'help', 'what do you do']:
        return 'I can answer basic questions, respond to greetings, and have simple conversations. Try asking me something!'

    # --- Time / Date (static reply since no live data) ---
    elif text in ['what is the time', 'time', 'current time']:
        return 'I cannot fetch live time, but you can check the clock on your device!'

    elif text in ['what is today', 'date', "what's the date"]:
        return 'I do not have access to live data, but your device will show the date.'

    # --- AI / Tech questions ---
    elif text in ['what is ai', 'what is artificial intelligence']:
        return 'Artificial Intelligence is when machines are programmed to mimic human thinking and decision-making.'

    elif text in ['what is machine learning', 'ml']:
        return 'Machine Learning is a branch of AI where systems learn from data instead of being explicitly programmed.'

    elif text in ['what is python']:
        return 'Python is a popular programming language widely used in AI, data science, and web development.'

    # --- Fun / Small talk ---
    elif text in ['tell me a joke', 'joke']:
        return 'Why do programmers prefer dark mode? Because light attracts bugs!'

    elif text in ['are you human', 'are you a robot']:
        return 'I am a program, not a human. But I try my best to be helpful!'

    elif text in ['favourite color', 'favorite color', 'your color']:
        return 'Probably blue. The color of logic and code.'

    # --- Farewell ---
    elif text in ['bye', 'goodbye', 'see you', 'take care', 'cya']:
        return 'Goodbye! It was nice talking to you.'

    elif text in ['thanks', 'thank you', 'thankyou', 'thx']:
        return 'You are welcome! Is there anything else I can help with?'

    # --- Fallback ---
    else:
        return "I am not sure how to respond to that yet. Try asking something else!"


def main():
    print("=" * 45)
    print("      DBot - Rule-Based AI Chatbot")
    print("   DecodeLabs Internship | Project 1")
    print("=" * 45)
    print("Type 'exit' or 'quit' to stop the chatbot.\n")

    # Infinite loop - chatbot keeps running until exit command
    while True:
        raw_input_text = input('You: ')

        # Sanitize input before processing
        clean_input = raw_input_text.lower().strip()

        # Exit condition - clean break
        if clean_input in ['exit', 'quit', 'stop', 'close']:
            print('DBot: Alright, shutting down. Goodbye!')
            break

        # Skip if user typed nothing
        if clean_input == '':
            print('DBot: Please type something!')
            continue

        # Get and print the response
        reply = get_response(clean_input)
        print(f'DBot: {reply}')
        print()


if __name__ == "__main__":
    main()
