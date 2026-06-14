# chatbot_v2.py
# Rule-Based AI Chatbot - Version 2 (Dictionary-Based, Improved)
# DecodeLabs Industrial Training | Project 1
#
# Improvement over v1:
#   - Uses a dictionary (hash map) instead of long if-elif chain
#   - O(1) lookup time regardless of how many rules we add
#   - Keyword-based matching: more flexible than exact match only
#   - Cleaner structure; easier to expand

# -- Knowledge Base ---------------------------------------------------
# Each key is a keyword to look for in user input.
# Value is the bot's reply.
knowledge_base = {
    # Greetings
    'hello'        : 'Hello! How can I help you today?',
    'hi'           : 'Hi there! What can I do for you?',
    'hey'          : 'Hey! Ask me anything.',
    'good morning' : 'Good morning! Starting the day with AI, nice choice.',
    'good evening' : 'Good evening! Hope your day went well.',
    'good night'   : 'Good night! Rest well.',

    # Identity
    'your name'    : 'I am DBot, your rule-based AI assistant.',
    'who are you'  : 'I am DBot. A chatbot built without any ML - just logic!',
    'who made you' : 'An intern at DecodeLabs built me during the AI training program.',
    'who created'  : 'I was created by a DecodeLabs intern as Project 1.',

    # How are you
    'how are you'  : 'Running perfectly! No bugs detected so far.',
    'how r u'      : 'All systems okay! What about you?',

    # Capabilities
    'what can you do' : 'I can respond to greetings, answer basic questions, tell jokes, and chat a little.',
    'help'             : 'Try asking: who are you, tell me a joke, what is AI, how are you, etc.',

    # AI/Tech topics
    'what is ai'              : 'AI stands for Artificial Intelligence - programming machines to think and act like humans.',
    'artificial intelligence' : 'AI is a broad field covering machine learning, robotics, NLP, and more.',
    'machine learning'        : 'ML is a subset of AI where models learn patterns from data.',
    'deep learning'           : 'Deep Learning uses neural networks with many layers to solve complex problems.',
    'what is python'          : 'Python is a simple, powerful language - very popular in AI and data science.',
    'rule based'              : 'Rule-based AI uses explicit if-else logic instead of learning from data. That is exactly how I work!',

    # Fun / small talk
    'joke'            : 'Why do programmers prefer dark mode? Because light attracts bugs!',
    'another joke'    : "I told my computer I needed a break. Now it won't stop sending me Kit-Kat ads.",
    'are you human'   : 'Nope, I am 100% code. But I try to be helpful like one.',
    'are you a robot' : 'Technically yes - a very basic one that runs on Python and if-else.',
    'favorite color'  : 'Probably blue. The color of screens and logic.',
    'are you smart'   : 'I only know what I have been programmed to know. So... it depends on the programmer!',

    # Time / Date
    'what time'     : 'I cannot fetch live time, but your device clock has you covered!',
    'what is today' : 'I do not have real-time access, but check the date on your screen.',

    # Farewells & thanks
    'bye'       : 'Goodbye! Come back anytime.',
    'goodbye'   : 'See you later!',
    'see you'   : 'Take care!',
    'thank you' : 'You are welcome! Anything else?',
    'thanks'    : 'Happy to help!',
}


def match_response(user_input):
    """
    Try to find a matching keyword in the knowledge base.
    If an exact match fails, check if any key is contained in the input.
    This makes the bot slightly more flexible without ML.
    """
    clean = user_input.lower().strip()

    # 1. Try direct/exact match first (O(1) lookup)
    if clean in knowledge_base:
        return knowledge_base[clean]

    # 2. Try keyword-contains match
    for key in knowledge_base:
        if key in clean:
            return knowledge_base[key]

    # 3. Fallback if nothing matched
    return "Hmm, I don't know how to answer that yet. Try rephrasing or type 'help'."


def main():
    print('=' * 50)
    print('         DBot v2 - AI Chatbot (Improved)')
    print('    DecodeLabs Internship | Project 1')
    print('=' * 50)
    print("Commands: type 'help' for tips | 'exit' to quit\n")

    while True:
        raw = input('You: ')
        clean = raw.lower().strip()

        # Exit handling
        if clean in ['exit', 'quit', 'stop', 'bye', 'close']:
            print('DBot: Goodbye! It was nice chatting with you.')
            break

        # Empty input guard
        if not clean:
            print('DBot: Go on, type something!')
            continue

        response = match_response(clean)
        print(f'DBot: {response}\n')


if __name__ == '__main__':
    main()
