# Project 1 – Rule-Based AI Chatbot (DBot)

## Overview
A simple rule-based chatbot built in Python that responds to predefined user inputs using if-else logic (v1) and dictionary-based lookup (v2). The bot runs in a continuous loop until the user types an exit command.

## Objective
Build a chatbot that:
- Handles greetings and exit commands
- Uses if-else / dictionary logic for responses
- Runs in a continuous loop
- Sanitizes user input (lowercase + strip whitespace)
- Returns a fallback message for unknown inputs

## Files

| File | Description |
|------|-------------|
| `chatbot_v1.py` | Basic version using if-elif chain (as per original spec) |
| `chatbot_v2.py` | Improved version using a dictionary knowledge base + keyword matching |
| `DBot_Project1_Report.docx` | Full project report (objective, code, analysis, viva prep) |

## How to Run

```bash
python chatbot_v1.py
# or
python chatbot_v2.py
```

Type your message and press Enter. Type `exit`, `quit`, `stop`, or `close` to end the conversation.

## Sample Conversation

```
You: hello
DBot: Hello! How can I help you today?

You: who are you
DBot: I am DBot. A chatbot built without any ML - just logic!

You: what is ai
DBot: AI stands for Artificial Intelligence - programming machines to think and act like humans.

You: tell me a joke
DBot: Why do programmers prefer dark mode? Because light attracts bugs!

You: thank you
DBot: You are welcome! Anything else?

You: exit
DBot: Goodbye! It was nice chatting with you.
```

## Key Concepts Used
- Control flow (if-elif, dictionaries)
- String methods: `.lower()`, `.strip()`
- `while True` loop with `break` for exit
- Dictionary `.get()` style lookup for O(1) response matching
- Fallback handling for unrecognized input

## Tech Stack
- Python 3.x (no external libraries required)
