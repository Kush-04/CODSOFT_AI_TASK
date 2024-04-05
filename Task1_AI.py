def chatbot():
  # Welcome message
  print("Hello! I'm a simple chatbot. How can I help you today?")

  while True:
    # Get user input
    user_input = input("> ").lower()

    # Exit condition
    if user_input == "bye":
      print("Thank you for chatting! See you next time.")
      break

    # Greeting response
    if any(greeting in user_input for greeting in ["hi", "hello", "hey"]):
      print("Hi there! How can I be of service?")
      continue

    # Question about the chatbot
    if "what can you do" in user_input or "what do you do" in user_input:
      print("I can answer basic questions and respond to greetings for now. I'm still under development!")
      continue

    # Default response
    print("Sorry, I can't understand that yet. Try asking a different question or saying 'hi'.")

# Start the chatbot interaction
chatbot()
