import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load the pre-trained conversational model (BlenderBot)
model_name = "facebook/blenderbot-400M-distill"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Streamlit app configuration
st.set_page_config(page_title="Learnix", page_icon=":robot:", layout="wide")

# Apply a custom theme using markdown for a clean UI
st.markdown("""
    <style>
        .css-1aumxhk {background-color: #f0f4f7; padding: 5px; border-radius: 8px; font-family: 'Helvetica', sans-serif; font-size: 18px;}
        .css-1y4g2fu {font-size: 30px; color: #4CAF50;}
        .css-1gw2p9z {font-size: 22px; font-weight: bold; color: #00B0FF;}
    </style>
""", unsafe_allow_html=True)

# Title with Emojis
st.title("💬 **Learnix** 🤖")
st.subheader("🎓 Helping students, learners, and job-seekers with stress, time management, career guidance, and more! 💡")

# Add some information about the chatbot
st.markdown("""
    📚 **How can I help you today?**  
    You can ask me about:
    - Academic stress and study tips 🎯
    - Time management techniques ⏳
    - Career advice 💼
    - Resource management strategies 🔧
""")

# Initialize session state for conversation history
if "history" not in st.session_state:
    st.session_state.history = []

# Function to get a response from the chatbot
def get_response(user_input):
    # Add the latest user input to the conversation history
    st.session_state.history.append(f"User: {user_input}")

    # Keep only the last 3 exchanges for better response quality
    context = " ".join(st.session_state.history[-3:])

    # Tokenize the input
    inputs = tokenizer(context, return_tensors="pt", truncation=True, max_length=512)

    # Generate response from the model
    output_ids = model.generate(**inputs, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    bot_response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Append chatbot response to history
    st.session_state.history.append(f"Bot: {bot_response}")

    return bot_response

# Create a text input box in the main area with a placeholder
user_input = st.text_input("📝 **Ask your question here:**", key="input", placeholder="Type your question...")

# Display the chatbot's response
if user_input:
    with st.spinner("🤖 Thinking... please wait..."):
        # Get chatbot response
        response = get_response(user_input)
        
        # Display the conversation
        st.write("### 💬 Conversation:")
        for message in st.session_state.history[-6:]:  # Display last few exchanges
            if message.startswith("User:"):
                st.write(f"**You:** {message[6:]}")
            else:
                st.write(f"**Learnix:** {message[5:]}")

else:
    st.write("### 🙋‍♂️ Ask me anything! I'm here to help. 😊")

# Optional: Add some interactive buttons to guide users
if st.button("✨ Need advice on Time Management?"):
    st.write("⏳ Time management is all about prioritizing tasks and staying organized. Break your work into manageable chunks, set specific goals, and use a timer to track your progress. 📝")

if st.button("✨ Need Career Guidance?"):
    st.write("💼 A great career starts with understanding your skills, passions, and goals. Focus on networking, learning, and continuously improving your skillset. 📈")

# Add emojis and helpful messages throughout the app to make it more engaging
st.markdown("<br><br><h3 style='color:#4CAF50;'>✨ Your journey starts here! 💡</h3>", unsafe_allow_html=True)

# Footer message with an encouraging note
st.markdown("""
    <br><br>
    <p style="text-align:center; font-size:18px; color:#00B0FF;">
        📍 **Stay curious, stay motivated!**  
        The answers you seek are just a question away. 😊
    </p>
""", unsafe_allow_html=True)
