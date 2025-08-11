import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory

# --- 1. Model and Provider Configuration ---
# We define all our available models in a dictionary for easy management.
AVAILABLE_MODELS = {
    "Together.xyz": {
        "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo": "Llama 3.2 90B",
        "Qwen/Qwen2.5-72B-Instruct-Turbo": "Qwen 2.5 72B",
    },
    "OpenAI": {
        "gpt-4o": "GPT-4o",
        "gpt-4o-mini": "GPT-4o Mini",
    },
    "Google": {
        "gemini-1.5-pro-latest": "Gemini 1.5 Pro",
        "gemini-1.5-flash-latest": "Gemini 1.5 Flash",
    }
}

# --- 2. Central Function to Create the LLM "Brain" ---
# This function creates the correct LLM object based on the user's selection.
def get_llm(provider, model_name):
    """Initializes and returns the selected LLM chain."""
    try:
        if provider == "OpenAI":
            api_key = st.secrets["OPENAI_API_KEY"]
            return ChatOpenAI(model_name=model_name, openai_api_key=api_key)

        elif provider == "Google":
            api_key = st.secrets["GOOGLE_API_KEY"]
            return ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)

        elif provider == "Together.xyz":
            api_key = st.secrets["TOGETHER_API_KEY"]
            return ChatOpenAI(
                model=model_name,
                openai_api_key=api_key,
                openai_api_base="https://api.together.xyz/v1"
            )
    except Exception as e:
        st.error(f"Error initializing model: {e}. Make sure your API key is set correctly in st.secrets.")
        return None

# --- 3. Streamlit App Interface ---
st.title("🗣️ Multi-Model Conversational Chatbot")

# --- Sidebar for Model Selection ---
with st.sidebar:
    st.header("Model Selection")
    # Dropdown for provider
    selected_provider = st.selectbox("Choose a provider:", list(AVAILABLE_MODELS.keys()))
    
    # Dropdown for model, dynamically updated based on provider
    provider_models = AVAILABLE_MODELS[selected_provider]
    selected_model_key = st.selectbox("Choose a model:", list(provider_models.keys()), format_func=lambda key: provider_models[key])

# --- 4. Session State and Conversation Management ---
# We now store the model choice and the conversation chain in the session state.
if "selected_model" not in st.session_state or st.session_state.selected_model != selected_model_key:
    st.session_state.selected_model = selected_model_key
    
    # When the model changes, create a new chain and get the knowledge cutoff.
    with st.spinner(f"Initializing {provider_models[selected_model_key]}..."):
        llm = get_llm(selected_provider, selected_model_key)
        if llm:
            conversation = ConversationChain(
                llm=llm,
                memory=ConversationBufferWindowMemory(k=3, return_messages=True)
            )
            
            # Ask the knowledge cutoff question
            cutoff_response = conversation.predict(input="What is your knowledge cutoff date? Answer with only the date.")
            
            # Reset chat history and start with the new model's introduction
            st.session_state.messages = [
                {"role": "assistant", "content": f"Hi! I'm {provider_models[selected_model_key]}. My knowledge cutoff is roughly {cutoff_response}. How can I help you?"}
            ]
            st.session_state.conversation = conversation
        else:
            st.session_state.messages = [{"role": "assistant", "content": "Could not initialize model. Please check settings."}]

# --- 5. Display Chat History ---
for message in st.session_state.get("messages", []):
    with st.chat_message(message["role"]):
        st.write(message["content"])

# --- 6. Handle User Input and Generate Response ---
if prompt := st.chat_input("Your question"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Generate and display assistant response
    if "conversation" in st.session_state:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.conversation.predict(input=prompt)
                st.write(response)
                # Add assistant response to history
                st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.error("The conversation chain is not initialized. Please select a valid model.")
