from flask import Flask, render_template, request, redirect, url_for, session
import os
from langchain.schema import AIMessage, HumanMessage
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI, ChatAnthropic

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Secret key for session encryption

def get_content_from_url(url):
    try:
        loader = WebBaseLoader(url)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        document_chunks = text_splitter.split_documents(documents)
        
        content = "\n".join([chunk.page_content for chunk in document_chunks])
        return content

    except Exception as e:
        return f"Error loading content from the URL: {e}"

def generate_response(user_input, website_content):
    api_key = session.get('api_key')
    api_provider = session.get('api_provider')
    
    if api_provider == 'openai':
        model_name = "gpt-3.5-turbo"  # You can change this to any available OpenAI model
        llm = ChatOpenAI(model_name=model_name, openai_api_key=api_key)
    elif api_provider == 'anthropic':
        model_name = "claude-v1"  # You can change this to any available Anthropic model
        llm = ChatAnthropic(model=model_name, anthropic_api_key=api_key)
    else:
        return "Invalid API provider selected."
    
    prompt = f"""
    The following is content from a website:
    {website_content}

    Based on this content, please answer the user's question:
    {user_input}
    """
    
    ai_message = llm.invoke(prompt)
    return ai_message.content

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        website_url = request.form.get('website_url')
        api_key = request.form.get('api_key')
        api_provider = request.form.get('api_provider')
        
        if not website_url:
            error = "Please enter a website URL"
            return render_template('index.html', error=error)
        if not api_key:
            error = "Please enter your API key"
            return render_template('index.html', error=error)
        if not api_provider:
            error = "Please select an API provider"
            return render_template('index.html', error=error)
        
        website_content = get_content_from_url(website_url)
        session['website_url'] = website_url
        session['website_content'] = website_content
        session['api_key'] = api_key
        session['api_provider'] = api_provider
        session['chat_history'] = []
        return redirect(url_for('chat'))
    return render_template('index.html')

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if 'website_content' not in session:
        return redirect(url_for('index'))
    website_content = session['website_content']
    chat_history = session.get('chat_history', [])
    if request.method == 'POST':
        user_query = request.form.get('user_query')
        if user_query:
            chat_history.append({'role': 'user', 'content': user_query})
            response = generate_response(user_query, website_content)
            chat_history.append({'role': 'ai', 'content': response})
            session['chat_history'] = chat_history
    return render_template('chat.html', chat_history=chat_history)

if __name__ == '__main__':
    app.run(debug=True)
