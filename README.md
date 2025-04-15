# 🧠 Chatbot with Contextual Retrieval (RAG)

This is a Flask-based AI assistant that answers questions based on your documents using embeddings.  
It combines **semantic search**, **keyword matching**, and **zero-shot classification** to retrieve contextually relevant answers.

---

## 🚀 Features
- ✅ Retrieval-Augmented Generation (RAG) from your text files
- ✅ Supports `.txt`, `.pdf`, `.docx` inputs
- ✅ Uses `pgvector` and `sentence-transformers`
- ✅ Chat interface with Markdown support
- ✅ Ready-to-deploy with Nginx + SSL

---

## 📦 Installation

```bash
git clone https://github.com/yourusername/rag-chatbot.git
cd rag-chatbot

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🧠 Ollama + LLaMA 3 Setup

This app uses [Ollama](https://ollama.com) to run local large language models (LLMs) like `llama3:8b`.

### 🟢 Install Ollama (Linux/macOS)

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

<sub>Or download manually:</sub>

```bash
wget https://ollama.com/download/OllamaLinux.tar.gz
tar -xzf OllamaLinux.tar.gz
sudo mv ollama /usr/local/bin/
```

---

### 📦 Pull the Model

```bash
ollama run llama3:8b
```

> This will download the model and run an interactive shell.  
> Once downloaded, the Flask app will be able to query it on `localhost:11434`.

---

### 🛠️ Configuration

Make sure your `.env` file contains:

```env
OLLAMA_MODEL=llama3:8b
```

---

## ⚙️ Environment Setup

Create your `.env` file:

```bash
cp .env.example .env
```

Edit `.env` and set:
- PostgreSQL credentials
- Path to your document directory
- Model names (optional)

---

## 🧠 Database Setup

Use PostgreSQL + `pgvector`. You can initialize with:

```bash
psql -U postgres -f init_database.sql
```

This script will:
- Create a database
- Enable `pgvector` extension
- Create a `data` table with `vector(768)` column

---

## 📚 Load Data

Place your categorized documents under `data/` directory (subfolders = categories):

```bash
python load.py
```

---

## 💬 Run the App

```bash
python app.py
```

Then go to:

📍 http://localhost:5000/chat

> It includes a built-in UI with dark/light mode, chat history, and Markdown support.

---

## 🌐 Deploy to Production (Optional)

For Nginx reverse proxy, HTTPS, and rate limiting:

➡️ See [`deployment/README.md`](deployment/README.md)

Includes:
- Nginx config with reverse proxy
- Let's Encrypt SSL setup
- Favicon/static file configuration

---

## 📁 Project Structure

```
├── app.py               # Flask API
├── load.py              # Load documents into PostgreSQL with embeddings
├── init_database.sql    # Creates database + pgvector setup
├── templates/chat.html  # Chat UI
├── data/                # Place your categorized documents here
├── deployment/          # Nginx + SSL configuration
```

---

## 🛡️ Security Tips

- Use `ALLOWED_IP` in `.env` to restrict access.
- Always use HTTPS in production.
- Protect `.env` file and server credentials.

---

## 📄 License

This project is licensed under the **MIT License**.
You are free to use, modify, and distribute it with attribution.

