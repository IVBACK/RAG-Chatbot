# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Import necessary libraries
import os
import psycopg2
from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from docx import Document

# Load database and data directory configuration from environment
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DATA_DIR = os.getenv("DATA_DIR")

# Load the sentence-transformers model for embedding generation
model = SentenceTransformer(os.getenv("EMBEDDING_MODEL", "all-mpnet-base-v2"))

# Function to reset the database: truncate the table and reset ID sequence
def reset_database():
    try:
        conn = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD)
        cur = conn.cursor()
        cur.execute("ALTER TABLE data ALTER COLUMN embedding TYPE vector(768);")
        cur.execute("TRUNCATE TABLE data RESTART IDENTITY;")
        conn.commit()
        cur.close()
        conn.close()
        print("Database has been reset.")
    except Exception as e:
        print(f"Database reset error: {e}")

# Function to rebuild the vector index for embedding column
def update_index():
    try:
        conn = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD)
        cur = conn.cursor()
        cur.execute("DROP INDEX IF EXISTS data_embedding_idx;")
        cur.execute("CREATE INDEX ON data USING ivfflat (embedding vector_cosine_ops);")
        cur.execute("ANALYZE data;")
        conn.commit()
        cur.close()
        conn.close()
        print("Index updated.")
    except Exception as e:
        print(f"Index update error: {e}")

# Function to read and extract text from a PDF file
def read_pdf(path):
    try:
        reader = PdfReader(path)
        return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    except Exception as e:
        return f"PDF read error: {e}"

# Function to read and extract text from a DOCX file
def read_docx(path):
    try:
        doc = Document(path)
        return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    except Exception as e:
        return f"DOCX read error: {e}"

# Function to read and return content of a TXT file
def read_txt(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"TXT read error: {e}")
        return ""

# Split a block of text into smaller chunks of `chunk_size` words
def split_text_chunks(text, chunk_size=100):
    words = text.split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return [chunk.strip() for chunk in chunks if len(chunk.strip()) > 10]

# Insert a batch of content, embeddings, and categories into the database
def batch_insert_to_database(contents, embeddings, categories):
    try:
        conn = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD)
        cur = conn.cursor()
        insert_query = "INSERT INTO data (content, embedding, category) VALUES %s"
        values = [(content, embedding.tolist(), category) for content, embedding, category in zip(contents, embeddings, categories)]
        execute_values(cur, insert_query, values)
        conn.commit()
        cur.close()
        conn.close()
        print(f"{len(contents)} records successfully inserted.")
    except Exception as e:
        print(f"Database insert error: {e}")

# Process all supported files in the directory, extract and embed content
def process_files(directory_path):
    contents = []
    embeddings = []
    categories = []

    for root, _, files in os.walk(directory_path):
        for file_name in files:
            if file_name.endswith(('.pdf', '.docx', '.txt')):
                file_path = os.path.join(root, file_name)
                category = os.path.basename(root)

                if file_name.endswith('.pdf'):
                    content = read_pdf(file_path)
                elif file_name.endswith('.docx'):
                    content = read_docx(file_path)
                elif file_name.endswith('.txt'):
                    content = read_txt(file_path)
                else:
                    continue

                if content:
                    chunks = split_text_chunks(content)
                    for chunk in chunks:
                        embedding = model.encode(chunk, normalize_embeddings=True)
                        contents.append(chunk)
                        embeddings.append(embedding)
                        categories.append(category)

    if contents:
        batch_insert_to_database(contents, embeddings, categories)

# Run the main steps when executed as script
if __name__ == "__main__":
    reset_database()
    process_files(DATA_DIR)
    update_index()
