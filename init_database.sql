-- Drop the database if it exists
DROP DATABASE IF EXISTS your_database_name;
CREATE DATABASE your_database_name;

-- Connect to the database
\c your_database_name

-- Load the pgvector extension (for vector similarity search)
CREATE EXTENSION IF NOT EXISTS vector;

-- Create main table
DROP TABLE IF EXISTS data;
CREATE TABLE data (
    id SERIAL PRIMARY KEY,
    content TEXT,
    category TEXT,
    embedding vector(768)  -- 768-dimensional embedding (MPNet-compatible)
);

-- Optional: Create ivfflat index for vector search (faster queries after ANALYZE)
CREATE INDEX data_embedding_idx ON data USING ivfflat (embedding vector_cosine_ops);

-- Analyze for performance
ANALYZE data;
