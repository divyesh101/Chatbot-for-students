# back.py
import os
import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import torch
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
import google.generativeai as genai
from datetime import datetime
import json
import pickle

@dataclass
class UserInfo:
    """User information for context"""
    name: str
    college: str
    degree: str
    year: int
    career_goals: str
    has_internship: bool
    has_placement: bool

@dataclass
class ChatConfig:
    """Configuration for the chatbot"""
    embedding_model_name: str = 'all-MiniLM-L6-v2'
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    max_history: int = 6
    gemini_api_key: str = os.getenv("GEMINI_API")  # Replace with your API key
    log_file: str = "chat_history.txt"
    user_data_file: str = "user_data.json"
    database_file: str = "faiss_db.pkl" # Added database file path

# In the UserManager class, modify these methods:
class UserManager:
    """Manages user information storage and retrieval"""
    def __init__(self, user_data_file: str):
        self.user_data_file = user_data_file
        self.ensure_file_exists()
    
    def ensure_file_exists(self):
        """Create user data file if it doesn't exist"""
        if not os.path.exists(self.user_data_file):
            os.makedirs(os.path.dirname(self.user_data_file), exist_ok=True)
            with open(self.user_data_file, 'w', encoding='utf-8') as f:
                json.dump({}, f)
    
    def save_user_info(self, user_info: UserInfo):
        """Save user information to JSON file"""
        try:
            # First ensure the file exists with valid JSON
            self.ensure_file_exists()
            
            # Read existing data
            try:
                with open(self.user_data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                data = {}
            
            # Update data
            data[user_info.name] = {
                "college": user_info.college,
                "degree": user_info.degree,
                "year": user_info.year,
                "career_goals": user_info.career_goals,
                "has_internship": user_info.has_internship,
                "has_placement": user_info.has_placement,
                "last_updated": datetime.now().isoformat()
            }
            
            # Write back to file
            with open(self.user_data_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
            return True
        except Exception as e:
            logging.error(f"Error saving user info: {str(e)}")
            return False


class ChatLogger:
    """Logger for chat interactions"""
    def __init__(self, log_file: str):
        self.log_file = log_file
        
    def log_interaction(self, question: str, answer: str, user_info: Optional[UserInfo] = None):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, 'a', encoding='utf-8') as f:
            user_context = ""
            if user_info:
                user_context = f"\nUser: {user_info.name} | College: {user_info.college} | Degree: {user_info.degree} | Year: {user_info.year} | Career Goals: {user_info.career_goals}"
            f.write(f"\n[{timestamp}]{user_context}\nQ: {question}\nA: {answer}\n{'-'*50}")

class ChatMemory:
    """Manages chat history"""
    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.history = []
        
    def add_interaction(self, question: str, answer: str):
        self.history.append({"question": question, "answer": answer})
        if len(self.history) > self.max_history:
            self.history.pop(0)
            
    def get_history(self) -> List[Dict[str, str]]:
        return self.history
    
    def clear_history(self):
        self.history = []

class QuestionGenerator:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.generation_config = {
            "temperature": 0.1,
            "max_output_tokens": 8192,
        }
        self.model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=self.generation_config,
            safety_settings={'HATE': 'BLOCK_NONE','HARASSMENT': 'BLOCK_NONE','SEXUAL' : 'BLOCK_NONE','DANGEROUS' : 'BLOCK_NONE'}
        )
        
        self.default_questions = [
            "What are some other skills I should focus on to improve my chances?",
            "What resources or platforms can help me in my career journey?",
            "Are there any specific companies or organizations I should target for internships/placements?",
            "What are some common interview questions asked for this career path?"
        ]
    
    async def generate_questions(
        self, 
        question: str, 
        answer: str, 
        user_info: Optional[UserInfo] = None
    ) -> List[str]:
        """Generate follow-up questions based on the conversation"""
        try:
            chat = self.model.start_chat(history=[])
            prompt = f"""Generate 4 simple, practical follow-up questions, that a college student may ask, based on this conversation about career advice:

Question: {question}
Answer: {answer}

Focus the questions on:
1. Skills development (What skills are needed, how to improve)
2. Resources and platforms (Where to find internships, jobs, etc.)
3. Specific target companies/organizations
4. Common interview questions

Keep the language simple and student-friendly. Format each question on a new line.

NOTE: YOU MUST STRICTLY REPLY IN HINGLISH"""

            response = chat.send_message(prompt).text
            
            # Extract questions
            questions = [q.strip() for q in response.split('\n') if q.strip()]
            
            # Return default questions if we don't get exactly 4 valid questions
            if len(questions) != 4:
                return self.default_questions
            
            return questions
            
        except Exception as e:
            logging.error(f"Error generating questions: {str(e)}")
            return self.default_questions

class GeminiRAG:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.generation_config = {
            "temperature": 0.1,
            "max_output_tokens": 8192,
        }
        self.model = genai.GenerativeModel(
            model_name="gemini-2.0-flash-exp",
            generation_config=self.generation_config,
            safety_settings={'HATE': 'BLOCK_NONE','HARASSMENT': 'BLOCK_NONE','SEXUAL' : 'BLOCK_NONE','DANGEROUS' : 'BLOCK_NONE'}
        )
    
    def create_context(self, relevant_docs: List[Dict[str, Any]]) -> str:
        """Creates a context string from relevant documents"""
        context_parts = []
        for doc in relevant_docs:
            context_parts.append(f"Section: {doc['metadata']['section']}\n{doc['content']}")
        return "\n\n".join(context_parts)

    async def get_answer(
        self, 
        question: str, 
        context: str,
        user_info: Optional[UserInfo] = None
    ) -> str:
        try:
            chat = self.model.start_chat(history=[])
            
            # Simplified prompt to reduce chances of recitation
            prompt = f"""As a career counselor, provide a helpful response based on:

Context: {context}

{f'''User Background:
- Student at {user_info.college}
- Studying {user_info.degree} (Year {user_info.year})
- Goals: {user_info.career_goals}
- {'Has internship experience' if user_info.has_internship else 'No internship yet'}
- {'Has placement' if user_info.has_placement else 'Seeking placement'}''' if user_info else ''}

Question: {question}

Provide practical advice with specific examples and actionable steps."""
            
            try:
                response = chat.send_message(prompt)
                if response.text:
                    return response.text
                else:
                    return "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."
            except Exception as chat_error:
                logging.error(f"Chat error: {str(chat_error)}")
                return "I encountered an error while processing your question. Please try again with a simpler question."
            
        except Exception as e:
            logging.error(f"Error generating answer: {str(e)}")
            return "An error occurred. Please try again later."

class CustomEmbeddings(Embeddings):
    """Custom embeddings using SentenceTransformer"""
    def __init__(self, model_name: str, device: str):
        self.model = SentenceTransformer(model_name)
        self.model.to(device)
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        with torch.no_grad():
            embeddings = self.model.encode(texts, convert_to_tensor=True)
            return embeddings.cpu().numpy().tolist()
            
    def embed_query(self, text: str) -> List[float]:
        with torch.no_grad():
            embedding = self.model.encode([text], convert_to_tensor=True)
            return embedding.cpu().numpy().tolist()[0]

class ProductDatabase:
    """Handles document storage and retrieval"""
    def __init__(self, config: ChatConfig):
        self.embeddings = CustomEmbeddings(
            model_name=config.embedding_model_name,
            device=config.device
        )
        self.vectorstore = None
        self.config = config
        self.load_database()
        
    def load_database(self):
        """Loads the FAISS database from file"""
        try:
            if os.path.exists(self.config.database_file):
                with open(self.config.database_file, "rb") as f:
                    self.vectorstore = pickle.load(f)
                print("Database loaded successfully from file.")
            else:
                print("Database file not found. Please run setup.py to create it.")
        except Exception as e:
            logging.error(f"Error loading database: {str(e)}")
            print(f"Error loading database: {str(e)}")
            self.vectorstore = None
    
    def process_markdown(self, markdown_content: str):
        """Process markdown content and create vector store"""
        try:
            sections = markdown_content.split('\n## ')
            documents = []
            
            if sections[0].startswith('# '):
                intro = sections[0].split('\n', 1)[1]
                documents.append({
                    "content": intro,
                    "section": "Introduction"
                })
            
            for section in sections[1:]:
                if section.strip():
                    title, content = section.split('\n', 1)
                    documents.append({
                        "content": content.strip(),
                        "section": title.strip()
                    })
            
            texts = [doc["content"] for doc in documents]
            metadatas = [{"section": doc["section"]} for doc in documents]
            
            if self.vectorstore is None:
                self.vectorstore = FAISS.from_texts(
                    texts=texts,
                    embedding=self.embeddings,
                    metadatas=metadatas
                )
            else:
                self.vectorstore.add_texts(texts=texts, metadatas=metadatas, embedding=self.embeddings)
            
        except Exception as e:
            raise Exception(f"Error processing markdown content: {str(e)}")
        
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant documents"""
        if not self.vectorstore:
            raise ValueError("Database not initialized. Please process documents first.")
        
        try:
            docs = self.vectorstore.similarity_search(query, k=k)
            return [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs]
        except Exception as e:
            logging.error(f"Error during search: {str(e)}")
            return []