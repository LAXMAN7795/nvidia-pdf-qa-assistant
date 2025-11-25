# PDF Question Answering with NVIDIA LLMs
A Streamlit RAG Application using NVIDIA NIM, FAISS & LangChain

This project is a PDF Question-Answering (RAG) application built using:

NVIDIA NIM LLMs (ChatNVIDIA, NVIDIAEmbeddings)

Streamlit for the UI

FAISS for vector search

LangChain (Classic + Community) for chains & retrievers

Recursive text splitting for document chunking

Users can upload or load PDF files, embed them, and ask questions. The app retrieves relevant sections and generates responses using NVIDIA’s LLaMA-3.2 Instruct model.

🚀 Features

🧠 NVIDIA LLM (LLaMA-3.2 Instruct) for accurate responses

📚 RAG Pipeline with FAISS vector search

📄 PDF directory loader

✂️ Recursive character text splitting

🔍 Context-aware question answering

⚡ Fast inference using NVIDIA API

🎛️ Simple and clean Streamlit UI

📁 Project Structure
📦 your-repo-name
│
├── pdfs/                     # Folder containing PDF documents
├── finalapp.py               # Main Streamlit application
├── requirements.txt          # Python dependencies
├── .env                      # NVIDIA_API_KEY (not committed)
└── README.md                 # Project documentation

🔧 Installation
1️⃣ Clone the Repository
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name

2️⃣ Create & Activate Virtual Environment
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # macOS/Linux

3️⃣ Install Dependencies
pip install -r requirements.txt

🔑 Environment Variable

Create a .env file inside the project:

NVIDIA_API_KEY=your_api_key_here


Get your NVIDIA NIM API key from:

🔗 https://build.nvidia.com/explore/discover

▶️ Running the App
streamlit run finalapp.py


The app will open in your browser at:

http://localhost:8501

🧠 How It Works (RAG Pipeline)

Load PDFs from /pdfs

Split documents into overlapping text chunks

Generate vector embeddings using NVIDIAEmbeddings

Store embeddings in FAISS

Retrieve relevant chunks using similarity search

Feed retrieved context to ChatNVIDIA

Generate the answer

This follows the Retrieval-Augmented Generation (RAG) architecture.

🛠️ Tech Stack
Component	Technology
LLM	NVIDIA NIM (ChatNVIDIA)
Embeddings	NVIDIAEmbeddings
Backend	Python
Framework	Streamlit
RAG	LangChain Classic + Community
Vector Search	FAISS
Document Handling	PyPDFDirectoryLoader
📦 requirements.txt (recommended)
streamlit
python-dotenv
langchain
langchain-community
langchain-core
langchain-classic
langchain-text-splitters
langchain-nvidia-ai-endpoints
faiss-cpu
pypdf

🧑‍💻 Author

Laxman Sannu Gouda
📧 laxman.sg0104@gmail.com

⭐ Support
If you like this project, consider giving it a star ⭐ on GitHub — it motivates me to improve & add new features!
