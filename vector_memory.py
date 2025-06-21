import os
import re
import numpy as np
from pinecone import Pinecone, ServerlessSpec
import google.generativeai as genai
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown

# --- CONFIGURATION ---
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = "gcp-starter" # Replace with your Pinecone environment if different
INDEX_NAME = "vibe-memory"
MODEL_EMBEDDING = "models/embedding-001"
MODEL_CHAT = "gemini-2.5-flash-lite-preview-06-17"

# --- SYSTEM PROMPT ---
SYSTEM_PROMPT = """
# Identity & Role
You are "Vibe", a personalized AI assistant. Your primary purpose is to assist the user while continuously learning about them to provide more helpful and contextually-aware responses over time. You are observant, curious, and your memory is persistent.

# Core Directives
1.  **Prioritize User Context**: You will be provided with a `[MEMORY CONTEXT]` block containing facts retrieved from your long-term vector memory. These are facts you already know about the user. You MUST use this context to inform your response.
2.  **Answer the User's Query**: Directly address the user's `[CURRENT QUERY]` in a clear and helpful manner.
3.  **Identify New Memories**: As you converse, actively listen for new, permanent information about Matthew. This could be a preference, a new project he's working on, a personal detail, a goal, or a change to existing information.
4.  **Signal Memory Updates**: When you identify a new piece of information that should be saved, you MUST include a special block in your response formatted EXACTLY as `[SAVE_MEMORY]New fact to be remembered.[/SAVE_MEMORY]`.
    - The fact inside the block must be a concise, atomic statement.
    - You can have multiple `[SAVE_MEMORY]` blocks in a single response if you learn multiple things.
    - DO NOT add this block for trivial or temporary information (e.g., "User asked what time it is"). Only save significant, long-term facts.
    - If a new fact contradicts an old one, state the new fact clearly. For example: `[SAVE_MEMORY]User now prefers C# over Python for game development.[/SAVE_MEMORY]`
"""

# --- INITIAL DATA ---
INITIAL_FACTS = [
    "The user's name is Alex.",
    "Alex is a software developer.",
    "Alex is interested in learning about artificial intelligence.",
    "Alex enjoys hiking on the weekends.",
    "Alex lives in a city with a vibrant tech scene.",
    "Alex's favorite programming language is Python."
]

def main():
    """The main function to run the Vibe assistant."""
    console = Console()
    console.print("[bold green]Vibe AI Assistant Initializing...[/bold green]")

    # Initialize services
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        console.print("[green]Pinecone initialized.[/green]")
    except Exception as e:
        console.print(f"[bold red]Error initializing Pinecone: {e}[/bold red]")
        return

    # Setup Pinecone Index
    if INDEX_NAME not in pc.list_indexes().names():
        console.print(f"Creating Pinecone index '{INDEX_NAME}'...")
        pc.create_index(
            name=INDEX_NAME, 
            dimension=768, 
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
        console.print(f"Index created. Seeding with initial data...")
        seed_initial_data(pc)
    else:
        console.print(f"Pinecone index '{INDEX_NAME}' already exists.")

    index = pc.Index(INDEX_NAME)

    # Start chat loop
    console.print("\n[bold cyan]Welcome. I'm Vibe. How can I help you today?[/bold cyan]")
    console.print("[italic bright_black]Type 'exit' or 'quit' to end the session.[/italic bright_black]")
    chat_session = genai.GenerativeModel(MODEL_CHAT, system_instruction=SYSTEM_PROMPT).start_chat()

    while True:
        user_input = console.input("[bold yellow]You: [/bold yellow]")
        if user_input.lower() in ["exit", "quit"]:
            console.print("[bold green]Goodbye![/bold green]")
            break
        
        if not user_input.strip():
            continue

        # 1. Memory Retrieval
        retrieved_memories = retrieve_memories(user_input, index)

        # 2. Context Assembly
        context_prompt = f"[MEMORY CONTEXT]\n{retrieved_memories}\n\n[CURRENT QUERY]\n{user_input}"

        # 3. LLM Interaction
        response = chat_session.send_message(context_prompt)
        response_text = response.text

        # 4. Response Generation & Display
        cleaned_response = re.sub(r'\[SAVE_MEMORY\].*?\[/SAVE_MEMORY\]', '', response_text).strip()
        console.print(f"[bold yellow]Gemini:[/bold yellow] {cleaned_response}")

        # 5. Memory Extraction & Storage
        new_facts = re.findall(r'\[SAVE_MEMORY\](.*?)\[/SAVE_MEMORY\]', response_text)
        if new_facts:
            console.print(f"\n[italic bright_black]Saving {len(new_facts)} new memories...[/italic bright_black]")
            for fact in new_facts:
                upsert_memory(fact, index)
            console.print("[italic bright_black]Memories saved.[/italic bright_black]\n")


def normalize_vector(v):
    """Normalizes a vector to a magnitude of 1 (L2 normalization)."""
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm

def get_embedding(text):
    """Generates a normalized embedding for the given text."""
    embedding = genai.embed_content(model=MODEL_EMBEDDING, content=text)['embedding']
    return normalize_vector(embedding).tolist()

def upsert_memory(fact, index):
    """Embeds a fact and upserts it into the Pinecone index."""
    embedding = get_embedding(fact)
    index.upsert(vectors=[(str(hash(fact)), embedding, {"text": fact})])

def retrieve_memories(query, index, top_k=15):
    """Retrieves the most relevant memories from Pinecone."""
    query_embedding = get_embedding(query)
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    
    if results['matches']:
        return "\n".join([f"- {match['metadata']['text']}" for match in results['matches']])
    return "No relevant memories found."

def seed_initial_data(pc):
    """Seeds the Pinecone index with the initial set of facts."""
    index = pc.Index(INDEX_NAME)
    console = Console()
    console.print(f"Seeding {len(INITIAL_FACTS)} initial facts into '{INDEX_NAME}'...")
    for fact in INITIAL_FACTS:
        upsert_memory(fact, index)
    console.print("[bold green]Initial data seeding complete.[/bold green]")


if __name__ == "__main__":
    main()
