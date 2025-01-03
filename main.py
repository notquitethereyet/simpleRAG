from preprocess import extract_text_from_pdf
from embed import initialize_pinecone, get_embeddings
from store import create_index, upsert_data
from search import search
from cock import get_search_query
import tkinter as tk
from tkinter import filedialog

def main():
    # Create root window and hide it
    root = tk.Tk()
    root.withdraw()

    # Initialize Pinecone
    pc = initialize_pinecone()

    # Ensure the index is created
    create_index(pc)

    while True:
        # Menu options
        print("\nChoose an option:")
        print("1. Select a PDF to process")
        print("2. Ask a query")
        print("3. Exit")

        choice = input("\nEnter your choice (1/2/3): ").strip()

        if choice == "1":
            # Open file picker dialog
            pdf_path = filedialog.askopenfilename(
                title="Select PDF file",
                filetypes=[("PDF files", "*.pdf")]
            )

            # Check if a file was selected
            if not pdf_path:
                print("No file selected. Sayonara...")
                continue

            # Preprocess the PDF
            text = extract_text_from_pdf(pdf_path)
            sentences = text.split(". ")

            # Get embeddings
            embeddings = get_embeddings(sentences)

            print("Processing PDF... Churning text.... Crunching numbers....")
            # Upsert data into Pinecone
            upsert_data(sentences, embeddings, pc)
            print("PDF ingested successfully!")

        elif choice == "2":
            # User input for query
            user_input = input("Ask me something: ").strip()

            # First perform the search
            results = search(user_input, pc)

            # Build context from results
            context = user_input + "\n\nRelevant context:\n"
            if isinstance(results, str):  # Check for fallback message
                context += "No relevant results found."
            else:
                for match in results['matches']:
                    context += f"- {match['metadata']['text']}\n"

            # Get inferred query with context
            inferred_query = get_search_query(context)
            print(f"\nInferred query: {inferred_query}")

        elif choice == "3":
            print("Sayonara...")
            break

        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()
