from langchain_community.document_loaders import PyPDFLoader

# Replace 'yourfile.pdf' with your actual PDF filename
loader = PyPDFLoader("notes.pdf")

pages = loader.load()

print(f"Total pages loaded: {len(pages)}")
print("\n--- First page content preview ---\n")
print(pages[1].page_content[:500])