# Multi_PDFs_ChatApp_AI_Agent
F/w: Langchain | Model: Google Gemini Pro | Vector DB | FAISS Deployment | Streamlit





### Gemini API

##### Overview
The **Gemini API** is a lightweight, privacy-focused protocol designed for browsing text-based resources. It combines simplicity with TLS encryption for secure, low-bandwidth communication.

##### Key Features
- **Minimalist Design**: Simple, text-based responses.
- **Privacy-Focused**: No cookies or client-side tracking.
- **Secure**: TLS encryption is mandatory.
- **Efficient**: Low bandwidth usage for quick communication.

##### Protocol Basics
- **Port**: Default `1965`
- **Media Type**: `text/gemini`
- **Request Format**: Single-line requests
- **Response Codes**:
  - `20`: Success
  - `30`: Redirect
  - `40`: Client error
  - `50`: Server error

##### Use Cases
- Static content hosting (e.g., blogs, documentation).
- Browsing on low-bandwidth or minimalist devices.
- Privacy-first communication platforms.

##### References
- [Gemini Specification](gemini://gemini.circumlunar.space/docs/specification.gmi)


# FAISS Vector Store - Key Attributes and Methods

## **Attributes**
| **Attribute**          | **Description**                                                      |
|-------------------------|----------------------------------------------------------------------|
| `index`                | The FAISS index used for similarity searches.                       |
| `embedding_function`   | Function used to generate embeddings (e.g., `OpenAIEmbeddings`).    |
| `docstore`             | Stores metadata and documents associated with vectors.              |
| `index_to_docstore_id` | Maps vector indices to document IDs in the `docstore`.              |

---

## **Key Methods**
| **Method**                  | **Description**                                                                 |
|-----------------------------|---------------------------------------------------------------------------------|
| `from_texts(texts, embedding)` | Creates a vector store from a list of `texts` using the specified `embedding`. |
| `add_texts(texts, metadatas)`  | Adds new texts and metadata to the vector store.                               |
| `similarity_search(query, k)`  | Finds the top `k` documents similar to a given `query`.                        |
| `similarity_search_by_vector(vector, k)` | Searches for similar documents using a raw embedding vector.         |
| `save_local(path)`           | Saves the FAISS index and metadata to a local directory.                        |
| `load_local(path, embedding)` | Loads a FAISS index and metadata from a local directory.                        |
| `as_retriever()`             | Converts the FAISS store into a retriever for LangChain chains.                 |

---
