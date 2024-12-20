# SUMMIT : Summarization Using Modular, Multi-layered Interpretable Trees

![SUMMIT image](https://github.com/user-attachments/assets/e96f046a-d21c-47f7-ae62-54bb7d3fb1dd)


Naive RAG cannot operate on contexts that are higher than the data that is chunked. For example, you can't ask "what did this document say", you can only ask for specific bits of data or facts within the document. Humans also structure our memory hierarchically. We don't remember a set of facts, we often remember higher level details, and can drill down to the lower level facts if we need to.

SUMMIT aims to incorporate these insights into a novel tree based structure that can enhance normal RAG. We build trees of summaries per important entity in the chunks, and use these new summary chunks as enhancements to the original chunks to help answer questions for long documents.

# Method explained

![SUMMIT explained](https://github.com/user-attachments/assets/013271da-a404-42d2-8b50-57335adf642a)


While the notebook contains a more comprehensive look into the methods, here's a high level overview:
1. we chunk and get semantic vectors for the document using SBERT
2. we then extract key entities from the document
3. for each key entity, we segment the chunks that correspond to it in contiguous blocks
4. we summarise these chunks, and then recursively apply this segmenting and summarising method until we make a tree of chunks of a specific size.
5. finally we add all of these summary chunks back into a flattened list of chunks, and run RAG on this

# Usage

To test out this paper, go to the SUMMIT.ipynb file in the repo. The code is organised in a functional manner to make it easy to experiment with and tweak.

Some key functions:

### Chunking the data

```def process_document(text: str, max_tokens: int = 500)``` and ```def chunk_by_tokens(
    text: str,
    max_tokens: int = 500,
    overlap_sentences: int = 1,
    level: int = 0
)``` 

### Extracting key entities
```extract_top_named_entities(text_chunks, top_n=10)``` we use spacy for NER to extract key entities

### Segmenting chunks for summarization

```segment_chunks(chunks: List[Chunk], threshold: float = 0.5, visualize: bool = False)``` segments chunks when it sees jumps in the cosine distance between chunks. See the image for an explainer on how it works:
![segmenting example](https://github.com/user-attachments/assets/be505c41-a521-4ff2-b6f2-1663221fa642)

### Summary chunk tree generation
```
class ChunkTree:
   def __init__(self, entity: str, level_0_chunks: List[Chunk], chunk_indices: List[int], client, threshold: float = 0.5, max_height: int = 3):
  ```
is the main function that deals with creating the tree. 

and then ```process_document_with_chunk_tree(document: str, client, top_n: int = 10, max_tokens: int = 500)``` ties it all together
