from text_chunk import Chunk
from typing import List, Iterator, Tuple
import re
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
import matplotlib.pyplot as plt
import numpy as np
import time
import spacy
from collections import defaultdict

def split_into_sentences(text: str) -> List[Tuple[str, int, int]]:
    """Split text into sentences and return with their start/end indices."""
    sentence_pattern = r'[.!?]+[\s]{1,2}(?=[A-Z])|[.!?]+$'

    sentences = []
    current_pos = 0

    matches = list(re.finditer(sentence_pattern, text))

    if not matches:
        return [(text.strip(), 0, len(text))]

    for i, match in enumerate(matches):
        end_pos = match.end()
        sentence = text[current_pos:end_pos].strip()
        if sentence:
            sentences.append((sentence, current_pos, end_pos))
        current_pos = end_pos

    if current_pos < len(text):
        final_sentence = text[current_pos:].strip()
        if final_sentence:
            sentences.append((final_sentence, current_pos, len(text)))

    return sentences

def count_tokens(text: str) -> int:
    """Approximate token count."""
    return len(text.split())

def chunk_by_tokens(
    text: str,
    max_tokens: int = 500,
    overlap_sentences: int = 1,
    level: int = 0
) -> Iterator[Chunk]:
    """
    Chunk document by token count, ensuring sentences aren't split mid-way.

    Args:
        text: Input text to chunk.
        max_tokens: Maximum tokens per chunk.
        overlap_sentences: Number of sentences to overlap between chunks.
        level: The tree level for these chunks.

    Yields:
        Chunk objects.
    """
    sentences = split_into_sentences(text)
    if not sentences:
        return

    current_chunk_sentences = []
    current_token_count = 0
    chunk_counter = 1

    for sentence in sentences:
        sentence_tokens = count_tokens(sentence[0])

        if current_token_count + sentence_tokens > max_tokens and current_chunk_sentences:
            chunk_text = ' '.join(s[0] for s in current_chunk_sentences)
            start_idx = current_chunk_sentences[0][1]
            end_idx = current_chunk_sentences[-1][2]

            yield Chunk(
                id=f"level{level}_{chunk_counter}",  # Structured ID
                text=chunk_text,
                start_idx=start_idx,
                end_idx=end_idx,
                token_count=current_token_count,
                level=level
            )
            chunk_counter += 1

            if overlap_sentences > 0:
                current_chunk_sentences = current_chunk_sentences[-overlap_sentences:]
                current_token_count = sum(count_tokens(s[0]) for s in current_chunk_sentences)
            else:
                current_chunk_sentences = []
                current_token_count = 0

        current_chunk_sentences.append(sentence)
        current_token_count += sentence_tokens

    if current_chunk_sentences:
        chunk_text = ' '.join(s[0] for s in current_chunk_sentences)
        start_idx = current_chunk_sentences[0][1]
        end_idx = current_chunk_sentences[-1][2]

        yield Chunk(
            id=f"level{level}_{chunk_counter}",  # Structured ID
            text=chunk_text,
            start_idx=start_idx,
            end_idx=end_idx,
            token_count=current_token_count,
            level=level
        )


def process_document(text: str, max_tokens: int = 500) -> List[Chunk]:
    """Split a document into chunks based on token limits."""
    return list(chunk_by_tokens(text, max_tokens=max_tokens))


def summarise_chunks(chunks, client) -> bool:
    """summarise a list of chunks"""

    joined_chunks = "\n".join([chunk.text for chunk in chunks])
    prompt = f"""
    summarise this list of chunks in one paragraph. Also please give a short succinct context to situate this chunk for the purposes of improving search retrieval of the chunk.:
    { joined_chunks }
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0
    )

    return response.choices[0].message.content

def segment_chunks(chunks: List[Chunk], threshold: float = 0.5, visualize: bool = False) -> List[List[Chunk]]:
    """
    Segment a list of Chunk objects based on semantic differences using precomputed SBERT embeddings.

    Args:
    - chunks (List[Chunk]): A list of Chunk objects to segment.
    - threshold (float): Threshold for cosine distance to define boundaries.
    - visualize (bool): Whether to display a graph of cosine distances and boundaries.

    Returns:
    - List[List[Chunk]]: Segmented groups of chunks.
    """
    if len(chunks) < 2:
        return [chunks]  # Return as a single segment if fewer than two chunks

    # Use precomputed embeddings
    embeddings = [chunk.embedding for chunk in chunks]

    # Compute cosine distances between consecutive embeddings
    distances = [
        cosine_distances([embeddings[i]], [embeddings[i + 1]])[0, 0]
        for i in range(len(embeddings) - 1)
    ]

    # Identify boundaries where distances exceed the threshold
    boundaries = [i + 1 for i, dist in enumerate(distances) if dist > threshold]

    # Visualize distances and boundaries if requested
    if visualize:
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(distances)), distances, marker='o', linestyle='-', color='b', label='Cosine Distance')
        for boundary in boundaries:
            plt.axvline(x=boundary - 1, color='r', linestyle='--', label='Boundary')
        plt.title("Cosine Distances Between Consecutive Chunks")
        plt.xlabel("Chunk Index")
        plt.ylabel("Cosine Distance")
        plt.legend()
        plt.grid(True)
        plt.show()

    # Segment the chunks using boundaries
    segmented_chunks = []
    start = 0
    for boundary in boundaries:
        segmented_chunks.append(chunks[start:boundary])
        start = boundary
    segmented_chunks.append(chunks[start:])  # Add the remaining chunks

    return segmented_chunks



# Initialize spaCy NER model
nlp = spacy.load("en_core_web_sm")


def extract_top_named_entities(text_chunks, top_n=10):
    """
    Extract named entities from text chunks and return the top N entities by occurrence,
    along with the chunk indices where they appear.

    Args:
    - text_chunks (list of str): List of text chunks to process.
    - top_n (int): Number of top entities to return based on frequency.

    Returns:
    - dict: A dictionary of the top N entities and their associated chunk indices.
    """
    # Step 1: Extract entities and map them to chunk indices
    entity_dict = defaultdict(set)
    for i, chunk in enumerate(text_chunks):
        doc = nlp(chunk)
        for ent in doc.ents:
            entity_dict[ent.text].add(i)  # Map entities to chunk indices

    # Step 2: Count entity occurrences
    entity_counts = count_entities(entity_dict)

    # Step 3: Get the top N entities
    top_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]

    # Step 4: Filter entity_dict to only include top N entities
    top_entity_dict = {entity: entity_dict[entity] for entity, _ in top_entities}

    return top_entity_dict



def count_entities(entity_dict):
    """
    Count the frequency of each entity from the NER output.

    Args:
    - entity_dict (dict): A dictionary where keys are entities and values are sets of chunk indices.

    Returns:
    - dict: A dictionary of entities and their frequencies.
    """
    # Count the size of each entity's chunk set
    entity_counts = {entity: len(chunks) for entity, chunks in entity_dict.items()}
    return entity_counts

def get_top_k_similar_chunks(query: str, chunks: List[Chunk], sbert_model, k: int = 5, layer_0_only: bool = False) -> List[Chunk]:
    """
    Retrieve the top k most similar chunks to a query based on cosine similarity.

    Args:
    - query (str): The input query string.
    - chunks (List[Chunk]): A list of Chunk objects to search.
    - sbert_model: The preloaded SentenceTransformer model for encoding.
    - k (int): The number of most similar chunks to return.
    - layer_0_only (bool): If True, restrict the search to only layer 0 chunks.

    Returns:
    - List[Chunk]: The top k most similar chunks.
    """
    start_time = time.perf_counter()

    # Filter to layer 0 chunks if layer_0_only is True
    if layer_0_only:
        chunks = [chunk for chunk in chunks if chunk.level == 0]
        print(f"Filtered to {len(chunks)} layer 0 chunks.")

    # Compute the embedding for the query
    query_embedding = sbert_model.encode(query)
    print(f"Query embedding computed in {time.perf_counter() - start_time:.4f} seconds")

    # Extract embeddings from chunks, try to change to chunk[0]
    chunk_embeddings = np.array([chunk.embedding for chunk in chunks])
    print(f"Embeddings extracted in {time.perf_counter() - start_time:.4f} seconds")

    # Compute cosine similarities
    similarities = cosine_similarity([query_embedding], chunk_embeddings)[0]
    print(f"Cosine similarities computed in {time.perf_counter() - start_time:.4f} seconds")

    # Get the indices of the top k most similar chunks
    top_k_indices = np.argsort(similarities)[-k:][::-1]
    print(f"Top-k indices computed in {time.perf_counter() - start_time:.4f} seconds")

    # Retrieve and return the top k chunks
    top_k_chunks = [chunks[i] for i in top_k_indices]
    print(f"Function completed in {time.perf_counter() - start_time:.4f} seconds")

    return top_k_chunks

def generate_answer(query, context) -> bool:
    """Generate an answer based on context provided"""
    prompt = f"""
    Use the context provided to synthesize and answer to the quetion. Don't hallucinate.
    Question : {query}
    Context : {context}

    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0
    )
    return response.choices[0].message.content
