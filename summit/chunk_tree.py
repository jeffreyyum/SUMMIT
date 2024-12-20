from typing import List
from dataclasses import dataclass
from process import summarise_chunks, count_tokens, segment_chunks
from text_chunk import Chunk

@dataclass
class ChunkTree:
    def __init__(self, entity: str, level_0_chunks: List[Chunk], chunk_indices: List[int], client, threshold: float = 0.5, max_height: int = 3):
        """
        Initialize the ChunkTree for a specific entity.

        Args:
        - entity (str): The name of the entity this tree represents.
        - level_0_chunks (List[Chunk]): The global list of level 0 chunks.
        - chunk_indices (List[int]): Indices of level 0 chunks associated with this entity.
        - client: OpenAI API client for generating summaries.
        - threshold (float): Threshold for cosine distance to define segmentation boundaries.
        - max_height (int): Maximum allowed height of the tree.
        """
        self.entity = entity
        self.level_0_chunks = level_0_chunks
        self.chunk_indices = chunk_indices  # References to level 0 chunk indices
        self.client = client
        self.threshold = threshold
        self.max_height = max_height
        self.root = None  # Root node of the tree

    def build_tree(self, visualize=False):
        """
        Build the hierarchical tree for the entity.

        Args:
        - visualize (bool): Whether to visualize cosine distances and boundaries.
        """
        # Retrieve the actual chunks for this entity
        entity_chunks = [self.level_0_chunks[i] for i in self.chunk_indices]

        # Build the tree recursively
        self.root = self._build_recursive_tree(entity_chunks, current_depth=0, visualize=visualize)

    def _build_recursive_tree(self, chunks: List[Chunk], current_depth: int, visualize=False) -> Chunk:
        """
        Recursively construct a tree of summary chunks with a maximum height.

        Args:
        - chunks (List[Chunk]): List of Chunk objects at the current level.
        - current_depth (int): Current depth of the tree.
        - visualize (bool): Whether to visualize cosine distances and boundaries.

        Returns:
        - Chunk: Root node of the tree for the current entity.
        """
        if current_depth >= self.max_height:
            # At max depth, combine all remaining chunks into a single root summary node
            print(f"Max depth reached. Creating root summary node with {len(chunks)} chunks.")
            root_summary_text = summarise_chunks(chunks, self.client)
            return Chunk(
                id=f"{self.entity}_root",
                text=root_summary_text,
                token_count=count_tokens(root_summary_text),
                children=chunks,
                level=current_depth,
                entity=self.entity
            )

        if len(chunks) <= 1:
            # Base case: If only one chunk, return it as the root node
            print(f"Stopping recursion at depth {current_depth}.")
            return chunks[0] if chunks else None

        print(f"Segmenting {len(chunks)} chunks at depth {current_depth}...")
        # Segment chunks into groups and optionally visualize
        segments = segment_chunks(chunks, threshold=self.threshold, visualize=visualize)

        # Separate single chunks
        single_chunks = [segment[0] for segment in segments if len(segment) == 1]
        non_single_segments = [segment for segment in segments if len(segment) > 1]

        # Summarize non-single segments
        summaries = [summarise_chunks(segment, self.client) for segment in non_single_segments]

        # Wrap summaries as Chunk objects and link parent-child relationships
        summary_chunks = []
        for i, (summary_text, segment) in enumerate(zip(summaries, non_single_segments)):
            print(f"Creating summary chunk {i + 1} for {len(segment)} segments.")
            summary_chunk = Chunk(
                id=f"{self.entity}_{current_depth + 1}_{i + 1}",
                text=summary_text,
                token_count=count_tokens(summary_text),
                children=segment,
                level=current_depth + 1,
                entity=self.entity
            )
            # Link the children to the parent summary chunk
            for child in segment:
                child.parent = summary_chunk
            summary_chunks.append(summary_chunk)

        # Add single chunks to the next level without summarization
        print(f"Carrying over {len(single_chunks)} single chunks to the next layer.")
        next_level_chunks = single_chunks + summary_chunks

        print(f"Recursing on {len(next_level_chunks)} chunks...")
        # Recur on the combined list of single chunks and summaries
        return self._build_recursive_tree(next_level_chunks, current_depth + 1, visualize=visualize)

    def flatten_tree(self) -> List[Chunk]:
        """
        Flatten the hierarchical tree into a list of chunks.

        Returns:
        - List[Chunk]: A flattened list of all chunks in the tree.
        """
        flattened_chunks = []

        def traverse(chunk):
            if chunk:
                flattened_chunks.append(chunk)  # Add the current chunk
                for child in chunk.children:
                    traverse(child)

        if self.root:
            traverse(self.root)

        return flattened_chunks


def process_document_with_chunk_tree(document: str, client, top_n: int = 10, max_tokens: int = 500):
    """
    Process a document, extract top entities, and build ChunkTrees.

    Args:
    - document (str): Input document text.
    - client: OpenAI API client for summarization.
    - top_n (int): Number of top entities to process.
    - max_tokens (int): Maximum tokens per chunk.

    Returns:
    - dict: Dictionary of ChunkTrees for each entity.
    """
    # process into level 0 chunks
    level_0_chunks = process_document(document, max_tokens=100)

    # extract top entities
    top_entities = extract_top_named_entities([chunk.text for chunk in level_0_chunks], top_n=3)
    print(top_entities)

    # build out chunk trees
    chunk_trees = []
    for entity, indices in top_entities.items():
        tree = ChunkTree(entity, level_0_chunks, list(indices), client, max_height=2)
        tree.build_tree(visualize=True)
        chunk_trees.append(tree)

    final_chunks = level_0_chunks[:]  # Start with original Layer 0 chunks
    for tree in chunk_trees:
        final_chunks.extend(tree.flatten_tree())

    return final_chunks