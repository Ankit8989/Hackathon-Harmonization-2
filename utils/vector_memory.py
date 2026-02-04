"""
Vector Memory Module for Schema History
========================================

Uses FAISS for efficient similarity search of historical schema mappings.
Enables auto-learning of mapping patterns from previous harmonization runs.

Stretch Goal Implementation.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config import VECTOR_MEMORY_CONFIG, BASE_DIR
from utils.logger import get_logger

logger = get_logger("VectorMemory")


class VectorMemory:
    """
    Vector-based memory store for schema history and learned mappings.
    
    Uses FAISS for efficient similarity search and SentenceTransformers
    for generating embeddings of column names and descriptions.
    """
    
    def __init__(self, index_path: Optional[str] = None):
        """
        Initialize the vector memory store.
        
        Args:
            index_path: Optional path to existing FAISS index
        """
        self.config = VECTOR_MEMORY_CONFIG
        self.index_path = Path(index_path) if index_path else Path(self.config.index_path)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.metadata_path = self.index_path.with_suffix('.json')
        
        self.index = None
        self.embeddings_model = None
        self.metadata: List[Dict[str, Any]] = []
        
        self._initialized = False
        
        if self.config.enabled:
            self._initialize()
    
    def _initialize(self):
        """Initialize FAISS index and embedding model"""
        try:
            import faiss
            from sentence_transformers import SentenceTransformer
            
            logger.info("Initializing vector memory...")
            
            # Load or create FAISS index
            if self.index_path.exists():
                logger.info(f"Loading existing index from {self.index_path}")
                self.index = faiss.read_index(str(self.index_path))
                self._load_metadata()
            else:
                logger.info("Creating new FAISS index")
                self.index = faiss.IndexFlatL2(self.config.dimension)
            
            # Load embedding model
            logger.info(f"Loading embedding model: {self.config.embedding_model}")
            self.embeddings_model = SentenceTransformer(self.config.embedding_model)
            
            self._initialized = True
            logger.info("Vector memory initialized successfully")
            
        except ImportError as e:
            logger.warning(f"Vector memory dependencies not available: {str(e)}")
            logger.warning("Install with: pip install faiss-cpu sentence-transformers")
            self._initialized = False
        except Exception as e:
            logger.error(f"Failed to initialize vector memory: {str(e)}")
            self._initialized = False
    
    def _load_metadata(self):
        """Load metadata from JSON file"""
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
    
    def _save_metadata(self):
        """Save metadata to JSON file"""
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
    
    def _save_index(self):
        """Save FAISS index to file"""
        import faiss
        faiss.write_index(self.index, str(self.index_path))
    
    def _embed(self, text: str) -> np.ndarray:
        """
        Generate embedding for text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        return self.embeddings_model.encode([text])[0]
    
    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Array of embedding vectors
        """
        return self.embeddings_model.encode(texts)
    
    def add_mapping(
        self,
        source_column: str,
        target_column: str,
        confidence: float,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add a column mapping to the memory.
        
        Args:
            source_column: Source column name
            target_column: Target column name
            confidence: Mapping confidence score
            context: Optional context (data type, sample values, etc.)
            
        Returns:
            True if successful
        """
        if not self._initialized:
            return False
        
        try:
            # Create text for embedding
            text = f"{source_column} maps to {target_column}"
            if context:
                text += f" (type: {context.get('data_type', 'unknown')})"
            
            # Generate embedding
            embedding = self._embed(text).astype('float32')
            
            # Add to FAISS index
            self.index.add(embedding.reshape(1, -1))
            
            # Store metadata
            self.metadata.append({
                'source_column': source_column,
                'target_column': target_column,
                'confidence': confidence,
                'context': context or {},
                'text': text,
                'timestamp': str(np.datetime64('now'))
            })
            
            # Save
            self._save_index()
            self._save_metadata()
            
            logger.debug(f"Added mapping: {source_column} -> {target_column}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add mapping: {str(e)}")
            return False
    
    def add_mappings_batch(
        self,
        mappings: List[Dict[str, Any]]
    ) -> int:
        """
        Add multiple mappings in batch.
        
        Args:
            mappings: List of mapping dictionaries
            
        Returns:
            Number of successfully added mappings
        """
        if not self._initialized:
            return 0
        
        try:
            texts = []
            new_metadata = []
            
            for mapping in mappings:
                source = mapping.get('source_column', '')
                target = mapping.get('target_column', '')
                confidence = mapping.get('confidence', 0)
                context = mapping.get('context', {})
                
                if not source or not target or target == 'UNMAPPED':
                    continue
                
                text = f"{source} maps to {target}"
                if context:
                    text += f" (type: {context.get('data_type', 'unknown')})"
                
                texts.append(text)
                new_metadata.append({
                    'source_column': source,
                    'target_column': target,
                    'confidence': confidence,
                    'context': context,
                    'text': text,
                    'timestamp': str(np.datetime64('now'))
                })
            
            if texts:
                # Generate embeddings in batch
                embeddings = self._embed_batch(texts).astype('float32')
                
                # Add to FAISS index
                self.index.add(embeddings)
                
                # Store metadata
                self.metadata.extend(new_metadata)
                
                # Save
                self._save_index()
                self._save_metadata()
                
                logger.info(f"Added {len(texts)} mappings to vector memory")
            
            return len(texts)
            
        except Exception as e:
            logger.error(f"Failed to add batch mappings: {str(e)}")
            return 0
    
    def search_similar(
        self,
        source_column: str,
        data_type: Optional[str] = None,
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for similar column mappings.
        
        Args:
            source_column: Source column name to search for
            data_type: Optional data type for context
            k: Number of results to return
            
        Returns:
            List of similar mappings with scores
        """
        if not self._initialized or self.index.ntotal == 0:
            return []
        
        try:
            # Create search text
            text = source_column
            if data_type:
                text += f" (type: {data_type})"
            
            # Generate query embedding
            query = self._embed(text).astype('float32').reshape(1, -1)
            
            # Search FAISS index
            k = min(k, self.index.ntotal)
            distances, indices = self.index.search(query, k)
            
            # Build results
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(self.metadata):
                    result = self.metadata[idx].copy()
                    # Convert L2 distance to similarity score (0-1)
                    result['similarity'] = 1 / (1 + distance)
                    result['rank'] = i + 1
                    results.append(result)
            
            # Filter by similarity threshold
            results = [
                r for r in results
                if r['similarity'] >= self.config.similarity_threshold
            ]
            
            logger.debug(f"Found {len(results)} similar mappings for '{source_column}'")
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return []
    
    def suggest_mapping(
        self,
        source_column: str,
        data_type: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Suggest a mapping for a source column based on history.
        
        Args:
            source_column: Source column name
            data_type: Optional data type
            
        Returns:
            Best matching mapping or None
        """
        results = self.search_similar(source_column, data_type, k=1)
        
        if results and results[0]['similarity'] >= self.config.similarity_threshold:
            suggestion = results[0]
            logger.info(
                f"Suggested mapping for '{source_column}': "
                f"'{suggestion['target_column']}' "
                f"(similarity: {suggestion['similarity']:.2%})"
            )
            return suggestion
        
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the vector memory"""
        if not self._initialized:
            return {'initialized': False}
        
        return {
            'initialized': True,
            'total_mappings': self.index.ntotal,
            'index_path': str(self.index_path),
            'embedding_model': self.config.embedding_model,
            'dimension': self.config.dimension,
            'similarity_threshold': self.config.similarity_threshold
        }
    
    def clear(self):
        """Clear all mappings from memory"""
        if not self._initialized:
            return
        
        import faiss
        
        # Reset index
        self.index = faiss.IndexFlatL2(self.config.dimension)
        self.metadata = []
        
        # Save
        self._save_index()
        self._save_metadata()
        
        logger.info("Vector memory cleared")


class AutoLearningMapper:
    """
    Auto-learning mapper that improves over time using vector memory.
    """
    
    def __init__(self, vector_memory: Optional[VectorMemory] = None):
        """
        Initialize the auto-learning mapper.
        
        Args:
            vector_memory: Optional existing VectorMemory instance
        """
        self.memory = vector_memory or VectorMemory()
        self.learning_enabled = self.memory._initialized
    
    def learn_from_mappings(
        self,
        mappings: List[Dict[str, Any]]
    ) -> int:
        """
        Learn from successful mappings.
        
        Args:
            mappings: List of successful column mappings
            
        Returns:
            Number of mappings learned
        """
        if not self.learning_enabled:
            return 0
        
        # Filter high-confidence mappings
        high_confidence = [
            m for m in mappings
            if m.get('confidence', 0) >= 0.8 and m.get('target_column') != 'UNMAPPED'
        ]
        
        return self.memory.add_mappings_batch(high_confidence)
    
    def suggest_mappings(
        self,
        source_columns: List[Dict[str, Any]]
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        """
        Suggest mappings for source columns.
        
        Args:
            source_columns: List of source column info
            
        Returns:
            Dictionary of column name to suggested mapping
        """
        suggestions = {}
        
        for col_info in source_columns:
            name = col_info.get('name', '')
            dtype = col_info.get('data_type', col_info.get('dtype'))
            
            suggestion = self.memory.suggest_mapping(name, dtype)
            suggestions[name] = suggestion
        
        return suggestions
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get auto-learning statistics"""
        stats = self.memory.get_statistics()
        stats['learning_enabled'] = self.learning_enabled
        return stats


# Singleton instance
_vector_memory_instance: Optional[VectorMemory] = None


def get_vector_memory() -> VectorMemory:
    """
    Get the singleton VectorMemory instance.
    
    Returns:
        VectorMemory instance
    """
    global _vector_memory_instance
    if _vector_memory_instance is None:
        _vector_memory_instance = VectorMemory()
    return _vector_memory_instance


def get_auto_learning_mapper() -> AutoLearningMapper:
    """
    Get an AutoLearningMapper with the shared vector memory.
    
    Returns:
        AutoLearningMapper instance
    """
    return AutoLearningMapper(get_vector_memory())


