#!/usr/bin/env python3
"""
Test suite for ZamAI Business Tools
"""

import pytest
import os
import sys
import tempfile
import shutil
from pathlib import Path
import numpy as np

# Add the demos directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "demos"))

from enhanced_business_tools import BusinessDocumentProcessor

class TestBusinessDocumentProcessor:
    """Test the Business Document Processor"""
    
    @pytest.fixture
    def processor(self):
        """Create a test processor instance"""
        # Use a temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock the paths to use temp directory
            processor = BusinessDocumentProcessor()
            processor.documents_path = Path(temp_dir) / "documents"
            processor.embeddings_path = Path(temp_dir) / "embeddings"
            processor.index_path = Path(temp_dir) / "faiss_index"
            processor._initialize_storage()
            yield processor
    
    def test_initialization(self, processor):
        """Test processor initialization"""
        assert processor.phi3_model is not None
        assert processor.embeddings_model is not None
        assert processor.client is not None
        assert processor.documents_path.exists()
        assert processor.embeddings_path.exists()
        assert processor.index_path.exists()
    
    def test_embeddings_generation(self, processor):
        """Test embeddings generation"""
        text = "This is a test document for embeddings generation."
        embeddings = processor.generate_embeddings(text)
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (768,)  # Expected embedding dimension
        assert np.allclose(np.linalg.norm(embeddings), 1.0, atol=1e-5)  # Should be normalized
    
    def test_document_storage(self, processor):
        """Test document storage functionality"""
        test_text = "This is a test business document."
        test_analysis = "This is a test analysis."
        test_embeddings = np.random.randn(768).astype(np.float32)
        test_embeddings = test_embeddings / np.linalg.norm(test_embeddings)
        
        doc_id = processor._store_document(
            test_text, "Contract", test_analysis, test_embeddings
        )
        
        assert doc_id in processor.document_index
        assert processor.document_index[doc_id]["type"] == "Contract"
        
        # Check if files were created
        doc_file = processor.documents_path / f"{doc_id}.json"
        embeddings_file = processor.embeddings_path / f"{doc_id}.npy"
        
        assert doc_file.exists()
        assert embeddings_file.exists()
    
    def test_document_processing(self, processor):
        """Test document processing pipeline"""
        test_document = """
        BUSINESS CONTRACT
        
        This agreement is between Company A and Company B.
        The contract is valid from January 1, 2024 to December 31, 2024.
        Total value: $50,000
        Payment terms: Net 30 days
        """
        
        # Mock the inference client to avoid actual API calls in tests
        class MockClient:
            def text_generation(self, **kwargs):
                return "Mock analysis: This is a business contract between Company A and Company B with a value of $50,000."
            
            def feature_extraction(self, text, **kwargs):
                # Return a mock embedding
                embedding = np.random.randn(768).astype(np.float32)
                return embedding / np.linalg.norm(embedding)
        
        processor.client = MockClient()
        
        result = processor.process_business_document(
            test_document, "Contract", "Information Extraction"
        )
        
        assert "analysis" in result
        assert "document_id" in result
        assert "processing_time" in result
        assert result["embeddings_generated"] is True
        assert result["stored_for_retrieval"] is True
    
    def test_content_generation(self, processor):
        """Test content generation functionality"""
        # Mock the inference client
        class MockClient:
            def text_generation(self, **kwargs):
                return "Mock generated content: This is a professional business email."
        
        processor.client = MockClient()
        
        result = processor.generate_business_content(
            "Email", 
            "Draft a meeting invitation for next week",
            "Monthly team meeting"
        )
        
        assert "generated_content" in result
        assert "content_type" in result
        assert result["generation_successful"] is True
    
    def test_document_search(self, processor):
        """Test document search functionality"""
        # Add some test documents first
        processor.client = type('MockClient', (), {
            'text_generation': lambda self, **kwargs: "Mock analysis",
            'feature_extraction': lambda self, text, **kwargs: np.random.randn(768).astype(np.float32)
        })()
        
        # Process a few test documents
        test_docs = [
            ("Contract between A and B", "Contract"),
            ("Invoice for services", "Invoice"),
            ("Meeting report", "Report")
        ]
        
        for doc_text, doc_type in test_docs:
            processor.process_business_document(doc_text, doc_type, "Information Extraction")
        
        # Search for documents
        results = processor.search_similar_documents("contract agreement", top_k=2)
        
        assert len(results) >= 1
        if "error" not in results[0]:
            assert "document_id" in results[0]
            assert "similarity_score" in results[0]
    
    def test_processing_stats(self, processor):
        """Test processing statistics"""
        stats = processor.get_processing_stats()
        
        assert "Documents Processed" in stats
        assert "Total Processing Time" in stats
        assert "Models in Use" in stats
    
    def test_document_library(self, processor):
        """Test document library functionality"""
        library_df = processor.get_document_library()
        
        assert library_df is not None
        # If no documents, should have a message
        if len(processor.document_index) == 0:
            assert "No documents stored yet" in str(library_df.iloc[0, 0])

class TestBusinessToolsIntegration:
    """Integration tests for the business tools"""
    
    def test_environment_variables(self):
        """Test that required environment variables can be loaded"""
        # These should be set in the CI environment or locally
        required_vars = [
            "PHI3_BUSINESS_MODEL",
            "EMBEDDINGS_MODEL"
        ]
        
        for var in required_vars:
            # Should be set or have defaults
            value = os.getenv(var)
            # Just check it's not None - defaults are handled in the code
            assert value is not None or var in ["HF_TOKEN"]  # HF_TOKEN might not be set in tests
    
    def test_prompt_generation(self):
        """Test prompt generation for different document types"""
        processor = BusinessDocumentProcessor()
        
        prompt = processor._create_analysis_prompt(
            "Test document", "Contract", "Information Extraction"
        )
        
        assert "Contract" in prompt
        assert "Information Extraction" in prompt
        assert "Test document" in prompt
    
    def test_faiss_index_operations(self):
        """Test FAISS index operations"""
        with tempfile.TemporaryDirectory() as temp_dir:
            processor = BusinessDocumentProcessor()
            processor.index_path = Path(temp_dir)
            processor._initialize_storage()
            
            # Test loading empty index
            faiss_index = processor._load_faiss_index()
            assert faiss_index is not None
            assert faiss_index.ntotal == 0
            
            # Test adding embeddings
            test_embedding = np.random.randn(768).astype(np.float32)
            test_embedding = test_embedding / np.linalg.norm(test_embedding)
            
            faiss_index.add(test_embedding.reshape(1, -1))
            assert faiss_index.ntotal == 1

def test_module_imports():
    """Test that all required modules can be imported"""
    try:
        import gradio
        import numpy
        import pandas
        import faiss
        import huggingface_hub
        assert True
    except ImportError as e:
        pytest.fail(f"Required module import failed: {e}")

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
