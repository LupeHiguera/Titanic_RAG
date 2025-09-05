#!/usr/bin/env python3
"""
Test cases for witnesses API endpoint functionality.
Following TDD approach: write test first, confirm it fails, then implement.
"""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add project root to path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from app import app

class TestWitnessesAPI:
    """Test cases for the /witnesses API endpoint."""
    
    @pytest.fixture
    def client(self):
        """Create test client for FastAPI app."""
        return TestClient(app)
    
    def test_witnesses_endpoint_exists(self, client):
        """Test that /witnesses endpoint exists and returns 200."""
        response = client.get("/witnesses")
        assert response.status_code == 200
        
    def test_witnesses_returns_json_with_required_fields(self, client):
        """Test that /witnesses returns JSON with witnesses list and count."""
        response = client.get("/witnesses")
        data = response.json()
        
        # Check required fields exist
        assert "witnesses" in data
        assert "total_count" in data
        
        # Check data types
        assert isinstance(data["witnesses"], list)
        assert isinstance(data["total_count"], int)
        
    def test_witnesses_search_filtering(self, client):
        """Test that search parameter filters witnesses correctly."""
        # Get all witnesses first
        all_response = client.get("/witnesses")
        all_data = all_response.json()
        all_witnesses = all_data["witnesses"]
        
        # Test search filtering (should work with partial matches)
        if len(all_witnesses) > 0:
            # Use part of first witness name for search
            search_term = all_witnesses[0][:3].lower()  # First 3 chars
            
            search_response = client.get(f"/witnesses?search={search_term}")
            search_data = search_response.json()
            
            # All returned witnesses should contain the search term
            for witness in search_data["witnesses"]:
                assert search_term in witness.lower()
                
            # Count should match list length
            assert search_data["total_count"] == len(search_data["witnesses"])
    
    def test_witnesses_case_insensitive_search(self, client):
        """Test that search is case insensitive."""
        # Test with uppercase search term
        response_upper = client.get("/witnesses?search=ISMAY")
        response_lower = client.get("/witnesses?search=ismay")
        
        # Should return same results regardless of case
        assert response_upper.json() == response_lower.json()
    
    def test_witnesses_empty_search_returns_all(self, client):
        """Test that empty search returns all witnesses."""
        all_response = client.get("/witnesses")
        empty_search_response = client.get("/witnesses?search=")
        
        assert all_response.json() == empty_search_response.json()
    
    def test_witnesses_no_match_returns_empty(self, client):
        """Test that search with no matches returns empty list."""
        response = client.get("/witnesses?search=NONEXISTENTWITNESS123")
        data = response.json()
        
        assert data["witnesses"] == []
        assert data["total_count"] == 0
    
    def test_witnesses_list_is_sorted(self, client):
        """Test that witnesses list is returned in alphabetical order."""
        response = client.get("/witnesses")
        data = response.json()
        witnesses = data["witnesses"]
        
        # Check if list is sorted (should be same when re-sorted)
        assert witnesses == sorted(witnesses)
        
    def test_witnesses_integration_with_vector_db(self, client):
        """Test that witnesses endpoint returns data from actual vector database."""
        response = client.get("/witnesses")
        data = response.json()
        
        # Should have at least the witnesses we know exist (Ismay variants)
        witnesses = data["witnesses"]
        witness_names_lower = [w.lower() for w in witnesses]
        
        # We know these witnesses exist in our current dataset
        assert any("ismay" in name for name in witness_names_lower)
        assert data["total_count"] > 0

if __name__ == "__main__":
    # Run the tests directly
    pytest.main([__file__, "-v"])