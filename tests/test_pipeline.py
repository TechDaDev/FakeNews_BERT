import unittest
import torch
import numpy as np
from src.prediction import decode_predictions, aggregate_chunk_predictions, LABEL_REAL, LABEL_FAKE
from src.article_extraction import extract_article
import os

class TestPipeline(unittest.TestCase):
    
    def test_label_mapping(self):
        """Verify that 0 is Real and 1 is Fake in decode_predictions."""
        # Simulated logits where index 0 is high
        logits_real = torch.tensor([[5.0, -5.0]])
        res_real = decode_predictions(logits_real)
        self.assertEqual(res_real['predicted_label'], "REAL")
        self.assertEqual(res_real['class_index'], LABEL_REAL)
        self.assertGreater(res_real['real_probability'], 0.99)
        
        # Simulated logits where index 1 is high
        logits_fake = torch.tensor([[-5.0, 5.0]])
        res_fake = decode_predictions(logits_fake)
        self.assertEqual(res_fake['predicted_label'], "FAKE")
        self.assertEqual(res_fake['class_index'], LABEL_FAKE)
        self.assertGreater(res_fake['fake_probability'], 0.99)

    def test_aggregation(self):
        """Verify that aggregation prefers fake if one chunk is very strong."""
        chunks = [
            {'real_probability': 0.9, 'fake_probability': 0.1, 'predicted_label': 'REAL'},
            {'real_probability': 0.9, 'fake_probability': 0.1, 'predicted_label': 'REAL'},
            {'real_probability': 0.1, 'fake_probability': 0.9, 'predicted_label': 'FAKE'} # One very fake chunk
        ]
        # With default strong threshold 0.75, this should be FAKE
        agg = aggregate_chunk_predictions(chunks, threshold_strong=0.75)
        self.assertEqual(agg['predicted_label'], "FAKE")
        self.assertEqual(agg['max_fake_prob'], 0.9)
        
        # Test case where none are strong
        chunks_weak = [
            {'real_probability': 0.6, 'fake_probability': 0.4, 'predicted_label': 'REAL'},
            {'real_probability': 0.6, 'fake_probability': 0.4, 'predicted_label': 'REAL'}
        ]
        agg_weak = aggregate_chunk_predictions(chunks_weak, threshold_strong=0.75)
        self.assertEqual(agg_weak['predicted_label'], "REAL")

    def test_extraction_validation(self):
        """Test extraction logic with a mock-like short text."""
        # This is more of an integration test if it hits the web, 
        # but we can test the validation logic by passing known content if we had a way.
        # For now, let's just check the method exists and returns the right keys.
        url = "https://www.google.com" # unlikely to be a news article
        res = extract_article(url)
        self.assertIn('clean_text', res)
        self.assertIn('extraction_method', res)
        self.assertIn('warnings', res)

    def test_bert_decoding_calibration(self):
        """Verify temperature scaling affects confidence."""
        logits = torch.tensor([[2.0, -2.0]]) # index 0 is high
        res_normal = decode_predictions(logits, temperature=1.0)
        res_calibrated = decode_predictions(logits, temperature=2.0) # higher temp = softer probs
        
        self.assertLess(res_calibrated['confidence'], res_normal['confidence'])
        self.assertEqual(res_calibrated['predicted_label'], res_normal['predicted_label'])

if __name__ == '__main__':
    unittest.main()
