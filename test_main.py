import unittest
from unittest.mock import patch, MagicMock
from main import verify_claim

class TestVerifyClaim(unittest.TestCase):
    @patch('wikipediaapi.Wikipedia')
    def test_verify_claim(self, mock_wikipedia):
        # Mock the page object and its methods/attributes
        mock_page = MagicMock()
        mock_page.exists.return_value = True
        mock_page.summary = "COVID-19 vaccines do not cause infertility."
        mock_wikipedia.return_value.page.return_value = mock_page

        claim = "COVID-19 vaccines cause infertility"
        result = verify_claim(claim)
        self.assertIn("Wikipedia Summary", result)
        self.assertEqual(result["Wikipedia Summary"], ["COVID-19 vaccines do not cause infertility."])

if __name__ == '__main__':
    unittest.main()
