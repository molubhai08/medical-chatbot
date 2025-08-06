# pdfextractor.py

import fitz  # PyMuPDF
import requests



class PDFExtractor:
    def __init__(self, groq_api_key, model="llama3-70b-8192", max_chunk_len=2000):
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {groq_api_key}",
            "Content-Type": "application/json"
        }
        self.model = model
        self.max_chunk_len = max_chunk_len

    def extract_text(self, file_path):
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text

    def summarize_chunk(self, chunk):

        payload = {
            "model": "llama3-70b-8192",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a clinical language model designed to extract the core medical data from a medical report."
                },
                {
                    "role": "user",
                    "content": f"""
        Given the report text, extract and return only the main clinical information in structured JSON format as follows:

        {{
        "Test Results": [],
        }}

        Only include what's mentioned in the report. If any section is not available, leave it as an empty string or list. Be concise but medically accurate.
        
        IMPORTANT: Return ONLY the JSON object with no additional text, explanations, or formatting whatsoever.

        Here is the report:
        {chunk}
        """
                }
            ],
            "temperature": 0
        }


        response = requests.post(self.api_url, headers=self.headers, json=payload)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return "[Summary failed for this chunk]"

    def summarize_text(self, text):
        chunks = [text[i:i + self.max_chunk_len] for i in range(0, len(text), self.max_chunk_len)]
        summaries = [self.summarize_chunk(chunk) for chunk in chunks]
        return  summaries # "\n".join(summaries)

    def extract_and_summarize(self, file_path):
        text = self.extract_text(file_path)
        summary = self.summarize_text(text)
        return summary
    

