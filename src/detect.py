"""
Factual consistency detection module
"""

import json
import sys
from tqdm import tqdm
from model import Model
from prompts import VERIFICATION_PROMPT
import re

def parse_json_response(response):
    """Parse JSON response from model"""
    try:
        # Clean up response for better parsing
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.endswith("```"):
            response = response[:-3]
        
        # Find JSON object
        start = response.find("{")
        end = response.rfind("}") + 1
        if start != -1 and end != 0:
            json_str = response[start:end].replace("\n", "").strip()
            return json.loads(json_str)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Failed to parse JSON: {e}")
        return None
    return None

def verify_sentence(model, document, sentence, max_retries=3):
    """Verify factual consistency of a single sentence"""
    prompt = VERIFICATION_PROMPT.format(document=document, sentence=sentence)
    
    for attempt in range(max_retries):
        try:
            response = model.run(prompt)
            if response is None:
                continue
            parsed = parse_json_response(response)
            if parsed and "answer" in parsed and "reasoning" in parsed:
                answer = parsed["answer"].lower()
                if answer in ["yes", "no"]:
                    return {
                        "answer": answer,
                        "reasoning": parsed["reasoning"],
                        "raw_response": response
                    }
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                return None
    
    return None

def detect_factual_consistency(model_name, input_file, output_file, api_key=None):
    """Main function to detect factual consistency"""
    # Load data
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Initialize model
    model = Model(model_name, api_key)
    
    results = []
    
    for item in tqdm(data, desc="Processing documents"):
        document = item["document"]
        summary_sentences = item.get("summary_sentences", [])
        
        verifications = []
        verifications_orig = []
        
        if summary_sentences:
            for sentence in tqdm(summary_sentences, desc="Verifying sentences", leave=False):
                result = verify_sentence(model, document, sentence)
                verifications.append(result)
                verifications_orig.append(result["raw_response"] if result else None)
        
        # Update item with verification results
        item["detection"] = verifications
        item["detection_orig"] = verifications_orig
        results.append(item)
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python detect.py <model_name> <input_file> <output_file>")
        print("Example: python detect.py gpt-4 data.json results.json")
        sys.exit(1)
    
    model_name = sys.argv[1]
    input_file = sys.argv[2]
    output_file = sys.argv[3]
    
    detect_factual_consistency(model_name, input_file, output_file) 