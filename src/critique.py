"""
Critique generation module for factually inconsistent sentences
"""

import json
import sys
from tqdm import tqdm
from model import Model
from prompts import CRITIQUE_PROMPT

def generate_critique(model, topic, document, sentence, max_retries=3):
    """Generate critique for a single sentence"""
    prompt = CRITIQUE_PROMPT.format(
        topic=topic,
        document=document,
        summary=sentence
    )
    
    for attempt in range(max_retries):
        try:
            response = model.run(prompt)
            if response is not None:
                return response.strip()
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                return None
    
    return None

def get_verification_score(verification):
    """Extract verification score from verification result"""
    if verification is None:
        return None
    
    if isinstance(verification, dict):
        if "answer" in verification:
            return int(verification["answer"].lower() == "yes")
        elif "verification" in verification:
            return verification["verification"]
    
    return verification

def generate_critiques(model_name, data_file, output_file, api_key=None):
    """Main function to generate critiques"""
    # Load data
    with open(data_file, 'r') as f:
        data = json.load(f)
    # Initialize model
    model = Model(model_name, api_key)
    
    results = []
    
    for i, item in enumerate(tqdm(data, desc="Generating critiques")):
        document = item["document"]
        summary_sentences = item.get("summary_sentences", [])
        topic = item.get("topic", "Unknown")
        
        # Get verification scores
        score = [int(s["answer"] == "yes") for s in item.get("detection", [])]
        assert len(score) == len(summary_sentences)

        critiques = []
        
        if summary_sentences:
            for j, sentence in enumerate(tqdm(summary_sentences, desc="Processing sentences", leave=False)):
                if j < len(score):
                    s = score[j]
                    if s is None:
                        critique = None
                    else:
                        # Convert to binary score
                        if isinstance(s, dict):
                            s = int(s.get("answer", "no").lower() == "yes")
                        elif isinstance(s, str):
                            s = int(s.lower() == "yes")
                        else:
                            s = int(s)
                        
                        if s == 0:  # Factually inconsistent
                            critique = generate_critique(model, topic, document, sentence)
                        else:  # Factually consistent
                            critique = None
                else:
                    critique = None
                
                critiques.append(critique)
        
        # Update item with critiques
        item["critique"] = critiques
        results.append(item)
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Critiques saved to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python critique.py <model_name> <data_file> <output_file>")
        print("Example: python critique.py gpt-4 verification.json critiques.json")
        sys.exit(1)
    
    model_name = sys.argv[1]
    data_file = sys.argv[2]
    output_file = sys.argv[3]
    
    generate_critiques(model_name, data_file, output_file) 