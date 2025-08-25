"""
Summary refinement module based on critiques
"""

import json
import sys
from tqdm import tqdm
from model import Model
from prompts import REFINEMENT_PROMPT

def format_feedback(feedbacks):
    """Format feedback list into a string"""
    if not feedbacks:
        return ""
    
    feedback_str = ""
    for i, feedback in enumerate(feedbacks, 1):
        if feedback is not None:
            feedback_str += f"{i}. {feedback}\n"
    
    return feedback_str.strip()

def refine_summary(model, topic, document, summary, feedbacks, max_retries=3):
    """Refine a summary based on feedback"""
    if not feedbacks:
        return summary
    
    feedback_str = format_feedback(feedbacks)
    if not feedback_str:
        return summary
    
    prompt = REFINEMENT_PROMPT.format(
        topic=topic,
        document=document,
        summary=summary,
        feedback=feedback_str
    )
    
    for attempt in range(max_retries):
        try:
            response = model.run(prompt)
            if response is not None:
                return response.strip()
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                return summary  # Return original summary if refinement fails
    
    return summary

def refine_summaries(model_name, input_file, output_file, api_key=None):
    """Main function to refine summaries"""
    # Load data
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Initialize model
    model = Model(model_name, api_key)
    
    results = []
    
    for item in tqdm(data, desc="Refining summaries"):
        document = item["document"]
        summary = item.get("summary", "")
        topic = item.get("topic", "Unknown")
        
        # Get critiques/feedback
        feedbacks = item.get("critique_final", [])
        if not feedbacks:
            feedbacks = item.get("explanation", [])
        
        # Filter out None values
        valid_feedbacks = [f for f in feedbacks if f is not None]
        
        if valid_feedbacks:
            refined_summary = refine_summary(
                model, topic, document, summary, valid_feedbacks
            )
        else:
            refined_summary = summary
        
        # Update item with refined summary
        item["refined_summary"] = refined_summary
        results.append(item)
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Refined summaries saved to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python refine.py <model_name> <input_file> <output_file>")
        print("Example: python refine.py gpt-4 data_with_critiques.json refined_summaries.json")
        sys.exit(1)
    
    model_name = sys.argv[1]
    input_file = sys.argv[2]
    output_file = sys.argv[3]
    
    refine_summaries(model_name, input_file, output_file) 