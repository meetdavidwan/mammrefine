"""
Factual consistency detection with debate between multiple agents
"""

import json
import sys
from tqdm import tqdm
from model import Model
from prompts import VERIFICATION_DEBATE_PROMPT
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

def format_agent_answers(verifications):
    """Format agent answers for debate prompt"""
    formatted_answers = []
    for i, verification in enumerate(verifications):
        if verification and "answer" in verification and "reasoning" in verification:
            answer = verification["answer"]
            reasoning = verification["reasoning"].strip()
            
            # Clean up reasoning
            for prefix in ["no.", "no,", "no", "yes.", "yes,", "yes"]:
                if reasoning.lower().startswith(prefix):
                    reasoning = reasoning[len(prefix):].strip()
                    break
            
            if reasoning:
                reasoning = reasoning[0].upper() + reasoning[1:]
            
            formatted_answers.append(
                f"Agent {i+1}'s answer: {{\"reasoning\": {reasoning}, \"answer\": {answer}}}"
            )
    
    return "\n".join(formatted_answers)

def verify_sentence_debate(model, document, sentence, verifications, max_retries=3):
    """Verify factual consistency with debate"""
    # Check if all agents agree
    valid_verifications = [v for v in verifications if v is not None]
    if not valid_verifications:
        return None
    
    answers = [v["answer"].lower() for v in valid_verifications]
    if len(set(answers)) <= 1:  # All agents agree
        return valid_verifications[0]
    
    # Agents disagree, use debate
    agent_answers = format_agent_answers(valid_verifications)
    prompt = VERIFICATION_DEBATE_PROMPT.format(
        document=document, 
        sentence=sentence, 
        agent_answers=agent_answers
    )
    
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

def detect_factual_consistency_debate(model_name, input_files, output_file, api_key=None):
    """Main function to detect factual consistency with debate"""
    # Load all input files
    data_sets = []
    for input_file in input_files:
        with open(input_file, 'r') as f:
            data_sets.append(json.load(f))
    
    # Verify all datasets have the same structure
    if len(set(len(data) for data in data_sets)) > 1:
        raise ValueError("All input files must have the same number of items")
    
    # Initialize model
    model = Model(model_name, api_key)
    
    results = []
    num_changed = 0
    total = 0
    
    for i, item in enumerate(tqdm(data_sets[0], desc="Processing documents")):
        document = item["document"]
        summary_sentences = item.get("summary_sentences", [])
        
        # Get verifications from all agents
        verifications_all = [data[i]["detection"] for data in data_sets]
        
        verifications = []
        if summary_sentences:
            for j, sentence in enumerate(tqdm(summary_sentences, desc="Verifying sentences", leave=False)):
                # Get verifications for this sentence from all agents
                sentence_verifications = [v[j] for v in verifications_all if v[j] is not None]
                
                if sentence_verifications:
                    result = verify_sentence_debate(model, document, sentence, sentence_verifications)
                    if result != sentence_verifications[0]:  # Check if debate changed the answer
                        num_changed += 1
                    verifications.append(result)
                    total += 1
                else:
                    verifications.append(None)
        
        # Update item with verification results
        item["detection"] = verifications
        results.append(item)
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Changed {num_changed}/{total} answers")
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python detect_debate.py <model_name> <input_file1> <input_file2> ... <output_file>")
        print("Example: python detect_debate.py gpt-4 agent1.json agent2.json debate_results.json")
        sys.exit(1)
    
    model_name = sys.argv[1]
    input_files = sys.argv[2:-1]
    output_file = sys.argv[-1]
    
    detect_factual_consistency_debate(model_name, input_files, output_file) 