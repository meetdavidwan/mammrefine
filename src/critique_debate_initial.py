"""
Critique selection module to choose the best critique from multiple sources.
This script reads critique data from multiple files, and for each sentence where
critiques differ, it uses a model to select the best one.
"""

import json
import sys
import re
from tqdm import tqdm
from prompts import CRITIQUE_DEBATE_PROMPT
from model import Model

def select_best_critique(model, topic, document, summary, critique1, critique2, max_retries=3):
    """
    Selects the best critique between two options using the model.
    This function incorporates the model call, JSON parsing, and retry logic.
    """
    prompt = CRITIQUE_DEBATE_PROMPT.format(
        topic=topic,
        document=document,
        summary=summary,
        response1=critique1,
        response2=critique2,
    )

    for attempt in range(max_retries):
        try:
            response = model.run(prompt)
            if response is not None:
                # Use regex to robustly find the JSON object in the response
                json_match = re.search(r"\{.*\}", response, re.DOTALL)
                if not json_match:
                    raise ValueError("No JSON object found in the response")
                
                response_json = json.loads(json_match.group(0))
                answer = str(response_json["answer"])
                
                if answer not in ["1", "2"]:
                    raise ValueError(f"Invalid answer: {answer}. Must be '1' or '2'.")
                
                reasoning = response_json["reasoning"]
                return {"answer": int(answer), "reasoning": reasoning}

        except (ValueError, KeyError, json.JSONDecodeError) as e:
            print(f"Attempt {attempt + 1} failed for sentence. Error: {e}. Retrying...")
            if attempt == max_retries - 1:
                print("Max retries reached. Failing for this sentence.")
                return None
    return None

def generate_critiques(model_name, data_files, output_file, api_key=None):
    """
    Main function to process and select the best critiques from multiple source files.
    """
    # Load all data files
    try:
        data_all = [json.load(open(file)) for file in data_files]
        if not data_all:
            raise ValueError("No data files were provided or loaded.")
    except Exception as e:
        print(f"Error loading data files: {e}")
        sys.exit(1)

    # Initialize model for selection
    model = Model(model_name, api_key)
    
    # Use the first file as the base for iteration and final output structure
    base_data = data_all[0]
    results = []
    
    for i, item in enumerate(tqdm(base_data, desc="Processing items")):
        document = item["document"]
        summary_sentences = item.get("summary_sentences", [])
        topic = item.get("topic", "Unknown")
        
        critique_team_answers = []
        critique_finals = []

        if summary_sentences:
            for j, sentence in enumerate(tqdm(summary_sentences, desc="Processing sentences", leave=False)):
                # Collect all non-empty critiques for the current sentence from all files
                critiques_for_sentence = []
                for data_source in data_all:
                    try:
                        # Ensure the data_source has the item and the critique for the sentence
                        critique = data_source[i]["critique"][j]
                        if critique:
                            critiques_for_sentence.append(critique)
                    except (IndexError, KeyError):
                        # This source might be missing the item or the critique, so we skip it
                        continue
                
                # Get unique critiques while preserving their order
                unique_critiques = list(dict.fromkeys(critiques_for_sentence))
                
                team_answer = None
                final_critique = None

                if len(unique_critiques) > 1:
                    # If critiques differ, ask the model to choose the best one between the first two
                    team_answer = select_best_critique(
                        model, topic, document, sentence, unique_critiques[0], unique_critiques[1]
                    )
                    if team_answer and "answer" in team_answer:
                        # Select the critique based on the model's answer
                        chosen_index = team_answer["answer"] - 1
                        final_critique = unique_critiques[chosen_index]
                elif len(unique_critiques) == 1:
                    # If all available critiques are identical, just use that one
                    final_critique = unique_critiques[0]

                critique_team_answers.append(team_answer)
                critique_finals.append(final_critique)

        # Update the base item with the new selection results
        item["critique_team_answer"] = critique_team_answers
        item["critique_final"] = critique_finals
        results.append(item)
    
    # Save the consolidated results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Critique selection complete. Results saved to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python critique.py <model_name> <data_file_1> <data_file_2> ... <output_file>")
        print("Example: python critique.py gpt-4 critique_run1.json critique_run2.json final_critiques.json")
        sys.exit(1)
    
    model_name = sys.argv[1]
    output_file = sys.argv[-1]
    data_files = sys.argv[2:-1]
    
    generate_critiques(model_name, data_files, output_file)