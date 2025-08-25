"""
Summary refinement module that selects the best summary from multiple agents.
If agents disagree on the refinement, a model is used to select the best one.
"""

import json
import sys
import re
from tqdm import tqdm
from model import Model

from prompts import REFINEMENT_DEBATE_INITIAL_PROMPT

def select_best_summary(model, topic, document, summary1, summary2, max_retries=3):
    """
    Selects the best summary between two options using the model.
    This function handles the model call, JSON parsing, and retry logic.
    """
    prompt = REFINEMENT_DEBATE_INITIAL_PROMPT.format(
        topic=topic,
        document=document,
        response1=summary1,
        response2=summary2,
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
            print(f"Attempt {attempt + 1} failed. Error: {e}. Retrying...")
            if attempt == max_retries - 1:
                print("Max retries reached. Failing for this item.")
                return None
    return None

def refine_summaries_with_selection(model_name, input_files, output_file, api_key=None):
    """
    Main function to process summaries from multiple files and select the best one.
    """
    # Load all input files
    try:
        data_sets = [json.load(open(file)) for file in input_files]
        if not data_sets:
            raise ValueError("No input files were provided or loaded.")
        if len(set(len(data) for data in data_sets)) > 1:
            raise ValueError("All input files must have the same number of items.")
    except Exception as e:
        print(f"Error loading data files: {e}")
        sys.exit(1)
    
    # Initialize model for the selection task
    model = Model(model_name, api_key)
    
    # Use the first file as the base for iteration
    base_data = data_sets[0]
    results = []
    
    for i, item in enumerate(tqdm(base_data, desc="Refining and selecting summaries")):
        document = item["document"]
        topic = item.get("topic", "Unknown")
        original_summary = item.get("summary", "")
        
        # Collect all non-empty refined summaries for the current item
        summaries_all = []
        for data in data_sets:
            try:
                refined_summary = data[i].get("refined_summary")
                if refined_summary and refined_summary.strip():
                    summaries_all.append(refined_summary)
            except (IndexError, KeyError):
                continue
        
        # Get unique summaries while preserving order
        unique_summaries = list(dict.fromkeys(summaries_all))
        
        team_answer = None
        final_summary = None

        if len(unique_summaries) > 1:
            # If summaries differ, ask the model to choose the best one
            team_answer = select_best_summary(
                model, topic, document, unique_summaries[0], unique_summaries[1]
            )
            if team_answer and "answer" in team_answer:
                # Select the summary based on the model's answer
                chosen_index = team_answer["answer"] - 1
                final_summary = unique_summaries[chosen_index]
            else:
                # Fallback to the first summary if selection fails
                final_summary = unique_summaries[0]
        elif len(unique_summaries) == 1:
            # If all agents agree on the refinement, use that one
            final_summary = unique_summaries[0]
        else:
            # If no valid refinements exist, keep the original summary
            final_summary = original_summary

        # Update the base item with the new selection results
        item["refined_summary_team_answer"] = team_answer
        item["refined_summary_final"] = final_summary
        results.append(item)
    
    # Save the consolidated results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Summary selection complete. Results saved to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python refine.py <model_name> <input_file1> <input_file2> ... <output_file>")
        print("Example: python refine.py gpt-4 refined1.json refined2.json final_refinements.json")
        sys.exit(1)
    
    model_name = sys.argv[1]
    input_files = sys.argv[2:-1]
    output_file = sys.argv[-1]
    
    refine_summaries_with_selection(model_name, input_files, output_file)