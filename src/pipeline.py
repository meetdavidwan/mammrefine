"""
Main pipeline for MammRefine: Multi-Agent Multi-Modal Refinement
"""

import json
import argparse
import os
from tqdm import tqdm
from model import Model
from detect import detect_factual_consistency
from detect_debate import detect_factual_consistency_debate
from critique import generate_critiques
from critique_debate import generate_critiques_debate
from refine import refine_summaries
from refine_debate import refine_summaries_debate

def run_pipeline(model_name, input_file, output_dir, use_debate=False, api_key=None):
    """Run the complete MammRefine pipeline"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 50)
    print("MammRefine Pipeline")
    print("=" * 50)
    
    # Step 1: Factual Consistency Detection
    print("\nStep 1: Factual Consistency Detection")
    detection_output = os.path.join(output_dir, "detection_results.json")
    
    if use_debate:
        print("Running detection with debate (simplified - single agent)")
        detect_factual_consistency(model_name, input_file, detection_output, api_key)
    else:
        detect_factual_consistency(model_name, input_file, detection_output, api_key)
    
    # Step 2: Critique Generation
    print("\nStep 2: Critique Generation")
    critique_output = os.path.join(output_dir, "critique_results.json")
    
    if use_debate:
        print("Running critique generation with debate (simplified - single agent)")
        generate_critiques(model_name, input_file, detection_output, critique_output, api_key)
    else:
        generate_critiques(model_name, input_file, detection_output, critique_output, api_key)
    
    # Step 3: Summary Refinement
    print("\nStep 3: Summary Refinement")
    refinement_output = os.path.join(output_dir, "refinement_results.json")
    
    if use_debate:
        print("Running refinement with debate (simplified - single agent)")
        refine_summaries(model_name, critique_output, refinement_output, api_key)
    else:
        refine_summaries(model_name, critique_output, refinement_output, api_key)
    
    print(f"\nPipeline completed! Results saved to {output_dir}")
    return refinement_output

def main():
    parser = argparse.ArgumentParser(description="MammRefine Pipeline")
    parser.add_argument("--model", type=str, required=True, 
                       help="Model name (e.g., gpt-4, claude-3-sonnet)")
    parser.add_argument("--input", type=str, required=True,
                       help="Input JSON file with documents and summaries")
    parser.add_argument("--output", type=str, required=True,
                       help="Output directory for results")
    parser.add_argument("--debate", action="store_true",
                       help="Use debate mode (requires multiple agents)")
    parser.add_argument("--api-key", type=str, default=None,
                       help="API key for the model (or set environment variable)")
    
    args = parser.parse_args()
    
    # Set API key from environment if not provided
    if not args.api_key:
        if "gpt" in args.model:
            args.api_key = os.getenv("OPENAI_API_KEY")
        elif "claude" in args.model:
            args.api_key = os.getenv("ANTHROPIC_API_KEY")
    
    if not args.api_key:
        print("Warning: No API key provided. Set environment variable or use --api-key")
    
    run_pipeline(args.model, args.input, args.output, args.debate, args.api_key)

if __name__ == "__main__":
    main() 