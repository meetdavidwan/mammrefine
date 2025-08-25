#!/bin/bash

# We show an example of running {DATASET}

DATASET="mediasum"

# STEP 1: Detect with GPT-4o and Claude-3-sonnet
python src/detect.py \
    gpt-4o \
    data/{DATASET}.json \
    output/{DATASET}_detect_gpt4o_r1.json

python src/detect.py \
    claude-3-sonnet \
    data/{DATASET}.json \
    output/{DATASET}_detect_claude_r1.json

for i in {1..10}; do
    python src/detect_debate.py \
        gpt-4o \
        output/{DATASET}_detect_gpt4o_r${i}.json \
        output/{DATASET}_detect_claude_r${i}.json \
        output/{DATASET}_detect_gpt4o_r${i+1}.json
    
    python src/detect_debate.py \
        claude-3-sonnet \
        output/{DATASET}_detect_claude_r${i}.json \
        output/{DATASET}_detect_gpt4o_r${i}.json \
        output/{DATASET}_detect_claude_r${i+1}.json
done

# we just take gpt4o r10 as the final detection results.

# STEP 2: Critique with 2 GPT-4os
python src/critique.py \
    gpt-4o \
    output/{DATASET}_detect_gpt4o_r10.json \
    output/{DATASET}_critique_gpt4o_1.json

python src/critique.py \
    gpt-4o \
    output/{DATASET}_detect_gpt4o_r10.json \
    output/{DATASET}_critique_gpt4o_2.json

python src/critique_debate_initial.py \
    gpt-4o \
    output/{DATASET}_critique_gpt4o_1.json \
    output/{DATASET}_critique_gpt4o_2.json \
    output/{DATASET}_critique_gpt4o_1_r1.json

python src/critique_debate_initial.py \
    gpt-4o \
    output/{DATASET}_critique_gpt4o_1_r1.json \
    output/{DATASET}_critique_gpt4o_2_r1.json \
    output/{DATASET}_critique_gpt4o_1_r2.json

for i in {1..10}; do
    python src/critique_debate.py \
        gpt-4o \
        output/{DATASET}_critique_gpt4o_1_r${i}.json \
        output/{DATASET}_critique_gpt4o_2_r${i}.json \
        output/{DATASET}_critique_gpt4o_1_r${i+1}.json
    
    python src/critique_debate.py \
        gpt-4o \
        output/{DATASET}_critique_gpt4o_2_r${i+1}.json \
        output/{DATASET}_critique_gpt4o_1_r${i}.json \
        output/{DATASET}_critique_gpt4o_2_r${i+2}.json
done

# STEP 3: Refine with 2 Claude-3-sonnets

python src/refine.py \
    claude-3-sonnet \
    output/{DATASET}_critique_gpt4o_1_r10.json \
    output/{DATASET}_refine_claude_1.json

python src/refine.py \
    claude-3-sonnet \
    output/{DATASET}_critique_gpt4o_2_r10.json \
    output/{DATASET}_refine_claude_2.json

python src/refine_debate_initial.py \
    claude-3-sonnet \
    output/{DATASET}_refine_claude_1.json \
    output/{DATASET}_refine_claude_2.json \
    output/{DATASET}_refine_claude_1_r1.json

python src/refine_debate_initial.py \
    claude-3-sonnet \
    output/{DATASET}_refine_claude_2.json \
    output/{DATASET}_refine_claude_1.json \
    output/{DATASET}_refine_claude_2_r1.json


for i in {1..10}; do
    python src/refine_debate.py \
        claude-3-sonnet \
        output/{DATASET}_refine_claude_1_r${i}.json \
        output/{DATASET}_refine_claude_2_r${i}.json \
        output/{DATASET}_refine_claude_1_r${i+1}.json

    python src/refine_debate.py \
        claude-3-sonnet \
        output/{DATASET}_refine_claude_2_r${i}.json \
        output/{DATASET}_refine_claude_1_r${i}.json \
        output/{DATASET}_refine_claude_2_r${i+1}.json
done