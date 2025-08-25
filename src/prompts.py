"""
Prompts for the MammRefine pipeline
"""

VERIFICATION_PROMPT = """Document: {document}
Sentence: {sentence}
Determine if the sentence is factually consistent with the document provided above. A sentence is factually consistent if it can be entailed (either stated or implied) by the document. Please briefly explain the reason within 50 words. Output your answer in json format, with the format as follows: {{"reasoning": "", "answer": ""}}. Please strictly output in JSON format. Only answer yes or no in the "answer" field."""

VERIFICATION_DEBATE_PROMPT = """Document: {document}
Sentence: {sentence}

Determine if the sentence is factually consistent with the document provided above. A sentence is factually consistent if it can be entailed (either stated or implied) by the document. Please briefly explain the reason within 50 words. Output your answer in json format, with the format as follows: {{"reasoning": "", "answer": ""}}. Please strictly output in JSON format. Only answer yes or no in the "answer" field.

Carefully review the following solutions from other agents as additional information, and provide your own answer and step-by-step reasoning to the question.

{agent_answers}"""

CRITIQUE_PROMPT = """I summarized the following document on the topic: '{topic}':
{document}

Summary of the above document on topic '{topic}':
{summary}

Reason about the factually inconsistent span in the sentence. A span is factually inconsistent if it cannot be substantiated by the document. Give reasons for the factual inconsistency, point to the error span by stating "The error span: <span from sentence>" and end your answer with a suggested fix to the summary."""

CRITIQUE_DEBATE_INITIAL_PROMPT = """I summarized the following document on the topic: '{topic}':
{document}

Summary of the above document on topic '{topic}':
{summary}

Reason about the factually inconsistent span in the sentence. A span is factually inconsistent if it cannot be substantiated by the document. Give reasons for the factual inconsistency, point to the error span by stating "The error span: <span from sentence>" and end your answer with a suggested fix to the summary.

## Response 1:
{response1}

## Response 2:
{response2}

Select the best response that best identifies the factually inconsistent span in the sentence and provides a suggested fix to the summary. Please briefly explain the reason within 50 words. Output your answer in json format, with the format as follows: {{"reasoning": "", "answer": ""}}. Please strictly output in JSON format. Only answer 1 or 2 in the "answer" field."""

CRITIQUE_DEBATE_PROMPT = """Document: {document}
Summary of the above document on topic '{topic}':
{summary}

Reason about the factually inconsistent span in the sentence. A span is factually inconsistent if it cannot be substantiated by the document. Give reasons for the factual inconsistency, point to the error span by stating "The error span: <span from sentence>" and end your answer with a suggested fix to the summary.

## Response 1:
{response1}

## Response 2:
{response2}

Select the best response that best identifies the factually inconsistent span in the sentence and provides a suggested fix to the summary. Please briefly explain the reason within 50 words. Output your answer in json format, with the format as follows: {{"reasoning": "", "answer": ""}}. Please strictly output in JSON format. Only answer 1 or 2 in the "answer" field.

Carefully review the following solutions from other agents as additional information, and provide your own answer and step-by-step reasoning to the question.

{agent_answers}"""


REFINEMENT_PROMPT = """I summarized the following document on the topic '{topic}':
{document}

Summary of the above document on topic '{topic}':
{summary}

Feedback for the above summary:
{feedback}

Edit the user response such that the refinement doesn't have any errors mentioned in the feedback. Make the minimum number of changes when doing the refinement. Do not include a preamble."""

REFINEMENT_DEBATE_INITIAL_PROMPT = """Document: {document}

Summarize the provided document focusing on "{topic}". The summary should be less than 50 words in length.

## Response 1:
{response1}

## Response 2:
{response2}

Select the best summary that contains the least amount of factual inconsistencies. Consistency in this context implies that all information presented in the summary is substantiated by the document. Please briefly explain the reason within 50 words. Output your answer in json format, with the format as follows: {{"reasoning": "", "answer": ""}}. Please strictly output in JSON format. Only answer 1 or 2 in the "answer" field."""

REFINEMENT_DEBATE_PROMPT = """Document: {document}

Summarize the provided document focusing on "{topic}". The summary should be less than 50 words in length.

## Response 1:
{response1}

## Response 2:
{response2}

Select the best summary that contains the least amount of factual inconsistencies. Consistency in this context implies that all information presented in the summary is substantiated by the document. Please briefly explain the reason within 50 words. Output your answer in json format, with the format as follows: {{"reasoning": "", "answer": ""}}. Please strictly output in JSON format. Only answer 1 or 2 in the "answer" field.

Carefully review the following solutions from other agents as additional information, and provide your own answer and step-by-step reasoning to the question.

{agent_answers}"""