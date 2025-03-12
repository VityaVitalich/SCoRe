import sys

sys.path.append('../')
sys.path.append('./')

import os
import json
from functools import partial
from collections.abc import Iterable

from prompts.prompt_schemas import compose_chat_messages
# ^ referencing your helper that builds a system-prompt + instructions + few-shot + user block
# And `tokenizer.apply_chat_template(...)` that finalizes the prompt string.

def star_correction_initial_generation_prompt(
    sample,
    tokenizer,
    question_col="question",
    few_shot_prompts=None,
    *args,
    **kwargs
):

    # 1) System role: instructions about generating a chain-of-thought
    system_prompt = (
        "You are a helpful reasoning assistant in general domain question answering. "
        "Please reason through the question step by step very shortly before giving a final answer."
    )

    # 2) High-level instructions for the user block
    #    Feel free to rephrase for your domain or tasks
    instructions = (
        "Generate a short chain-of-thought rationale very shortly, and then provide the final answer.\n"
        "Step-by-step reasoning:\n"
        "Final Answer:\n"
    )

    # 3) The user question itself
    question_text = sample.get(question_col, "")
    user_question = f"Question:\n{question_text}\n\nReason step by step very shortly, then conclude with the answer."

    # 4) Build the message sequence
    messages = compose_chat_messages(
        system_prompt=system_prompt,
        instructions=instructions,
        user_question=user_question,
        few_shot_prompts=few_shot_prompts  # Optionally include a few chain-of-thought exemplars
    )

    # 5) Return the final merged prompt
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


def star_correction_prompt(
    sample,
    tokenizer,
    question_col="question",
    initial_answer_col="inital_answer",
    few_shot_prompts=None,
    *args,
    **kwargs
):

    # 1) System role: instructions
    system_prompt = (
        "You are a helpful reasoning assistant in general domain question answering. "
        "Your task is to correct the initial response if it is incorrect."
    )

    # 2) High-level instructions
    instructions = (
        "Below is the question and the initial answer. "
        "Generate a correction to the initial answer if it is incorrect"
        "Disregard the information you already have, look for other options. "
        "Do not use the information that does not match your criteria."
        "Step-by-step reasoning:\n"
        "Final Answer:\n"
    )

    # 3) The user block includes both the question and the correct answer (as the 'hint')
    question_text = sample.get(question_col, "")

    all_correction_prompts = []
    for initial_answer in sample[initial_answer_col]:
        user_question = (
            f"Question:\n{question_text}\n\n"
            f"Initial Answer: {initial_answer}\n\n"
            "Write a correction if the initial answer is incorrect."
        )

        # 4) Build the message sequence
        messages = compose_chat_messages(
            system_prompt=system_prompt,
            instructions=instructions,
            user_question=user_question,
            few_shot_prompts=few_shot_prompts  # Could also omit or add exemplars
        )

        all_correction_prompts.append(tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        ))

    return all_correction_prompts