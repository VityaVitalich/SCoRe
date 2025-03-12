
import os
import json
from functools import partial
from collections.abc import Iterable

from prompt_schemas import compose_chat_messages



def gather_full_conversation(
    sample,
    question_col="question",
    generation_col="self_refine_initial_generation",
    feedback_prefix="self_refine_feedback_",
    refinement_prefix="self_refine_refinement_"
):
    """
    Gathers the entire conversation history from 'sample' in chronological order:
    - The user’s question'
    - Initial generation self_refine_initial_generation   (user)
    - For each iteration i:
      * feedback_i                (user)
      * refinement_i              (user)

    Returns a list of dicts with "role" and "content".
    """

    messages = []

    # 1) Add the user's question
    user_question = sample.get(question_col, "")
    if user_question:
        messages.append({
            "role": "user",
            "content": f"User question: {user_question}"
        })

    generation_text = sample.get(generation_col, None)[0]
    messages.append({
        "role": "user",
        "content": f"Initial Answer: {generation_text}"
    })


    # 2) Identify how many steps we have. We’ll gather them until we don’t find a generation_i.
    i = 0
    while True:
        fb_key = f"{feedback_prefix}{i}"
        ref_key = f"{refinement_prefix}{i}"


        # Check if there's feedback_i
        feedback_text = sample.get(fb_key, None)
        if feedback_text is not None:
            messages.append({
                "role": "user",
                "content": f"Feedback {i}: {feedback_text[0]}"
            })
        else:
            break

        # Check if there's refinement_i
        refinement_text = sample.get(ref_key, None)
        if refinement_text is not None:
            messages.append({
                "role": "user",
                "content": f"Refinement {i}: {refinement_text[0]}"
            })
        else:
            break

        i += 1

    return messages


def initial_generation_prompt(
    sample,
    tokenizer,
    question_col="question",
    few_shot_prompts=None,
    *args,
    **kwargs
):

    # 1) Define the system prompt—e.g., specifying the role, policies, or style for the AI assistant
    system_prompt = (
        "You are a helpful AI assistant. Your goal is to answer the user's question precisely "
        "and concisely. You can provide details, but avoid digressions."
    )

    # 2) Provide instructions describing how to answer. 
    #    These might include guidelines for formatting, referencing sources, etc.
    instructions = (
        "Answer the following question concisely.\n"
    )

    # 3) Extract the user’s question from the sample
    user_question = sample.get(question_col, "")

    # 4) Compose the messages. The function `compose_chat_messages` should handle:
    messages = compose_chat_messages(
        system_prompt=system_prompt,
        instructions=instructions,
        user_question=user_question,
        few_shot_prompts=few_shot_prompts,  # Optionally include a few chain-of-thought exemplars
    )

    # 5) Return the final merged prompt with the chat template
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

def refinement_prompt(
    sample,
    tokenizer,
    conversation_gather_func=gather_full_conversation,
    few_shot_prompts=None,
    *args,
    **kwargs
):
    """
    Creates a prompt asking the model to refine its previous answer using the *latest* feedback.

    :param sample: A dictionary with the entire conversation so far.
    :param tokenizer: A tokenizer or utility with method 'apply_chat_template'
    :param next_refinement_index: The index for the new refinement,
                                  e.g. 0 for refinement_0, 1 for refinement_1, etc.
    :param few_shot_prompts: Optional chain-of-thought exemplars
    :return: A prompt for the model to generate a refined answer
    """

    # 1) System prompt for refining
    system_prompt = (
        "You are refining the assistant's previous answer based on the latest feedback. "
        "Incorporate the critique to address any errors or omissions."
        "Make the answer concise and accurate while using only helpful context."
    )

    # 2) Instructions
    instructions = (
        "Below is the entire conversation so far. Focus on the latest feedback. "
        "Review and correct the given answer based on the information provided."
        "Update or correct the most recent assistant answer as necessary to produce a refined answer."
        "Do not use the information that does not match your criteria."
        "Output only the refined answer"
    )

    # 3) Gather the entire conversation
    conversation_messages = conversation_gather_func(sample)

    # 4) Compose final messages
    messages = [
        {"role": "user", "content": system_prompt},
        {"role": "user", "content": instructions}
    ]
    messages.extend(conversation_messages)

    # Optionally add few_shot_prompts
    if few_shot_prompts:
        messages.extend(few_shot_prompts)

    # 5) Apply the chat template
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


def feedback_prompt(
    sample,
    tokenizer,
    conversation_gather_func=gather_full_conversation,
    few_shot_prompts=None,
    *args,
    **kwargs
):
    """
    Creates a prompt asking the model to provide feedback on the *latest* answer
    in the conversation (be it an initial generation or a refinement).

    :param sample: A dictionary storing the entire conversation so far.
    :param tokenizer: A tokenizer or utility with method 'apply_chat_template'
    :param next_feedback_index: The index for the new feedback, e.g., 0 for feedback_0, 1 for feedback_1, etc.
    :param few_shot_prompts: Optional chain-of-thought exemplars
    :return: A prompt for the model to produce the new feedback
    """

    # 1) The "system prompt" specifically for providing feedback
    system_prompt = (
        "You are reviewer. You will provide feedback on the assistant's most recent answer. "
        "Identify any factual error, and suggest how they might be improved."
    )

    # 2) The "instructions" for how to provide feedback
    instructions = (
        "Below is the entire conversation so far, including the user’s question, all previous generations, "
        "feedback, and refinements. Now think step-by-step and provide a new piece of feedback, focusing "
        "on the most recent assistant answer correctness."
    )

    # 3) Gather the entire conversation
    conversation_messages = conversation_gather_func(sample)

    messages = [
        {"role": "user", "content": system_prompt},
        {"role": "user", "content": instructions}
    ]
    messages.extend(conversation_messages)

    # Optionally include few_shot_prompts (chain-of-thought exemplars) if desired
    if few_shot_prompts:
        # Insert them in the correct places or roles as needed
        messages.extend(few_shot_prompts)

    # 5) Finally, apply the chat template
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
