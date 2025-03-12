# prompt_builders/star_qa_builder.py

import sys
sys.path.append('../')
sys.path.append('./')

import os
import json
from functools import partial
from collections.abc import Iterable

from .base import BasePromptBuilder
from prompts.prompt_schemas import compose_chat_messages


class QAPromptBuilder(BasePromptBuilder):



    # For the *initial generation* step
    system_prompt_init = (
        "You are a helpful reasoning assistant in general domain question answering. "
        "Please reason through the question step by step very shortly before giving a final answer."
    )
    initial_instructions = (
        "Generate a short chain-of-thought rationale very shortly, and then provide the final answer.\n"
        "Step-by-step reasoning:\n"
        "Final Answer:\n"
    )

    # For the *correction* step
    system_prompt_corr = (
        "You are a helpful reasoning assistant in general domain question answering. "
        "Your task is to correct the initial response if it is incorrect."
    )
    correction_instructions = (
        "Consider the question and the initial answer. "
        "Generate a correction to the initial answer if it is incorrect. "
        "Disregard the information you already have, look for other options. "
        "Do not use the information that does not match your criteria."
        "Think about your correction step-by-step and output answer in following format:\n"
        "Step-by-step reasoning:\n"
        "Final Answer:\n"
    )

    def _create_user_question(self, question_text: str):

        return (
            f"Question:\n{question_text}\n\n"
            "Reason step by step very shortly, then conclude with the answer. "
            "Strictly follow format Step-by-step reasoning: and Final Answer:"
        )

    def build_initial_generation_prompt(
        self,
        sample,
        tokenizer,
        question_col="question",
        few_shot_prompts=None,
        tokenize=False,
        *args,
        **kwargs
    ):

        # Build user question
        question_text = sample.get(question_col, "")
        user_question = self._create_user_question(question_text)

        # Compose messages
        messages = compose_chat_messages(
            system_prompt=self.system_prompt_init,
            instructions=self.initial_instructions,
            user_question=user_question,
            few_shot_prompts=few_shot_prompts
        )

        # Return merged prompt
        return tokenizer.apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=True
        )

    def build_correction_prompt(
        self,
        sample,
        tokenizer,
        question_col="question",
        initial_answer_col="inital_answer",
        few_shot_prompts=None,
        tokenize=False,
        *args,
        **kwargs
    ):

        question_text = sample.get(question_col, "")
        user_question = self._create_user_question(question_text)
        initial_answers = sample.get(initial_answer_col, [])

        all_correction_prompts = []
        for init_ans in initial_answers:
            messages = [
                {"role": "user", "content": self.initial_instructions},
                {"role": "user", "content": user_question},
                {"role": "assistant", "content": init_ans},
                {"role": "user", "content": self.correction_instructions},
            ]

            # Convert messages to final text
            final_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=tokenize,
                add_generation_prompt=True
            )
            all_correction_prompts.append(final_prompt)

        return all_correction_prompts

    def build_correction_messages_with_final_answer(
        self,
        question,
        init_answer,
        correction,
        few_shot_prompts=None,
        *args,
        **kwargs
    ):
        """
        New helper method for a single conversation that ends 
        with the final correction as the assistant's last message.
        
        The 'question' is optional but let's incorporate it so we have context. 
        The idea is to mirror the logic of 'build_correction_prompt' 
        but actually include the final corrected answer as the assistant response.
        """

        # 1) system -> self.system_prompt_corr
        #messages.append({"role": "system", "content": self.system_prompt_corr})

        user_question = self._create_user_question(question)
        
        
        messages = [
                {"role": "user", "content": self.initial_instructions},
                {"role": "user", "content": user_question},
                {"role": "assistant", "content": init_answer},
                {"role": "user", "content": self.correction_instructions},
                {"role": "assistant", "content": correction},
            ]

        return messages


