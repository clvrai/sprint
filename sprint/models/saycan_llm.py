from sprint.datasets.large_language_model import LargeLanguageModel
import torch
from sentence_transformers import SentenceTransformer


class SaycanPlanner(LargeLanguageModel):
    prompt_start = "Robot: Hi there, I'm a robot operating in a house.\n"
    prompt_start += "Robot: You can ask me to do various tasks and I'll tell you the sequence of actions I would do to accomplish your task.\n"
    starter = "Human: How would you "
    prompt = prompt_start + starter
    prompt += "put the box with keys on the sofa next to the newspaper?\n"
    prompt += "Robot: "
    prompt += "1. Pick up the keys on the center table.\n"
    prompt += "2. Put the keys in the box.\n"
    prompt += "3. Pick up the box with keys.\n"
    prompt += "4. Put the box with keys on the sofa close to the newspaper.\n"

    prompt += starter + "cool a slice of lettuce and put it on the counter?\n"
    prompt += "Robot: "
    prompt += "1. Pick up the knife from in front of the tomato.\n"
    prompt += "2. Cut the lettuce on the counter.\n"
    prompt += "3. Set the knife down on the counter in front of the toaster.\n"
    prompt += "4. Pick up a slice of the lettuce from the counter.\n"
    prompt += "5. Put the lettuce slice in the refrigerator. take the lettuce slice out of the refrigerator.\n"
    prompt += "6. Set the lettuce slice on the counter in front of the toaster.\n"

    prompt += starter + "put the book on the table on the couch?\n"
    prompt += "Robot: "
    prompt += "1. Pick up the book on the table, in front of the chair.\n"
    prompt += "2. Place the book on the left cushion of the couch.\n"

    prompt += starter + "put the book on the table on the couch?\n"
    prompt += "Robot: "
    prompt += "1. Pick up the fork from the table.\n"
    prompt += "2. Put the fork in the sink and fill the sink with water, then empty the water from the sink and remove the fork.\n"
    prompt += "3. Put the fork in the drawer.\n"

    prompt += starter + "put two boxes of tissues on the barred rack?\n"
    prompt += "Robot: "
    prompt += "1. Take the box of tissues from the makeup vanity.\n"
    prompt += "2. Put the tissues on the barred rack.\n"
    prompt += "3. Take the box of tissues from the top of the toilet.\n"
    prompt += "4. Put the tissues on the barred rack.\n"

    prompt += starter + "put a heated glass from the sink onto the wooden rack?\n"
    prompt += "Robot: "
    prompt += "1. Pick up the glass from the sink.\n"
    prompt += "2. Heat the glass in the microwave.\n"
    prompt += "3. Put the glass on the wooden rack.\n"

    prompt += (
        starter + "look at the box from the far side of the bed under the lamp light?\n"
    )
    prompt += "Robot: "
    prompt += "1. Pick up the box from the far side of the bed.\n"
    prompt += "2. Hold the box and turn on the lamp.\n"

    prompt += starter
    prompt_mid_1 = "Robot: "
    prompt_mid_fn = lambda self, index, text: f"{index+1}. {text}\n"

    all_next_skill_prompt_start = prompt  # [: -len(starter)]
    all_next_skill_prompt_mid = "\nPredict the next skill correctly by choosing from the following next skills: "
    all_next_skill_aggregate_skills = (
        lambda self, text: f"{text.replace(text[-1], ';')}"
    )

    def __init__(self, config):
        config.llm_max_new_tokens = 30
        config.llm_next_skill_temp = 0.8
        config.llm_summary_temp = None
        super().__init__(config)
        self.threshold = 0.95
        print(f"SayCan Prompt:\n{self.prompt}")
        self.next_skill_top_p = 0.8

    def preprocess_llm_inputs_for_logprob(
        self,
        first_annotations: list[list[str]],
        second_annotations: list[str],
        high_level_tasks: list[str],
    ):
        modified_prompts_without_end = []
        only_part_twos = []
        for prompt_part_one, prompt_part_two, high_level_task in zip(
            first_annotations, second_annotations, high_level_tasks
        ):
            with_mid_annotations = [
                self.prompt_mid_fn(i, annotation)
                for i, annotation in enumerate(prompt_part_one)
            ]
            next_i = len(prompt_part_one)

            second_with_mid_annotations = self.prompt_mid_fn(next_i, prompt_part_two)
            high_level_task_question = high_level_task.lower()[:-1] + "?\n"
            modified_prompts_without_end.append(
                self.prompt
                + high_level_task_question  # task name
                + self.prompt_mid_1
                + "".join(with_mid_annotations)
                + second_with_mid_annotations
            )
            only_part_two = (
                # " " + prompt_part_two[0] + "\n" + second_with_mid_annotations[1:]
                # " " + prompt_part_two + "\n" + second_with_mid_annotations[4:] # get rid of the number and first letter (" 1. P" for example)
                second_with_mid_annotations[
                    2:
                ]  # get rid of the number (" 1." for example)
            )
            only_part_twos.append(only_part_two)
        all_tokenized_prompts_no_end = self.tokenizer(
            modified_prompts_without_end,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        tokenized_second_parts = self.tokenizer(
            only_part_twos, padding=True, truncation=True, return_tensors="pt"
        )
        return (
            all_tokenized_prompts_no_end,
            tokenized_second_parts,
        )

    def _get_logprobs(
        self,
        first_annotation: list[str],
        sample_second_annotations: list[str],
        high_level_tasks: list[str],
    ):
        all_first_annotations = [first_annotation] * len(sample_second_annotations)
        high_level_tasks_repeated = high_level_tasks * len(sample_second_annotations)
        # use batch size
        for i in range(0, len(sample_second_annotations), self.llm_batch_size):
            (
                batch_tokenized_prompt_no_end,
                batch_tokenized_second_part,
            ) = self.preprocess_llm_inputs_for_logprob(
                all_first_annotations[i : i + self.llm_batch_size],
                sample_second_annotations[i : i + self.llm_batch_size],
                high_level_tasks_repeated[i : i + self.llm_batch_size],
            )
            batch_input_ids = batch_tokenized_prompt_no_end.input_ids
            batch_attention_mask = batch_tokenized_prompt_no_end.attention_mask
            batch_second_part_attention_mask = (
                batch_tokenized_second_part.attention_mask
            )
            logprobs = self._get_non_generated_logprobs_hf(
                batch_input_ids,
                batch_attention_mask,
                batch_second_part_attention_mask,
            )
            if i == 0:
                all_logprobs = logprobs.clone()
            else:
                all_logprobs = torch.cat((all_logprobs, logprobs), dim=0)
        return all_logprobs.cpu()

    def get_saycan_logprobs(
        self,
        first_annotation: list[str],
        sample_second_annotations: list[str],
        high_level_tasks: list[str],
    ):
        return self._get_logprobs(
            first_annotation, sample_second_annotations, high_level_tasks
        )
