import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from threading import Lock
from sprint.utils.utils import process_skill_strings


class LargeLanguageModel:
    summary_prompt_start = "Instructions: summarize the following ordered steps describing common household tasks.\n\n"

    summary_prompt_start += "Task Steps:\n"
    summary_prompt_start += (
        "1: Pick up the smaller knife on the counter to the left of the stove.\n"
    )
    summary_prompt_start += "2: Slice the tomato with the smaller knife.\n"
    summary_prompt_start += "3: Put the knife in the sink.\n"
    summary_prompt_start += "4: Pick up a slice of tomato from the countertop.\n"
    summary_prompt_start += (
        "5: Heat up the slice of tomato in the microwave, removing it afterwards.\n"
    )
    summary_prompt_start += "Summary: Microwave the tomato slice after slicing it with the smaller knife on the counter.\n\n"

    summary_prompt_start += "Task Steps:\n"
    summary_prompt_start += "1: Pick up the vase.\n"
    summary_prompt_start += "2: Turn on the lamp.\n"
    summary_prompt_start += "Summary: Look at the vase under the light.\n\n"

    summary_prompt_start += "Task Steps:\n"
    summary_prompt_start += "1: Grab the pencil off of the desk.\n"
    summary_prompt_start += "2: Put the pencil in the bowl.\n"
    summary_prompt_start += "3: Grab the container off of the desk.\n"
    summary_prompt_start += "4: Put the container down at the back of the desk.\n"
    summary_prompt_start += "Summary: Put a bowl with a pencil in it on the desk.\n\n"

    summary_prompt_start += "Task Steps:\n"
    summary_prompt_start += "1: Pick up the bar of soap from the back of the toilet.\n"
    summary_prompt_start += "2: Put the bar of soap in to the sink, turn on the faucet to rinse off the soap, pick up the soap out of the sink.\n"
    summary_prompt_start += (
        "3: Put the soap in the cabinet under the sink and on the left.\n"
    )
    summary_prompt_start += (
        "Summary: Put a rinsed bar of soap in the cabinet under the sink.\n\n"
    )

    summary_prompt_start += "Task Steps:\n"
    summary_prompt_mid = lambda self, index, text: f"{index+1}: {text}\n"
    summary_prompt_end = "Summary:"

    def __init__(self, config):
        assert (
            "opt" in config.llm_model or "gpt" in config.llm_model or "llama" in config.llm_model
        ), "No tokenizer support for non-gpt/opt/llama models"
        self.config = config
        self.llm_gpus = config.llm_gpus
        self.llm_max_new_tokens = config.llm_max_new_tokens
        self.llm_batch_size = config.llm_batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.llm_model,
            model_max_length=2048,
            use_fast=True,
        )

        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.pad_token = self.tokenizer.eos_token
        if len(self.llm_gpus) == 1:
            self.model = AutoModelForCausalLM.from_pretrained(
                config.llm_model,
                torch_dtype=torch.float16,
                pad_token_id=self.tokenizer.eos_token_id,
            ).to(self.llm_gpus[0])
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                config.llm_model,
                torch_dtype=torch.float16,
                pad_token_id=self.tokenizer.eos_token_id,
                device_map="auto",
            )
        self.lock = Lock()
        self.next_skill_top_p = 0.9
        self.next_skill_temp = 0.8
        self.ret_tensor_type = "pt"
        self.do_sample = True
        self.top_p = 0.9
        self.temp = 0.8
        self.num_beams = 1

    def _get_logprobs_hf(
        self,
        input_prompt_input_ids: torch.Tensor,
        input_prompt_attn_mask: torch.Tensor,
        second_skill_attn_mask: torch.Tensor,
    ):
        """Returns the average per-token sequential logprobs for only the generated text.

        Args:
            input_prompt_input_ids (torch.Tensor): tokens of the prompts
            input_prompt_attn_mask (torch.Tensor): attn mask of the prompts
            second_skill_attn_mask (torch.Tensor): attn mask corresponding to last (i.e., generated) sentence of the prompt.

        Returns:
            torch.tensor: logprobs of the generated text
        """
        second_skill_start_pos = second_skill_attn_mask.sum(-1)
        with torch.no_grad():
            logits = (
                self.model(
                    input_prompt_input_ids.to(self.llm_gpus[0]),
                    attention_mask=input_prompt_attn_mask.to(self.llm_gpus[0]),
                    return_dict=True,
                )
                .logits.cpu()
                .float()
            )
        input_ids = input_prompt_input_ids
        if self.tokenizer.bos_token_id is not None:
            # the start token is attended to
            second_skill_start_pos -= 1
            # every logit but the last one because the logits correspond to distributions over the NEXT token given the token at the position
            logits = logits[:, :-1]
            # shifted_input_ids to disregard start token
            input_ids = input_prompt_input_ids[:, 1:]
        logprobs = torch.log_softmax(logits, dim=-1)
        token_specific_logprobs = logprobs.gather(2, input_ids.unsqueeze(2)).squeeze(2)
        token_logprobs = []
        for i in range(len(second_skill_start_pos)):
            token_logprobs.append(
                torch.mean(token_specific_logprobs[i, -second_skill_start_pos[i] :])
            )
        return torch.tensor(token_logprobs)

    def process_hf_generation(self, hf_output: dict):
        """Postprocesses the output of the HF generation function (batched token ids) to create a list of strings.

        Args:
            hf_output (dict): The output of the HF generation function.

        Returns:
            list: A list of strings, each corresponding to a generated text sequence.
        """
        generated_tokens = hf_output["sequences"][:, -self.llm_max_new_tokens :].cpu()
        model_texts = self.tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        generated_texts = []
        special_eos = "."
        for i in range(len(model_texts)):
            model_text = model_texts[i]
            if special_eos in model_text:
                model_text = model_text[: model_text.index(special_eos)]
            generated_texts.append(model_text.strip())
        return process_skill_strings(generated_texts)

    def _get_summaries(self, input_ids, attention_mask):
        """
        Generates summaries for the given tokenized input_ids and attention_mask.
        Returns actual summary strings.
        """
        responses = self.model.generate(
            input_ids.to(self.llm_gpus[0]),
            attention_mask=attention_mask.to(self.llm_gpus[0]),
            return_dict_in_generate=True,
            early_stopping=True,
            max_new_tokens=self.llm_max_new_tokens,
            do_sample=self.do_sample,
            top_p=self.top_p,
            temperature=self.temp,
            num_beams=self.num_beams,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        new_labels = self.process_hf_generation(responses)
        return new_labels

    def get_summaries_and_logprobs(self, input_sentences):
        """
        Returns both summaries and associated logprobs of those generated summaries.

        :param input_sentences: list of list of sentences
        :return: list of summaries and list of logprobs
        """
        all_summaries = []
        all_logprobs = []
        for i in tqdm(range(0, len(input_sentences), self.llm_batch_size)):
            (
                tokenized_prompts,
                tokenized_conditional_prompts,
                tokenized_prompt_no_end,
            ) = self.preprocess_llm_inputs(input_sentences[i : i + self.llm_batch_size])
            # generate the logprobs
            batch_prompt_annotation_ids = tokenized_prompt_no_end.input_ids
            batch_prompt_annotation_attn_mask = tokenized_prompt_no_end.attention_mask
            batch_second_annotation_attn_mask = (
                tokenized_conditional_prompts.attention_mask
            )
            batch_logprobs = self._get_logprobs_hf(
                batch_prompt_annotation_ids,
                batch_prompt_annotation_attn_mask,
                batch_second_annotation_attn_mask,
            )
            # generate the summary
            batch_annotation_ids = tokenized_prompts.input_ids
            batch_annotation_attn_mask = tokenized_prompts.attention_mask
            new_labels = self._get_summaries(
                batch_annotation_ids, batch_annotation_attn_mask
            )
            all_summaries.extend(new_labels)
            all_logprobs.append(batch_logprobs)
        if len(all_logprobs) > 1:
            all_logprobs = torch.cat(all_logprobs, dim=0)
        else:
            all_logprobs = all_logprobs[0]

        return all_summaries, all_logprobs

    def get_summaries_only(self, input_sentences: list[list[str]]):
        """This generates summaries of a list of lists of given input sentences.
        Each inner list is considered a single set of sentences to be summarized.

        :param input_sentences: list of list of sentences
        :return: list of summaries and list of logprobs
        """
        all_summaries = []
        for i in range(0, len(input_sentences), self.llm_batch_size):
            tokenized_prompts = self.preprocess_llm_inputs_summary(
                input_sentences[i : i + self.llm_batch_size]
            )
            # generate the summary
            batch_annotation_ids = tokenized_prompts.input_ids
            batch_annotation_attn_mask = tokenized_prompts.attention_mask
            new_labels = self._get_summaries(
                batch_annotation_ids, batch_annotation_attn_mask
            )
            all_summaries.extend(new_labels)

        return all_summaries

    def preprocess_llm_inputs_summary(self, all_annotations: list[str]):
        """Preprocesses LLM inputs for summarization with the huggingface API

        Args:
            all_annotations (list[str]): list of skill annotations to be summarized

        Returns:
            torch.tensor: a tensor of tokenized, padded, and truncated input annotation strings
        """
        modified_prompts = []
        for primitive_annotations in all_annotations:
            with_mid_annotations = [
                self.summary_prompt_mid(i, annotation)
                for i, annotation in enumerate(primitive_annotations)
            ]
            modified_prompts.append(
                self.summary_prompt_start
                + "".join(with_mid_annotations)
                + self.summary_prompt_end
            )
        all_tokenized_prompts = self.tokenizer(
            modified_prompts,
            padding=True,
            truncation=True,
            return_tensors=self.ret_tensor_type,
        )
        return all_tokenized_prompts

    def preprocess_llm_inputs(self, all_annotations: list[str]):
        """Preprocesses LLM inputs for generation AND logprob generation with the huggingface API.

        Args:
            all_annotations (list[str]): list of skill annotations to be summarized

        """
        modified_prompts = []
        conditional_prompts = []
        modified_prompts_without_end = []
        for primitive_annotations in all_annotations:
            with_mid_annotations = [
                self.summary_prompt_mid(i, annotation)
                for i, annotation in enumerate(primitive_annotations)
            ]
            modified_prompts.append(
                self.summary_prompt_start
                + "".join(with_mid_annotations)
                + self.summary_prompt_end
            )
            modified_prompts_without_end.append(
                self.summary_prompt_start + "".join(with_mid_annotations)
            )
            conditional_prompts.append("".join(with_mid_annotations[1:]))
        all_tokenized_prompts = self.tokenizer(
            modified_prompts,
            padding=True,
            truncation=True,
            return_tensors=self.ret_tensor_type,
        )
        tokenized_conditional_prompts = self.tokenizer(
            conditional_prompts,
            padding=True,
            truncation=True,
            return_tensors=self.ret_tensor_type,
        )
        tokenized_prompt_no_end = self.tokenizer(
            modified_prompts_without_end,
            padding=True,
            truncation=True,
            return_tensors=self.ret_tensor_type,
        )
        return (
            all_tokenized_prompts,
            tokenized_conditional_prompts,
            tokenized_prompt_no_end,
        )

    def _get_non_generated_logprobs_hf(
        self,
        input_prompt_input_ids: torch.Tensor,
        input_prompt_attn_mask: torch.Tensor,
        second_skill_attn_mask: torch.Tensor,
    ):
        second_skill_start_pos = second_skill_attn_mask.sum(-1)
        with torch.no_grad():
            # so that even if multiple threads are using it at once, our maximium batch size won't be exceeded
            with self.lock:
                logits = (
                    self.model(
                        input_prompt_input_ids.to(self.llm_gpus[0]),
                        attention_mask=input_prompt_attn_mask.to(self.llm_gpus[0]),
                        return_dict=True,
                    )
                    .logits.cpu()
                    .float()
                )
        input_ids = input_prompt_input_ids
        if self.tokenizer.bos_token_id is not None:
            # the start token is attended to
            second_skill_start_pos -= 1
            # every logit but the last one because the logits correspond to distributions over the NEXT token given the token at the position
            logits = logits[:, :-1]
            # shifted_input_ids to disregard start token
            input_ids = input_prompt_input_ids[:, 1:]
        logprobs = torch.log_softmax(logits, dim=-1)
        token_specific_logprobs = logprobs.gather(2, input_ids.unsqueeze(2)).squeeze(2)
        token_logprobs = []
        for i in range(len(second_skill_start_pos)):
            token_logprobs.append(
                torch.mean(token_specific_logprobs[i, -second_skill_start_pos[i] :])
                # torch.sum(token_specific_logprobs[i, -second_skill_start_pos[i] :])
            )
        return torch.tensor(token_logprobs)

