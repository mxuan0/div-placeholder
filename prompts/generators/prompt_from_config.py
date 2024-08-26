# from prompts.configs.llava_generator_diverse_description import prompts as llava_generator_diverse_description_prompts
from prompts.configs.llava_generator_diverse_description_5_describe import prompts as llava_generator_diverse_description_prompts
# from prompts.configs.llava_generator_diverse_description_1_sent import prompts as llava_generator_diverse_description_prompts


MODEL_TO_PROMPTS = {
    "llava-hf/llava-v1.6-mistral-7b-hf": llava_generator_diverse_description_prompts
}


class PromptFromConfig:
    def __init__(self, model_type, prompt_formatter=None):
        if model_type not in MODEL_TO_PROMPTS:
            raise ValueError(f"No prompts configure for {model_type}")

        self._prompts = MODEL_TO_PROMPTS[model_type]

    def prompts(self):
        return self._prompts
