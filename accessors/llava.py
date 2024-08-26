from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration, LlavaNextForConditionalGeneration, BitsAndBytesConfig
from prompts.generators.prompt_from_config import PromptFromConfig
from utils import list_lambda
from functools import partial

import torch, pdb


# FORMAT = "Give the response in one sentence "
FORMAT = ""

def mistral_7b_prompt_fomatter(prompt: str):
    return f"[INST] <image>\n{prompt} {FORMAT}[/INST]"


def mistral_7b_response_extractor(response: str, prompt: str):
    start_marker = f"[INST]  \n{prompt} {FORMAT}[/INST]"
    return response.split(start_marker)[-1].replace('\n', ' ').strip()
    
    # return response.replace('\n', ' ').strip()

PROMPT_FORMATTER = {
    "llava-hf/llava-v1.6-mistral-7b-hf": {
        "formatter": mistral_7b_prompt_fomatter,
        "response_extractor": mistral_7b_response_extractor
    }
}


class LlavaAccessor:
    def __init__(
        self,
        args,
        device=None, 
        model_type="llava-hf/llava-v1.6-mistral-7b-hf", 
        cache_dir='/data/jingdong'
    ):
        self.args = args
        self.device = device
        if not device:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
        if model_type not in PROMPT_FORMATTER:
            raise ValueError("model_type not implemented")

        self.model_type = model_type
        self.cache_dir = cache_dir

        self._create_captioning_prompts()
        self.model, self.processor = self._load_from_hf()
        
    def _load_from_hf(self):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        model = LlavaNextForConditionalGeneration.from_pretrained(
            self.model_type, 
            device_map=self.device, 
            cache_dir=self.cache_dir,
            quantization_config=quantization_config,
            # torch_dtype=torch.float16,
            # use_flash_attention_2=True
        )
        
        processor = AutoProcessor.from_pretrained(
            self.model_type,
            cache_dir=self.cache_dir
        )

        return model, processor

    def _create_captioning_prompts(self):
        if self.args.prompt_from_config:
            prompt_generator = PromptFromConfig(self.model_type)
        else:
            raise ValueError("llava captioner need prompts")

        self.prompts = prompt_generator.prompts()
        self.formatted_prompts = list_lambda(self.prompts, PROMPT_FORMATTER[self.model_type]['formatter'])

    def generate_caption(self, images):
        if not images:
            return []

        n_image = len(images)
        n_prompts = len(self.formatted_prompts)

        expanded_images = images * n_prompts
        expanded_prompts = []

        for prompt in self.formatted_prompts:
            for _ in range(n_image):
                expanded_prompts.append(prompt)

        # We can simply feed images in the order they have to be used in the text prompt
        # Each "<image>" token uses one image leaving the next for the subsequent "<image>" tokens
        inputs = self.processor(
            text=expanded_prompts, 
            images=expanded_images,
            padding=True, 
            return_tensors="pt"
        ).to(self.device)

        # Generate
        generate_ids = self.model.generate(**inputs, max_new_tokens=25)
        del inputs
        generated_text = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        del generate_ids
        
        return [
            list_lambda(
                generated_text[i*n_image : (i+1)*n_image], 
                partial(PROMPT_FORMATTER[self.model_type]['response_extractor'], prompt=self.prompts[i])
            )
            for i in range(n_prompts)
        ]
