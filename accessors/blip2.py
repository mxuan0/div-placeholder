from PIL import Image
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
from lavis.models import load_model_and_preprocess


class Blip2Accessor:
    def __init__(
        self, 
        device=None, 
        model_dict={'loader':'hugging_face', 'model_type': "Salesforce/blip2-opt-6.7b-coco"}, 
        cache_dir='/data/jingdong'
    ):
        self.device = device
        if not device:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
        self.model_dict = model_dict
        self.cache_dir = cache_dir

        if model_dict['loader'] == 'lavis':
            self.model, self.vis_processors, self.txt_processors = self._load_from_lavis()
        elif model_dict['loader'] == 'hugging_face':
            self.model, self.processor = self._load_from_hf()
        else:
            raise NotImplementedError("Invalid model loader")

    def _load_from_lavis(self):
        return load_model_and_preprocess(
            name=self.model_dict['model_name'], 
            model_type=self.model_dict['model_type'], 
            is_eval=True, 
            device=self.device
        )

    def _load_from_hf(self):
        processor = Blip2Processor.from_pretrained(self.model_dict['model_type'], cache_dir=self.cache_dir)
        model = Blip2ForConditionalGeneration.from_pretrained(
            self.model_dict['model_type'], 
            device_map=self.device, 
            torch_dtype=torch.float16, 
            cache_dir=self.cache_dir
        ) 
        return model, processor

    def generate_caption(self, images):
        if not images:
            return []
            
        if self.model_dict['loader'] == "lavis":
            images = vis_processors["eval"](images).unsqueeze(0).to(device)
            return model.generate({"image": image})

        elif self.model_dict['loader'] == "hugging_face":
            inputs = self.processor(images=images, return_tensors="pt").to(self.device, torch.float16)
            generated_ids = self.model.generate(**inputs)
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
            
            return [text.strip() for text in generated_text]
