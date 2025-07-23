from transformers import AutoImageProcessor, SiglipForImageClassification
from app.config import model_name

model = SiglipForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)
