import torch
from transformers import ViTFeatureExtractor, ViTModel
from transformers import CLIPProcessor, CLIPModel


class Embedder:
    def __init__(self, model_type='clip', representation='cls',
                 model_kind='base', patch=32, device='cpu'):
        # Load DINO model and feature extractor
        self.feature_type = representation
        self.device = 'cuda' if (torch.cuda.is_available()
                                 and device == 'cuda') else 'cpu'
        self.model_type = model_type
        
        if model_type == 'dino':
            self.model_path = f'facebook/dino-vit{model_kind}{patch}'
            print(f'Loading DINO model from {self.model_path}...')
            
            self.feature_extractor = ViTFeatureExtractor.from_pretrained(self.model_path)
            self.model = ViTModel.from_pretrained(self.model_path)
            self.emb_dim = 768
        elif model_type == 'clip':
            self.model_path = f"openai/clip-vit-{model_kind}-patch{patch}"
            print(f'Loading CLIP model from {self.model_path}...')
            
            self.feature_extractor = CLIPProcessor.from_pretrained(self.model_path)
            self.model = CLIPModel.from_pretrained(self.model_path)
            self.emb_dim = self.model.projection_dim
        
        self.model.eval()
        self.model.to(self.device)
        print(f'Using {self.device} device')
        
    def set_params(self, feature_type, device):
        self.feature_type = feature_type
        new_device = 'cuda' if (torch.cuda.is_available() and device == 'cuda') else 'cpu'
        
        if self.device != new_device:
            self.device = new_device
            self.model.to(self.device)
            print(f'Using {self.device} device')

    def image_embedding(self, image):
        self.model.eval()
        
        with torch.no_grad():
            inputs = self.feature_extractor(images=image,
                                            return_tensors="pt")
            
            if self.model_type == 'dino':
                outputs = self.model(**inputs.to(self.device))
                features = outputs.last_hidden_state.detach().squeeze(0)
            elif self.model_type == 'clip':
                outputs = self.model.get_image_features(**inputs.to(self.device))
                features = outputs.detach().squeeze(0)
        
        # L2 normalize features
        features = features / features.norm(dim=-1, keepdim=True)
        # Apply Softmax with Torch
        features = torch.nn.functional.softmax(features, dim=-1)
        
        if self.device == 'cuda':
            features = features.cpu()
            
        if self.model_type == 'dino':
            if self.feature_type == 'cls':
                feature = features[0]
            else:
                feature = torch.mean(features, dim=0)
            return feature
        elif self.model_type == 'clip':
            return features
