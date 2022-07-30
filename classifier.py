import torch, torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import numpy as np
import os

from PIL import Image

class SnakeClassification():

    def __init__(self, img_path, model_path) -> None:
        self.image_path = img_path
        self.model_path = model_path
        self.identity_dict = {"0":"Python", "1":"Russel"}

    def predict(self):

        test_transform = transforms.Compose([
                                    transforms.Resize(size=256),
                                    transforms.CenterCrop(size=224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                        [0.229, 0.224, 0.225])
                                            ])
        test_image = Image.open(self.image_path).convert('RGB')
        
        test_image_tensor = test_transform(test_image)
        test_image_tensor = test_image_tensor.view(1, 3, 224, 224)
        model = torch.load(self.model_path)

        with torch.no_grad():
            model.eval()
            # Model outputs log probabilities
            out = model(test_image_tensor)
            ps = torch.exp(out)

            topk, topclass = ps.topk(2, dim=1)
            cls = topclass.cpu().numpy()[0][0]
            score = topk.cpu().numpy()[0][0]

            result = f'Identified as a {self.identity_dict[str(cls)]} with {score*100:.3f}% of confidence'
            #for i in range(2):
            #    print("Predcition", i+1, ":", idx_to_class[topclass.cpu().numpy()[0][i]], ", Score: ", topk.cpu().numpy()[0][i])
            return result

#predict_img = Predict('images\Screenshot (195).png', '_model_8.pt')

#result = predict_img.predict()
#print(result)