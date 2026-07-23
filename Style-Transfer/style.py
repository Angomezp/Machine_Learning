import argparse
import os
import sys
import time
import re

import streamlit as st

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import torch.onnx


try:
    from . import utils  # Para ejecución local
except ImportError:
    import utils  # Para Streamlit Cloud

from transformer_net import TransformerNet
from transformer_net_v2 import TransformerNet as TransformerNetV2

from vgg19 import Vgg19

# Configuración automática del dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

# Mensaje informativo (opcional pero recomendado)
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    print(f"🚀 Usando GPU: {gpu_name}")
else:
    print("⚠️  GPU no disponible, usando CPU")


@st.cache_resource

def load_model(model_path):
    with torch.no_grad():
            style_name = os.path.basename(model_path).lower()
            if "michelangelo" in style_name:
                style_model = TransformerNetV2()
            else:
                style_model = TransformerNet()
            state_dict = torch.load(model_path)
            # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
            for k in list(state_dict.keys()):
                if re.search(r'in\d+\.running_(mean|var)$', k):
                    del state_dict[k]
            style_model.load_state_dict(state_dict)
            style_model.to(device)
            style_model.eval()
            return style_model
    
def stylize(style_model, content_image, output_image):
    
    content_image = utils.load_image(content_image)
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device)

    with torch.no_grad():
      output = style_model(content_image).cpu()
    utils.save_image(output_image, output[0])

  
