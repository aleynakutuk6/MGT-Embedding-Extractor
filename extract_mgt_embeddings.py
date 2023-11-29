import os
import json
import torch
import random
import numpy as np
from tqdm import tqdm

from multigraph_transformer import MultiGraphTransformer
from mgt_preprocessor import MGTPreprocessor

def save_mgt_embeds(root_dir, qd_classes, is_absolute):
    
    print("Loading the model...")
    
    # Initialize parameters
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model_prep = MGTPreprocessor(device=device)
    
    # Load models
    model = MultiGraphTransformer(
        n_classes=345, coord_input_dim=2, feat_input_dim=2, 
        feat_dict_size=103, n_layers=4, n_heads=8, embed_dim=256, 
        feedforward_dim=1024, normalization='batch', dropout=0.25,
        mlp_classifier_dropout=0.25)
    
    model.load_state_dict(
        torch.load(
            "weights/multigraph_transformer.pth", 
            map_location=torch.device('cpu'))["network"], strict=True)
    model = model.to(device).eval()
        
    print("Reading the dataset...")

    for set_type in os.listdir(root_dir):
        
        if set_type not in ["train", "valid", "test"]:
            continue
        
        paths_dict = {}
        
        folder_path = os.path.join(root_dir, set_type)   
        for folder_name in os.listdir(folder_path):
            
            if folder_name not in ["vectors"]:
                continue
                
            file_path = os.path.join(folder_path, folder_name)   
            for filename in tqdm(os.listdir(file_path)):
                
                
                class_name, img_id, rest = filename.split("_")
                inst_id = rest.split(".")[0]
                class_id = qd_classes.index(class_name)
                
                path = os.path.join(file_path, filename)
                
                with open(path, "r") as f:
                    data = json.load(f)
                
                qd_data = data["stroke"]
                sketch = np.asarray(qd_data)
                
                inps = model_prep.preprocess(sketch, is_absolute=is_absolute)
                with torch.no_grad():
                    out, embs = model(*inps)
                    embs = embs.squeeze(0)
                    out = torch.softmax(out, dim=1).squeeze(0)
                    max_idx = out.argmax().data.item()
                    cls_score = out[max_idx].data.item()
                    cls_name = qd_classes[max_idx].lower()
                    
                data["mgt_embeds_from_linear"] = embs.cpu().numpy().tolist()
                # data["mgt_scores"] = out.cpu().numpy().tolist()
                # data["mgt_pred"] = {"class": max_idx, "score": cls_score}
                
                with open(path, "w") as f:
                    json.dump(data, f)
                   

f = open('/userfiles/akutuk21/Sketchformer/prep_data/quickdraw/list_quickdraw.txt', 'r')
lines = f.readlines()
qd_classes = [cls.replace("\n", "") for cls in lines]

root_dir = "../generate_scene_dataset/scene_coords-new"
is_absolute=False
save_mgt_embeds(root_dir, qd_classes, is_absolute)



