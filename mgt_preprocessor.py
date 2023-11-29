import os
import torch
import numpy as np

from draw_scene import *

from utils.sketch import (
    relative_to_absolute, normalize_to_scale, get_absolute_bounds)

class MGTPreprocessor:
    
    def __init__(self, seq_len: int=100, device="cpu"):
        self.seq_len         = seq_len
        self.device          = device
        self.norm_flag       = 100
        self.stroke_end_flag = 101
        self.pad_flag        = 102

        
    def preprocess(self, sketch: np.ndarray, is_absolute: bool= False):
        assert self.stroke_end_flag - self.norm_flag == 1
        # sketch is a relative-stroke-3 format np.ndarray
        
        if not is_absolute:
            sketch = relative_to_absolute(sketch)
        
        # sketch_img = draw_sketch(sketch, "sketch.png", is_absolute=is_absolute, white_bg=True, max_dim=224)
            
        xmin, ymin, xmax, ymax = get_absolute_bounds(sketch)
        sketch[:, 0] -= xmin
        sketch[:, 1] -= ymin
        sketch = normalize_to_scale(
            sketch, is_absolute=True, scale_factor=255.0)
        
        coords = sketch[:,:2].astype(int).astype(float)
        flags  = sketch[:,2] + self.norm_flag
        sk_len = coords.shape[0]
        pads   = self.generate_padding_mask(sk_len)
        poses  = self.generate_position_embedding(sk_len)
        
        if self.seq_len is not None and self.seq_len > 0:
            if self.seq_len <= sk_len:
                coords = coords[:self.seq_len, ...]
                flags = flags[:self.seq_len]
                flags[-1] = self.stroke_end_flag
                sk_len = self.seq_len
            else:
                plen = self.seq_len - coords.shape[0]
                coords = np.concatenate(
                    [coords, np.full([plen, 2], -1)], axis=0)
                coords[sk_len, 0] = 0
                coords[sk_len, 1] = 0
                flags = np.concatenate(
                    [flags, np.full((plen,), self.pad_flag)], axis=0)
        
        attn_mask1 = self.produce_adjacent_matrix_2_neighbors(
            flags, sk_len)
        attn_mask2 = self.produce_adjacent_matrix_4_neighbors(
            flags, sk_len)
        attn_mask3 = self.produce_adjacent_matrix_joint_neighbors(
            flags, sk_len)
        
        # convert to tensors
        coords = torch.from_numpy(coords).float().to(self.device)
        flags  = torch.from_numpy(flags).long().to(self.device)
        poses  = torch.from_numpy(poses).long().to(self.device)
        attn_mask1 = torch.from_numpy(attn_mask1).float().to(self.device)
        attn_mask2 = torch.from_numpy(attn_mask2).float().to(self.device)
        attn_mask3 = torch.from_numpy(attn_mask3).float().to(self.device)
        pads = torch.from_numpy(pads).long().to(self.device)

        # add batch dimension
        coords = coords.unsqueeze(0)
        flags  = flags.unsqueeze(0)
        poses  = poses.unsqueeze(0)
        attn_mask1 = attn_mask1.unsqueeze(0)
        attn_mask2 = attn_mask2.unsqueeze(0)
        attn_mask3 = attn_mask3.unsqueeze(0)
        pads = pads.unsqueeze(0)

        return (coords, flags, poses, 
                attn_mask1, attn_mask2, attn_mask3, pads)
    
    
    def preprocess_batch(self, sketches: list):
        
        (coords, flags, poses, 
         attn_mask1, attn_mask2, attn_mask3,
         pads) = self.preprocess(sketches[0])
        
        for sketch in sketches[1:]:
            (new_coords, new_flags, new_poses, 
             new_attn_mask1, new_attn_mask2, new_attn_mask3,
             new_pads) = self.preprocess(sketch)
            
            coords = torch.cat([coords, new_coords], dim=0)
            flags = torch.cat([flags, new_flags], dim=0)
            poses = torch.cat([poses, new_poses], dim=0)
            attn_mask1 = torch.cat([attn_mask1, new_attn_mask1], dim=0)
            attn_mask2 = torch.cat([attn_mask2, new_attn_mask2], dim=0)
            attn_mask3 = torch.cat([attn_mask3, new_attn_mask3], dim=0)
            pads = torch.cat([pads, new_pads], dim=0)
        
        return (coords, flags, poses, 
                attn_mask1, attn_mask2, attn_mask3, pads)
            
    
    def generate_padding_mask(self, stroke_length: int):
        if self.seq_len is None or self.seq_len < 0:
            return np.ones([stroke_length, 1], dtype=int).tolist()
        else:
            padding_mask = np.ones([self.seq_len, 1], dtype=int)
            if stroke_length < self.seq_len:
                padding_mask[stroke_length:] = 0
            return padding_mask
        
    
    def generate_position_embedding(self, stroke_length: int=100):
        if self.seq_len is None or self.seq_len < 0:
            pos_emb = np.arange(stroke_length)
        else:
            pos_emb = np.arange(self.seq_len)
        return pos_emb
    
    
    def produce_adjacent_matrix_2_neighbors(self, flag_bits, stroke_len):
        adja_matr = np.zeros([flag_bits.shape[0], flag_bits.shape[0]], int)
        adja_matr[ : ][ : ] = -1e10
        adja_matr[0][0] = 0
        if (flag_bits[0] == 100):
            adja_matr[0][1] = 0
    
        for idx in range(1, stroke_len):
            adja_matr[idx][idx] = 0
    
            if (flag_bits[idx - 1] == 100):
                adja_matr[idx][idx - 1] = 0
    
            if idx == stroke_len - 1:
                break
    
            if (flag_bits[idx] == 100):
                adja_matr[idx][idx + 1] = 0
    
        return adja_matr
    
    
    def produce_adjacent_matrix_4_neighbors(self, flag_bits, stroke_len):
        adja_matr = np.zeros([flag_bits.shape[0], flag_bits.shape[0]], int)
        adja_matr[ : ][ : ] = -1e10
    
        adja_matr[0][0] = 0
        if (flag_bits[0] == 100):
            adja_matr[0][1] = 0
            if (flag_bits[1] == 100):
                adja_matr[0][2] = 0
    
    
        for idx in range(1, stroke_len):
            adja_matr[idx][idx] = 0
    
            if (flag_bits[idx - 1] == 100):
                adja_matr[idx][idx - 1] = 0
                if (idx >= 2) and (flag_bits[idx - 2] == 100):
                    adja_matr[idx][idx - 2] = 0
    
            if idx == stroke_len - 1:
                break

            if (idx <= (stroke_len - 2)) and (flag_bits[idx] == 100):
                adja_matr[idx][idx + 1] = 0
                if (idx <= (stroke_len - 3)) and (flag_bits[idx + 1] == 100):
                    adja_matr[idx][idx + 2] = 0
    
        return adja_matr


    def produce_adjacent_matrix_joint_neighbors(self, flag_bits, stroke_len):
        adja_matr = np.zeros([flag_bits.shape[0], flag_bits.shape[0]], int)
        adja_matr[ : ][ : ] = -1e10
        
        adja_matr[0][0] = 0
        adja_matr[0][stroke_len - 1] = 0
        adja_matr[stroke_len - 1][stroke_len - 1] = 0
        adja_matr[stroke_len - 1][0] = 0
    
        assert flag_bits[0] == 100 or flag_bits[0] == 101
    
        if (flag_bits[0] == 101) and stroke_len >= 2:
            adja_matr[0][1] = 0
    
        for idx in range(1, stroke_len):
            assert flag_bits[idx] == 100 or flag_bits[idx] == 101
            adja_matr[idx][idx] = 0
    
            if (flag_bits[idx - 1] == 101):
                adja_matr[idx][idx - 1] = 0
    
            if (idx == stroke_len - 1):
                break

            if (idx <= (stroke_len - 2)) and (flag_bits[idx] == 101):
                adja_matr[idx][idx + 1] = 0
    
        return adja_matr