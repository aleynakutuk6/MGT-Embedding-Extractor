from utils.sketch import *
import copy
import numpy as np
import cv2


def draw_sketch(
    sketch: np.ndarray, save_path: str=None, margin: int=10, keep_size: bool=False,
    is_absolute: bool=False, max_dim: int=512, white_bg: bool=False):
    
    assert max_dim > 2*margin # pads 50 to each side
    
    if white_bg:
        canvas = np.full((max_dim, max_dim, 1), 255, dtype=np.uint8)
        fill_color = (0, 0, 0)
    else:
        canvas = np.zeros((max_dim, max_dim, 1), dtype=np.uint8)
        fill_color = (255, 255, 255)
    
    
    if not is_absolute:
        abs_sketch = relative_to_absolute(copy.deepcopy(sketch))
    else:
        abs_sketch = copy.deepcopy(sketch)
    
    if not keep_size:
        xmin, ymin, xmax, ymax = get_absolute_bounds(abs_sketch)
        abs_sketch[:,0] -= xmin
        abs_sketch[:,1] -= ymin
        abs_sketch = normalize_to_scale(
            abs_sketch, is_absolute=True, scale_factor=max_dim-2*margin)
        if abs_sketch is None:
            return None
        abs_sketch[:,:2] += margin # pads margin px to top and left sides
    
    for i in range(1, abs_sketch.shape[0]):
        if abs_sketch[i-1, -1] > 0.5: continue # stroke end
        px, py = int(abs_sketch[i-1, 0]), int(abs_sketch[i-1, 1])
        x, y   = int(abs_sketch[i, 0]), int(abs_sketch[i, 1])
        canvas = cv2.line(canvas, (px, py), (x, y), color=fill_color, thickness=2)
    
    if save_path is not None:
        cv2.imwrite(save_path, canvas)
    
    return canvas
