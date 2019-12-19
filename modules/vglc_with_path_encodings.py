import numpy as np
import json
from PIL import Image

tiles_path = '../data/tiles'

char2int_smb = { 
    "X": 0,
    "S": 1,
    "-": 2,
    "?": 3,
    "Q": 4,
    "E": 5,
    "<": 6,
    ">": 7,
    "[": 8,
    "]": 9,
    "o": 10,
    "P": 11,  # for smb path (from Anurag's email)
}

char2int_ki = {
    "T": 0,
    "M": 1,
    "D": 2,
    "#": 3,
    "H": 4,
    "*": 5,  # for ki background, different from vglc (from Anurag's email)
    "P": 6,  # for ki path (from Anurag's email)
}

int2char_smb = {v:k for k, v in char2int_smb.items()}
int2char_ki = {v:k for k, v in char2int_ki.items()}

chars2pngs_smb = {    
    "-": Image.open(f'{tiles_path}/smb-background.png'),
    "X": Image.open(f'{tiles_path}/smb-unpassable.png'),
    "S": Image.open(f'{tiles_path}/smb-breakable.png'),
    "?": Image.open(f'{tiles_path}/smb-question.png'),
    "Q": Image.open(f'{tiles_path}/smb-question.png'),
    "o": Image.open(f'{tiles_path}/smb-coin.png'),
    "E": Image.open(f'{tiles_path}/smb-enemy.png'),
    "<": Image.open(f'{tiles_path}/smb-tube-top-left.png'),
    ">": Image.open(f'{tiles_path}/smb-tube-top-right.png'),
    "[": Image.open(f'{tiles_path}/smb-tube-lower-left.png'),
    "]": Image.open(f'{tiles_path}/smb-tube-lower-right.png'),
    "P": Image.open(f'{tiles_path}/smb-path.png')  # self-created
}

chars2pngs_ki = {
    "#": Image.open(f'{tiles_path}/ki-unpassable.png'),
    "T": Image.open(f'{tiles_path}/ki-passable.png'),
    "M": Image.open(f'{tiles_path}/ki-moving-platform.png'),
    "H": Image.open(f'{tiles_path}/ki-hazard.png'),
    "*": Image.open(f'{tiles_path}/ki-background.png'),
    "D": Image.open(f'{tiles_path}/ki-door.png'),
    "P": Image.open(f'{tiles_path}/ki-path.png')  # self-created
}

class Encoding():
    char2int_smb = char2int_smb
    char2int_ki = char2int_ki
    int2char_smb = int2char_smb
    int2char_ki = int2char_ki
    chars2pngs_smb = chars2pngs_smb
    chars2pngs_ki = chars2pngs_ki

def array_from_json(path:str):
    with open(path) as f:
        chunks = np.array(json.load(f))
    print(f'{len(chunks)} chunks loaded from {path}.')
    return chunks

def array_to_image(array, game):
    """
    Convert a 16-by-16 array of integers into a PIL.Image object
    param: array: a 16-by-16 array of integers
    """
    images = []
    for i in array:
        image = Image.new('RGB',(16 * 16, 16 * 16))
        for row, seg in enumerate(i):
            for col, tile in enumerate(seg):
                if game == 'smb':
                    image.paste(chars2pngs_smb[int2char_smb[tile]], (col * 16, row * 16))
                elif game == 'ki':
                    image.paste(chars2pngs_ki[int2char_ki[tile]], (col * 16, row * 16))
        images.append(image)
    return images

