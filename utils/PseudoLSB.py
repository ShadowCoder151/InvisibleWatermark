import cv2
from typing import List
import numpy as np
import random




def LSB_pipeline(image:cv2.Mat, text:str):
    """Get the secret key generated"""
    def generate_secret_key():
        return random.randint(1000000, 9999999)

    def get_binary(text:str):
        length = len(text)
        arr = []
        for ch in text:
            co = bin(ord(ch))[2:].zfill(8)
            arr.append(co)
        return ''.join(arr)
    
    """LSB Watermarking"""
    def lsb_mark(image: cv2.Mat, binary:str, key:int):
        random.seed(key)

        # Get the dimensions of the image
        h, w, ch = image.shape
        image = image.copy()

        pos = random.sample(range(h * w * ch), len(binary))

        idx3 = lambda val: ((val // ch) // w, (val // ch) % w, val % ch)

        # Modify LSB
        point = 0
        length = len(binary)
        for idx, point in enumerate(pos):
            y, x, z = idx3(point)
            image[y, x, z] = (image[y, x, z] & 0xFE) | int(binary[idx])
        
        return image
    
    text_size = len(text) * 8
    img_size = image.size

    if text_size > img_size:
        raise ValueError(f'Text size is to large. Cannot fit in image. Available bits {img_size}, needed {text_size} bits')
    
    key = generate_secret_key()
    bin_text = get_binary(text)
    new_img = lsb_mark(image, bin_text, key)
    return new_img


def INVERSE_LSB_pipeline(image:cv2.Mat):
    # Extract the bit stream
    def extract_stream(image:cv2.Mat):
        arr = []
        bit_length, point = 0, -1

        flag = False
        h, w, c = image.shape
        for y in range(h):
            for x in range(w):
                pixel  = image[y, x]
                for z in range(3):
                    arr.append(str(pixel[z] & 1))
                    if len(arr) == 16 and point == -1:
                        bit_length = int(''.join(arr), 2)
                        arr = []
                        flag = True
                    point += flag
                    if point == bit_length:
                        return arr

    # Process the bit stream and extract the text
    def process_stream(stream: List[int]):
        string = []
        for i in range(0, len(stream), 8):
            byte = ''.join(stream[i:i+8])

            order = int(byte, 2)
            string.append(chr(order))
        
        return ''.join(string)
    

    stream = extract_stream(image)
    final_text = process_stream(stream)

    return final_text


# Get the image difference (convincing purposes to user that image is embedded)
def stats(og:cv2.Mat, enc:cv2.Mat):
    def mse(img_a: cv2.Mat, img_b: cv2.Mat):
        err = np.sum((img_a - img_b) ** 2)
        err /= float(img_a.shape[0] * img_b.shape[1])
        return err
    

    mse_val = mse(og, enc)
    if mse_val == 0:
        return 100
    
    return 20 * np.log10(255.0/np.sqrt(mse_val))


image = cv2.imread('./input/image1.png')

text = "I am a good boy..."


res = LSB_pipeline(image, text)
cv2.imshow('Original', image)
cv2.imshow('Image after embedding', res)

# print(stats(image, res))

# cv2.imwrite('./output/image_encode.jpg', res)
cv2.waitKey(0)
cv2.destroyAllWindows()

# print('Text extracted: ', INVERSE_LSB_pipeline(res))

