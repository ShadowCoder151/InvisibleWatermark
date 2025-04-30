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
        arr = [bin(length)[2:].zfill(16)]
        for ch in text:
            co = bin(ord(ch))[2:].zfill(8)
            arr.append(co)
        return ''.join(arr)
    
    """LSB Watermarking"""
    def lsb_mark(image: cv2.Mat, binary:str, key:int):
        random.seed(key)

        h, w, ch = image.shape
        total_bits = h * w * ch
        image_marked = image.copy()

        header_bits = binary[:16]
        idx = 0
        for y in range(h):
            for x in range(w):
                for c in range(ch):
                    if idx < 16:
                        image_marked[y, x, c] = (image_marked[y, x, c] & 0xFE) | int(header_bits[idx])
                        idx += 1
                    else:
                        break
                if idx >= 16:
                    break
            if idx >= 16:
                break

        payload_bits = binary[16:]
        if len(payload_bits) > total_bits - 16:
            raise ValueError("Not enough capacity for payload bits")

        positions = random.sample(range(16, total_bits), len(payload_bits))
        idx3 = lambda val: ((val // ch) // w, (val // ch) % w, val % ch)

        for i, bit in enumerate(payload_bits):
            y, x, c = idx3(positions[i])
            image_marked[y, x, c] = (image_marked[y, x, c] & 0xFE) | int(bit)

        return image_marked
    
    text_size = 16 + len(text) * 8
    img_size = image.size

    if text_size > img_size:
        raise ValueError(f'Text size is to large. Cannot fit in image. Available bits {img_size}, needed {text_size} bits')
    
    key = generate_secret_key()
    bin_text = get_binary(text)
    new_img = lsb_mark(image, bin_text, key)
    return new_img, key


def INVERSE_LSB_pipeline(image:cv2.Mat, key:int):
    # Extract the bit stream
    def extract_stream(image:cv2.Mat, key:int):
        random.seed(key)
        h, w, ch = image.shape
        total_bits = h * w * ch

        # Read the first 16 bits sequentially to get length
        header = []
        idx = 0
        for y in range(h):
            for x in range(w):
                for c in range(ch):
                    if idx < 16:
                        header.append(str(image[y, x, c] & 1))
                        idx += 1
                    else:
                        break
                if idx >= 16:
                    break
            if idx >= 16:
                break

        payload_length = int(''.join(header), 2) * 8
        # Sample same random positions for payload
        positions = random.sample(range(16, total_bits), payload_length)
        idx3 = lambda val: ((val // ch) // w, (val // ch) % w, val % ch)

        bits = []
        for pos in positions:
            y, x, c = idx3(pos)
            bits.append(str(image[y, x, c] & 1))

        return bits

    # Process the bit stream and extract the text
    def process_stream(stream: List[str]):
        string = []
        for i in range(0, len(stream), 8):
            byte = ''.join(stream[i:i+8])

            order = int(byte, 2)
            string.append(chr(order))
        
        return ''.join(string)
    

    stream = extract_stream(image, key)
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


res , key= LSB_pipeline(image, text)
cv2.imshow('Original', image)
cv2.imshow('Image after embedding', res)

print(stats(image, res))

# cv2.imwrite('./output/image_encode.jpg', res)
cv2.waitKey(0)
cv2.destroyAllWindows()

print('Text extracted: ', INVERSE_LSB_pipeline(res, key))

