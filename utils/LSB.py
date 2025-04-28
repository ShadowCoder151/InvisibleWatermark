import cv2
from typing import List


def LSB_pipeline(image:cv2.Mat, text:str):
    """Get the binarized version of the text"""

    def get_binary(text:str):
        length = len(text)

        # Placing the 16 bit text length header
        arr = [bin(length * 8)[2:].zfill(16)]
        for ch in text:
            co = bin(ord(ch))[2:].zfill(8)
            arr.append(co)
        return ''.join(arr)
    
    """LSB Watermarking"""
    def lsb_mark(image: cv2.Mat, binary:str):
        # Get the dimensions of the image
        h, w, ch = image.shape
        image = image.copy()

        # Modify LSB
        point = 0
        length = len(binary)
        for y in range(h):
            for x in range(w):
                for z in range(3):
                    image[y, x, z] = (image[y, x, z] & 0xFE) | int(binary[point])
                    point += 1
                    if point == length:
                        return image
        
        return image
    
    text_size = 16 + len(text) * 8
    img_size = image.size

    if text_size > img_size:
        raise ValueError(f'Text size is to large. Cannot fit in image. Available bits {img_size}, needed {text_size} bits')
    bin_text = get_binary(text)
    new_img = lsb_mark(image, bin_text)
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




image = cv2.imread('./input/image1.png')

text = "I am a good boy..."


res = LSB_pipeline(image, text)
cv2.imshow('Original', image)
cv2.imshow('Image after embedding', res)

cv2.imwrite('./output/image_encode.jpg', res)
cv2.waitKey(0)
cv2.destroyAllWindows()

print('Text extracted: ', INVERSE_LSB_pipeline(res))
