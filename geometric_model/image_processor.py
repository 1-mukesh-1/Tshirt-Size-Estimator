import cv2
from config import IMAGE_PROCESSING as IP

class ImageProcessor:
    @staticmethod
    def enhance_image(image):
        l, a, b = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2LAB))
        
        clahe = cv2.createCLAHE(
            clipLimit=IP['enhancement']['clahe']['clip_limit'],
            tileGridSize=IP['enhancement']['clahe']['tile_grid_size']
        )
        
        enhanced = cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)
        
        return cv2.fastNlMeansDenoisingColored(
            enhanced, None,
            h=IP['enhancement']['denoising']['h_luminance'],
            hColor=IP['enhancement']['denoising']['photo_render'],
            templateWindowSize=IP['enhancement']['denoising']['search_window'],
            searchWindowSize=IP['enhancement']['denoising']['block_size']
        )