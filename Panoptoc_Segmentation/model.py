from transformers import Mask2FormerForUniversalSegmentation, AutoImageProcessor

def load_model(task='semantic'):
    if task == 'instance':
        image_processor = AutoImageProcessor.from_pretrained(
            'facebook/mask2former-swin-large-coco-instance'
        )
        model = Mask2FormerForUniversalSegmentation.from_pretrained(
            'facebook/mask2former-swin-large-coco-instance'
        )
    
    if task == 'semantic':
        image_processor = AutoImageProcessor.from_pretrained(
            'facebook/mask2former-swin-large-ade-semantic'
        )
        model = Mask2FormerForUniversalSegmentation.from_pretrained(
            'facebook/mask2former-swin-large-ade-semantic'
        )

    if task == 'panoptic':
        image_processor = AutoImageProcessor.from_pretrained(
            'facebook/mask2former-swin-large-coco-panoptic'
        )
        model = Mask2FormerForUniversalSegmentation.from_pretrained(
            'facebook/mask2former-swin-large-coco-panoptic'
        )

    return model, image_processor