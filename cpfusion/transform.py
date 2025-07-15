from torchvision.transforms import Compose, Resize, ToTensor, Normalize

def transform():
    return Compose([
        # Resize((image_size, image_size)), 
        ToTensor(), 
        # Normalize((0.5,), (0.5,))
    ])