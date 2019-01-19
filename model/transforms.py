from torchvision import transforms

def transform_1():
    transform=transforms.Compose([
            transforms.ToPILImage('RGB'),
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor()
        ])
    return transform