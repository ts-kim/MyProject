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

def transform_2():
    transform=transforms.Compose([
            transforms.ToPILImage('RGB'),
            transforms.RandomResizedCrop(size=256, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor()
        ])
    return transform
    

def transform_gray():
    transform=transforms.Compose([
        transforms.ToPILImage('RGB'),
        transforms.Grayscale(3),
        transforms.Resize(224),
        transforms.ToTensor()
    ])
    