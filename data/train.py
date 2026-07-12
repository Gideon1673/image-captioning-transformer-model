import torch
from PIL import Image

from config import IMAGE_DIR
from data.dataset import ImageTransform

train_transform = ImageTransform(
    image_size=224,
    resize_size=256,
    train=True,
    horizontal_flip_probability=0.5,
)

validation_transform = ImageTransform(
    image_size=224,
    resize_size=256,
    train=False,
)

test_transform = ImageTransform(
    image_size=224,
    resize_size=256,
    train=False,
)


def main():
    transform = ImageTransform(
        image_size=224,
        resize_size=256,
        train=False,
    )

    image_names = [
        "1000268201_693b08cb0e.jpg",
        "1001773457_577c3a7d70.jpg",
    ]

    image_tensors = []

    for image_name in image_names:
        image = Image.open(IMAGE_DIR / image_name)
        image_tensor = transform(image)
        image_tensors.append(image_tensor)

    images = torch.stack(
        image_tensors,
        dim=0,
    )

    print(images.shape)


if __name__ == "__main__":
    main()
