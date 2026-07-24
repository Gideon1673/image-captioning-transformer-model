# Giải thích code by ThangHQ

Flow từ đầu đến cuối của chương trình này như sau

## Data preparation

Chỉ chạy 1 lần duy nhất hàm `prepare_training_data`, hàm này làm 3 việc

1. Gọi `prepare_dataset`: Tạo file `.json` lưu mapping giữa image_name và các captions của nó
2. Chia dataset thành 3 tập: Train + Test + Val dùng hàm `split_dataset`
3. Resize lại tất cả ảnh tập Train dùng hàm `resize_image`: Kích thước ảnh config được trong code. Hiện tại resize ảnh thành vuông 224\*224. Thực hiện resize bằng cách centre crop (giữ lại phần trung tâm kích thước 224*224, phần còn lại cắt hết)

Sample output bước 1:

```
Tổng số ảnh: 3
Tổng số caption: 15
Caption ít nhất mỗi ảnh: 5
Caption nhiều nhất mỗi ảnh: 5
Caption trung bình mỗi ảnh: 5.00
Project root: /home/gideon/image-captioning-transformer-model
Image directory: /home/gideon/image-captioning-transformer-model/datasets/raw/Images
Caption file: /home/gideon/image-captioning-transformer-model/datasets/raw/captions.txt
Output file: /home/gideon/image-captioning-transformer-model/datasets/processed/image_to_captions.json
Đã lưu mapping tại: /home/gideon/image-captioning-transformer-model/datasets/processed/image_to_captions.json
```

Sample output bước 2:

```
===== DATASET SPLIT STATISTICS =====
Tổng số ảnh: 3
Tổng số caption: 15

Train: 2 ảnh (66.67%), 10 caption
Validation: 0 ảnh (0.00%), 0 caption
Test: 1 ảnh (33.33%), 5 caption

Train file: /home/gideon/image-captioning-transformer-model/datasets/processed/train.json
Validation file: /home/gideon/image-captioning-transformer-model/datasets/processed/validation.json
Test file: /home/gideon/image-captioning-transformer-model/datasets/processed/test.json
```

Sample output bước 3:

```
------------------------------------------------------------
[1/2] Thành công: 47871819_db55ac4699.jpg
[2/2] Thành công: 42637987_866635edf6.jpg

===== KẾT QUẢ =====
Tổng số ảnh: 2
Resize thành công: 2
Resize thất bại: 0
Đã cập nhật JSON: /home/gideon/image-captioning-transformer-model/datasets/processed/train.json
```

QUESTION

1. Tại sao lại chỉ resize ảnh train? Nếu vậy ảnh khi inference đưa vào kích thước khác thì sao model đoán được tốt? Đây là vô tình hay cố ý?
2. Việc center crop sẽ gây mất dữ liệu ở góc. Đây là vô tình hay cố ý, hay là sai sót nhưng buộc phải chấp nhận?


Sample output đoạn PyTorch detect CUDA

```
Device: cuda:0
GPU: NVIDIA GeForce 930MX
```

## Transpose, chuyển về phân phối chuẩn

IMAGE_TRANSFORM thực hiện 2 bước sau

1. Convert về CHW
2. Normalize ảnh: Bản thân `ToTensor()` đã chuyển [0, 255] về [0, 1] rồi. Sau đó normalize để scale các con số không bị lệch nhiều --> Giúp training stable hơn.
3. Vector chứa `std` và `mean` đã được tính sẵn dựa trên toàn bộ ảnh trong dataset để đưa về phân phối chuẩn.

QUESTION

1. Tìm hiểu hàm class `ImageCaptionDataset` được dùng ở đâu?
2. `HWC` và `CHW` là gì? Trong code có hàm `ToTensor()` để chuyển từ HWC --> CHW, tại sao phải làm như thế?
3. `transforms.Normalize(mean=[...], std=[...])` hàm này trong code đang hard-code `mean` và `std` value. Tại sao chọn bộ mean và std này?

HWC and CHW refer to the physical order in which multidimensional image data (tensors) is stored in a computer's linear memory.

- **HWC (Height, Width, Channels)**: Also known as *channel-last* or interleaved format.
- **CHW (Channels, Height, Width)**: Also known as *channel-first* or planar format.

Images are essentially 3D matrices containing a height, a width, and color channels (like Red, Green, and Blue). However, RAM and GPU memory are strictly 1-dimensional. To store a 3D matrix, the system must "flatten" it into a 1D sequence of bytes. The data format dictates the logic of how this flattening occurs.

- **HWC (Interleaved)**: The memory stores all channel values for a single pixel before moving to the next pixel. If you read the linear memory sequentially, you see: R1, G1, B1, then R2, G2, B2, and so on.
- **CHW (Planar)**: The memory stores an entire channel (a 2D plane of pixels) before moving to the next channel. Reading sequentially, you see all Red values (R1, R2, R3...), followed by all Green values (G1, G2, G3...), and finally all Blue values.

[Tensor Physical Layouts on Memory](https://leimao.github.io/blog/Tensor-Physical-Layout-on-Memory/)

*(Note: In deep learning, you will often see this written as NHWC and NCHW, where N stands for the Batch size—the number of images processed at once).*

**Tại sao lại chọn HWC/CHW?** Phụ thuộc vào target hardware là CPU/GPU mà sử dụng, để tận dụng processing speed, cache locality, and hardware utilization during computation tối ưu theo từng loại hardware.

1. HWC dùng cho CPUs --> Libs hay dùng loại này là OpenCV, Matplotlib, PIL/Pillow, and TensorFlow/Keras (which initially optimized for legacy architectures before GPUs dominated).
2. CHW dùng cho GPUs --> PyTorch, ONNX, and NVIDIA's cuDNN library (built specifically from the ground up for GPU acceleration).

Ở trong PyTorch thì nó cung cấp sẵn hàm để convert HWC --> CHW rồi, ví dụ

```python
from torchvision import transforms
from PIL import Image

image_hwc = Image.open("dataset/sample.jpg")

# Automatically converts PIL Image (HWC) to a PyTorch Tensor (CHW)
transform = transforms.ToTensor()
tensor_chw = transform(image_hwc)
```

Do mình đang dùng thư viện Pillow (PIL) để load ảnh, nên phải dùng `transforms.ToTensor()` để convert về CHW.

## Patches

Project này sử dụng kiến trúc Vision Transform (ViT), thực chất xuất phát từ kiến trúc Transformer ban đầu được dùng cho text.

Đặc điểm của text là của thể tokenize by word, ví dụ "I", "Have", "a", "pen", etc. 

Nhưng với ảnh nếu một bức ảnh 224\*224 mà coi mỗi pixel == token thì KHÔNG thể đủ infrastructure để tính toán được --> Họ chia bức ảnh thành nhiều mảnh gọi là *patches*, với kích thước (trong project này) là 16*16. Lúc này patch == token trong kiến trúc ViT.

Config của patch size trong `Config.py`