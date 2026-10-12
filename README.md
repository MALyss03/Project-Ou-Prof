# CNN Research Lab — CS535

Ứng dụng nghiên cứu cục bộ bằng Streamlit cho bài toán phân loại ảnh **mèo/chó**, dựa trên hai tài liệu CS535 được cung cấp. Hệ thống giữ nguyên ý tưởng quan trọng của notebook: biến đổi ảnh **no-upscale + padding**, chia dữ liệu phân tầng theo tỷ lệ **70/15/15**, và không dùng tập independent test để lựa chọn siêu tham số.

## Tính năng

- Tải archive Microsoft Cats and Dogs (~786 MB) hoặc sử dụng dataset đã có sẵn.
- Quét ảnh, bỏ qua file hỏng, loại bản sao chính xác theo SHA-256; nếu nội dung ảnh trùng nhưng nhãn Cat/Dog mâu thuẫn thì loại cả bản trùng khỏi manifest.
- Một split train/validation/test có seed cố định, dùng chung cho các run.
- Tự xây CNN với các tham số depth, filters, kernel, pooling, dropout, BatchNorm và activation.
- So sánh ResNet-18, MobileNetV3-Small, EfficientNet-B0 và ConvNeXt-Tiny với ba chiến lược: from scratch, feature extraction (đóng băng backbone) và fine-tuning.
- Lưu cấu hình, môi trường phần mềm, seed, fingerprint của split, checkpoint, learning history và validation predictions.
- Bảng so sánh thí nghiệm, learning curves, phân tích lỗi có confidence cao, và dự đoán một ảnh do người dùng tải lên.
- Đánh giá internal test cuối cùng và challenge set ngoài gồm `cat/`, `dog/`, `ood/`.

Tài liệu HTML và notebook gốc nằm trong thư mục `references/`.

## 1. Cài đặt môi trường

Khuyến nghị Python 3.10 hoặc 3.11. Phiên bản PyTorch/CUDA tùy máy tính; nếu cần GPU CUDA, hãy cài bộ PyTorch và TorchVision phù hợp từ trang cài đặt chính thức của PyTorch trước khi cài các thư viện còn lại.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Trên macOS/Linux, kích hoạt môi trường bằng `source .venv/bin/activate`.

## 2. Khởi chạy web

Tại thư mục `cnn_research_lab`, chạy:

```bash
streamlit run app.py
```

Mở URL cục bộ được in trong terminal (thường là `http://localhost:8501`).

## 3. Chuẩn bị dataset

### Cách A — Tải từ giao diện

Mở **Dataset & QC** và nhấn **Tải dataset Microsoft**. Archive khoảng 786 MB; hãy bảo đảm còn vài GB dung lượng đĩa cho archive và ảnh đã giải nén.

Nguồn chính thức: [Microsoft Cats and Dogs](https://www.microsoft.com/en-us/download/details.aspx?id=54765).

### Cách B — Dùng dữ liệu đã tải

Cấu trúc được hỗ trợ:

```text
data/raw/PetImages/
  Cat/*.jpg
  Dog/*.jpg
```

Hoặc chỉ định trực tiếp thư mục có hai thư mục `Cat/` và `Dog/` ở bên trong trang **Dataset & QC**.

Sau khi quét, hệ thống tạo:

```text
data/processed/manifest.csv
data/processed/split_manifest.csv
data/processed/dataset_report.json
```

Tập dữ liệu được chia phân tầng theo nhãn: 70% train, 15% validation, 15% internal test, seed mặc định 42. Các run tiếp tục dùng cùng split đã lưu. Chỉ quét lại khi thực sự muốn tạo một phiên bản split mới.

## 4. Huấn luyện và so sánh

Để kiểm tra pipeline, nên bắt đầu với **Configurable CNN (from scratch)**, kích thước ảnh 128, batch size 32 và 1–3 epoch. Với kiến trúc TorchVision, có thể chọn train from scratch, feature extraction (chỉ cập nhật classifier, giữ BatchNorm của backbone ở eval mode) hoặc fine-tuning toàn bộ mạng. Cấu hình mặc định giới hạn train ở 5.000 ảnh để chạy thử nhanh; có thể chọn **Toàn bộ train split** cho thí nghiệm đầy đủ.

Mỗi run được lưu trong `experiments/<run_id>/`, gồm `config.json`, `checkpoint.pt`, `history.csv`, `validation_metrics.json` và `validation_predictions.csv`. Hãy chỉ thay đổi một biến chính trong mỗi thí nghiệm (ví dụ kernel size 3, 5, 7), đồng thời giữ nguyên các điều kiện còn lại.

## 5. Bảo vệ independent test khỏi rò rỉ dữ liệu

Trong quá trình huấn luyện, hệ thống **không đánh giá internal test**. Hãy chọn mô hình/siêu tham số bằng validation trước. Khi đã chốt mô hình, mở **Đánh giá cuối & Challenge** để chạy test. Sau lần đánh giá cuối đầu tiên, ứng dụng khóa run cuối cho split hiện tại nhằm tránh so sánh test metrics giữa nhiều mô hình.

Khóa được lưu tại `data/processed/final_test_lock.json`. Nếu cần thực hiện một nghiên cứu mới với split mới, hãy lưu trữ báo cáo cũ, tạo phiên bản dataset/split mới rõ ràng và ghi lại quyết định này trong báo cáo.

Challenge set tùy chọn:

```text
challenge_set/
  cat/   # ảnh mèo khó, có nhãn
  dog/   # ảnh chó khó, có nhãn
  ood/   # ví dụ cáo, sói, thú nhồi bông; không có nhãn mèo/chó
```

Ảnh `cat/` và `dog/` được tính metric phân loại nhị phân. Với OOD chỉ báo cáo tóm tắt độ tự tin; bộ phân loại hai lớp vẫn luôn ép ảnh vào Cat hoặc Dog, vì vậy confidence không phải bằng chứng rằng ảnh thuộc phân phối huấn luyện.

Oxford-IIIT Pet chưa được tự động tải/chuyển nhãn bởi MVP này. Trước khi dùng làm external test, cần chuẩn hóa vào `cat/` và `dog/`, thống nhất quy tắc gán nhãn và kiểm tra ảnh trùng/gần trùng với tập chính.

## 6. Lưu ý tái lập

- Mỗi run lưu seed, thiết bị, số tham số, thời gian chạy, phiên bản Python/PyTorch/TorchVision/NumPy/Pandas và SHA-256 của split manifest.
- Seed cố định giúp tái lập nhưng không bảo đảm kết quả giống tuyệt đối giữa mọi phần cứng/phiên bản thư viện.
- Lần đầu sử dụng pretrained weights có thể cần kết nối Internet.
- Tiền xử lý theo notebook: chuyển RGB, giảm kích thước khi cần nhưng không phóng lớn ảnh nhỏ, padding thành ảnh vuông, chuẩn hóa ImageNet; augmentation chỉ áp dụng trên train.
- Không đưa ra kết luận về độ chính xác hay mức cải thiện cho đến khi các thí nghiệm thực sự chạy xong.

## Khắc phục sự cố

- **Không tìm thấy Cat/Dog:** chọn đúng thư mục có `Cat/` và `Dog/`, hoặc thư mục cha chứa `PetImages/`.
- **Thiếu bộ nhớ:** giảm image resolution/batch size hoặc dùng model nhỏ hơn.
- **CPU chạy chậm:** thử 1.000–2.500 ảnh và một epoch trước; dùng môi trường có GPU CUDA cho thí nghiệm đầy đủ.
- **Tải pretrained thất bại:** chạy mô hình from-scratch trước và kiểm tra kết nối Internet.
- **Challenge labels không được nhận:** tên thư mục phải là `cat`, `dog` và tùy chọn `ood` (không phân biệt chữ hoa/thường).
