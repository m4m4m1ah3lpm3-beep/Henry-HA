# PROMPT BÀN GIAO — DÁN NGUYÊN VĂN VÀO AI (ChatGPT / Claude / Gemini)

> Cách dùng: mở một cuộc trò chuyện mới, dán toàn bộ phần trong khung dưới đây
> vào tin nhắn đầu tiên, rồi hỏi tiếp bằng tiếng Việt bình thường.

---

Tôi đang làm đồ án môn ADY201m (FPT HCMC) về phát hiện và phân đoạn tổn thương
trên ảnh siêu âm. Nhóm trưởng đã xử lý xong dữ liệu và giao lại cho tôi phần
huấn luyện mô hình. Dưới đây là toàn bộ bối cảnh. Hãy đọc kỹ và ghi nhớ, sau đó
giúp tôi theo đúng những gì đã được thiết lập — ĐỪNG đề xuất chia lại dữ liệu
hay xử lý lại ảnh, vì phần đó đã chốt và cả nhóm dùng chung.

## 1. Dữ liệu

Hai bộ ảnh siêu âm, đã được chuẩn hóa và chia sẵn, tôi chỉ việc tải về và dùng:

**TN3K** (nhân tuyến giáp, ảnh xám, mọi ảnh đều có tổn thương):
- train 2519 / val 360 / test 614 (tổng 3493)
- Tập test là tập test CHÍNH THỨC của tác giả, giữ nguyên để so sánh được với
  các bài báo. Val tách từ trainval theo tỉ lệ 7:1.
- Không có ảnh trùng lặp. Không có ảnh bình thường (normal).

**BUSI** (vú, có 3 lớp benign / malignant / normal):
- train 544 / val 117 / test 117 (tổng 778)
- Chia 70/15/15 phân tầng theo lớp, seed 42.
- Trong tập test có 20 ảnh normal (không tổn thương) — dùng để đo báo động giả.
- Đã loại 2 ảnh trùng lặp có nhãn mâu thuẫn (benign_0433 và malignant_0145).

## 2. Cấu trúc thư mục sau khi giải nén

```
TN3K_split/            (và BUSI_split/ y hệt)
├─ images/train/  val/  test/     ảnh gốc
├─ labels/train/  val/  test/     nhãn YOLO (.txt), 1 lớp duy nhất id=0
├─ masks/ train/  val/  test/     mask nhị phân (0 và 255), dùng cho phân đoạn
├─ data.yaml                      cấu hình cho YOLO
└─ SPLIT_INFO.txt
```

Quy ước quan trọng:
- Ảnh, mask và nhãn của cùng một ảnh có CÙNG TÊN FILE (khác đuôi).
  Ví dụ: images/train/benign_0001.png ↔ masks/train/benign_0001.png
  ↔ labels/train/benign_0001.txt
- Ảnh normal của BUSI có mask đen hoàn toàn và file nhãn RỖNG. YOLO coi đây là
  ảnh nền (background), điều này là cố ý, không phải lỗi.
- YOLO chỉ có 1 lớp tên là `lesion` (id = 0). Không phân biệt benign/malignant.
- Trong data.yaml, dòng `path:` là đường dẫn tuyệt đối trên máy của nhóm trưởng.
  Tôi cần sửa lại cho khớp máy mình.

## 3. Mô hình cần huấn luyện

1. **YOLOv8** — phát hiện, cho ra bounding box quanh tổn thương.
2. **U-Net** — phân đoạn, tô từng pixel.
3. **Swin-UNet** — phân đoạn, kiến trúc transformer.
4. **SAM** — KHÔNG huấn luyện, chỉ chạy suy luận trên tập test, dùng box do
   YOLO dự đoán làm prompt.

Mỗi bộ dữ liệu huấn luyện riêng (không gộp TN3K và BUSI lại với nhau).

## 4. Câu hỏi nghiên cứu (quyết định tôi cần lưu gì)

- RQ1: pipeline YOLO+SAM so với U-Net và Swin-UNet, cái nào phân đoạn tốt hơn.
- RQ2: mô hình có báo động giả trên ảnh không tổn thương không (20 ảnh normal
  trong tập test BUSI).
- RQ3: train trên bộ này, test trên bộ kia thì kết quả tụt bao nhiêu.
- RQ4: giảm dữ liệu huấn luyện xuống 10%, 25%, 50%, 100% thì điểm số đổi thế nào.
  Chỉ lấy mẫu con từ TẬP TRAIN; val và test giữ nguyên.
- Bước cuối: hồi quy dự đoán điểm Dice từ đặc trưng ảnh (contrast,
  edge_sharpness, lesion_ratio) để giải thích vì sao mô hình sai ở ảnh nào.

## 5. Yêu cầu bắt buộc khi lưu kết quả

Vì RQ4 và phần hồi quy cần dữ liệu chi tiết, tôi phải lưu điểm Dice và IoU CHO
TỪNG ẢNH, không chỉ số trung bình. File CSV có các cột:

```
image_id, dataset, model, split, dice, iou, n_pred_boxes, n_true_lesions
```

`image_id` là tên file không có đuôi (ví dụ `benign_0001`), để sau này nối được
với bảng metadata của nhóm.

Ngoài ra: dùng seed 42, tập test chỉ chạy MỘT LẦN ở cuối, val dùng để chọn
checkpoint tốt nhất.

## 6. Việc tôi cần bạn giúp

Tôi chưa quen huấn luyện mô hình. Hãy hướng dẫn tôi từng bước một, giải thích
đơn giản, đưa code chạy được trên Google Colab (có GPU). Khi tôi gặp lỗi, hãy
giải thích lỗi đó nghĩa là gì trước khi sửa.

Bắt đầu bằng việc hỏi tôi đang ở bước nào và đang dùng bộ dữ liệu nào.

---

## GHI CHÚ RIÊNG CHO NGƯỜI NHẬN BÀN GIAO (không cần dán vào AI)

Những điều KHÔNG được tự ý làm, vì sẽ khiến kết quả của cả nhóm không so sánh
được với nhau:

1. Không chia lại train/val/test. Không xáo trộn, không đổi tỉ lệ.
2. Không chạy lại script xử lý ảnh (prep) hay script chia (make_split).
3. Không xóa hay thêm ảnh vào thư mục đã giải nén.
4. Không đổi tên file ảnh.
5. Không gộp hai bộ dữ liệu lại để train chung.
6. Không dùng tập test để chọn mô hình — chỉ dùng val.

Nếu thấy có gì bất thường trong dữ liệu, báo nhóm trưởng thay vì tự sửa.
