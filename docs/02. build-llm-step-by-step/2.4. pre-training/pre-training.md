### Pre-Training
Giai đoạn này mô hình học được:
- Ngữ pháp
- Ngữ nghĩa
- Cấu trúc câu
- Kiến thức thế giới
- Lối viết, giọng văn
- Mẫu suy luận (reasoning patterns)
- Quan hệ dài hạn trong văn bản (long-range dependency)

Toàn bộ kiến thức này tự xuất hiện nhờ scale lớn của dữ liệu + mô hình.



### Dữ liệu dùng trong Pre-training
Một mô hình LLM mạnh không thể được huấn luyện từ dữ liệu nhỏ.
Nó cần từ trăm tỷ đến trillions tokens.

 - Nguồn dữ liệu phổ biến:
 - Web Crawl (CommonCrawl)
 - Wikipedia
 - Sách (BookCorpus)
 - Báo khoa học (arXiv)
 - Code (GitHub, StackOverflow)
 - Tài liệu kỹ thuật
 - Tập dữ liệu tổng hợp: The Pile, RedPajama
 - Multilingual corpora
 - Dialogue corpus

Vì sao cần cực lớn?
→ Ngôn ngữ rất đa dạng.
→ Muốn generalize tốt, mô hình phải “thấy đủ nhiều tình huống”.


### Mục tiêu tối ưu (Loss Function)

Giảm loss = mô hình hiểu ngôn ngữ tốt hơn
Cross Entropy Loss Function (Next-token Prediction)

Quy trình loop
Lặp cho đến khi:
 - đạt max step
 - hết compute / đạt loss target.

**Dạng 1: Công thức theo phân phối xác suất**


$\mathcal{L} = - \sum_{t} \log P_\theta(x_t \mid x_{<t})$

**Dạng 2: Công thức theo one-hot vector**

$\mathcal{L} = - \sum_{t} \sum_{i} y_{t,i} \cdot \log(p_{t,i})$

Trong đó:
- $x_t$: token thật tại vị trí t  
- $P_\theta(x_t \mid x_{<t})$: xác suất mô hình dự đoán token đúng  
- $y_{t,i}$: phần tử one-hot cho token đúng  
- $p_{t,i}$: xác suất mô hình dự đoán cho token i  


### Kiến trúc mô hình trong Pre-training

Các LLM hiện đại (GPT-3, GPT-4, LLaMA, GPT-OSS) dùng:

- Decoder-only Transformer
- [Multi-head Self-Attention](/docs/02.%20build-llm-step-by-step/2.4.%20pre-training/multi-head.md)
- Feed-Forward Networks (FFN)
- Residual Connections
- LayerNorm
- Positional Encoding (RoPE hoặc Sinusoidal)

Những kỹ thuật quan trọng:
- Rotary Embedding (RoPE): giúp mở rộng context tốt hơn
- [FlashAttention](/docs/02.%20build-llm-step-by-step/2.4.%20pre-training/flash-attention.md) / FlexAttention: tăng tốc
- RMSNorm thay LayerNorm (nhiều model mới dùng)
- Tensor Parallel + Pipeline Parallel
- Mixed Precision (FP16/BF16)