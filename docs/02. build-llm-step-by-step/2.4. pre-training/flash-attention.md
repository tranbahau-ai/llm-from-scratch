
### FlashAttention

![alt](/images/flashattn.jpg)



**Cải thiện:** Giảm memory từ O(N²) xuống O(N), tăng 3-5x tốc độ.

### 1. Vì sao Attention truyền thống chậm?


Vấn đề:
**1. Ma trận T × T quá lớn**
- T = 4k → 16 triệu phần tử
- T = 16k → 256 triệu phần tử
- Mỗi phần tử FP16 → hàng trăm MB RAM

**2. Bottleneck I/O GPU**
GPU phải liên tục đọc/ghi từ:
 - HBM (bộ nhớ chính của GPU — rất chậm)
 - SRAM (bộ nhớ trên chip — rất nhanh)


**FlashAttention** giải quyết đúng điểm này.

### 2. FlashAttention hoạt động
Sử dụng 2 kỹ thuật chính:

#### 1. Tiling — Chia Q,K,V thành các khối nhỏ

Thay vì tạo ma trận T×T:
 - Load một block Q vào SRAM
 - Load block K & V tương ứng
 - Tính attention block theo block
 - Không bao giờ lưu toàn bộ **scores**


#### 2. Tính softmax từng phần
hay vì phải biết toàn bộ `scores[i, :]` trước khi softmax, FlashAttention duy trì 3 biến:
`m` = giá trị lớn nhất chạy (running max)
`s` = tổng exp chạy (running sum)
`o` = output chạy (running output)

Mỗi khi xử lý thêm 1 block K,V:

```
m_new = max(m, max(score))
s = s * exp(m - m_new) + sum(exp(score - m_new))
o = o * exp(m - m_new) + exp(score - m_new) @ V_block
m = m_new
```

Cuối cùng:

```
output = o / s
```



##### (Pseudocode)
```python
for blockQ in Q:
    load blockQ into SRAM
    m = -inf
    s = 0
    o = 0
    for (blockK, blockV):
        score = blockQ @ blockK.T
        m_new = max(m, max(score))
        s = s * exp(m - m_new) + sum(exp(score - m_new))
        o = o * exp(m - m_new) + exp(score - m_new) @ blockV
        m = m_new
    output = o / s

```

[GitHub: FlashAttention](https://github.com/Dao-AILab/flash-attention/tree/main)

### Code example use 

```python
import torch
from flash_attn import flash_attn_func

Q = torch.randn(1, 4096, 128, device="cuda", dtype=torch.float16)
K = torch.randn(1, 4096, 128, device="cuda", dtype=torch.float16)
V = torch.randn(1, 4096, 128, device="cuda", dtype=torch.float16)

out = flash_attn_func(Q, K, V, dropout_p=0.0)
print(out.shape)
```
