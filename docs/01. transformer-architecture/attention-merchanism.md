## Cơ chế attention

Cơ chế attention giúp mô hình **tập trung vào các phần quan trọng** của đầu vào khi tạo đầu ra. Thay vì xử lý tất cả token như nhau, attention **học xem token nào quan trọng hơn** tại mỗi thời điểm.

**Attention** tính điểm tương đồng giữa:
- Query (Q)
- Key (K)  
    → Điểm càng cao → token đó càng quan trọng  
    Điểm này sẽ dùng để kết hợp Value (V) tạo ra **đại diện đầu ra**.

### Công thức tổng quát / Self-Attention 

$$
Attention(Q,K,V) = \text{softmax} (\frac{Q \cdot K^\top}{\sqrt{d_k}}) * V
$$

Trong đó:
 **Query (Q) :** Token đang đặt câu hỏi
 **Key (K):**    Token chứa thông tin
 **Value (V):**  Thông tin thực sự cần lấy
${\sqrt{d_k}}$: dimension của key (để scaling)

 **softmax:**	 Tính mức độ “nên tập trung vào token nào nhất”

### 🧠 Detail explaination:
**📐 1. Dot Product (Tích vô hướng)**

$$
\text{score} = Q \cdot K^\top
$$

**Example:**

$$
Q = [1, 2, 3], \quad K = [2, 1, 0]
$$
$$
Q \cdot K = 1\times2 + 2\times1 + 3\times0 = 4
$$

**Q** = ↓ hướng cần tìm (câu hỏi)  
**K** = dữ liệu có sẵn (các đáp án)
Q · K.T = độ giống nhau giữa câu hỏi và từng đáp án
→ Nếu **Q & K** hướng giống nhau → score cao (relevant)  
→ Nếu khác hướng → score thấp (not relevant)

####  Code Example (PyTorch)

```python
import torch

Q = torch.tensor([[1.0, 2.0, 3.0]])  # (1 x 3)
K = torch.tensor([[2.0, 1.0, 0.0],
                  [1.0, 2.0, 3.0]])  # (2 x 3)

# compute similarity
score = torch.matmul(Q, K.T)
print(score)  # tensor([[4., 14.]])
```

**🧠 2. Scaling (Stabilization/Tính ổn định)**
To prevent large values when vector dimension is big:

$$
\text{scores} = \frac{Q \cdot K^\top}{\sqrt{d_k}}
$$

**Lí do:**
Nếu số chiều của vector lớn → Dot product quá cao → gradient exploded → model học không ổn định.  
Chia cho $\sqrt{d_k}$ ​ để **ổn định training**.

**🧠 3.  Softmax**
$$
weights=softmax(scores)
$$

→ Softmax biến độ liên quan thành **trọng số xác suất**


**🧠 4.  Final Step** 

$\text{scores} = \frac{Q \cdot K^\top}{\sqrt{d_k}}$  ->  $\text{weights}=softmax(scores)$  ->  $\text{output}=weights * V$

→ Multiply với V để lấy **contextual embedding**


##### Code example for Simple Self-Attention (no training)


```python
import torch
import torch.nn.functional as F

# Example input (T tokens, each 3-dim embedding)
X = torch.tensor([
    [0.4, 0.1, 0.8],  # "Your"
    [0.5, 0.8, 0.6],  # "journey"
    [0.5, 0.8, 0.6],  # "starts"
])

# Compute attention scores
scores = torch.matmul(X, X.T)    # (T x T)
weights = F.softmax(scores, dim=-1)

# Compute new contextualized embeddings
Z = torch.matmul(weights, X)     # Weighted sum
print(Z)

#tensor([
# [0.4656, 0.5592, 0.6688],
# [0.4765, 0.6358, 0.6469],
# [0.4765, 0.6358, 0.6469]
#])

```


#### 🧠 **Causal Attention (dùng trong GPT)** 
**Dùng để đảm bảo chỉ nhìn token phía trước, không nhìn tương lai**  
Mô hình chỉ được phép “dự đoán từng token một” → **token by token generation**
Masking giúp che những token chưa xuất hiện

$$
\frac{Q \cdot K^\top}{\sqrt{d_k}} + \text{mask}
$$

![alt text](/images/image-01.png)


### Multi-Head Attention

Thay vì dùng 1 attention, dùng nhiều "heads" song song:

$$
 \text{MHA(Q,K,V)} = Concat(head1​,…,head n​) * Wo​
$$

![alt text](/images/image-02.png)

**Lợi ích**: Mỗi head học các mối quan hệ khác nhau (ngữ pháp, ngữ nghĩa, v.v.)
Each head learns different relationships:
 - syntax head (ngữ pháp)
 - semantic head (nghĩa)
 - positional head (vị trí)


### Tổng kết
Attention giúp mô hình **tập trung, suy luận và linh hoạt**. Đây là **core của Transformer & GPT**.