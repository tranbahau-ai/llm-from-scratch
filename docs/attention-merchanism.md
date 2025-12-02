Cơ chế attention giúp mô hình **tập trung vào các phần quan trọng** của đầu vào khi tạo đầu ra. Thay vì xử lý tất cả token như nhau, attention **học xem token nào quan trọng hơn** tại mỗi thời điểm.

**Attention** tính điểm tương đồng giữa:
- Query (Q)
- Key (K)  
    → Điểm càng cao → token đó càng quan trọng  
    Điểm này sẽ dùng để kết hợp Value (V) tạo ra **đại diện đầu ra**.

### Công thức tổng quát / Foundation 

$$
Attention(Q,K,V) = \text{softmax} (\frac{Q \cdot K^\top}{\sqrt{d_k}}) * V
$$
Trong đó:
 **Query (Q) :** Token đang đặt câu hỏi
 **Key (K):** Token chứa thông tin
 **Value (V):** Thông tin thực sự cần lấy
 **softmax:**	Tính mức độ “nên tập trung vào token nào nhất”

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
$$
\text{score} = Q \cdot K^\top
$$
**Q** = ↓ hướng cần tìm (câu hỏi)  
**K** = dữ liệu có sẵn (các đáp án)
`Q · K.T` = **độ giống nhau giữa câu hỏi và từng đáp án**
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
**biến các raw scores** thành **xác suất** (giá trị từ 0 đến 1 và tổng = 1).
=> mô hình biết nên chú ý bao nhiêu % vào từng token.

**🧠 4.  Final Step -> nhân trọng số đó với V** 

$\text{scores} = \frac{Q \cdot K^\top}{\sqrt{d_k}}$  ->  $\text{weights}=softmax(scores)$  ->  $\text{output}=weights * V$

**nhân trọng số đó với V** — tức là **thông tin thật sự cần lấy**.

### 🧠 **Causal Attention (dùng trong GPT)**
**Dùng để đảm bảo chỉ nhìn token phía trước, không nhìn tương lai**  
Mô hình chỉ được phép “dự đoán từng token một” → **token by token generation**
Masking giúp che những token chưa xuất hiện

$$
\frac{Q \cdot K^\top}{\sqrt{d_k}} + \text{mask}
$$

### Tổng kết
Attention giúp mô hình **tập trung, suy luận và linh hoạt**. Đây là **core của Transformer & GPT**.