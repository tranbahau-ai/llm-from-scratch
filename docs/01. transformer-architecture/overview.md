## 2.1 Tại Sao Cần Transformer?

Trước Transformer, các mô hình RNN và LSTM gặp khó khăn với:

- Xử lý tuần tự chậm (không song song hóa được)
- Khó học phụ thuộc xa trong câu dài
- Vấn đề vanishing/exploding gradient

## 2.2. Core Idea
Ý tưởng là so sánh các từ với nhau đôi một, bao gồm cả chính nó (self), để tìm ra mức độ quan trọng của mỗi từ mà mô hình nên chú ý tới (thể hiện qua trọng số). Giúp mô hình hiểu đúng ý nghĩa của từ trong ngữ cảnh cụ thể, thay vì chỉ dựa vào ý nghĩa tổng quát của từ đó khi đứng riêng lẻ.

Transformer giải quyết vấn đề này bằng cơ chế [self-attention](/docs/01.%20transformer-architecture/attention-merchanism.md)
, cho phép mô hình xử lý toàn bộ chuỗi đồng thời và học mối quan hệ giữa các từ cách xa nhau.



![alt text](/images/image-03.png)

