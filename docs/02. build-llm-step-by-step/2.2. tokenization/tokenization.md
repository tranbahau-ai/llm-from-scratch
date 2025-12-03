
## Tokenization

### 1. Why Tokenization?

Ngôn ngữ tự nhiên có:
 - Quá nhiều từ (trên 1 triệu dạng từ)
 - Nhiều từ hiếm
 - Các biến thể (run, running, runs, runner)
 - Từ dài khó xử lý (internationalization)


Nếu:
 - Vocab quá lớn → embedding lớn
 - Vocab quá nhỏ → câu bị tách thành quá nhiều token → sequence dài



### 2. Tokenization works
![alt text](/images/image-04.png)


Thuật toán
- [Byte-Pair Encoding (Most Popular)](/docs/02.%20build-llm-step-by-step/02.1.%20prepare-dataset/byte-pair-encoding.md)

- WordPiece
- Unigram