### Byte-Pair Encoding BPE

#### Cách BPE được “học” từ corpus
##### 1. Find the base vocabulary
 - Sau bước normalization + pre-tokenization, tokenizers lấy tập hợp các “ký tự cơ bản” xuất hiện trong corpus. 

###### Example

```
 corpus = ["hug", "pug", "pun", "bun", "hugs"]
 base_vocabs = ["b","g","h","n","p","s","u"]
```

 - Với mô hình thực tế, base vocabulary thường là các byte (tức 256 giá trị byte), để đảm bảo mọi ký tự Unicode được mã hóa — đó là “byte-level BPE”.

##### 2. Count all adjacent character pairs in the corpus
Từ base vocab, BPE học các “merge-rules" bằng cách lặp nhiều vòng loop:
  - Ở mỗi vòng, tính tần suất xuất hiện của mọi cặp token kề nhau trong tất cả từ của corpus.
  - Cặp xuất hiện nhiều nhất được chọn để gộp thành token mới — token này được thêm vào vocabulary.
  - Corpus cập nhật → mọi nơi xuất hiện cặp đó sẽ được thay bằng token mới.
  - Quá trình lặp cho đến khi vocabulary đạt kích thước mong muốn (ví dụ 30–50 tokens trong ví dụ đơn giản; hoặc vài chục nghìn trong mô hình thực tế).

##### Example: Corpus related to Fruit

```
"banana" (8 lần)
"band"   (5 lần)
"banner" (3 lần)
"can"    (6 lần)
"candy"  (4 lần)
```
**1-** Tách từng từ thành ký tự cơ bản

```
b a n a n a
b a n d
b a n n e r
c a n
c a n d y
```

**2-** Đếm tần suất các cặp ký tự liền nhau
  Liệt kê một vài cặp:
  Ví dụ cặp `a n` xuất hiện trong:
```
banana → 2 lần
band → 1 lần
banner → 1 lần
can → 1 lần
candy → 1 lần
```

**TỔNG:** 6 lần → cặp cực kỳ phổ biến.

**3-** Gộp cặp phổ biến nhất
Gộp `a` + `n` thành token mới `an`

Cập nhật corpus: Thêm token `an` vào vocabulary.
```
b an a n a
b an d
b an n e r
c an
c an d y
```

**4-** Lặp lại: đếm cặp mới xuất hiện sau khi gộp

Giờ các cặp mới cần xét gồm:
- `b an`
- `an a`
- `an d`
- `c an`
- `n a`
- `n e`
- `e r`

Tính lại tần suất: 
| Pairs  | Số lần |
| ---- | ------ |
| b an | 3      |
| an a | 2      |
| an d | 3      |
| c an | 2      |
| an n | 1      |
| n a  | 1      |

Các cặp phổ biến nhất = **b an và an d (3 lần)**.
Giả sử thuật toán chọn `b an` → `ban`.

**5-** Gộp tiếp

Gộp `b` + `an` → `ban`
Cập nhật corpus:

```
ban a n a
ban d
ban n e r
c an
c an d y
```

**6-** Tiếp tục vòng lặp (training)
Các rule tiếp theo có thể sẽ là:
`an a` → `ana`
`ban` + `ana` → `banana`

Kết thúc quá trình, vocabulary sẽ là:
```
a, n, b, c, d, e, r, y (base tokens)
an
ban
ana
banana
can
candy
```


#### Code example

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# Initialize tokenizer
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

# Train tokenizer
trainer = BpeTrainer(
    vocab_size=50000,
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
)

# Train trên dữ liệu
files = ["corpus1.txt", "corpus2.txt"]
tokenizer.train(files, trainer)

tokenizer.save("llm_tokenizer.json")
```

[GitHub: Tokenizers](https://github.com/huggingface/tokenizers)