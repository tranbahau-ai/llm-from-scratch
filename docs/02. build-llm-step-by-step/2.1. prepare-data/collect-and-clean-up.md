### Thu Thập và Chuẩn Bị Dữ Liệu

#### 1. Nguồn Dữ Liệu

**Dữ liệu cần thiết** (hàng trăm GB đến TB):
- **Common Crawl**: Dữ liệu web crawl công khai
- **Books**: Project Gutenberg, sách điện tử
- **Wikipedia**: Bách khoa toàn thư
- **Code repositories**: GitHub, StackOverflow
- **Academic papers**: ArXiv, PubMed
- **Conversations**: Reddit, Twitter, forums


#### 2. Làm Sạch Dữ Liệu

```python
import re
from bs4 import BeautifulSoup

def clean_text(text):
    # Remove HTML tags
    text = BeautifulSoup(text, 'html.parser').get_text()
    
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Remove special characters
    text = re.sub(r'[^\w\s.,!?]', '', text)
    
    # Remove white-space
    text = ' '.join(text.split())
    
    return text
```

#### 3. Deduplication

Loại bỏ văn bản trùng lặp để tránh overfitting, tăng coverage nội dung:

```python
from datasketch import MinHash, MinHashLSH

def deduplicate_texts(texts, threshold=0.8):
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    unique_texts = []
    
    for idx, text in enumerate(texts):
        m = MinHash(num_perm=128)
        for word in text.split():
            m.update(word.encode('utf8'))
        
        # Kiểm tra duplicate
        result = lsh.query(m)
        if not result:
            lsh.insert(f"doc_{idx}", m)
            unique_texts.append(text)
    
    return unique_texts
```

[Next Step: Tokenization](/docs/02.%20build-llm-step-by-step/2.2.%20tokenization/tokenization.md)