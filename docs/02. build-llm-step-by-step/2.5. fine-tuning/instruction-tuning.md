### 1. Instruction Tuning


```python
# Định dạng instruction dataset
instruction_data = [
    {
        "instruction": "Viết email xin nghỉ phép",
        "input": "Nghỉ 3 ngày từ 10/12",
        "output": "Kính gửi Quý Anh/Chị,..."
    },
    # ...
]

def format_instruction(sample):
    prompt = f"""### Instruction:
{sample['instruction']}

### Input:
{sample['input']}

### Response:
{sample['output']}"""
    return prompt
```