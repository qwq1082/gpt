# 导入必要的库
from transformers import pipeline

# 加载ChatGPT模型
model = pipeline('text-generation', model='gpt-neo-2.7B')

# 生成文本
generated_text = model("Hello, how are you?", max_length=30, do_sample=True, temperature=0.7)[0]['generated_text']

print(generated_text)
