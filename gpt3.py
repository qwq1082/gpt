# -*- coding: utf-8 -*-
from numpy import true_divide
from transformers import GPT2LMHeadModel
from transformers import AutoTokenizer
from transformers import TextGenerationPipeline
from transformers import pipeline

model = GPT2LMHeadModel.from_pretrained("D:\GPT\SkyTextTiny")
tokenizer = AutoTokenizer.from_pretrained("D:\GPT\SkyTextTiny", trust_remote_code=True)
text_generator = TextGenerationPipeline(model, tokenizer, device=0)
max_new_tokens = 20

while True:
    print("Please input your question:")
    input_str = input()
    answer = text_generator(input_str, max_new_tokens=max_new_tokens, do_sample=True)
    print(answer)