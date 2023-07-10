# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 13:42:06 2023

@author: RANGER
"""

import tkinter as tk
from tkinter import scrolledtext
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the pre-trained LLM model and tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# Create the GUI
window = tk.Tk()
window.title("LLM Text Generation")
window.geometry("800x600")

# Create a scrolled text widget for input and output
input_text = scrolledtext.ScrolledText(window, width=100, height=10)
input_text.pack(pady=10)

output_text = scrolledtext.ScrolledText(window, width=100, height=20)
output_text.pack(pady=10)

# Generate text based on the input
def generate_text():
    # Get the input text from the widget
    input_str = input_text.get("1.0", tk.END).strip()
    
    # Tokenize the input text
    input_tokens = tokenizer.encode(input_str, return_tensors="pt")
    input_tokens = input_tokens.to(device)
    
    # Generate text using the LLM model
    output_tokens = model.generate(input_tokens, max_length=100, num_return_sequences=1)
    
    # Decode the generated tokens
    output_str = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    
    # Update the output text widget
    output_text.delete("1.0", tk.END)
    output_text.insert(tk.END, output_str)

# Create a button to generate text
generate_button = tk.Button(window, text="Generate", command=generate_text)
generate_button.pack(pady=10)

# Start the GUI main loop
window.mainloop()
