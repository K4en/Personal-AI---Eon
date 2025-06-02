# Personal-AI---Eon

Project summary:
Personal AI assistant project. Using ChatGpt to help with coding and planning the project out. Also using it to explain new concepts and learn along the way as I don't like to blindly copy-paste code, rather type-copy even if it takes longer.

The idea was to build an AI assistant similar to Iron Man - Jarvis and while that's a longshot and not within the capabilities of today's technology or my resources it's still a fun project to do for learning.

Assistant Main Functions:
So far the assistant app.py has a prompt, memory, knowledge and task handling function through Flask using.

Personal Notes:
- I also had troubles with running LlaMA3 or actually anything through Ollama. It worked at some point and then just stopped working again. I'm hoping it's something with Ollama and once I plug in my own model it will work fine. Finger's crossed.

Project log:
- Built a basic framework for AI assistant
- 27/05/25 - Trained the "mistralai/Mistral-7B-Instruct-v0.2" model on my previous conversations with GPT with a Perplexity: 2.353214740753174 score. Went through a few couple trials and errors to find a model that can be trained without running out of memory on my RTX 4070 Ti using HuggingFace transformers.
- 02/06/25 - infer.py results:
{'loss': 1.6683, 'grad_norm': 0.8919278979301453, 'learning_rate': 0.00016161616161616162, 'epoch': 0.62} 
{'loss': 1.389, 'grad_norm': 0.9381101131439209, 'learning_rate': 0.00012121212121212122, 'epoch': 1.22}   
{'loss': 1.3026, 'grad_norm': 1.0599125623703003, 'learning_rate': 8.080808080808081e-05, 'epoch': 1.84}   
{'loss': 1.1853, 'grad_norm': 1.078960657119751, 'learning_rate': 4.0404040404040405e-05, 'epoch': 2.44}   
{'train_runtime': 540.7393, 'train_samples_per_second': 2.846, 'train_steps_per_second': 0.183, 'train_loss': 1.356082473138366, 'epoch': 3.0}
