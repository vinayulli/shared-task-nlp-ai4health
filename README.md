# shared-task-nlp-ai4health

for summarization, run inference_summarization.ipynb notebook

for Q&A, run inference_Q&A.ipynb notebook 

for Information Extraction, follow below steps. 

1. Download GGUF file - wget --no-check-certificate -c "https://www.dropbox.com/scl/fi/1f8t15k3azie270q91i7z/qwen-ie-finetuned.gguf?rlkey=gj72btv01m39np8ekppt12hbk&st=8hjxitl6&dl=1" -O qwen-ie-finetuned.gguf
2. Change the Path of gguf file in Modelfile after "from" command
3. create ollama model using - ollama create qwen3-ie-finetuned -f Modelfile  # ollama create modelname(this should be mentioned in dspy.LM) -f Modelfile
   
