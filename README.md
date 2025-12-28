"# medical-chat-bot"


Steps to run the project:
1. Create an env
```bash
 conda create -n mchatbot python==3.11 -y
 ```

2. Activate the env
```bash
 conda activate mchatbot
 ```

3. Install the requirements
```bash
 pip install -r requirements.txt
 ```

4. download llama-2-7b-chat.ggmlv3.q4_0 model from hugging face and save it in the model folder. In case if you want to use other model, change the model name in the app.py file

5. Create Pinecone account.

6. Create a Pinecone index called medical-chatbot. if you choose a different name, change the index name in the app.py file and store_index.py file

7. Create a .env file and add the following variables
- PINECONE_API_KEY

8. Run the store_index.py file to create the Pinecone vector store
Note: this will take a while to run (approx 20 mins). depends on how large the pdf file is.
```bash
python store_index.py
```

9. Run the app.py file to start the server
```bash
python app.py
```
Note: The model will take a while to give response depending on your computer's performance.


### Tech Stack:
- Python
- Flask
- Langchain
- Pinecone
- Hugging Face
- Llama 2
