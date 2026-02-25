from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    task="conversational",
    temperature=0.5,
    max_new_tokens=100
)

chat_model = ChatHuggingFace(llm=llm)

result = chat_model.invoke(
    [HumanMessage(content="What is the height of Mount Everest?")]
)

print(result.content)
