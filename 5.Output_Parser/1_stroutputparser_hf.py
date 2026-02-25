from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

load_dotenv()

llm = HuggingFaceEndpoint(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    max_new_tokens=1024,
    temperature=0.7,
)

model = ChatHuggingFace(llm=llm)

template1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=["topic"],
)

template2 = PromptTemplate(
    template="Write a 5 line summary on the following text.\n\n{text}",
    input_variables=["text"],
)

parser = StrOutputParser()


chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({"topic": "black hole"})
print(result)
