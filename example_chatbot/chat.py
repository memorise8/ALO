from dotenv import load_dotenv
import os
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langfuse.decorators import observe
import os

load_dotenv()

OPENAI_API_TYPE = os.getenv("OPENAI_API_TYPE")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")
OPENAI_MODEL_ID = os.getenv("OPENAI_MODEL_ID")


@observe()
def ask_chat(topic: str):
    # RAG 사용하여 관련 문서들 검색하기
    # relevant_docs = retrieve_docs(topic)  # `retrieve_docs` 함수를 호출
    # relevant_content = " ".join([doc.page_content for doc in relevant_docs])

    # 렐러번트 컨텐츠와 주제를 포함한 프롬프트 구성
    full_prompt = f"{topic} 에 대해 설명해주세요."

    model = AzureChatOpenAI(model=OPENAI_MODEL_ID, api_version=OPENAI_API_VERSION)
    prompt_template = ChatPromptTemplate.from_template(full_prompt)
    chain = prompt_template | model

    response = chain.invoke({"topic": topic})

    return {
        'response': response.content
    }