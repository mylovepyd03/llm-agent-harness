"""
1단계: Ollama와 기본 대화 확인
도구 호출 없이, 그냥 메시지 보내고 응답 받는 것부터 확인.
"""
import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.2"


def chat(messages):
    response = requests.post(
        OLLAMA_URL,
        json={"model": MODEL, "messages": messages, "stream": False},
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


if __name__ == "__main__":
    messages = [{"role": "user", "content": "너는 누구야? 한 문장으로 답해줘."}]
    answer = chat(messages)
    print("모델 응답:", answer)
