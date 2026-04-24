from typing import Dict
import os
import json
import requests

API_KEYs = os.getenv("API_KEY")
url = "https://api.longcat.chat/openai/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {API_KEYs}",
    "Content-Type": "application/json"
}



def build_prompt(text: str) -> str:
    return f"""
    你是一个情绪分析助手。请分析用户输入文本的情绪状态。
    
    请输出 JSON，字段包括：
    1. label：只能是 positive、neutral、negative、mixed 之一
    2. sentiment_score：整数，范围从 -3 到 +3
       -3 表示强烈负面，0 表示中性，+3 表示强烈正面
    3. emotion：只能是 happiness、sadness、anger、fear、disgust、surprise、neutral、uncertain 之一
    4. reason：用一句中文解释判断依据
    5. confidence：只能是 high、medium、low 之一
    
    用户文本：
    「{text}」
    
    请严格输出 JSON，不要输出多余解释。
    """.strip()

def call_llm(prompt: str) -> str:
    """
    这里替换成你自己的 LLM API 调用逻辑。
    返回模型原始字符串输出。
    """
    data = {
        "model": "LongCat-Flash-Chat",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 1000,
        "temperature": 0.2
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]

    except requests.exceptions.Timeout:
        raise RuntimeError("LLM API 请求超时，请检查网络或稍后重试")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"网络或 API 请求失败: {e}")
    except (KeyError, TypeError, ValueError) as e:
        raise RuntimeError(f"API 返回数据格式异常: {e}")
    except Exception as e:
        raise RuntimeError(f"调用 LLM 时发生未知错误: {e}")


def parse_response(raw_response: str) -> Dict[str, any]:
    try:
        data = json.loads(raw_response)
        return {
            "label": data.get("label", "unknown"),
            "sentiment_score": data.get("sentiment_score", 0),
            "emotion": data.get("emotion", "uncertain"),
            "reason": data.get("reason", "No reason provided."),
            "confidence": data.get("confidence", "low")
        }
    except Exception:
        return {
            "label": "unknown",
            "sentiment_score": 0,
            "emotion": "uncertain",
            "reason": f"Failed to parse response: {raw_response}",
            "confidence": "low"
        }

def analyze_text_emotion(text: str) -> Dict[str, str]:
    prompt = build_prompt(text)
    raw_response = call_llm(prompt)
    result = parse_response(raw_response)
    return result

def myTest():
    samples = [
        "我今天还好",
        "真的没事，你不用管我",
        "我太开心了",
        "烦死了",
        "算了，我自己消化吧",
        "我有点累，但问题不大",
        "我今天心情特别好",
        "为什么总是这样",
        "哈哈，太有意思了",
        "没关系，我已经习惯了"
    ]
    for s in samples:
        print(f"{s} -> {analyze_text_emotion(s)}")