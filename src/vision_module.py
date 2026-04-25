from typing import Dict, Any
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from transformers import pipeline


DEVICE = "cuda"

EMOTION_PROMPTS = {
    "happiness": [
        "a photo of a person with a happy facial expression",
        "a photo showing a smiling and joyful person"
    ],
    "sadness": [
        "a photo of a person with a sad facial expression",
        "a photo showing a depressed or unhappy person"
    ],
    "anger": [
        "a photo of a person with an angry facial expression",
        "a photo showing anger or frustration"
    ],
    "fear": [
        "a photo of a person with a fearful facial expression",
        "a photo showing anxiety or fear"
    ],
    "disgust": [
        "a photo of a person with a disgusted facial expression",
        "a photo showing disgust or dislike"
    ],
    "surprise": [
        "a photo of a person with a surprised facial expression",
        "a photo showing surprise or shock"
    ],
    "neutral": [
        "a photo of a person with a neutral facial expression",
        "a photo showing a calm and neutral emotion"
    ]
}


EMOTION_TO_LABEL = {
    "happiness": "positive",
    "joy": "positive",
    "happy": "positive",

    "neutral": "neutral",

    "sadness": "negative",
    "sad": "negative",
    "anger": "negative",
    "angry": "negative",
    "fear": "negative",
    "disgust": "negative",

    "surprise": "mixed"
}


EMOTION_TO_SCORE = {
    "happiness": 2,
    "joy": 2,
    "happy": 2,

    "neutral": 0,

    "sadness": -2,
    "sad": -2,
    "anger": -2,
    "angry": -2,
    "fear": -2,
    "disgust": -2,

    "surprise": 1
}

_clip_model = None
_clip_processor = None

def load_clip():
    global _clip_model, _clip_processor

    if _clip_model is None or _clip_processor is None:
        model_name = "openai/clip-vit-base-patch32"
        _clip_model = CLIPModel.from_pretrained(model_name).to(DEVICE)
        _clip_processor = CLIPProcessor.from_pretrained(model_name)
        _clip_model.eval()

    return _clip_model, _clip_processor

def analyze_image_emotion_clip(image_path: str) -> Dict[str, Any]:
    model, processor = load_clip()

    image = Image.open(image_path).convert("RGB")

    all_prompts = []
    prompt_to_emotion = []

    for emotion, prompts in EMOTION_PROMPTS.items():
        for prompt in prompts:
            all_prompts.append(prompts)
            prompt_to_emotion.append(emotion)

    inputs = processor(text=all_prompts, images=image, return_tensors="pt", padding=True).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)[0]

    emotion_score = {}
    for idx, emotion in enumerate(prompt_to_emotion):
        emotion_score.setdefault(emotion, [])
        emotion_score[emotion].append(float(probs[idx].item()))

    averaged_scores = {
        emotion: sum(scores)/ len(scores)
        for emotion, scores in emotion_score.items()
    }

    sort_scores = sorted(
        averaged_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    top_emotion, top_score = sort_scores[0]
    second_emotion, second_score = sort_scores[1]
    margin = top_score - second_score

    if margin >= 0.15:
        confidence = "high"
    elif margin >= 0.05:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "label": EMOTION_TO_LABEL.get(top_emotion, "mixed"),
        "sentiment_score": EMOTION_TO_SCORE.get(top_emotion, 0),
        "emotion": top_emotion,
        "reason": f"CLIP zero-shot 图文相似度最高的情绪为 {top_emotion}，最高分为 {top_score:.3f}，与第二名 {second_emotion} 的差距为 {margin:.3f}。",
        "confidence": confidence,
        "method": "clip_zero_shot",
        "scores": averaged_scores
    }


def myTest_clip():
    len  = 7
    for i in range(len):
        result = analyze_image_emotion_clip(f"../data/test_image/{i + 1}.png")
        print(result)

# FER
_fer_pipe = None

def load_fer_model():
    global _fer_pipe

    if _fer_pipe is None:
        model_name = "mo-thecreator/vit-Facial-Expression-Recognition"
        device_id = 0

        _fer_pipe = pipeline("image-classification", model=model_name, device=device_id)

    return _fer_pipe

def norm_fer_emotion(raw_label: str) -> str:
    label = raw_label.lower()
    mapping = {
        "happy": "happiness", "joy": "happiness",
        "sad": "sadness", "angry": "anger",
        "fear": "fear", "fearful": "fear",
        "disgust": "disgust", "disgusted": "disgust",
        "surprise": "surprise", "surprised": "surprise",
        "neutral": "neutral"
    }
    return mapping.get(label,"uncertain")

def analyze_image_emotion_fer(image_path: str) -> Dict[str, Any]:
    fer_pipe = load_fer_model()
    outputs = fer_pipe(image_path)

    outputs = sorted(outputs, key=lambda x: x["score"], reverse=True)

    raw_emotion = outputs[0]["label"]
    score = float(outputs[0]["score"])

    emotion = norm_fer_emotion(raw_emotion)

    if score >= 0.75:
        confidence = "high"
    elif score >= 0.45:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "label": EMOTION_TO_LABEL.get(emotion, "mixed"),
        "sentiment_score": EMOTION_TO_SCORE.get(emotion, 0),
        "emotion": emotion,
        "reason": f"本地 FER 模型预测人脸表情为 {emotion}，原始标签为 {raw_emotion}，置信分数为 {score:.3f}。",
        "confidence": confidence,
        "method": "fer_classifier",
        "raw_outputs": outputs[:5]
    }

def myTest_fer():
    len  = 7
    for i in range(len):
        result = analyze_image_emotion_fer(f"../data/test_image/{i + 1}.png")
        print(result)

def fuse_image_results(clip_result: Dict[str, Any], fer_result: Dict[str, Any]) -> Dict[str, Any]:
    clip_emotion = clip_result["emotion"]
    fer_emotion = fer_result["emotion"]

    clip_label = clip_result["label"]
    fer_label = fer_result["label"]

    if clip_emotion == fer_emotion:
        return{
            "label": clip_label,
            "sentiment_score": clip_result["sentiment_score"],
            "emotion": clip_emotion,
            "reason": f"CLIP 与 FER 均判断图像情绪为 {clip_emotion}，整图语义与人脸表情一致，因此置信度较高。",
            "confidence": "high",
            "method": "clip_zero_shot + fer_classifier",
            "clip_result": clip_result,
            "fer_result": fer_result
        }
    if clip_label == fer_label:
        return {
            "label": clip_label,
            "sentiment_score": min(
                clip_result["sentiment_score"], fer_result["sentiment_score"]
            )if clip_label == "negative" else max(
                clip_result["sentiment_score"], fer_result["sentiment_score"]
            ),
            "emotion": mixed,
            "reason": f"CLIP 判断为 {clip_emotion}，FER 判断为 {fer_emotion}，二者情绪极性均为 {clip_label}，但具体类别不同，因此标记为 mixed。",
            "confidence": "medium",
            "method": "clip_zero_shot + fer_classifier",
            "clip_result": clip_result,
            "fer_result": fer_result
        }
    return{
        "label": "mixed",
        "sentiment_score": 0,
        "emotion": "uncertain",
        "reason": f"CLIP 判断为 {clip_emotion}/{clip_label}，FER 判断为 {fer_emotion}/{fer_label}，二者情绪极性不一致，说明视觉信号内部存在冲突。",
        "confidence": "low",
        "method": "clip_zero_shot + fer_classifier",
        "clip_result": clip_result,
        "fer_result": fer_result
    }

def analyze_image_emotion(image_path: str) -> Dict[str, Any]:
    clip_result = analyze_image_emotion_clip(image_path)
    fer_result = analyze_image_emotion_fer(image_path)

    return fuse_image_results(clip_result, fer_result)

def myTest():
    len  = 7
    for i in range(len):
        result = analyze_image_emotion(f"../data/test_image/{i + 1}.png")
        print(result)