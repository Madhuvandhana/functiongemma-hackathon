<img src="assets/banner.png" alt="Logo" style="border-radius: 30px; width: 100%;">

## Context
- Cactus runs Google DeepMind's FunctionGemma at up to 3000 toks/sec prefill speed on M4 Macs.
- While decode speed reaches 200 tokens/sec, all without GPU, to remain energy-efficient. 
- FunctionGemma is great at tool calling, but small models are not the smartest for some tasks. 
- There is a need to dynamically combine edge and cloud (Gemini Flash) to get the best of both worlds. 
- Cactus develops various strategies for choosing when to fall back to Gemini or FunctionGemma.

## Challenge
- FunctionGemma is just a tool-call model, but tool calling is the core of agentic systems. 
- You MUST design new strategies that decide when to stick with on-device or fall to cloud. 
- You will be objectively ranked on tool-call correctness, speed and edge/cloud ratio (priortize local). 
- You can focus on prompting, tool description patterns, confidence score algorithms, anything!
- Please ensure at least 1 team member has a Mac, Cactus runs on Macs, mobile devices and wearables.

---

# ⚡ Hybrid On-Device Function Calling Engine

### FunctionGemma (Cactus) + Gemini Cloud Fallback

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![F1 Score](https://img.shields.io/badge/F1%20Score-0.967-brightgreen)
![Hackathon Score](https://img.shields.io/badge/Hackathon-91.6%25-success)
![Edge First](https://img.shields.io/badge/Edge-AI-orange)
![On--Device](https://img.shields.io/badge/Inference-On--Device-blueviolet)

A production-grade hybrid function-calling system that prioritizes **deterministic parsing + on-device LLM inference**, and falls back to the cloud only when necessary.

# 🏗 Architecture

```
User Input
   ↓
Fast Deterministic Parser (0ms)
   ↓ (if valid)
Return On-Device Result
   ↓ (else)
FunctionGemma via Cactus (On-Device LLM)
   ↓ (confidence ≥ threshold?)
Return On-Device Result
   ↓ (else)
Gemini Cloud Fallback
```

---

# 🔥 Key Features

## ✅ Deterministic Intent Layer

Instantly resolves:

* `set_alarm`
* `set_timer`
* `get_weather`
* `send_message`
* `create_reminder`
* `play_music`
* `search_contacts`

Includes:

* Absolute time detection (`10:30am`)
* Relative time detection (`in 5 minutes`)
* Pronoun resolution (`find Bob and text him`)
* Multi-intent splitting (`and`, `then`, `also`)
* Required argument validation
* Argument type coercion

---

## 📱 On-Device LLM (FunctionGemma via Cactus)

* Zero network dependency
* Structured function calling
* Confidence scoring
* Fast inference (~200–400ms typical)

```python
from cactus import cactus_init, cactus_complete, cactus_destroy
```

---

## ☁️ Cloud Fallback (Gemini 2.5 Flash)

Used only when:

* Local confidence < threshold
* Required arguments missing
* Ambiguous multi-intent cases

```python
model = "gemini-2.5-flash"
```

---

# 🧠 Core API

## 1️⃣ On-Device Only

```python
generate_cactus(messages, tools)
```

---

## 2️⃣ Cloud Only

```python
generate_cloud(messages, tools)
```

---

## 3️⃣ Hybrid Mode (Recommended)

```python
generate_hybrid(messages, tools, confidence_threshold=0.60)
```
---

# 🛠 Example Usage

```python
if __name__ == "__main__":
    tools = [{
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name",
                }
            },
            "required": ["location"],
        },
    }]

    messages = [
        {"role": "user", "content": "What is the weather in San Francisco?"}
    ]

    hybrid = generate_hybrid(messages, tools)
    print_result("Hybrid", hybrid)
```

---

# 📊 Benchmark Results

## Local Benchmark Suite (49 Scenarios)

| Difficulty  | Avg F1   | Avg Time  | On-Device Rate |
| ----------- | -------- | --------- | -------------- |
| Easy        | 1.00     | 148ms     | 100%           |
| Medium      | 1.00     | 309ms     | 100%           |
| Hard        | 1.00     | 321ms     | 100%           |
| **Overall** | **1.00** | **271ms** | **100%**       |

* Total cases: 49
* Cloud fallback usage: 0%
* Local score: **85.8%**

---

# 🏆 Hackathon Performance

Live evaluation results:

* 🎯 **F1 Score:** 0.967
* ⚡ **Average Latency:** 234ms
* 📱 Majority resolved on-device
* ☁️ Minimal cloud fallback

**Final Score: 91.6%**

---

# Why This Approach Works

Most assistants either:

* Rely entirely on cloud LLMs (higher latency + cost), or
* Use pure rules (low flexibility).

This hybrid system:

✔ Deterministic when possible
✔ LLM when needed
✔ Cloud only as fallback
✔ Validates before execution
✔ Confidence-gated responses

Result:

* Real-time performance
* Edge-AI friendly
* Cost-efficient
* Production-ready reliability

---

# Requirements

* Python 3.10+
* Cactus runtime
* FunctionGemma weights
* Google GenAI SDK
* `GEMINI_API_KEY` environment variable

---

#  Ideal Use Cases

* On-device AI assistants
* Android system assistants
* Edge IoT agents
* Autonomous action agents
* Hackathon demo systems

---

# Design Principles

1. Deterministic > Probabilistic when possible
2. Validate before trusting LLM output
3. Confidence gating prevents hallucinated calls
4. Edge-first, cloud-second

