from google.genai.types import SafetySetting, HarmCategory, HarmBlockThreshold

# 定义了适用于Gemini系列模型的标准安全设置。
# 此设置旨在最大程度地减少因内容安全策略导致的空响应问题。
# 通过将所有主要有害类别的阈值设置为BLOCK_NONE，可以允许更广泛的内容通过，
# 这在许多开发和测试场景中非常有用。
#
# 支持的类别包括：
# - HARM_CATEGORY_HATE_SPEECH: 仇恨言论
# - HARM_CATEGORY_DANGEROUS_CONTENT: 危险内容
# - HARM_CATEGORY_HARASSMENT: 骚扰
# - HARM_CATEGORY_SEXUALLY_EXPLICIT: 色情内容
#
# 此配置可用于原生Gemini API调用和通过OpenAI兼容终结点进行的调用。

GEMINI_SAFETY_SETTINGS = [
    SafetySetting(category=HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=HarmBlockThreshold.BLOCK_NONE),
    SafetySetting(category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=HarmBlockThreshold.BLOCK_NONE),
    SafetySetting(category=HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=HarmBlockThreshold.BLOCK_NONE),
    SafetySetting(category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=HarmBlockThreshold.BLOCK_NONE),
]

# 用于OpenAI兼容模式的字典格式
GEMINI_SAFETY_SETTINGS_FOR_OPENAI = [
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
]
