"""
Prompt templates for different datasets and tasks.
Centralized prompt management for consistent formatting across the project.
"""

# SkinCon dataset prompts
SKINCON_PROMPT = """List the observed clinical concepts in a single sentence starting with 'Presence of', 
and separate the concepts with commas. Put your answer in <rule>...</rule>."""

# Trajectory dataset prompts
TRAJECTORY_PROMPT_TYPE1 = """You are a medical vision-language assistant specializing in dermatology. 
Given the dermatology image, analyze the visual findings and provide a structured diagnosis following this format:

<rule>Describe the key clinical features and visual observations that support the diagnosis,and dianostic rationale</rule>
<diagnosis>Provide only the single most likely and specific fine-grained diagnosis. 
Do not include explanations, multiple options, or extra punctuation.</diagnosis>
<taxonomy>Briefly classify the condition (benign/malignant) and any relevant clinical taxonomy.</taxonomy>

Ensure your response is medically accurate, concise, and strictly follows the specified format."""

TRAJECTORY_PROMPT_TYPE2 = """You are a medical vision-language assistant specializing in dermatology. 
Given the dermatology image, analyze the visual findings and provide a structured diagnostic reasoning following this format:

<rule1>Describe the key clinical features and visual observations that support the primary diagnosis.</rule1>
<diagnosis1>State the most likely diagnosis based on Rule 1.</diagnosis1>
<differential diagnosis>Identify at least one differential diagnosis and provide its diagnostic rule (Rule 2) for comparison.</differential diagnosis>
<differential diagnosis reasoning>Compare the observed lesion features against both Rule 1 and Rule 2, 
and explain why the final diagnosis is more consistent with the findings.</differential diagnosis reasoning>
<final diagnosis>Provide only the single most likely and specific fine-grained diagnosis after considering the differential diagnosis. 
Do not include explanations, multiple options, or extra punctuation.</final diagnosis>
<taxonomy>Briefly classify the condition (benign/malignant) and provide any relevant clinical taxonomy.</taxonomy>

Ensure your response is medically accurate, concise, and strictly follows the specified format."""

# Trajectory dataset prompts
TRAJECTORY_PROMPT_TYPE1_V2 = """You are a medical vision-language assistant specializing in dermatology. 
Given the dermatology image, analyze the visual findings and provide a structured diagnosis following this format:

<thinking>Describe the key clinical features, visual observations, and the diagnostic rationale that support the diagnosis.</thinking>
<diagnosis>Provide the most likely and specific fine-grained diagnosis, and briefly classify the condition (benign or malignant) along with any relevant clinical taxonomy.</diagnosis>

Ensure your response is medically accurate, concise, and strictly follows the specified format."""

TRAJECTORY_PROMPT_TYPE2_V2 = """You are a medical vision-language assistant specializing in dermatology. 
Given the dermatology image, analyze the visual findings and provide a structured diagnostic reasoning following this format:

<thinking>
Begin by describing the characteristic clinical features and visual observations of the lesion. 
Based on these features, propose the most likely diagnosis. 
Then, identify at least one plausible alternative diagnosis and describe its defining features. 
Compare the observed lesion with these alternatives, and explain step by step why the final diagnosis is more consistent with the findings. 
Conclude the reasoning with the single condition that best matches the clinical presentation.
</thinking>

<diagnosis>
Provide only the single most likely and specific fine-grained diagnosis. 
Also briefly classify the condition (benign or malignant) and indicate any relevant clinical taxonomy (e.g., subtype or disease family). 
Do not include explanations, multiple options, or extra punctuation.
</diagnosis>

Ensure your response is medically accurate, concise, and strictly follows the specified format."""
# RL dataset prompts (if needed for future use)
RL_DIAGNOSIS_PROMPT = """Analyze this dermatology image and provide a diagnosis."""


# Function to get prompt by type
def get_prompt(prompt_type: str) -> str:
    """
    Get prompt template by type.
    
    Args:
        prompt_type: Type of prompt ('skincon', 'trajectory_type1', 'trajectory_type2', 'rl')
    
    Returns:
        Prompt template string
    """
    prompts = {
        'skincon': SKINCON_PROMPT,
        'trajectory_type1': TRAJECTORY_PROMPT_TYPE1,
        'trajectory_type2': TRAJECTORY_PROMPT_TYPE2,
        'rl': TRAJECTORY_PROMPT_TYPE1,
        'trajectory_type1_v2': TRAJECTORY_PROMPT_TYPE1_V2,
        'trajectory_type2_v2': TRAJECTORY_PROMPT_TYPE2_V2,
    }
    
    if prompt_type not in prompts:
        raise ValueError(f"Unknown prompt type: {prompt_type}. Available types: {list(prompts.keys())}")
    
    return prompts[prompt_type]

# Function to get trajectory prompt based on record type
def get_trajectory_prompt(record_type: str) -> str:
    """
    Get appropriate trajectory prompt based on record type.
    
    Args:
        record_type: Type of trajectory record ('type1', 'type2', 'type3')
    
    Returns:
        Appropriate prompt template
    """
    if record_type == "type1":
        return TRAJECTORY_PROMPT_TYPE1
    else:
        return TRAJECTORY_PROMPT_TYPE2

def get_trajectory_prompt_v2(record_type: str) -> str:
    """
    Get appropriate trajectory prompt based on record type.
    
    Args:
        record_type: Type of trajectory record ('type1', 'type2', 'type3')
    
    Returns:
        Appropriate prompt template
    """
    if record_type == "type1":
        return TRAJECTORY_PROMPT_TYPE1_V2
    else:
        return TRAJECTORY_PROMPT_TYPE2_V2

def get_message(prompt: str) -> list:
    
    messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": ""},  # Placeholder for image
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
    return messages

def get_message_verl(prompt: str) -> list:
    messages = [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ]
    return messages


def get_message_with_image(prompt: str, image: str) -> list:
    
    messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},  # Placeholder for image
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
    return messages

def get_question_prompt(question_text: str, options_str: str, question_type: int) -> list:
    if question_type == 3:
        # reflection
        question3 = (f"""You are a medical vision-language assistant specializing in dermatology. Given the dermatology image, answer: {question_text} {options_str}
                    \nProvide necessary reasoning and only answer the question in the following format:
                    <thinking>
                        Begin by describing the characteristic clinical features and visual observations of the lesion. 
                        Based on these features, choose the most likely diagnosis from the options. 
                        Then, identify at least one plausible alternative diagnosis from other options and describe its defining features. 
                        Compare the observed lesion with these alternatives, and explain step by step why the final diagnosis is more consistent with the findings. 
                        Conclude the reasoning with the single condition that best matches the clinical presentation.
                        </thinking>
                    <final diagnosis>Provide only the single most likely option. Also provide the lesion condition (benign, malignant or precancerous in situ). </final diagnosis>
                    Ensure your response is medically accurate, concise, and strictly follows the specified format.""")
        return question3
    elif question_type == 2:
        # CoT
        question2 = (f"You are a medical vision-language assistant specializing in dermatology. Given the dermatology image, answer: {question_text} {options_str}"
                " \nProvide necessary reasoning and only answer the question in the following format:"
                "<thinking>Describe the key clinical features and visual observations that support your diagnosis.</thinking>"
                "<final diagnosis>Provide only the single most likely option.</final diagnosis>"
                "Ensure your response is medically accurate, concise, and strictly follows the specified format.")
        return question2
    elif question_type == 1:
        # No CoT
        question1 = (f"You are a medical vision-language assistant specializing in dermatology. Given the dermatology image, answer: {question_text} {options_str}"
                " \nProvide only answer the question in the following format, do not include any other text:"
                "<final diagnosis>Provide only the single most likely option.</final diagnosis>"
                "Ensure your response is medically accurate, concise, and strictly follows the specified format.")
        return question1
    elif question_type == 4:
        # fine-grained
        question4 = (
                "You are a medical vision-language assistant specializing in dermatology. "
                f"Given the dermatology image, answer: {question_text} {options_str}\n"
                "Provide necessary reasoning and only answer the question in the following format:\n"
                "<thinking>Describe the key clinical features and visual observations that support your diagnosis.</thinking>\n"
                "<final diagnosis>Provide only the single most likely option without reasoning. "
                "If multiple options are plausible, always choose the most fine-grained (i.e., most specific or detailed) option available among them. "
                "Also provide the lesion condition (benign, malignant or precancerous in situ). </final diagnosis>\n"
                "Ensure your response is medically accurate, concise, and strictly follows the specified format."
            )
        return question4
    elif question_type == 5:
        # reflection
        question5 = (f"""You are a medical vision-language assistant specializing in dermatology. Given the dermatology image, answer: {question_text} {options_str}
                    \nProvide necessary reasoning and only answer the question in the following format:
                    <thinking>
                        Begin by describing the characteristic clinical features and visual observations of the lesion. 
                        Based on these features, choose the most likely diagnosis from the options. 
                        Then, identify at least one plausible alternative diagnosis from other options and describe its defining features. 
                        Compare the observed lesion with these alternatives, and explain step by step why the final diagnosis is more consistent with the findings. 
                        Conclude the reasoning with the single condition that best matches the clinical presentation.
                        </thinking>
                    <final diagnosis>Provide only the single most likely option without reasoning.  </final diagnosis>
                    Ensure your response is medically accurate, concise, and strictly follows the specified format.""")
        return question5
