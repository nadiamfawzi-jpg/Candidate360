import json
import os

MODEL_NAME = "openrouter/free"


def get_client(api_key=""):
    key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    if not key:
        return None

    try:
        from openai import OpenAI
    except ImportError as error:
        raise ImportError(
            "The OpenAI client is not installed. Add openai>=1.30 to "
            "requirements.txt and reboot the Streamlit app."
        ) from error

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=key
    )
    return client


def get_llm_response(prompt, system_message, api_key=""):
    client = get_client(api_key)
    if client is None:
        raise ValueError("Add OPENAI_API_KEY to Streamlit Secrets.")

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )
    response = completion.choices[0].message.content
    return response


def get_json_response(prompt, system_message, api_key=""):
    response = get_llm_response(prompt, system_message, api_key)
    clean_response = response.strip()

    if clean_response.startswith("```"):
        clean_response = clean_response.replace("```json", "").replace("```", "").strip()

    start = clean_response.find("{")
    end = clean_response.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("The API did not return the expected JSON. Please try again.")

    return json.loads(clean_response[start:end + 1])


def build_interview(cv_text, job_description, target_role, api_key=""):
    prompt = f"""
Create the CV analysis and six interview questions in one response.

Target role: {target_role}

CV:
{cv_text[:12000]}

Job description:
{job_description[:8000]}

Return JSON only with this exact structure:
{{
  "cv_analysis": {{
    "match_summary": "short evidence-based summary",
    "cv_strengths": ["strength 1", "strength 2", "strength 3"],
    "missing_or_unclear": ["gap 1", "gap 2", "gap 3"],
    "skills_to_prove": ["skill 1", "skill 2", "skill 3"],
    "match_score": 0
  }},
  "questions": [
    {{
      "type": "Behavioural",
      "question": "question text",
      "purpose": "what it tests"
    }}
  ]
}}

Return exactly six fair, role-related questions using a balanced mix of
introduction, behavioural, situational, role-specific, CV-evidence and
closing questions. The match score must be from 0 to 100 and is only a
practice estimate. Use only supplied evidence. Do not infer protected or
personal characteristics.
"""
    system_message = (
        "You are a careful CV analyst and supportive interviewer. "
        "Do not invent evidence. Return valid JSON only."
    )
    result = get_json_response(prompt, system_message, api_key)
    return {
        "cv_analysis": result.get("cv_analysis", {}),
        "questions": result.get("questions", [])
    }


def analyse_cv_job(cv_text, job_description, target_role, api_key=""):
    prompt = f"""
Target role: {target_role}

CV:
{cv_text[:14000]}

Job description:
{job_description[:10000]}

Return JSON only with this structure:
{{
  "match_summary": "short evidence-based summary",
  "cv_strengths": ["strength 1", "strength 2", "strength 3"],
  "missing_or_unclear": ["gap 1", "gap 2", "gap 3"],
  "skills_to_prove": ["skill 1", "skill 2", "skill 3"],
  "match_score": 0
}}

The match_score must be from 0 to 100 and is only a practice estimate.
Use only evidence present in the supplied text. Do not infer age, gender,
race, health, religion, personality or other protected information.
"""
    system_message = "You are a careful CV analyst supporting interview practice. Do not invent evidence."
    return get_json_response(prompt, system_message, api_key)


def create_questions(cv_text, job_description, target_role, api_key=""):
    prompt = f"""
Create six interview questions for this candidate and role.

Target role: {target_role}
CV: {cv_text[:10000]}
Job description: {job_description[:8000]}

Include a balanced mix of introduction, behavioural, situational,
role-specific, CV-evidence and closing questions.

Return JSON only:
{{
  "questions": [
    {{"type": "Behavioural", "question": "question text", "purpose": "what it tests"}}
  ]
}}
"""
    system_message = "You are a supportive professional interviewer. Ask fair, job-related questions only."
    result = get_json_response(prompt, system_message, api_key)
    return result.get("questions", [])


def evaluate_answer(question, answer, cv_text, job_description, api_key=""):
    prompt = f"""
Question: {question}
Candidate answer: {answer}
Relevant CV: {cv_text[:7000]}
Job description: {job_description[:5000]}

Evaluate the answer for interview practice. Use the answer itself as evidence.
Do not judge accent, identity, facial appearance, personality or employability.

Return JSON only:
{{
  "relevance": 0,
  "specificity": 0,
  "structure": 0,
  "job_alignment": 0,
  "overall": 0,
  "strengths": ["specific strength", "specific strength"],
  "improvements": ["actionable improvement", "actionable improvement"],
  "missing_evidence": ["missing point"],
  "improved_answer": "a concise improved example that does not invent experience",
  "follow_up_question": "one useful follow-up question"
}}

All five scores must be integers from 0 to 100.
"""
    system_message = "You are an evidence-based interview answer coach. Be supportive, specific and honest."
    return get_json_response(prompt, system_message, api_key)


def create_session_summary(target_role, cv_analysis, answer_results, api_key=""):
    prompt = f"""
Target role: {target_role}
CV and job analysis: {json.dumps(cv_analysis, ensure_ascii=False)}
Answer results: {json.dumps(answer_results, ensure_ascii=False)}

Return JSON only:
{{
  "headline": "short session headline",
  "overall_progress": "short honest summary",
  "top_strengths": ["strength 1", "strength 2", "strength 3"],
  "priority_actions": ["action 1", "action 2", "action 3"],
  "next_practice_plan": ["step 1", "step 2", "step 3"]
}}
Do not make a hiring recommendation. Do not use visual or voice-expression
predictions as evidence of competence, confidence or personality.
"""
    system_message = "You are the final interview practice coach. Summarise evidence without exaggeration."
    return get_json_response(prompt, system_message, api_key)
