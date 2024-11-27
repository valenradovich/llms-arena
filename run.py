import requests

def run_discussion(topic, num_turns=3, system_prompt=None):
    url = "http://127.0.0.1:8000/discuss"
    headers = {"Content-Type": "application/json"}
    data = {
        "topic": topic,
        "num_turns": num_turns,
        "system_prompt": system_prompt or "You are participating in a debate about AI. Your role is to: 1. Directly respond to the previous speaker's points 2. Provide a contrasting perspective 3. Keep responses concise (2-3 sentences) 4. Make compelling arguments to change the other participant's mind Do not repeat previous points and always acknowledge what was just said."
    }
    
    try:
        with requests.post(url, json=data, headers=headers, stream=True, timeout=30) as response:
            response.raise_for_status()
            for chunk in response.iter_content(chunk_size=8192, decode_unicode=True):
                if chunk:
                    print(chunk, end='', flush=True)
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    run_discussion(
        topic="Future of Work: Will technology create more jobs than it replaces?",
        num_turns=3,
        system_prompt="""You are participating in a debate about AI. Your role is to:
1. Directly respond to the previous speaker's points
2. Provide a contrasting perspective
3. Keep responses concise (2-3 sentences)
4. Make compelling arguments to change the other participant's mind
Do not repeat previous points and always acknowledge what was just said."""

    )
