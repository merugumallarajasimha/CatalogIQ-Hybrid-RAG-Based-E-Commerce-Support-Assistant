import os
from openai import OpenAI


def test_omniroute_connection():
    api_key = os.environ.get("OMNIROUTE_API_KEY")
    if not api_key:
        print("ERROR: OMNIROUTE_API_KEY environment variable not set")
        print("Set it before running: export OMNIROUTE_API_KEY=your_key")
        return False

    print("Testing Omniroute connection...")
    print(f"Base URL: http://localhost:20128/v1")
    print(f"API Key: {'*' * 8}...{api_key[-4:] if len(api_key) > 4 else '****'}")

    client = OpenAI(
        api_key=api_key,
        base_url="http://localhost:20128/v1",
    )

    try:
        response = client.chat.completions.create(
            model="auto",
            messages=[
                {"role": "user", "content": "say hello in 5 words"}
            ],
            max_tokens=50,
            temperature=0.1,
        )

        answer = response.choices[0].message.content
        # Sanitize for Windows console
        answer = answer.encode('ascii', 'replace').decode('ascii')
        model = response.model

        print(f"\n[OK] Connection successful!")
        print(f"Response: {answer}")
        print(f"Model used: {model}")
        print(f"Usage: {response.usage}")

        return True

    except Exception as e:
        print(f"\n[FAIL] Connection failed: {type(e).__name__}: {e}")
        return False


if __name__ == "__main__":
    test_omniroute_connection()