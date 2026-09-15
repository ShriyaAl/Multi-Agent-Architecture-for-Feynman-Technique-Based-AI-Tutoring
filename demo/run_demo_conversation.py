# demo/run_demo_conversation.py

from agents.diagnosis_agent.agent import DiagnosisAgent

def main():
    agent = DiagnosisAgent()

    # Simulate a short real conversation — edit this to test different cases
    question = "Heloise has dogs and cats in ratio 10:17, total 189 pets. She gives 10 dogs to Janet. How many dogs does she have left?"
    reference_answer = "70 - 10 = 60"

    conversation = [
        {"role": "teacher", "text": "Can you walk me through how you'd solve this?"},
        {"role": "student", "text": "So dogs and cats add up to 27 parts total."},
        {"role": "teacher", "text": "Good — what's each part worth?"},
        {"role": "student", "text": "189 divided by 27 is 7, so 10 times 7 is 70 dogs."},
    ]

    context_text = "\n".join(f"{t['role']}: {t['text']}" for t in conversation[:-1])
    latest_explanation = conversation[-1]["text"]

    print("Context so far:\n", context_text)
    print("\nDiagnosing latest student turn:\n", latest_explanation)

    result = agent.diagnose(
        question=question + "\n\nConversation so far:\n" + context_text,
        student_explanation=latest_explanation,
        reference_answer=reference_answer,
    )

    for dim in ["accuracy", "completeness", "clarity", "coherence"]:
        r = getattr(result, dim)
        print(f"\n{dim}: score={r.score}, gaps={r.matched_gap_ids}, reasoning={r.reasoning}")

if __name__ == "__main__":
    main()