# Multi-Agent-Architecture-for-Feynman-Technique-Based-AI-Tutoring

A multi-agent AI system that helps students actually understand concepts — not just get answers.

## The Problem

Most AI tutors give help the moment a student is stuck. That feels helpful, but studies show it can leave students unable to explain the concept once the AI is taken away — they got good at using the tool, not at the concept itself.

## The Idea

Based on the **Feynman Technique**: you don't really understand something until you can explain it simply, in your own words. So this system flips the usual order — students explain first, and AI only steps in once it knows exactly what's missing.

## How It Works

1. Student explains a concept, unprompted.
2. The system checks it's a genuine attempt.
3. The explanation is scored on 4 things: completeness, accuracy, clarity, coherence.
4. If something's missing, the system asks a targeted question — never the answer — to nudge the student toward it themselves.
5. This repeats a couple of times until the concept is mastered, or the system simplifies/flags it for the teacher.

## The Agents

- **Gate** – checks the response is real
- **Confidence** – notes how sure the student is
- **Diagnosis** – scores the explanation, finds the gap
- **Socratic** – asks the targeted follow-up question
- **Router** – decides what happens next
- **Reporting** – summarizes progress for teachers

## Why It's Different

Most AI tutors optimize for feeling helpful in the moment. This one optimizes for whether the understanding actually sticks once the AI is gone.