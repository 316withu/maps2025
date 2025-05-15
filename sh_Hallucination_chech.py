import os
import re
import argparse
import random
import pandas as pd
from dataclasses import dataclass
from typing import List
import openai
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────── Job 데이터 구조 ───────────────────────────
@dataclass
class Job:
    id: int
    p: int
    d: int
    w: int = 1

# ─────────────────────────── CSV 로드 ───────────────────────────
DEFAULT_CSV = "jobs_50_weight.csv"
DEFAULT_LOGX = "milp_decision_log.xlsx"  # few-shot 예시 로그

def load_jobs(csv: str) -> List[Job]:
    df = pd.read_csv(csv)
    return [
        Job(int(r.id), int(r.processing_time), int(r.due_date),
            int(r.weight) if "weight" in df.columns else 1)
        for _, r in df.iterrows()
    ]

# ─────────────────────────── 엑셀 → few-shot ───────────────────────────
def load_few_shot_all(xlsx: str) -> list[dict]:
    df = pd.read_excel(xlsx)
    examples = []
    for _, r in df.iterrows():
        user = (
            f"Remaining jobs: {r.job_available}\n"
            f"Current total WT: {r.total_WT}\n"
            "Choose ONE Job ID that should be processed next."
        )
        examples.append({"role": "user", "content": user})
        examples.append({"role": "assistant", "content": str(r.job_chosen)})
    return examples

# ─────────────────────────── 프롬프트 템플릿 ───────────────────────────
LOCAL_TMPL = """
You are scheduling jobs on three identical parallel machines (A, B, C).
All machines can start a new job at time {now}.

Remaining jobs ({n}):

{lines}

Choose **exactly ONE Job ID** that should be processed next so that the
final weighted total tardiness will be as small as possible.

Return ONLY the Job ID.
""".strip()

def build_prompt_local(remaining: List[Job], now: int) -> str:
    lines = "\n".join(
        f"- Job {j.id}: Weight {j.w}, Processing Time {j.p}, Due {j.d}"
        for j in remaining
    )
    return LOCAL_TMPL.format(now=now, n=len(remaining), lines=lines)

# ─────────────────────────── LLM 호출 + 할루시네이션 감지 ───────────────────────────
def llm_choose_one(remaining: List[Job], now: int, few_shot: list[dict], hallucination_log: dict) -> int:
    messages = few_shot + [{"role": "user", "content": build_prompt_local(remaining, now)}]

    try:
        reply = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY")).chat.completions.create(
            model="gpt-4o",
            max_tokens=200,
            temperature=0,
            messages=messages
        ).choices[0].message.content
    except Exception as e:
        print("[ERROR] GPT 호출 실패:", e)
        return random.choice(remaining).id

    m = re.search(r"\d+", reply)
    if m:
        chosen_id = int(m.group())
        if any(j.id == chosen_id for j in remaining):
            return chosen_id
        else:
            hallucination_log["count"] += 1
            hallucination_log["examples"].append({
                "invalid_id": chosen_id,
                "reply": reply,
                "remaining_ids": [j.id for j in remaining],
                "now": now
            })
            return random.choice(remaining).id
    else:
        hallucination_log["count"] += 1
        hallucination_log["examples"].append({
            "invalid_id": None,
            "reply": reply,
            "remaining_ids": [j.id for j in remaining],
            "now": now
        })
        return random.choice(remaining).id

# # ─────────────────────────── 스케줄링 루프 ───────────────────────────
# def schedule_local(jobs: List[Job], start: int, few_shot: list[dict]) -> list[int]:
#     remaining = jobs.copy()
#     mtime = [start] * 3
#     seq = []
#     hallucination_log = {"count": 0, "examples": []}

#     while remaining:
#         now = min(mtime)
#         cid = llm_choose_one(remaining, now, few_shot, hallucination_log)
#         chosen = next((j for j in remaining if j.id == cid), remaining[0])
#         remaining.remove(chosen)
#         seq.append(chosen.id)
#         mtime[mtime.index(now)] = now + chosen.p

#     print("\n[할루시네이션 발생 횟수]:", hallucination_log["count"])
#     if hallucination_log["count"] > 0:
#         for idx, h in enumerate(hallucination_log["examples"]):
#             print(f"\n📌 사례 {idx+1}")
#             print(" - 잘못된 ID:", h["invalid_id"])
#             print(" - GPT 응답 :", h["reply"])
#             print(" - 시간:", h["now"])
#             print(" - 실제 남은 잡들:", h["remaining_ids"])
#     return seq
def schedule_local(jobs: List[Job], start: int, few_shot: list[dict]) -> list[int]:
    remaining = jobs.copy()
    mtime = [start] * 3
    seq = []
    
    hallucination_log = {
        "count": 0,
        "examples": [],
        "total_decisions": 0
    }

    while remaining:
        now = min(mtime)
        hallucination_log["total_decisions"] += 1  # 선택 횟수 누적
        cid = llm_choose_one(remaining, now, few_shot, hallucination_log)
        chosen = next((j for j in remaining if j.id == cid), remaining[0])
        remaining.remove(chosen)
        seq.append(chosen.id)
        mtime[mtime.index(now)] = now + chosen.p

    # 최종 결과 요약 출력
    hallucination_rate = (hallucination_log["count"] / hallucination_log["total_decisions"]) * 100
    print("\n📊 [할루시네이션 요약]")
    print(f"- 총 결정 횟수     : {hallucination_log['total_decisions']}")
    print(f"- 할루시네이션 수  : {hallucination_log['count']}")
    print(f"- 발생 비율(%)     : {hallucination_rate:.2f}%")

    if hallucination_log["count"] > 0:
        for idx, h in enumerate(hallucination_log["examples"]):
            print(f"\n📌 사례 {idx+1}")
            print(" - 잘못된 ID:", h["invalid_id"])
            print(" - GPT 응답 :", h["reply"])
            print(" - 시간:", h["now"])
            print(" - 실제 남은 잡들:", h["remaining_ids"])

    return seq


# ─────────────────────────── WTT 계산 ───────────────────────────
def wtt_parallel(order: list[int], jobs: List[Job], start: int) -> int:
    jobmap = {j.id: j for j in jobs}
    mtime = [start] * 3
    total = 0
    for jid in order:
        j = jobmap[jid]
        m = mtime.index(min(mtime))
        finish = mtime[m] + j.p
        mtime[m] = finish
        total += j.w * max(0, finish - j.d)
    return total

# ─────────────────────────── 실행부 ───────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--log", default=DEFAULT_LOGX, help="MILP decision log (few-shot)")
    ap.add_argument("--start", type=int, default=0)
    args = ap.parse_args()

    jobs = load_jobs(args.csv)
    few_shot = load_few_shot_all(args.log)
    order = schedule_local(jobs, args.start, few_shot)
    score = wtt_parallel(order, jobs, args.start)

    print("\nLLM 결정 순서:", order)
    print("Weighted Total Tardiness:", score)
