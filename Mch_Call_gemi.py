import os, re, argparse, heapq, pandas as pd
from dataclasses import dataclass
from typing import List
from dotenv import load_dotenv
import google.generativeai as genai
load_dotenv()

@dataclass
class Job:
    id: int; p: int; d: int; w: int = 1

DEFAULT_CSV  = "jobs_50_weight.csv"
DEFAULT_LOGX = "milp_decision_log.xlsx"

def load_jobs(csv) -> List[Job]:
    df = pd.read_csv(csv)
    return [Job(int(r.id), int(r.processing_time), int(r.due_date),
                int(r.weight) if "weight" in df.columns else 1)
            for _, r in df.iterrows()]

def load_training_data(xlsx) -> list[dict]:
    df = pd.read_excel(xlsx)
    ex = []
    for _, r in df.iterrows():
        user = (
            f"Remaining jobs: {r.job_available}\n"
            f"Current total WT: {r.total_WT}\n"
            "Choose ONE Job ID that should be processed next."
        )
        ex += [{"role":"user","content":user},
               {"role":"model","content":str(r.job_chosen)}]
    return ex

PROMPT = """
Machine {mname} is idle at time {now}.
Remaining jobs ({n}):

{lines}

Choose exactly ONE Job ID that {mname} should process next so that the
final weighted total tardiness will be as small as possible.

Return ONLY the Job ID.
""".strip()

def build_prompt(machine, remaining, now):
    lines = "\n".join(
        f"- Job {j.id}: w={j.w}, p={j.p}, d={j.d}" for j in remaining
    )
    return PROMPT.format(mname=machine, now=now, n=len(remaining), lines=lines)

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash-latest")

def llm_choose_one(machine, remaining, now, few_shot) -> int:
    prompt = build_prompt(machine, remaining, now)

    # few-shot 예시를 prompt에 붙이기
    history = ""
    for i in range(0, len(few_shot), 2):
        history += f"{few_shot[i]['content']}\n{few_shot[i+1]['content']}\n"

    full_prompt = history + "\n" + prompt

    response = model.generate_content(full_prompt)
    reply = response.text
    m = re.search(r"\d+", reply)
    return int(m.group()) if m else remaining[0].id


def dispatch_machine_call(jobs: List[Job], start: int, few_shot) -> List[tuple]:
    mch_names = ["A", "B", "C"]
    heap = [(start, i) for i in range(3)]
    heapq.heapify(heap)
    remaining = jobs.copy()
    assignment = []

    while remaining:
        now, mi = heapq.heappop(heap)
        mch = mch_names[mi]
        cid = llm_choose_one(mch, remaining, now, few_shot)
        job = next((j for j in remaining if j.id == cid), remaining[0])
        remaining.remove(job)
        assignment.append((mch, job.id))
        heapq.heappush(heap, (now + job.p, mi))
    return assignment

def wtt_pairs(pairs, jobs, start):
    jobmap = {j.id: j for j in jobs}
    avail = {"A":start, "B":start, "C":start}
    total = 0
    for mch, jid in pairs:
        j = jobmap[jid]
        begin = avail[mch]
        finish = begin + j.p
        avail[mch] = finish
        total += j.w * max(0, finish - j.d)
    return total

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",  default=DEFAULT_CSV)
    ap.add_argument("--log",  default=DEFAULT_LOGX)
    ap.add_argument("--start", type=int, default=0)
    args = ap.parse_args()

    jobs  = load_jobs(args.csv)
    shots = load_training_data(args.log)
    pairs = dispatch_machine_call(jobs, args.start, shots)
    score = wtt_pairs(pairs, jobs, args.start)

    by_machine = {"A": [], "B": [], "C": []}
    for mch, jid in pairs:
        by_machine[mch].append(jid)

    print("training 예시 메시지 :", len(shots) // 2)
    print(
        "머신별 호출 순서     :\n"
        f'A: {by_machine["A"]}\n'
        f'B: {by_machine["B"]}\n'
        f'C: {by_machine["C"]}'
    )
    print("Weighted Total Tardiness:", score)
