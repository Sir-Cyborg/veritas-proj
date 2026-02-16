from rag import Augmentor, Retriever, Judger, run_court
import pandas as pd
import os



if __name__ == "__main__":
    
    path_in = "data/test_sets.xlsx"
    sheet = "poison"
    df_in = pd.read_excel(path_in, sheet_name=sheet, engine="openpyxl")
    questions = df_in["question"].tolist()
    
    augmentors = [
        Augmentor(type="openai", role_prompt="You are a conservative banking compliance expert."),
        Augmentor(type="openai", role_prompt="You are a pragmatic ICT operations engineer."),
        Augmentor(type="openai", role_prompt="You are a critical security auditor.")
        ]
    judger = Judger(augmentors[0].client)

    rows = []
    for _, r in df_in.iterrows():
        question = r["question"]
        context = r.get("poisoned context", "")
        answers = run_court(question=question, context=context, augmentors=augmentors)
        verdict = judger.judge(question=question, context=context, answers=answers)
        rows.append({"question": question, "verdict": verdict, "context": context, "poison_type": r.get("poison type", None), "answers_of_council": answers})
    
    df_out = pd.DataFrame(rows)
    path = "data/test_sets.xlsx"
    os.makedirs(os.path.dirname(path), exist_ok=True)

    sheet_name = "exp2"

    with pd.ExcelWriter(path, engine="openpyxl", mode="a", if_sheet_exists="new") as writer:
        df_out.to_excel(writer, sheet_name=sheet_name, index=False)
