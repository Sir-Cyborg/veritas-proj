from rag import Augmentor, Retriever, Judger, run_court
import pandas as pd
import os



if __name__ == "__main__":
    questions = [
    "How does the identity lifecycle management process work from joiner to leaver?",
    "What are the minimum steps for requesting and approving access to a system?",
    "Which controls are required for Privileged Access Management (PAM)?",
    "How should service accounts be created, used, and managed?",
    "What is the process for revoking access when an employee leaves?",
    "What operational backup requirements must be met?",
    "What implementation standards are required for backups?",
    "Describe the recovery execution procedure (who does what during a restore)",
    "How often must backup and recovery testing be performed, and what must be documented?",
    "What additional controls are recommended for ransomware resilience?",
    "What are the main change types (standard/normal/emergency) and how do they differ?",
    "What information must a change record contain at minimum?",
    "How is change risk assessed and who approves high-risk changes?",
    "What pre-implementation testing and controls are required?",
    "What is required in the post-implementation review and closure steps?",
    "What are the data classification levels defined?",
    "What rules should staff use to classify a new dataset?",
    "What handling requirements apply to higher classifications?",
    "What are the retention and disposal requirements for bank data?",
    "How should DLP events be reported and escalated?",
    "What are the objectives of ICT risk management in the bank?",
    "Describe the ICT risk governance structure and responsibilities",
    "What are the required steps of the ICT risk management lifecycle?",
    "How are risk acceptance and exceptions handled?",
    "What documentation/evidence must be maintained for ICT risks?",
    "What are the key mandatory ICT security control requirements?",
    "What are the acceptable use and user obligations?",
    "How does the bank manage risk acceptance and compensating controls?",
    "How are compliance, monitoring, and audit handled for ICT security?",
    "What governance principles guide the bank’s security program?",
    "What minimum data must be logged when registering an incident?",
    "WHow are incidents categorized and prioritized?",
    "Describe the incident handling workflow from detection to recovery",
    "What are the incident closure criteria and required documentation?",
    "What emergency change/access controls apply during incidents?",
    "What are the minimum security requirements across the SDLC?",
    "How must secrets and keys be managed in development and CI/CD?",
    "What are the logging/auditability and privacy-by-design requirements?",
    "What are the rules for using open-source and third-party components?",
    "How are exceptions handled and enforced in secure software development?",
    "How does the bank classify third parties by criticality?",
    "What due diligence is required before contract signature?",
    "What should be covered in the security and architecture assessment of a supplier?",
    "Which minimum contractual security clauses must be included?",
    "What are the required steps for third-party exit planning and portability?",
    "What are the main vulnerability sources and how are they ingested?",
    "How are vulnerabilities triaged and validated (including false positives)?",
    "How does the bank risk-rate and prioritize vulnerabilities?",
    "What are the remediation SLAs by severity/criticality?",
    "How are vulnerability exceptions and risk acceptance handled?",
    "What is the bank’s Data Protection Officer (DPO) contact email and escalation hotline?",
    "Which exact ISO 27001 controls (Annex A) are adopted as mandatory, with control IDs?",
    "What is the bank’s approved list of endpoint security vendors and versions (e.g., EDR product name/version)?",
    "What is the formal incident severity matrix with numerical thresholds (e.g., financial loss > €X, customers affected > Y)?",
    "What is the official cloud landing zone architecture and network CIDR ranges used by CBG?"
    ]



    retriever = Retriever()
    augmentors = [
        Augmentor(type="openai", role_prompt="You are a conservative banking compliance expert."),
        Augmentor(type="openai", role_prompt="You are a pragmatic ICT operations engineer."),
        Augmentor(type="openai", role_prompt="You are a critical security auditor.")
        ]
    judger = Judger(augmentors[0].client)

    rows = []
    for question in questions:
        context = retriever.retrieve(query=question, k=1)
        answers = run_court(question=question, context=context, augmentors=augmentors)
        verdict = judger.judge(question=question, context=context, answers=answers)
        rows.append({"question": question, "verdict": verdict, "context": context, "answers_of_council": answers})
    
    df = pd.DataFrame(rows)
    path = "data/test_sets.xlsx"
    os.makedirs(os.path.dirname(path), exist_ok=True)

    sheet_name = "exp1.1"  # scegli tu il nome

    with pd.ExcelWriter(path, engine="openpyxl", mode="a", if_sheet_exists="new") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
    print(df)