import os
import sys
from docx import Document
from rlm import RLM

# ------------------------------------------------------------------
# 1. HELPER: Load Protocol Text
# ------------------------------------------------------------------
def load_protocol_text(filepath):
    """Reads the full text from the DOCX protocol."""
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        sys.exit(1)
    
    doc = Document(filepath)
    full_text = []
    for para in doc.paragraphs:
        if para.text.strip():
            full_text.append(para.text)
    return "\n".join(full_text)

# ------------------------------------------------------------------
# 2. CONFIGURATION
# ------------------------------------------------------------------
PROTOCOL_PATH = "data/protocol_example.docx"
MODEL_NAME = "gpt-5.1"

# ------------------------------------------------------------------
# 3. PIPELINE EXECUTION
# ------------------------------------------------------------------
def run_icf_extraction():
    print(f"--- UHN ICF AUTOMATION PIPELINE ---")
    print(f"Loading protocol from: {PROTOCOL_PATH}")
    
    # Load the text
    protocol_content = load_protocol_text(PROTOCOL_PATH)
    print(f"Protocol loaded ({len(protocol_content)} chars). Initializing RLM...")

    # Initialize RLM with PERSISTENCE
    rlm = RLM(
        backend="openai",
        backend_kwargs={"model_name": MODEL_NAME},
        environment="local",   
        persistent=True,       # Keeps context_0 alive across calls
        verbose=True,          
        max_iterations=15      
    )

    # ----------------------------------------------------------------
    # Variable 1: Sponsor Name (The "Loader" Step)
    # CRITICAL FIX: We pass 'protocol_content' here to load it as context_0
    # ----------------------------------------------------------------
    print("\n\n>>> PROCESSING VARIABLE 1: SPONSOR NAME")
    var1_instruction = (
        "You are an expert medical writer for UHN. "
        "The protocol text has been loaded into 'context_0'. "
        "Identify the commercial or academic 'Sponsor' of this study. "
        "Return ONLY the exact name of the sponsor entity."
    )
    
    response_1 = rlm.completion(
        prompt=protocol_content,  # <--- CRITICAL: DATA LOADED HERE
        root_prompt=var1_instruction
    )
    print(f"\n[RLM OUTPUT]: {response_1.response}")
    print(f"[GROUND TRUTH]: Design Therapeutics, Inc") 

    # ----------------------------------------------------------------
    # Variable 2: Study Design Class
    # Now we can just refer to context_0 because it was loaded in Step 1
    # ----------------------------------------------------------------
    print("\n\n>>> PROCESSING VARIABLE 2: STUDY DESIGN CLASSIFICATION")
    var2_instruction = (
        "Refer to 'context_0'. Analyze the 'Study Design' or 'Methodology' section. "
        "Classify this study as either 'Interventional' (Clinical Trial) or 'Observational'. "
        "Output strictly one word: 'Interventional' or 'Observational'."
    )
    response_2 = rlm.completion(
        prompt=var2_instruction,  # <--- Lightweight prompt is fine now
    )
    print(f"\n[RLM OUTPUT]: {response_2.response}")
    print(f"[GROUND TRUTH]: Observational")

    # ----------------------------------------------------------------
    # Variable 3: Conflict of Interest
    # ----------------------------------------------------------------
    print("\n\n>>> PROCESSING VARIABLE 3: CONFLICT OF INTEREST STATEMENT")
    var3_instruction = (
        "Refer to 'context_0'. Based on the Sponsor identified earlier, draft the "
        "Conflict of Interest text using strictly this UHN template structure:\n"
        "'[Insert Sponsor Name], the sponsor of this study, will reimburse the "
        "hospital and researcher for the costs of doing this study.'\n"
        "Ensure the sponsor name is inserted correctly."
    )
    response_3 = rlm.completion(
        prompt=var3_instruction,
    )
    print(f"\n[RLM OUTPUT]: {response_3.response}")
    print(f"[GROUND TRUTH]: Design Therapeutics, Inc., the sponsor of this study, will reimburse the hospital and researcher for the costs of doing this study.")

    # ----------------------------------------------------------------
    # Variable 4: Primary Objective (Lay Summary)
    # ----------------------------------------------------------------
    print("\n\n>>> PROCESSING VARIABLE 4: PRIMARY OBJECTIVE (LAY SUMMARY)")
    var4_instruction = (
        "Refer to 'context_0'. Locate the 'Primary Objective' of the protocol. "
        "Draft a single sentence starting with 'The purpose of this study is to...' "
        "that explains this objective in simple lay terms (Grade 6 reading level) "
        "for a patient."
    )
    response_4 = rlm.completion(
        prompt=var4_instruction,
    )
    print(f"\n[RLM OUTPUT]: {response_4.response}")
    print(f"[GROUND TRUTH CHECK]: Compare with 'Introduction' section in Ground Truth ICF.")

    # ----------------------------------------------------------------
    # Variable 5: Total Number of Visits
    # ----------------------------------------------------------------
    print("\n\n>>> PROCESSING VARIABLE 5: TOTAL NUMBER OF VISITS")
    var5_instruction = (
        "Refer to 'context_0'. Find the 'Schedule of Assessments' or 'Schedule of Activities' table. "
        "Count the total number of distinct visits a participant must attend (including Screening, "
        "Treatment/Observation visits, and Follow-up). "
        "Return just the integer number."
    )
    response_5 = rlm.completion(
        prompt=var5_instruction,
    )
    print(f"\n[RLM OUTPUT]: {response_5.response}")
    print(f"[GROUND TRUTH CHECK]: Verify count against Protocol Schedule of Assessments table.")

    # Cleanup environment
    rlm.close()
    print("--- PIPELINE COMPLETE ---")

if __name__ == "__main__":
    if not os.path.exists("data"):
        os.makedirs("data")
        print("Created 'data' directory. Please place 'protocol_example.docx' there before running.")
    else:
        run_icf_extraction()