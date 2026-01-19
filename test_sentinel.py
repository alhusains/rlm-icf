import os
import sys
import json
import re
from pypdf import PdfReader
from rlm import RLM

# ------------------------------------------------------------------
# 1. HELPER: Load PDF Text
# ------------------------------------------------------------------
def load_pdf_text(filepath):
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        sys.exit(1)
    try:
        reader = PdfReader(filepath)
        full_text = []
        for i, page in enumerate(reader.pages[:50]): 
            text = page.extract_text()
            if text:
                full_text.append(f"--- PAGE {i+1} ---\n{text}")
        return "\n".join(full_text)
    except Exception as e:
        print(f"Error reading PDF: {e}")
        sys.exit(1)

# ------------------------------------------------------------------
# 2. CONFIGURATION
# ------------------------------------------------------------------
PROTOCOL_PATH = "data/Prot_000.pdf"
MODEL_NAME = "gpt-5.1"

# ------------------------------------------------------------------
# 3. SENTINEL PROMPT
# ------------------------------------------------------------------
SENTINEL_PROMPT = """
You are a strict Clinical Data Auditor.

TARGET VARIABLE: "{variable_name}"
INSTRUCTION: {instruction}

DATA SOURCE:
The protocol text is provided in 'context_0'. 
Use python to search it.

RESPONSE FORMAT:
Output strictly VALID JSON.
{{
    "answer": "Extracted value (Grade 6 level)",
    "quote": "Direct verbatim quote",
    "page_ref": "Page #",
    "status": "FOUND" | "NOT_FOUND"
}}

RULES:
1. If not found, status="NOT_FOUND".
2. DO NOT GUESS.
3. Start response with '{{'.
"""

# ------------------------------------------------------------------
# 4. TEST VARIABLES
# ------------------------------------------------------------------
TEST_VARIABLES = [
    {"name": "Study Title", "instruction": "Extract the full formal title."},
    {"name": "Protocol Number", "instruction": "Identify the Sponsor's Protocol Number."},
    {"name": "Investigational Drug", "instruction": "Identify the name of the study drug."},
    {"name": "COG Number", "instruction": "Identify the COG study number."},
    {"name": "Study Phase", "instruction": "What Phase is this clinical trial?"},
    {"name": "Target Population", "instruction": "Summarize the patient population."},
    {"name": "Amendment Date", "instruction": "Identify the release date of 'Amendment 5'."},
    # Traps
    {"name": "Parking Reimbursement", "instruction": "Find the parking reimbursement amount ($)."},
    {"name": "Dr. Clara Chan", "instruction": "Find the phone number for PI Dr. Clara Chan."},
    {"name": "Washout Period", "instruction": "Identify the placebo washout period duration."}
]

# ------------------------------------------------------------------
# 5. EXECUTION LOOP
# ------------------------------------------------------------------
def run_stateless_sentinel():
    print(f"--- UHN STATELESS SENTINEL TEST ---")
    
    print(f"Loading PDF from: {PROTOCOL_PATH}")
    # Load into memory ONCE
    protocol_content = load_pdf_text(PROTOCOL_PATH)
    print(f"Protocol loaded ({len(protocol_content)} chars).")

    for i, var in enumerate(TEST_VARIABLES, start=1):
        print(f"\n[{i}/10] Checking: {var['name']}...")

        # Initialize FRESH RLM for every question
        # persistent=False (default) means it forgets everything after completion()
        rlm = RLM(
            backend="openai",
            backend_kwargs={"model_name": MODEL_NAME},
            environment="local",
            verbose=True,  # You can see the logs fresh every time
            max_iterations=10
        )
        
        instruction_text = SENTINEL_PROMPT.format(
            variable_name=var['name'],
            instruction=var['instruction']
        )
        
        # We pass the protocol content EVERY time.
        # This initializes a fresh 'context_0' for just this question.
        result = rlm.completion(
            prompt=protocol_content, 
            root_prompt=instruction_text
        )
        
        expect = "NOT_FOUND" if i > 7 else "FOUND"
        validate_and_print(result.response, var['name'], expected_status=expect)

def validate_and_print(raw_response, var_name, expected_status):
    GREEN = "\033[92m"
    RED = "\033[91m"
    RESET = "\033[0m"

    try:
        json_match = re.search(r'\{.*\}', raw_response.replace('\n', ' '), re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            status = data.get("status", "UNKNOWN")
            answer = data.get("answer", "N/A")
            
            if status == expected_status:
                print(f"  Status: {status} --> {GREEN}[PASS]{RESET}")
            else:
                print(f"  Status: {status} (Expected {expected_status}) --> {RED}[FAIL]{RESET}")
            
            if status == "FOUND":
                print(f"  Answer: {answer}")

        else:
            raise ValueError("No JSON")

    except:
        print(f"  {RED}Error parsing JSON{RESET}")
        print(f"  Raw: {raw_response[:100]}...")

if __name__ == "__main__":
    if not os.path.exists("data"):
        os.makedirs("data")
    run_stateless_sentinel()