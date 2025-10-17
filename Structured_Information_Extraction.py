from __future__ import annotations
import dspy
from typing import Optional, List, Dict, TypedDict, Any
from pydantic import Field
import os
import json

# ---- Nested shapes (TypedDicts) ----
class Demographics(TypedDict, total=False):
    age: Optional[int] = Field(None, description="Patient age in years (0-130).")
    sex: Optional[str] = Field(None, description='Sex: "Male", "Female", "Other", or "Unknown".')

class Visit(TypedDict, total=False):
    date_time: Optional[str]  = Field(None, description="ISO 8601 or None")
    type: Optional[str] = Field(None, description="The type of visit")

class ConversationMetadata(TypedDict, total=False):
    timestamps: List[Any]    = Field(None, description="keep flexible (dict/list/None). Mostly keep null")# keep flexible (dict/list/None)
    speaker_labels: List[Any]   = Field(None, description="keep flexible (dict/list/None). Mostly keep null")# keep flexible (dict/list/None)

class MedicalIEOutput(TypedDict, total=False):
    patient_identifiers: Optional[Any] = Field(description="Patient Name")
    demographics: Demographics = Field(None, description="Patient demographics. Age and Sex details ")
    visit: Visit = Field(None, description="The type of visit and the date of the visit")
    chief_complaint: Optional[str] = Field(None, description="The chief complaint of the patient")
    onset_duration: Optional[str] = Field(None, description="The duration of the onset of the complaint")
    symptom_description: Optional[str] = Field(None, description="The description of the symptom by the patient")
    aggravating_factors: Optional[str] = Field(None, description="The aggravating factors of the symptom")
    relieving_factors: Optional[str] = Field(None, description="The relieving factors of the symptom")
    associated_symptoms: List[str] = Field(None, description="The associated symptoms of the complaint in a list")
    past_medical_history: Optional[str] = Field(None, description="The past medical history of the patient")
    past_surgical_history: Optional[str] = Field(None, description="The past surgical history of the patient")
    family_history: Optional[str] = Field(None, description="The family history of the patient")
    current_medications: List[str] = Field(None, description="The current medications of the patient")
    allergies: Optional[str] = Field(None, description="The allergies of the patient")
    social_history: List[str] = Field(None, description="The social history of the patient")
    functional_status: Optional[str] = Field(None, description="The functional status of the patient")
    vital_signs: Optional[str] = Field(None, description="The vital signs of the patient. Mostly keep null")    
    examination_findings: Optional[str] = Field(None, description="The examination findings of the patient")
    investigations: List[str] = Field(None, description="The investigations of the patient. Mostly keep null")    
    assessment_primary_diagnosis: Optional[str] = Field(None, description="The primary diagnosis of the patient")
    differential_diagnoses: List[str] = Field(None, description="The differential diagnoses of the patient")
    management_plan: Optional[str] = Field(None, description="The management plan of the patient. Mostly keep null")
    tests_referrals_planned: List[str] = Field(None, description="The tests and referrals planned for the patient. Mostly keep null")
    follow_up_plan: Optional[str] = Field(None, description="The follow-up plan for the patient. Mostly keep null")
    chronology_response_to_treatment: Optional[str] = Field(None, description="The chronology of the response to treatment. Mostly keep null")
    patient_concerns_preferences_consent: Optional[str] = Field(None, description="The patient's concerns and preferences. Mostly keep null")
    safety_issues_red_flags: Optional[str] = Field(None, description="The safety issues and red flags. Mostly keep null")
    coding_terms: Optional[str] = Field(None, description="The coding terms. Mostly keep null")
    conversation_metadata: ConversationMetadata = Field(None, description="The conversation metadata. Mostly keep null if not available")

class MedicalDialogueToStructured(dspy.Signature):
    """
    Extract a neutral, non-diagnostic, English-only clinical summary JSON from a multi-turn
    The extracted information should be available in the dialogue. Name, Age, Sex, Visit dates etc. Ensure you have these details otherwise return null. 
    health worker ↔ patient dialogue. Only use information explicitly stated. Do not hallucinate any information.
    - Keep text concise. Do not invent or infer beyond the provided dialogue.
    Return exactly the following keys (null-allowed if data is not available) and nesting:
    {
      "patient_identifiers": str description="Patient Name",
      "demographics": {"age": int|null description="Patient Age", "sex": "Male"|"Female"|...|null description="Patient Sex"},
      "visit": {"date_time": str|null description="Visit Date", "type": str|null description="Visit Type"},
      "chief_complaint": str|null description="Chief Complaint",
      "associated_symptoms": List[str] description="Associated Symptoms",
      "current_medications": List[str] description="Current Medications",
      "allergies": str|null description="Allergies",
      "social_history": List[str]| description="Social History",
      "functional_status": str|null description="Functional Status",
      "vital_signs": str|null description="Vital Signs",
      "examination_findings": str|null description="Examination Findings",
      "investigations": List[str]|null description="Investigations",
      "assessment_primary_diagnosis": str|null description="Primary Diagnosis",
      "differential_diagnoses": List[str]|null description="Differential Diagnoses",
      "management_plan": str|null description="Management Plan",
      "tests_referrals_planned": List[str]|null description="Tests and Referrals Planned",
      "follow_up_plan": str|null description="Follow Up Plan",
      "chronology_response_to_treatment": str|null description="Chronology of Response to Treatment",
      "patient_concerns_preferences_consent": str|null description="Patient Concerns and Preferences Consent",
      "safety_issues_red_flags": str|null description="Safety Issues and Red Flags",
      "coding_terms": str|null description="Coding Terms",
      "conversation_metadata": {"timestamps": List[str]|null description="Timestamps", "speaker_labels": List[str]|null description="Speaker Labels"    } description="Conversation Metadata"
    } description="Structured Output as a dictionary"       
    """

    # You can pass either raw text or structured turns. Use whichever you have.
    dialogue_text: str = dspy.InputField(desc="The full multi-turn dialogue concatenated into a single string.")
    # OR, if you have structured turns, add this instead/in addition:
    # dialogue_turns: List[Dict[str, str]] = dspy.InputField(desc="List of turns: [{'speaker': 'Patient'|'HW', 'text': '...'}].")

    # One big nested object as a Python dict
    structured: MedicalIEOutput = dspy.OutputField(
        desc="Structured output matching the exact schema above. Use null where info is missing."
    )

# Language Model Configuration

lm = dspy.LM(
    "openai/qwen3-ie:latest",         #  your Ollama model name (from `ollama list`)
    api_base="https://a0yy4b3et3iys4-11434.proxy.runpod.net/v1", # Ollama server URL - if you run locally use http://localhost:11434/v1
    api_key="local",                      # 
    model_type="chat" ,                    # Qwen chat template, 
    cache=False,
    max_tokens=30000
)
dspy.configure(lm=lm)
# Instantiate a predictor with this signature
ie = dspy.Predict(MedicalDialogueToStructured)


folder_location = r"C:\Users\ulliv\Downloads\test_data_release\test_data_release\Kannada\Dialogues"
list_of_files = os.listdir(folder_location)    
i=0
print(len(list_of_files))
output_folder = r"C:\Users\ulliv\Desktop\vinay\code\convert_to_gguf\kannada_output"
list_of_output_files = os.listdir(output_folder)
list_of_output_files = [file.split('.')[0] for file in list_of_output_files]
for file in list_of_files:
    if "structured_information_" + file.split('.')[0] in list_of_output_files:
        print("File already exists",file)
        continue
    else:
        print(file)
        print(i)
        i+=1
        dialogue_text = open(os.path.join(folder_location, file), "r", encoding="utf-8").read()
        dialogue_text = dialogue_text.lower()
        resp = ie(dialogue_text=dialogue_text)
        result = resp.structured
        with open(os.path.join(output_folder, f"structured_information_{file.split('.')[0]}.json"), "w") as f:
            json.dump(result, f, indent=2)