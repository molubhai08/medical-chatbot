import argparse
import anthropic
from transformers import pipeline
import groq
import openai, re, random, time, json, replicate, os
import base64
import tiktoken


llama2_url = "meta/llama-2-70b-chat"
llama3_url = "meta/meta-llama-3-70b-instruct"
mixtral_url = "mistralai/mixtral-8x7b-instruct-v0.1"
os.environ["GROQ_API_KEY"] = "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"

def load_huggingface_model(model_name):
    pipe = pipeline("text-generation", model=model_name, device_map="auto")
    return pipe

def inference_huggingface(prompt, pipe):
    response = pipe(prompt, max_new_tokens=100)[0]["generated_text"]
    response = response.replace(prompt, "")
    return response


import os
import time
import re
import base64
import json
import openai
import groq
import replicate
import anthropic
import tiktoken

llama2_url = "meta/llama-2-70b-chat"
llama3_url = "meta/meta-llama-3-70b-instruct"
mixtral_url = "mistralai/mixtral-8x7b-instruct-v0.1"

def query_model(model_str, prompt, system_prompt, tries=3, timeout=5.0, image_requested=False, scene=None, max_prompt_len=2**14, clip_prompt=False):
    if model_str not in [
        "gpt4", "gpt3.5", "gpt4o", "gpt-4o-mini", "gpt4v", "claude3.5sonnet", "o1-preview",
        "llama-2-70b-chat", "llama-3-70b-instruct", "mixtral-8x7b",
        "llama3-groq", "mixtral-groq", "gemma-groq", 
        "llama4-scout-groq", "llama4-maverick-groq"
    ] and "_HF" not in model_str:
        raise Exception(f"No model by the name {model_str}")

    for _ in range(tries):
        if clip_prompt:
            prompt = prompt[:max_prompt_len]
        try:
            # ========== OPENAI IMAGE MODELS ==========
            if image_requested and model_str in ["gpt4v", "gpt4o", "gpt-4o-mini"]:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", 
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": scene.image_url}},
                        ],
                    },
                ]
                model_map = {
                    "gpt4v": "gpt-4-vision-preview",
                    "gpt-4o-mini": "gpt-4o-mini",
                    "gpt4o": "gpt-4o",
                }
                response = openai.ChatCompletion.create(
                    model=model_map[model_str],
                    messages=messages,
                    temperature=0.05,
                    max_tokens=200,
                )
                return response["choices"][0]["message"]["content"].strip()

            # ========== OPENAI TEXT MODELS ==========
            if model_str in ["gpt4", "gpt3.5", "gpt4v", "gpt-4o-mini", "gpt4o"]:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ]
                model_map = {
                    "gpt4": "gpt-4-turbo-preview",
                    "gpt3.5": "gpt-3.5-turbo",
                    "gpt4v": "gpt-4-vision-preview",
                    "gpt4o": "gpt-4o",
                    "gpt-4o-mini": "gpt-4o-mini",
                }
                response = openai.ChatCompletion.create(
                    model=model_map[model_str],
                    messages=messages,
                    temperature=0.05,
                    max_tokens=200,
                )
                return re.sub(r"\s+", " ", response["choices"][0]["message"]["content"])

            # ========== OPENAI O1 Preview ==========
            if model_str == "o1-preview":
                response = openai.ChatCompletion.create(
                    model="o1-preview-2024-09-12",
                    messages=[{"role": "user", "content": system_prompt + prompt}],
                )
                return re.sub(r"\s+", " ", response["choices"][0]["message"]["content"])

            # ========== GROQ MODELS ==========
            if model_str in ["llama3-groq", "mixtral-groq", "gemma-groq"]:
                groq_client = groq.Groq(api_key=os.environ["GROQ_API_KEY"])
                model_mapping = {
                    "llama3-groq": "llama3-70b-8192",
                    "mixtral-groq": "mixtral-8x7b-32768",
                    "gemma-groq": "gemma-7b-it",
                }
                response = groq_client.chat.completions.create(
                    model=model_mapping[model_str],
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.05,
                    max_tokens=200,
                )
                return re.sub(r"\s+", " ", response.choices[0].message.content)

            if model_str in ["llama4-scout-groq", "llama4-maverick-groq"]:
                groq_client = groq.Groq(api_key=os.environ["GROQ_API_KEY"])
                model_map = {
                    "llama4-scout-groq": "meta-llama/llama-4-scout-17b-16e-instruct",
                    "llama4-maverick-groq": "meta-llama/llama-4-maverick-17b-128e-instruct",
                }

                if image_requested and scene and hasattr(scene, "image_url"):
                    image_path = scene.image_url
                    if not os.path.exists(image_path):
                        raise FileNotFoundError(f"Image not found: {image_path}")
                    with open(image_path, "rb") as img:
                        b64 = base64.b64encode(img.read()).decode("utf-8")
                    user_content = [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
                    ]
                else:
                    user_content = prompt

                response = groq_client.chat.completions.create(
                    model=model_map[model_str],
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=0.05,
                    max_tokens=400,
                )
                return re.sub(r"\s+", " ", response.choices[0].message.content)

            # ========== REPLICATE MODELS ==========
            if model_str == "llama-2-70b-chat":
                output = replicate.run(llama2_url, input={
                    "prompt": prompt,
                    "system_prompt": system_prompt,
                    "max_new_tokens": 200,
                })
                return re.sub(r"\s+", " ", ''.join(output))

            if model_str == "mixtral-8x7b":
                output = replicate.run(mixtral_url, input={
                    "prompt": prompt,
                    "system_prompt": system_prompt,
                    "max_new_tokens": 75,
                })
                return re.sub(r"\s+", " ", ''.join(output))

            if model_str == "llama-3-70b-instruct":
                output = replicate.run(llama3_url, input={
                    "prompt": prompt,
                    "system_prompt": system_prompt,
                    "max_new_tokens": 200,
                })
                return re.sub(r"\s+", " ", ''.join(output))

            # ========== ANTHROPIC ==========
            if model_str == "claude3.5sonnet":
                client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
                message = client.messages.create(
                    model="claude-3-5-sonnet-20240620",
                    system=system_prompt,
                    max_tokens=256,
                    messages=[{"role": "user", "content": prompt}],
                )
                return json.loads(message.to_json())["content"][0]["text"]

            # ========== HF Placeholder ==========
            if "HF_" in model_str:
                raise Exception("HF model support is not yet implemented.")

        except Exception as e:
            print(f"⚠️ Query failed: {str(e)}")
            time.sleep(timeout)
            continue

    raise Exception("Max retries: timeout")




class ScenarioMedQA:
    def __init__(self, scenario_dict) -> None:
        self.scenario_dict = scenario_dict
        self.tests = scenario_dict["OSCE_Examination"]["Test_Results"]
        self.diagnosis = scenario_dict["OSCE_Examination"]["Correct_Diagnosis"]
        self.patient_info  = scenario_dict["OSCE_Examination"]["Patient_Actor"]
        self.examiner_info  = scenario_dict["OSCE_Examination"]["Objective_for_Doctor"]
        self.physical_exams = scenario_dict["OSCE_Examination"]["Physical_Examination_Findings"]
    
    def patient_information(self) -> dict:
        return self.patient_info

    def examiner_information(self) -> dict:
        return self.examiner_info
    
    def exam_information(self) -> dict:
        exams = self.physical_exams
        exams["tests"] = self.tests
        return exams
    
    def diagnosis_information(self) -> dict:
        return self.diagnosis


class ScenarioLoaderMedQA:
    def __init__(self) -> None:
        with open("agentclinic_medqa.jsonl", "r") as f:
            self.scenario_strs = [json.loads(line) for line in f]
        self.scenarios = [ScenarioMedQA(_str) for _str in self.scenario_strs]
        self.num_scenarios = len(self.scenarios)
    
    def sample_scenario(self):
        return self.scenarios[random.randint(0, len(self.scenarios)-1)]
    
    def get_scenario(self, id):
        if id is None: return self.sample_scenario()
        return self.scenarios[id]
        


class ScenarioMedQAExtended:
    def __init__(self, scenario_dict) -> None:
        self.scenario_dict = scenario_dict
        self.tests = scenario_dict["OSCE_Examination"]["Test_Results"]
        self.diagnosis = scenario_dict["OSCE_Examination"]["Correct_Diagnosis"]
        self.patient_info  = scenario_dict["OSCE_Examination"]["Patient_Actor"]
        self.examiner_info  = scenario_dict["OSCE_Examination"]["Objective_for_Doctor"]
        self.physical_exams = scenario_dict["OSCE_Examination"]["Physical_Examination_Findings"]
    
    def patient_information(self) -> dict:
        return self.patient_info

    def examiner_information(self) -> dict:
        return self.examiner_info
    
    def exam_information(self) -> dict:
        exams = self.physical_exams
        exams["tests"] = self.tests
        return exams
    
    def diagnosis_information(self) -> dict:
        return self.diagnosis


class ScenarioLoaderMedQAExtended:
    def __init__(self) -> None:
        with open("agentclinic_medqa_extended.jsonl", "r") as f:
            self.scenario_strs = [json.loads(line) for line in f]
        self.scenarios = [ScenarioMedQAExtended(_str) for _str in self.scenario_strs]
        self.num_scenarios = len(self.scenarios)
    
    def sample_scenario(self):
        return self.scenarios[random.randint(0, len(self.scenarios)-1)]
    
    def get_scenario(self, id):
        if id is None: return self.sample_scenario()
        return self.scenarios[id]
        


class ScenarioMIMICIVQA:
    def __init__(self, scenario_dict) -> None:
        self.scenario_dict = scenario_dict
        self.tests = scenario_dict["OSCE_Examination"]["Test_Results"]
        self.diagnosis = scenario_dict["OSCE_Examination"]["Correct_Diagnosis"]
        self.patient_info  = scenario_dict["OSCE_Examination"]["Patient_Actor"]
        self.examiner_info  = scenario_dict["OSCE_Examination"]["Objective_for_Doctor"]
        self.physical_exams = scenario_dict["OSCE_Examination"]["Physical_Examination_Findings"]
    
    def patient_information(self) -> dict:
        return self.patient_info

    def examiner_information(self) -> dict:
        return self.examiner_info
    
    def exam_information(self) -> dict:
        exams = self.physical_exams
        exams["tests"] = self.tests
        return exams
    
    def diagnosis_information(self) -> dict:
        return self.diagnosis


class ScenarioLoaderMIMICIV:
    def __init__(self) -> None:
        with open("agentclinic_mimiciv.jsonl", "r") as f:
            self.scenario_strs = [json.loads(line) for line in f]
        self.scenarios = [ScenarioMIMICIVQA(_str) for _str in self.scenario_strs]
        self.num_scenarios = len(self.scenarios)
    
    def sample_scenario(self):
        return self.scenarios[random.randint(0, len(self.scenarios)-1)]
    
    def get_scenario(self, id):
        if id is None: return self.sample_scenario()
        return self.scenarios[id]


class ScenarioNEJMExtended:
    def __init__(self, scenario_dict) -> None:
        self.scenario_dict = scenario_dict 
        self.question = scenario_dict["question"] 
        self.image_url = scenario_dict["image_url"] 
        self.diagnosis = [_sd["text"] 
            for _sd in scenario_dict["answers"] if _sd["correct"]][0]
        self.patient_info = scenario_dict["patient_info"]
        self.physical_exams = scenario_dict["physical_exams"]

    def patient_information(self) -> str:
        patient_info = self.patient_info
        return patient_info

    def examiner_information(self) -> str:
        return "What is the most likely diagnosis?"
    
    def exam_information(self) -> str:
        exams = self.physical_exams
        return exams
    
    def diagnosis_information(self) -> str:
        return self.diagnosis


class ScenarioLoaderNEJMExtended:
    def __init__(self) -> None:
        with open("agentclinic_nejm_extended.jsonl", "r") as f:
            self.scenario_strs = [json.loads(line) for line in f]
        self.scenarios = [ScenarioNEJMExtended(_str) for _str in self.scenario_strs]
        self.num_scenarios = len(self.scenarios)
    
    def sample_scenario(self):
        return self.scenarios[random.randint(0, len(self.scenarios)-1)]
    
    def get_scenario(self, id):
        if id is None: return self.sample_scenario()
        return self.scenarios[id]


class ScenarioNEJM:
    def __init__(self, scenario_dict) -> None:
        self.scenario_dict = scenario_dict 
        self.question = scenario_dict["question"] 
        self.image_url = scenario_dict["image_url"] 
        self.diagnosis = [_sd["text"] 
            for _sd in scenario_dict["answers"] if _sd["correct"]][0]
        self.patient_info = scenario_dict["patient_info"]
        self.physical_exams = scenario_dict["physical_exams"]

    def patient_information(self) -> str:
        patient_info = self.patient_info
        return patient_info

    def examiner_information(self) -> str:
        return "What is the most likely diagnosis?"
    
    def exam_information(self) -> str:
        exams = self.physical_exams
        return exams
    
    def diagnosis_information(self) -> str:
        return self.diagnosis


class ScenarioLoaderNEJM:
    def __init__(self) -> None:
        with open("agentclinic_nejm.jsonl", "r") as f:
            self.scenario_strs = [json.loads(line) for line in f]
        self.scenarios = [ScenarioNEJM(_str) for _str in self.scenario_strs]
        self.num_scenarios = len(self.scenarios)
    
    def sample_scenario(self):
        return self.scenarios[random.randint(0, len(self.scenarios)-1)]
    
    def get_scenario(self, id):
        if id is None: return self.sample_scenario()
        return self.scenarios[id]


class PatientAgent:
    def __init__(self, scenario, backend_str="gpt4", bias_present=None) -> None:
        # disease of patient, or "correct answer"
        self.disease = ""
        # symptoms that patient presents
        self.symptoms = ""
        # conversation history between doctor and patient
        self.agent_hist = ""
        # language model backend for patient agent
        self.backend = backend_str
        # presentation of any form of bias
        self.bias_present = (None if bias_present == "None" else bias_present)
        # sample initial question from dataset
        self.scenario = scenario
        self.reset()
        self.pipe = None

        self.biases = ["recency", "frequency", "false_consensus", "self_diagnosis", "gender", "race", "sexual_orientation", "cultural", "education", "religion", "socioeconomic"]

    def generate_bias(self) -> str:
        """ 
        ================
        Cognitive biases 
        ================
        """
        if self.bias_present == "recency":
            return "\nRecently, you had a friend with similar symptoms to yours that was diagnosed with cancer. This affects how you interact with the doctor.\n"
        elif self.bias_present == "frequency":
            return "\nYou recently learned that most frequently, doctors diagnose this as cancer based on past cases. This affects how you interact with the doctor.\n"
        elif self.bias_present == "false_consensus":
            return "\nMost of your friends who heard about your symptoms believe that you have cancer. This affects how you interact with the doctor.\n"
        elif self.bias_present == "self_diagnosis":
            return "\nYou recently performed a thorough investigation of your symptoms online and have come to the conclusion that you should be diagnosed with cancer. While the doctor may believe otherwise you remain convinced you have cancer. This affects how you interact with the doctor.\n"
        elif self.bias_present == "gender":
            return "\nYou are uncomfortable with your doctor because you find out that they are a particular gender and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "race":
            return "\nYou are uncomfortable with your doctor because you find out that they are a particular race and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "sexual_orientation":
            return "\nYou are uncomfortable with your doctor because you find out that they are a particular sexual orientation and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "cultural":
            return "\nYou are uncomfortable with your doctor because you find out that they are from a particular culture and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "education":
            return "\nYou are uncomfortable with your doctor because you find out that they went to a low ranked medical school and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "religion":
            return "\nYou are uncomfortable with your doctor because you find out that they believe in a particular religion and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "socioeconomic":
            return "\nYou are uncomfortable with your doctor because you find out that they are from a particular socioeconomic background and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present is None:
            pass
        else:
            print("BIAS TYPE {} NOT SUPPORTED, ignoring bias...".format(self.bias_present))
        return ""

    def inference_patient(self, question) -> str:
        answer = query_model(self.backend, "\nHere is a history of your dialogue: " + self.agent_hist + "\n Here was the doctor response: " + question + "Now please continue your dialogue\nPatient: ", self.system_prompt())
        self.agent_hist += question + "\n\n" + answer + "\n\n"
        return answer

    def system_prompt(self) -> str:
        bias_prompt = ""
        if self.bias_present is not None:
            bias_prompt = self.generate_bias()
        base = """You are a patient in a clinic who only responds in the form of dialogue. You are being inspected by a doctor who will ask you questions and will perform exams on you in order to understand your disease. Your answer will only be 1-3 sentences in length."""
        symptoms = "\n\nBelow is all of your information. {}. \n\n Remember, you must not reveal your disease explicitly but may only convey the symptoms you have in the form of dialogue if you are asked.".format(self.symptoms)
        return base + bias_prompt + symptoms
    
    def reset(self) -> None:
        self.agent_hist = ""
        self.symptoms = self.scenario.patient_information()

    def add_hist(self, hist_str) -> None:
        self.agent_hist += hist_str + "\n\n"


class DoctorAgent:
    def __init__(self, scenario, backend_str="gpt4", max_infs=20, bias_present=None, img_request=False) -> None:
        # number of inference calls to the doctor
        self.infs = 0
        # maximum number of inference calls to the doctor
        self.MAX_INFS = max_infs
        # conversation history between doctor and patient
        self.agent_hist = ""
        # presentation information for doctor
        self.presentation = ""
        # language model backend for doctor agent
        self.backend = backend_str
        # presentation of any form of bias
        self.bias_present = (None if bias_present == "None" else bias_present)
        # prepare initial conditions for LLM
        self.scenario = scenario
        self.reset()
        self.pipe = None
        self.img_request = img_request
        self.biases = ["recency", "frequency", "false_consensus", "confirmation", "status_quo", "gender", "race", "sexual_orientation", "cultural", "education", "religion", "socioeconomic"]

    def generate_bias(self) -> str:
        """ 
        ================
        Cognitive biases 
        ================
        """
        if self.bias_present == "recency":
            return "\nRecently, you learned that a doctor with similar symptoms to the patient that diagnosed their patient with cancer. This affects how you interact with the patient.\n"
        elif self.bias_present == "frequency":
            return "\nYou recently learned that most frequently, doctors diagnose patients with similar symptoms with cancer based on past cases. This affects how you interact with the patient.\n"
        elif self.bias_present == "false_consensus":
            return "\nMost of your colleagues who heard about the patient's symptoms believe that they have cancer. This affects how you interact with the patient.\n"
        elif self.bias_present == "status_quo":
            return "\nYour hosptial has been diagnosing these patient's symptoms with cancer for the past 10 years. This affects how you interact with the patient.\n"
        elif self.bias_present == "confirmation":
            return "\nYou are initially confident that the patient has cancer. This affects how you interact with the patient.\n"
        elif self.bias_present == "gender":
            return "\nYou are uncomfortable with your patient because you find out that they are a particular gender and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "race":
            return "\nYou are uncomfortable with your patient because you find out that they are a particular race and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "sexual_orientation":
            return "\nYou are uncomfortable with your patient because you find out that they are a particular sexual orientation and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "cultural":
            return "\nYou are uncomfortable with your patient because you find out that they are from a particular culture and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "education":
            return "\nYou are uncomfortable with your patient because you find out that they are uneducated and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "religion":
            return "\nYou are uncomfortable with your patient because you find out that they believe in a particular religion and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "socioeconomic":
            return "\nYou are uncomfortable with your patient because you find out that they are from a particular socioeconomic background and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present is None:
            pass
        else:
            print("BIAS TYPE {} NOT SUPPORTED, ignoring bias...".format(self.bias_present))
        return ""

    def inference_doctor(self, question, image_requested=False) -> str:
        answer = str()
        if self.infs >= self.MAX_INFS: return "Maximum inferences reached"
        answer = query_model(self.backend, "\nHere is a history of your dialogue: " + self.agent_hist + "\n Here was the patient response: " + question + "Now please continue your dialogue\nDoctor: ", self.system_prompt(), image_requested=image_requested, scene=self.scenario)
        self.agent_hist += question + "\n\n" + answer + "\n\n"
        self.infs += 1
        return answer

    def system_prompt(self) -> str:
        bias_prompt = ""
        if self.bias_present is not None:
            bias_prompt = self.generate_bias()
        base = "You are a doctor named Dr. Agent who only responds in the form of dialogue. You are inspecting a patient who you will ask questions in order to understand their disease. You are only allowed to ask {} questions total before you must make a decision. You have asked {} questions so far. You can request test results using the format \"REQUEST TEST: [test]\". For example, \"REQUEST TEST: Chest_X-Ray\". Your dialogue will only be 1-3 sentences in length. Once you have decided to make a diagnosis please type \"DIAGNOSIS READY: [diagnosis here]\"".format(self.MAX_INFS, self.infs) + ("You may also request medical images related to the disease to be returned with \"REQUEST IMAGES\"." if self.img_request else "")
        presentation = "\n\nBelow is all of the information you have. {}. \n\n Remember, you must discover their disease by asking them questions. You are also able to provide exams.".format(self.presentation)
        return base + bias_prompt + presentation

    def reset(self) -> None:
        self.agent_hist = ""
        self.presentation += self.scenario.examiner_information()
        self.presentation += " , Patient Info" + self.scenario.get_patient_info()
        self.presentation += " , Physical Exam results" + self.scenario.get_phys_exams()

class ManualNEJMScenario:
    def __init__(self, scenario_dict):
        self.data = scenario_dict
        self.image_url = self.data.get("image_url")


    def get_question(self):
        return self.data["question"]

    def get_image_url(self):
        return self.data["image_url"]

    def get_answers(self):
        return self.data["answers"]
    
    def get_phys_exams(self):
        return self.data["physical_exams"]

    def get_patient_info(self):
        return self.data["patient_info"]

    def get_physical_exams(self):
        return self.data.get("physical_exams", "")

    def get_test_results(self):
        return self.data.get("test_results", {})

    def exam_information(self):  # 👈 Needed by MeasurementAgent
        return self.data.get("test_results", {})
    
    def examiner_information(self):
        return self.data["question"]

    def patient_information(self):
        return self.data["patient_info"]

    def diagnosis_information(self):
        for ans in self.data["answers"]:
            if ans["correct"]:
                return ans["text"]
        return "Unknown"

class MeasurementAgent:
    def __init__(self, scenario, backend_str="gpt4") -> None:
        # conversation history between doctor and patient
        self.agent_hist = ""
        # presentation information for measurement 
        self.presentation = ""
        # language model backend for measurement agent
        self.backend = backend_str
        # prepare initial conditions for LLM
        self.scenario = scenario
        self.pipe = None
        self.reset()

    def inference_measurement(self, question) -> str:
        answer = str()
        answer = query_model(self.backend, "\nHere is a history of the dialogue: " + self.agent_hist + "\n Here was the doctor measurement request: " + question, self.system_prompt())
        self.agent_hist += question + "\n\n" + answer + "\n\n"
        return answer

    def system_prompt(self) -> str:
        base = "You are an measurement reader who responds with medical test results. Please respond in the format \"RESULTS: [results here]\""
        presentation = "\n\nBelow is all of the information you have. {}. \n\n If the requested results are not in your data then you can respond with NORMAL READINGS.".format(self.information)
        return base + presentation
    
    def add_hist(self, hist_str) -> None:
        self.agent_hist += hist_str + "\n\n"

    def reset(self) -> None:
        self.agent_hist = ""
        self.presentation = self.scenario.examiner_information()



def compare_results(diagnosis, correct_diagnosis, moderator_llm, mod_pipe):
    answer = query_model(moderator_llm, "\nHere is the correct diagnosis: " + correct_diagnosis + "\n Here was the doctor dialogue: " + diagnosis + "\nAre these the same?", "You are responsible for determining if the corrent diagnosis and the doctor diagnosis are the same disease. Please respond only with Yes or No. Nothing else.")
    return answer.lower()


def main(api_key, replicate_api_key, inf_type, doctor_bias, patient_bias, doctor_llm, patient_llm, measurement_llm, moderator_llm, num_scenarios, dataset, img_request, total_inferences, anthropic_api_key=None):
    openai.api_key = api_key
    anthropic_llms = ["claude3.5sonnet"]
    replicate_llms = ["llama-3-70b-instruct", "llama-2-70b-chat", "mixtral-8x7b"]
    if patient_llm in replicate_llms or doctor_llm in replicate_llms:
        os.environ["REPLICATE_API_TOKEN"] = replicate_api_key
    if doctor_llm in anthropic_llms:
        os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key

    # Load MedQA, MIMICIV or NEJM agent case scenarios
    if dataset == "MedQA":
        scenario_loader = ScenarioLoaderMedQA()
    elif dataset == "MedQA_Ext":
        scenario_loader = ScenarioLoaderMedQAExtended()
    elif dataset == "NEJM":
        scenario_loader = ScenarioLoaderNEJM()
    elif dataset == "NEJM_Ext":
        scenario_loader = ScenarioLoaderNEJMExtended()
    elif dataset == "MIMICIV":
        scenario_loader = ScenarioLoaderMIMICIV()
    else:
        raise Exception("Dataset {} does not exist".format(str(dataset)))
    total_correct = 0
    total_presents = 0

    s = {
    "image_url": r"C:\Users\SARTHAK\Desktop\RemoteCare\day6\AgentClinic\try.png",
    "question": (
        "A 43-year-old woman presented to the dermatology clinic with an 8-year history of yellow-brown spots on her shins. "
        "The lesions had been asymptomatic, and she had not sought care for them until they had grown in size. "
        "She had no history of diabetes mellitus, hypertension, or thyroid disease. "
        "On physical examination, atrophic yellow-brown plaques with telangiectasias and irregular violaceous borders were observed on both shins. "
        "A skin biopsy of the right shin was performed. "
        "Histopathological analysis showed several layers of necrobiosis within the dermis, perivascular inflammatory-cell infiltrates, "
        "collagen degeneration, and findings consistent with granulomatous dermatitis. "
        "Which of the following is the most likely diagnosis?"
    ),
    "patient_info": (
        "For your role as a patient, here's what you need to know:\n"
        "- You are a 43-year-old woman.\n"
        "- For the past 8 years, you've noticed yellow-brown spots on your shins. "
        "These spots have grown in size over time but haven't caused you any discomfort.\n"
        "- You've never been diagnosed with diabetes, high blood pressure, or thyroid problems.\n"
        "- Recently, you decided to seek medical advice because the spots on your shins have gotten bigger.\n"
        "- During the examination, the doctor noted that you have atrophic yellow-brown plaques with a network of small blood vessels (telangiectasias) "
        "and irregular purple edges on both shins.\n"
        "- You underwent a skin biopsy on your right shin to help diagnose your condition."
    ),
    "physical_exams": (
        "Skin biopsy of the right shin: Histopathological analysis revealed several layers of necrobiosis within the dermis, "
        "perivascular inflammatory-cell infiltrates, collagen degeneration, and findings consistent with granulomatous dermatitis."
    ),
    "answers": [
        {"text": "Cutaneous sarcoidosis", "correct": False},
        {"text": "Granuloma annulare", "correct": False},
        {"text": "Necrobiosis lipoidica", "correct": True},
        {"text": "Pigmented purpuric dermatosis", "correct": False},
        {"text": "Stasis purpuric dermatosis", "correct": False}
    ]
}   
    # Load MedQA, MIMICIV or NEJM agent case scenarios
    scenario_loader = [ManualNEJMScenario(s)]

    # Pipeline for huggingface models
    if "HF_" in moderator_llm:
        pipe = load_huggingface_model(moderator_llm.replace("HF_", ""))
    else:
        pipe = None
    if num_scenarios is None: num_scenarios = scenario_loader.num_scenarios
    for _scenario_id in range(0, num_scenarios):
        total_presents += 1
        pi_dialogue = str()
        # Initialize scenarios (MedQA/NEJM)
        scenario = scenario_loader[_scenario_id]
        # Initialize agents
        meas_agent = MeasurementAgent(
            scenario=scenario,
            backend_str=measurement_llm)
        patient_agent = PatientAgent(
            scenario=scenario, 
            bias_present=patient_bias,
            backend_str=patient_llm)
        doctor_agent = DoctorAgent(
            scenario=scenario, 
            bias_present=doctor_bias,
            backend_str=doctor_llm,
            max_infs=total_inferences, 
            img_request=img_request)

        doctor_dialogue = ""
        for _inf_id in range(total_inferences):
            # Check for medical image request
            imgs = True 

            if img_request:
                imgs = "REQUEST IMAGES" in doctor_dialogue

            # Check if final inference
            if _inf_id == total_inferences - 1:
                pi_dialogue += "This is the final question. Please provide a diagnosis.\n"
            # Obtain doctor dialogue (human or llm agent)
            if inf_type == "human_doctor":
                doctor_dialogue = input("\nQuestion for patient: ")
            else: 
                doctor_dialogue = doctor_agent.inference_doctor(pi_dialogue, image_requested=imgs)
            print("Doctor [{}%]:".format(int(((_inf_id+1)/total_inferences)*100)), doctor_dialogue)
            # Doctor has arrived at a diagnosis, check correctness
            if "DIAGNOSIS READY" in doctor_dialogue:
                correctness = compare_results(doctor_dialogue, scenario.diagnosis_information(), moderator_llm, pipe) == "yes"
                if correctness: total_correct += 1
                print("\nCorrect answer:", scenario.diagnosis_information())
                print("Scene {}, The diagnosis was ".format(_scenario_id), "CORRECT" if correctness else "INCORRECT", int((total_correct/total_presents)*100))
                break
            # Obtain medical exam from measurement reader
            if "REQUEST TEST" in doctor_dialogue:
                pi_dialogue = meas_agent.inference_measurement(doctor_dialogue,)
                print("Measurement [{}%]:".format(int(((_inf_id+1)/total_inferences)*100)), pi_dialogue)
                patient_agent.add_hist(pi_dialogue)
            # Obtain response from patient
            else:
                if inf_type == "human_patient":
                    pi_dialogue = input("\nResponse to doctor: ")
                else:
                    pi_dialogue = patient_agent.inference_patient(doctor_dialogue)
                print("Patient [{}%]:".format(int(((_inf_id+1)/total_inferences)*100)), pi_dialogue)
                meas_agent.add_hist(pi_dialogue)
            # Prevent API timeouts
            time.sleep(1.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Medical Diagnosis Simulation CLI')
    parser.add_argument('--openai_api_key', type=str, required=False, help='OpenAI API Key')
    parser.add_argument('--replicate_api_key', type=str, required=False, help='Replicate API Key')
    parser.add_argument('--inf_type', type=str, choices=['llm', 'human_doctor', 'human_patient'], default='llm')
    parser.add_argument('--doctor_bias', type=str, help='Doctor bias type', default='None', choices=["recency", "frequency", "false_consensus", "confirmation", "status_quo", "gender", "race", "sexual_orientation", "cultural", "education", "religion", "socioeconomic"])
    parser.add_argument('--patient_bias', type=str, help='Patient bias type', default='None', choices=["recency", "frequency", "false_consensus", "self_diagnosis", "gender", "race", "sexual_orientation", "cultural", "education", "religion", "socioeconomic"])
    parser.add_argument('--doctor_llm', type=str, default='gpt4')
    parser.add_argument('--patient_llm', type=str, default='gpt4')
    parser.add_argument('--measurement_llm', type=str, default='gpt4')
    parser.add_argument('--moderator_llm', type=str, default='gpt4')
    parser.add_argument('--agent_dataset', type=str, default='MedQA') # MedQA, MIMICIV or NEJM
    parser.add_argument('--doctor_image_request', type=bool, default=False) # whether images must be requested or are provided
    parser.add_argument('--num_scenarios', type=int, default=None, required=False, help='Number of scenarios to simulate')
    parser.add_argument('--total_inferences', type=int, default=20, required=False, help='Number of inferences between patient and doctor')
    parser.add_argument('--anthropic_api_key', type=str, default=None, required=False, help='Anthropic API key for Claude 3.5 Sonnet')
    
    args = parser.parse_args()

    main(args.openai_api_key, args.replicate_api_key, args.inf_type, args.doctor_bias, args.patient_bias, args.doctor_llm, args.patient_llm, args.measurement_llm, args.moderator_llm, args.num_scenarios, args.agent_dataset, args.doctor_image_request, args.total_inferences, args.anthropic_api_key)



# cd "C:\Users\SARTHAK\Desktop\RemoteCare\day6\AgentClinic"
# python agentclinic.py --inf_type llm `
# --doctor_llm llama3-groq `
# --patient_llm llama3-groq `
# --measurement_llm llama3-groq `
# --moderator_llm llama3-groq `
# --num_scenarios 1


# python vision_try.py --inf_type llm --doctor_llm llama4-scout-groq --patient_llm llama3-groq --measurement_llm llama3-groq --moderator_llm llama3-groq --agent_dataset NEJM --doctor_image_request True --num_scenarios 1

# python vision_try.py --inf_type llm --doctor_llm llama4-scout-groq --patient_llm llama3-groq --measurement_llm llama3-groq --moderator_llm llama3-groq --doctor_image_request True --num_scenarios 1
