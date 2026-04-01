import os
import concurrent.futures
import time 
from typing import List

from google import genai
from google.genai import types as genai_types

from vlmgineer.prompts.prompt_composer import SinglePrompt
from vlmgineer.prompts.prompt_utils import compose_prompt_and_files

class GeminiAgent():
    def __init__(self, model_name, request_timeout=540):
        ### Parameters ###
        self.API_KEY = os.getenv("GEMINI_API")
        self.client = genai.Client(
            api_key=self.API_KEY,
            http_options=genai_types.HttpOptions(timeout=request_timeout * 1000),
        )
        self.contents = []
        self.session = self.client.chats.create(model=model_name)
    
    def send_chat_message(self, prompt : SinglePrompt, prompt_config=None):
        # upload files
        processed_prompt = compose_prompt_and_files(
            self.client,
            prompt.file_prompts,
            prompt.instruction_prompts
        )
        # send messagge
        response = self.session.send_message(
            message=processed_prompt,
            config=prompt_config,
        )
        return response

class SamplingCollection():
    def __init__(self, n_agents, max_workers, model_name):
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        self.agent_list = [GeminiAgent(model_name) for _ in range(n_agents)]
        self.MAX_WORKERS = max_workers # Max number of threads (adjust based on your system and API rate limits)

    def start_sampling_agent(self, prompts : List[SinglePrompt], prompt_config, timeout=540, stagger_delay=1.0):
        assert len(prompts) == len(self.agent_list), \
            "Number of prompts must match number of agents"
        start_time = time.time()
        results_dict = {} # Dictionary to store results keyed by index

        # Use ThreadPoolExecutor for I/O-bound tasks like API calls
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
            # Stagger submissions to avoid hitting API rate limits simultaneously
            future_to_index = {}
            for i in range(len(self.agent_list)):
                future = executor.submit(self.call_gemini_api, i, prompts[i], prompt_config[i])
                future_to_index[future] = i
                if i < len(self.agent_list) - 1:
                    time.sleep(stagger_delay)

            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_index):
                original_index = future_to_index[future]
                try:
                    # Get the result from the future with timeout
                    call_idx, result_text = future.result(timeout=timeout)
                    results_dict[call_idx] = result_text
                except concurrent.futures.TimeoutError:
                    print(f"Call {original_index} timed out after {timeout} seconds")
                    # We don't add timed-out results to the results dictionary
                except Exception as exc:
                    print(f'Call {original_index} generated an exception: {exc}')
                    # We don't add failed results to the results dictionary

        end_time = time.time()
        
        # Reindex results to be continuous (0, 1, 2, ...) without gaps
        final_results = {}
        for new_idx, old_idx in enumerate(sorted(results_dict.keys())):
            final_results[new_idx] = results_dict[old_idx]
        
        print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")
        print(f"Completed {len(final_results)} out of {len(self.agent_list)} calls")

        return final_results


    def call_gemini_api(self, index: int, prompts, prompt_config, max_retries=3) -> tuple[int, str]:
        """
        Makes a single call to the Gemini API and returns the index and result text.
        Handles potential errors during the API call with retry logic.
        """
        print(f"[Thread {index}] Starting call...")
        
        for attempt in range(max_retries + 1):
            try:
                response = self.agent_list[index].send_chat_message(prompts, prompt_config)
                # Safety check for response structure (can vary based on settings/errors)
                if response and response.text:
                    result_text = response.text
                    print(f"[Thread {index}] Finished successfully.")
                elif response and response.prompt_feedback:
                    result_text = f"Blocked: {response.prompt_feedback}"
                    print(f"[Thread {index}] Finished with blocking: {response.prompt_feedback}")
                else:
                    result_text = "Error: Received an empty or unexpected response."
                    print(f"[Thread {index}] Finished with empty/unexpected response.")
                    
                return index, result_text.strip() # Return index for tracking

            except Exception as e:
                if attempt < max_retries:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    print(f"[Thread {index}] Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"[Thread {index}] All {max_retries + 1} attempts failed. Final error: {e}")
                    return index, f"Error after {max_retries + 1} attempts: {e}"
        

if __name__ == "__main__":
    default_sampling_kwargs = {
        "model_name": "gemini-2.5-pro-preview-03-25",
        "task_name": "move_ball",
        "save_top_k": 3,
        "top_k": 40,
        "top_p": 0.98,
        "temperature": 1.0,
        "thinking": False,
    }

    sa = SamplingAgent(**default_sampling_kwargs)