# Homework

> [!IMPORTANT]  
> - Please upload all Week 5 deliverables into a single folder on Google Drive and share the link.
> - When submitting to OMA, please use the updated template. It includes a section to indicate each group member's contribution as a percentage.  


- **Due Date:** 2026-04-23
- **Submission Method:** Google Drive

**Objective:** To replace the mock functionality of your Phase 1 application concept with real logic that calls the Gemini API. You will demonstrate the use of instruct prompting, parameter tuning, and JSON structured output in your implementation.
This task directly applies the concepts from this week's session to your ongoing project, moving it from a static mock-up to a dynamic, AI-powered application.

**Context:** Phase 1 involved designing a mock application interface. Phase 2 requires implementing the core backend logic using Gemini, which will eventually power that interface.

**Instructions:**

This task involves two main stages: 
1. Developing and testing the core Gemini interaction logic independently. 
2. Integrating this logic into your Phase 1 Gradio structure.

### Stage 1: Develop and Test Core Gemini Logic

Focus first on writing and testing the Python code that interacts with the Gemini API, separate from your Gradio UI code.

1.  **Define the Core Task:** Based on your Phase 1 project idea, clearly define what specific task Gemini should perform. What input will it receive (conceptually, e.g., "a user's question," "a piece of text to analyze," "a request for creative content")? What output should it generate?

2.  **Create a Python Function:** Write a standalone Python function (e.g., `get_gemini_data`, `process_input_with_gemini`) that encapsulates the interaction with Gemini. This function should:
    *   Accept necessary input arguments (e.g., `user_prompt`, `input_text`).
    *   **Implement Instruct Prompting:** Inside the function, construct a clear and specific prompt for the Gemini API, guiding it to perform the defined task effectively. Consider persona, context, and constraints as discussed in the lecture/lab.
    *   **Implement Parameter Tuning:** Use `types.GenerateContentConfig` to set at least one relevant generation parameter (e.g., `temperature` to control creativity, `max_output_tokens` to limit length). Pass this config object to the API call.
    *   **Implement JSON Mode:** Configure the API call to return structured JSON output.
        *   Set `response_mime_type="application/json"` in the `GenerateContentConfig`.
        *   Define the expected JSON structure using a Pydantic `BaseModel` or a Python dictionary representing the JSON schema. Pass this schema via the `response_schema` argument in the `GenerateContentConfig`.
    *   **Make the API Call:** Use `client.models.generate_content()` with the prompt, model ID, and your configuration.
    *   **Process/Return the Result:** For this standalone testing stage, it's sufficient for the function to `print` the raw JSON string received in `response.text`, or optionally parse it (`json.loads`) and print the structured data.

3.  **Test Independently:** Call your function directly from Colab cell with sample inputs. Verify that:
    *   The API call executes without errors (basic execution).
    *   The output is a JSON string.
    *   The JSON structure generally matches the schema you defined (perfect adherence depends on the model and prompt).
    *   The content of the response is relevant to your prompt and task.
    *   Adjust your prompt, parameters, or schema definition as needed until you get satisfactory results from the standalone function.

**Example Snippet (Standalone Logic - Conceptual):**

```python
# Assume 'client', 'MODEL_ID', 'types', 'json', 'BaseModel', 'Field', etc. are imported

# Define your Pydantic Schema (Example: Task Extraction)
class TaskItem(BaseModel):
    task_description: str = Field(description="Clear description of the task")
    priority: Optional[str] = Field("medium", description="Priority ('high', 'medium', 'low')")

def extract_task_from_text(text_input):
    # 1. Instruct Prompt
    prompt = f"""
    Analyze the following text and extract the main task described.
    Determine its priority (high, medium, or low). If unsure, default to medium.
    Format the output strictly as JSON according to the required schema.

    Text: "{text_input}"
    """

    # 2. Parameter Tuning & JSON Config
    config = types.GenerateContentConfig(
        temperature=0.2, # Lower temperature for extraction tasks
        response_mime_type="application/json",
        response_schema=TaskItem # Provide schema
    )

    try:
        # 3. API Call
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt,
            config=config
        )

        # 4. Process/Return Result (for testing)
        print("--- Raw JSON Output ---")
        print(response.text)
        try:
            # Optional: Parse and print structured data
            parsed_data = json.loads(response.text)
            print("\n--- Parsed Data ---")
            print(f"Task: {parsed_data.get('task_description')}")
            print(f"Priority: {parsed_data.get('priority')}")
        except Exception as parse_e:
            print(f"\nCould not parse JSON: {parse_e}")

        return response.text # Return raw JSON for potential later use

    except Exception as e:
        print(f"An API error occurred: {e}")
        return None

# --- Test the function ---
sample_text = "Need to finish the report by Friday, it's critical!"
extract_task_from_text(sample_text)
```

### Stage 2: Integrate Gemini Logic into Gradio Application

Once your core Gemini function works reliably standalone, integrate it into your Phase 1 Gradio UI.

1.  **Prepare Your Function for Gradio:** Modify your standalone function slightly if needed:
    *   Ensure it accepts input directly from the Gradio input components defined in your Phase 1 UI.
    *   Modify the `return` value. Instead of just printing, parse the JSON response (`json.loads(response.text)`) and format the extracted data into a string suitable for display in your Gradio output component(s) (e.g., `gr.Markdown`, `gr.Textbox`).

2.  **Update Gradio Interface:**
    *   Open your Phase 1 Gradio code.
    *   Find the `gr.Interface(...)` definition.
    *   Change the `fn` argument to point to your *new*, tested Gemini interaction function (the one prepared in step 1 of this stage).
    *   Ensure the `inputs` and `outputs` arguments of `gr.Interface` match what your new function expects and returns.

3.  **Test the Integrated Application:** Run the Gradio application. Test the full user flow – enter input in the UI, trigger the function, and verify that the output displayed is generated by Gemini and formatted correctly.

**Deliverable:** Submit your updated Colab notebook containing the functional Gradio application powered by Gemini API calls to Google Drive. Ensure the code is well-commented, particularly the sections involving the API interaction and response processing.
