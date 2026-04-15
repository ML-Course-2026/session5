# Demo

**Objective:** Convert the Phase 1 mock Gradio application to use live Gemini API calls for generating support tips, demonstrating instruct prompting, parameter tuning, and JSON output.

**Original Mock App Summary:**
*   Takes inputs: Main Category, Subcategory, Severity, Age Group.
*   Outputs: A summary of selections and a *pre-defined* mock response based only on severity.

**Target Functionality:**
*   Take the same inputs.
*   Generate a prompt for Gemini using *all* inputs.
*   Call Gemini API requesting structured JSON output (e.g., disclaimer, list of suggestions).
*   Use parameters like temperature.
*   Display the summary of selections and the *actual Gemini-generated* response in the Gradio UI.

---

### Stage 1: Develop and Test Core Gemini Logic (Without Gradio)

First, we'll write the Python code to handle the API interaction and test it directly.

**Step 1.1: Setup (API Key, Client Initialization)**

Ensure you have run the setup cells from the lab (Parts 1.1 to 1.4) in your Colab notebook. This means you should have imported necessary libraries (`genai`, `types`, `json`, `BaseModel`, etc.) and successfully initialized the `client` object and defined `MODEL_ID`.

```python
%pip install -q google-genai gradio
```

```python
from google.colab import userdata
GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
```

```python
# Core Gemini and Colab libraries
from google.colab import userdata
from google import genai
from google.genai import types # types is used for specific configurations later

```


```python
# --- Ensure these are run first---
import json
from pydantic import BaseModel, Field
from typing import Optional, List

client = genai.Client(api_key=GOOGLE_API_KEY)
MODEL_ID = "gemini-2.5-flash-lite" # @param ["gemini-2.5-flash-lite", "gemini-2.5-flash","gemini-3.1-flash-lite-preview"] {"allow-input":true, isTemplate: true}
print("Client and Model ID assumed to be initialized.")
# --- End of Setup ---
```

**Step 1.2: Define JSON Output Structure (Pydantic)**

We want Gemini to return specific pieces of information in a predictable format. Let's define a Pydantic model for this.

```python
from pydantic import BaseModel, Field
from typing import List, Optional

class SupportResponse(BaseModel):
    """Defines the structure for Gemini's response."""
    disclaimer: str = Field(description="A mandatory disclaimer stating this is not professional advice.")
    suggestions: List[str] = Field(description="A list of 3-5 actionable support suggestions or resources.")
    category_acknowledged: Optional[str] = Field(None, description="Confirmation of the main topic addressed.")
```

**Step 1.3: Create the Gemini Interaction Function**

This function will take the user's selections, build a prompt, configure the API call, execute it, and print the result.

```python
def get_gemini_support_logic(main_cat, sub_cat, severity, age):
    """
    Generates support suggestions using Gemini based on user selections.
    Demonstrates instruct prompting, parameter tuning, and JSON mode.
    Prints the raw JSON response for testing.
    """
    print(f"\n--- Requesting Gemini Support ---")
    print(f"Inputs: Main={main_cat}, Sub={sub_cat}, Severity={severity}, Age={age}")

    # --- 1. Instruct Prompt Design ---
    # Remove placeholder prefixes like "-- Select ... --"
    main_cat_clean = main_cat.split(" ", 1)[-1].replace("--", "").strip() if main_cat else "General"
    sub_cat_clean = sub_cat.split(" ", 1)[-1].replace("--", "").strip() if sub_cat else "General Tips"
    severity_clean = severity.split(" ", 1)[-1].replace("--", "").strip() if severity else "Mild"
    age_clean = age.split(" ", 1)[-1].replace("--", "").strip() if age else "Adult"

    # Base prompt structure
    prompt = f"""
    Act as a supportive mental health assistant providing helpful suggestions.
    The user has indicated the following:
    - Main Topic: {main_cat_clean}
    - Specific Focus: {sub_cat_clean}
    - Severity Level: {severity_clean}
    - Age Group: {age_clean}

    Based ONLY on these selections, provide 3-5 concise, actionable, and appropriate support suggestions or resources relevant to the topic and focus area.
    Tailor the tone and complexity appropriately for the selected age group.
    """

    # Specific instructions based on severity
    if "Crisis" in severity:
        prompt += """
    IMPORTANT: Since the user indicated 'Crisis' severity, prioritize directing them to URGENT professional help resources (like crisis lines). Briefly list 1-2 immediate grounding techniques IF appropriate, but the main focus MUST be professional crisis resources.
    Include a very prominent disclaimer that this IS NOT a substitute for immediate professional help and they should contact emergency services or a crisis line NOW.
    """
    elif "Severe" in severity:
         prompt += """
    Since the user indicated 'Severe' severity, strongly recommend seeking professional help alongside any general tips. Include resources for finding therapists or counselors.
    Include a prominent disclaimer that this is not a substitute for professional diagnosis or treatment.
    """
    else: # Mild or Moderate
        prompt += """
    Keep the suggestions practical and encouraging.
    Include a clear disclaimer that this tool provides general suggestions and is NOT a substitute for professional medical advice or diagnosis.
    """

    # Final instruction for JSON output
    prompt += f"""
    Acknowledge the main topic addressed.
    Format your entire response *strictly* as a JSON object matching the required schema, including the disclaimer and a list of suggestions.
    """

    # --- 2. Parameter Tuning & JSON Configuration ---
    generation_config = types.GenerateContentConfig(
        temperature=0.6,  # Moderate temperature for helpful, slightly varied suggestions
        max_output_tokens=300, # Limit response length
        response_mime_type="application/json",
        response_schema=SupportResponse, # Use the Pydantic model
    )

    # --- 3. API Call ---
    # Note: No try-except here as per request for this stage's core example
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=prompt,
        config=generation_config,
        # safety_settings={'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE'} # Optional: Adjust safety if needed, use cautiously
    )

    # --- 4. Print Raw Result for Testing ---
    print("\n--- Gemini Raw JSON Response ---")
    print(response.text)

    # --- Optional: Basic Parsing Test ---
    # Note: json.loads will raise an error if response.text is not valid JSON
    parsed_data = json.loads(response.text)
    print("\n--- Parsed Data (Optional Check) ---")
    print(f"Disclaimer: {parsed_data.get('disclaimer')}")
    print(f"Suggestions: {parsed_data.get('suggestions')}")
    print(f"Category Ack: {parsed_data.get('category_acknowledged')}")

    return response.text # Return raw JSON for potential use later
```

**Step 1.4: Test the Standalone Function**

Call the function with different combinations of inputs, especially testing the "Crisis" severity.

```python
# --- Test Case 1: Normal Scenario ---
get_gemini_support_logic(
    main_cat="😰 Stress & Anxiety",
    sub_cat="🛠️ Coping Techniques",
    severity="😐 Moderate",
    age="🧑 Young Adult (20–30)"
)

# --- Test Case 2: Crisis Scenario ---
get_gemini_support_logic(
    main_cat="😔 Depression",
    sub_cat="📞 Professional Help Resources",
    severity="🚨 Crisis",
    age="🧒 Teen (13–19)"
)

# --- Test Case 3: Mild Scenario ---
get_gemini_support_logic(
    main_cat="😴 Sleep Issues",
    sub_cat="🛏️ Sleep Hygiene Tips",
    severity="🙂 Mild",
    age="👵 Senior (50+)"
)
```

**Review Stage 1:**
*   Does the function execute?
*   Does it print JSON output?
*   Does the JSON structure look like the `SupportResponse` model?
*   Are the suggestions relevant to the inputs?
*   Is the disclaimer present?
*   Does the "Crisis" scenario correctly prioritize professional help resources?
*   Adjust the prompt, parameters, or Pydantic schema if needed.

---

### Stage 2: Integrate Gemini Logic into Gradio Application

Now, we'll take the working logic from Stage 1 and connect it to the Gradio UI structure from Phase 1.

**Step 2.1: Prepare Code**

Copy the following into a *new* Colab cell or Python script:
*   All necessary imports (from Lab Part 1.3).
*   The API key retrieval and client initialization (from Lab Part 1.4 - the simple version).
*   The `SupportResponse` Pydantic model definition (from Stage 1.2).
*   The *final, tested version* of the `get_gemini_support_logic` function (from Stage 1.3).
*   The `format_selection_output` function from your original Phase 1 code.
*   The `check_all_selected` function from your original Phase 1 code.
*   The `build_app` function structure from your original Phase 1 code.
*   The `main()` function and `if __name__ == "__main__":` block from your original Phase 1 code.

**Step 2.2: Create the Gradio Callback Function**


```python
# Gradio for UI
import gradio as gr

```

This new function will be called when the button is clicked. It orchestrates getting the summary, calling the Gemini logic, parsing the JSON, and formatting the final output strings.

```python
def format_selection_output(main_cat, sub_cat, severity, age):
    """Formats the user's dropdown selections for display."""
    return f"""
### ✅ Your Selections:
- **Main Category:** {main_cat}
- **Subcategory:** {sub_cat}
- **Severity Level:** {severity}
- **Age Group:** {age}
---
"""


def get_gemini_support_for_gradio(main_cat, sub_cat, severity, age):
    """
    Gradio callback:
    - formats input
    - calls Gemini
    - formats structured response
    """

    # 1. Input summary
    formatted_summary = format_selection_output(main_cat, sub_cat, severity, age)

    # 2. Call Gemini (IMPORTANT: returns SupportResponse object)
    result = get_gemini_support_logic(main_cat, sub_cat, severity, age)

    # 3. Safety check
    if result is None:
        return formatted_summary, "❌ No response from Gemini."

    # 4. Extract fields (NO json.loads needed)
    disclaimer = result.disclaimer
    suggestions = result.suggestions
    category_ack = result.category_acknowledged

    # 5. Build output
    formatted_response = ""

    if category_ack:
        formatted_response += f"**Acknowledgement:** Addressing '{category_ack}'\n\n"

    formatted_response += f"**{disclaimer}**\n\n"
    formatted_response += "**Suggestions:**\n"

    for sugg in suggestions:
        formatted_response += f"- {sugg}\n"

    # 6. Return to Gradio
    return formatted_summary, formatted_response
```

**Step 2.3: Update the Gradio UI Build Function**

Modify the `build_app` function to use the new callback.

```python
# Keep this function from Phase 1
def check_all_selected(*values):
    """Checks if all dropdowns have a valid (non-default) selection."""
    for val in values:
        if val is None or val.startswith("--"):
            return gr.update(interactive=False)
    return gr.update(interactive=True)

# --- Original Dropdown Lists (keep these) ---
MAIN_CATEGORIES = [
    "-- 🧠 Select a mental health topic --", "😰 Stress & Anxiety", "😔 Depression", "😴 Sleep Issues",
    "💖 Emotional Well-being", "🩺 General Mental Health Support"
]
SUB_CATEGORIES_PHASE1 = [
    "-- 📚 Select a subtopic --", "🛠️ Coping Techniques", "🧘 Breathing Exercises", "🧠 Mindfulness Tips",
    "💪 Motivational Support", "📓 Journaling Prompts", "📞 Professional Help Resources",
    "🛏️ Sleep Hygiene Tips", "🌿 Relaxation Techniques", "🎧 Guided Sleep Meditation",
    "🧴 Self-care Activities", "🌈 Positive Affirmations", "👥 Relationship Advice", "📌 General Tips"
]
SEVERITY_LEVELS = [ "-- 🚦 Select severity --", "🙂 Mild", "😐 Moderate", "😟 Severe", "🚨 Crisis" ]
AGE_GROUPS = [ "-- 👤 Select age group --", "🧒 Teen (13–19)", "🧑 Young Adult (20–30)", "🧔 Adult (31–50)", "👵 Senior (50+)" ]
# --- End Dropdown Lists ---


def build_app(dark_mode=False):
    """Builds the Gradio UI, now using the Gemini callback."""
    theme = gr.themes.Base() if dark_mode else gr.themes.Default()

    with gr.Blocks(theme=theme, title="Mental Health Assistant v2.0 (Gemini)") as demo:
        gr.Markdown("## 🧠 Mental Health Support Assistant (Powered by Gemini)")
        gr.Markdown("Select from the dropdowns below to receive AI-generated support suggestions.")

        with gr.Accordion("Step 1: Choose Your Preferences", open=True):
            with gr.Row():
                with gr.Column():
                    dropdown_main = gr.Dropdown(MAIN_CATEGORIES, label="Main Category", value=MAIN_CATEGORIES[0])
                    dropdown_severity = gr.Dropdown(SEVERITY_LEVELS, label="Severity Level", value=SEVERITY_LEVELS[0])
                with gr.Column():
                    dropdown_sub = gr.Dropdown(SUB_CATEGORIES_PHASE1, label="Subcategory", value=SUB_CATEGORIES_PHASE1[0])
                    dropdown_age = gr.Dropdown(AGE_GROUPS, label="Age Group", value=AGE_GROUPS[0])

            get_btn = gr.Button("🎯 Get AI Support Tips", interactive=False, variant="primary") # Renamed button slightly

        gr.Markdown("---")
        gr.Markdown("### 📋 Your Input Summary + AI Suggestions")
        output_summary = gr.Markdown()
        output_response = gr.Markdown() # This will display the formatted Gemini response

        # Enable button only when valid selections (same logic as before)
        inputs = [dropdown_main, dropdown_sub, dropdown_severity, dropdown_age]
        for d in inputs:
            d.change(fn=check_all_selected, inputs=inputs, outputs=get_btn)

        # *** THE KEY CHANGE: Update the button's click function ***
        get_btn.click(
            fn=get_gemini_support_for_gradio, # Use the NEW callback function
            inputs=inputs,
            outputs=[output_summary, output_response] # Map to the two Markdown outputs
        )

    return demo
```

**Step 2.4: Run the Application**

Use the `main` function to launch the Gradio app.

```python
def main():
    dark_mode = False

    if 'client' not in globals() or client is None:
        print("❌ Error: Gemini client not initialized.")
        return

    print("🚀 Launching Gradio App...")

    try:
        app = build_app(dark_mode=dark_mode)
        app.launch(
            share=True,
            debug=True
        )
    except Exception as e:
        print("❌ Error during launch:")
        print(str(e))


if __name__ == "__main__":
    main()
```

**Testing Stage 2:**
*   Run the complete script.
*   Does the Gradio app launch?
*   Make selections in all dropdowns. Does the button become active?
*   Click the button.
    *   Does the summary appear correctly?
    *   Does the AI response appear after a short delay?
    *   Is the response formatted reasonably (disclaimer, suggestions)?
    *   Test with different inputs, including "Crisis".

