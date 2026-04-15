# Part 2

> [!NOTE]
> This material presents several use cases for working with multimodal LLMs from the Gemini family.
>
> Some examples were already covered in Session 4. Additional examples include generating structured JSON output, creating images, streaming content, uploading and processing files, and handling YouTube links.

---

## Introduction and Rationale

We begin our exploration of Large Language Models (LLMs) by interacting with them through an Application Programming Interface (API). This approach allows us to leverage powerful, pre-trained models without managing the underlying infrastructure.

### Why Start with an API?

Using an API like Google's Gemini API provides immediate access to sophisticated AI capabilities. It allows us to focus on *how to use* the model effectively for tasks like text generation, summarization, or structured data extraction, and integrate these capabilities into applications (such as the Gradio UIs we are building). This practical experience is valuable before delving into the model's internal workings. The underlying mechanisms, such as **tokenization**, **embeddings**, and the **model pipeline**, will be covered in the next session to provide a deeper understanding necessary for customization and advanced use cases.

### Why Use the Gemini API Specifically?

For this stage, the Gemini API serves as a practical example due to its:
-   **Advanced Capabilities:** Access to models demonstrating strong performance in various tasks.
-   **Multimodality:** Ability to process text, images, audio, and video inputs.
-   **Developer Experience:** A well-documented Python SDK (`google-genai`) simplifies interaction.
-   **Structured Output:** Features for generating formatted JSON output, useful for application integration.
-   **Accessibility:** includes a free tier suitable for learning and experimentation.

### Why Explore Hugging Face / Local Models Later?

While APIs are convenient, direct interaction with models (often sourced from platforms like Hugging Face, covered in Part 3) offers distinct advantages:
-   **Flexibility/Choice:** Access to thousands of specialized models.
-   **Privacy/Control:** Data remains within your environment when run locally.
-   **Customization:** Enables fine-tuning on specific datasets.
-   **Offline Use:** Models can run without internet connectivity.
-   **Cost Efficiency:** Potentially lower cost at high scale compared to API calls.

Understanding both API-based and direct model usage allows for informed decisions based on project requirements.

<details>
<summary><strong>Test Your Knowledge: APIs vs. Local Models</strong></summary>

**Q: If you are building an application that handles highly sensitive medical records that legally cannot leave your company's servers, should you use an API like Gemini or a local model?**<br>
**A:** You should use a local model. Sending sensitive data to an external API (like Google's or OpenAI's) often violates strict data privacy regulations unless specific enterprise agreements are in place. Local models ensure the data never leaves your infrastructure.
</details>

---
## Setup and Initialization

The first steps involve setting up the environment. Gemini uses API keys for authentication. here's a walk through how to create an API key, and using it in colab

### Create an API key

You can [create](https://aistudio.google.com/app/apikey) your API key using Google AI Studio with a single click.  

Remember to treat your API key like a password. Do not accidentally save it in a notebook or source file you later commit to GitHub. 
In Google Colab, it is recommended to store your key in Colab Secrets. here's how to

### Add your key to Colab Secrets

Add your API key to the Colab Secrets manager to securely store it.

1. Open your Google Colab notebook and click on the 🔑 **Secrets** tab in the left panel.
   
   <img src="https://storage.googleapis.com/generativeai-downloads/images/secrets.jpg" alt="The Secrets tab is found on the left panel." width=50%>

2. Create a new secret with the name `GOOGLE_API_KEY`.
3. Copy/paste your API key into the `Value` input box of `GOOGLE_API_KEY`.
4. Toggle the button on the left to allow notebook access to the secret.


### Setup your API Key

You create a client using your API key, but instead of pasting your key into the notebook, you'll read it from Colab Secrets.

```python
from google.colab import userdata
GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
```

*   **Explanation:** Accessing the Gemini API requires authentication. An API key is a unique secret credential that identifies your project or account to Google Cloud. This code retrieves the key securely stored as a Colab Secret named `GOOGLE_API_KEY`. Storing keys as secrets is crucial for security, preventing them from being exposed directly in the notebook code. You need to generate your own API key from Google AI Studio or Google Cloud Console and store it in Colab secrets for this code to work.

### Install SDK

```
%pip install google-genai -q
```

*   **Explanation:** This command installs or updates the necessary Python library, `google-genai`. This library, provided by Google, contains the functions and classes needed to interact with the Gemini API easily from Python code.  The `-q` makes the installation process quiet (less output)

<!-- The `-U` flag ensures you get the latest version, and `-q` makes the installation process quiet (less output). -->

### Initialize SDK client

```python
from google import genai
from google.genai import types # types is used for specific configurations later

# Initialize the client with the API key
client = genai.Client(api_key=GOOGLE_API_KEY)
```

*   **Explanation:** Here, we import the installed library (`genai`). The core of the interaction is the `Client` object. We create an instance of this client, passing our `GOOGLE_API_KEY` for authentication. This `client` object will be used for all subsequent calls to the API (e.g., generating content, managing files).

### Choose a model

Now choose a model. The Gemini API offers different models that are optimized for specific use cases, for more information check[Gemini models](https://ai.google.dev/gemini-api/docs/models)

```python
MODEL_ID = "gemini-2.5-flash" # @param["gemini-2.5-flash-lite", "gemini-3.1-flash-lite-preview"] {"allow-input":true, isTemplate: true}
```

*   **Explanation:** The Gemini family includes several models optimized for different tasks, performance levels, and input modalities. This line selects which specific model variant we want to use for our requests. `gemini-2.5-flash` is chosen here as a generally capable and efficient model. Other options like `gemini-3.1-flash-lite-preview` offer different features or performance characteristics. The model ID is stored in the `MODEL_ID` variable for easy reference in later API calls. The comment `# @param ...` enables an interactive dropdown menu in Colab for selecting the model.

## Send Text Prompts

The most basic interaction involves sending a text prompt and receiving a text response.

### Basic Text Generation

```python
from IPython.display import Markdown # Used for nice formatting of output

# Make the API call
response = client.models.generate_content(
    model=MODEL_ID,
    contents="What's the largest planet in our solar system?"
)

# Display the response text
Markdown(response.text)
```

-   **Explanation:** This code demonstrates a simple text-in, text-out request.
    -   `client.models.generate_content()`: This is the primary method for sending prompts to the selected model.
    -   `model=MODEL_ID`: Specifies which Gemini model to use (the one selected earlier).
    -   `contents=...`: This argument holds the input prompt. Here, it's a simple string.
    -   The API call returns a `response` object. The generated text is typically accessed via `response.text`.
    -   `Markdown(response.text)` displays the output using Markdown formatting for better readability in environments like Colab or Jupyter.

### Text Generation with Gradio Interface


```
%pip install gradio -q # Install Gradio if not already installed
```


```python
import gradio as gr
# from IPython.display import Markdown # Already imported

# Define the function that calls the Gemini API
def ask_model(prompt):
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=prompt
    )
    # Return the text part of the response
    # Gradio's Markdown component will render this
    return response.text

# Create the Gradio interface
gr.Interface(
    fn=ask_model, # The function to call when the user interacts
    inputs=gr.Textbox(lines=2, placeholder="Ask me something...", label="Prompt"), # Input component
    outputs=gr.Markdown(label="Response"), # Output component (displays Markdown)
    title="Gemini Model Q&A",
    description="Ask the Gemini model a question and see its response!"
).launch() # Launch the web UI
```

-   **Explanation:** This section wraps the basic text generation functionality in a simple web interface using the Gradio library.
    -   `import gradio as gr`: Imports the Gradio library.
    -   `ask_model(prompt)`: This function takes a `prompt` string (from the Gradio textbox) as input, calls the `client.models.generate_content` method just like before, and returns the `response.text`.
    -   `gr.Interface(...)`: This creates the user interface.
        -   `fn=ask_model`: Specifies the Python function to execute.
        -   `inputs=gr.Textbox(...)`: Defines the input field as a multi-line textbox.
        -   `outputs=gr.Markdown(...)`: Defines the output area, specifying that the returned text should be rendered as Markdown.
        -   `title`, `description`: Set the UI titles.
    -   `.launch()`: Starts the interactive Gradio web server and displays the UI. This allows users to interact with the Gemini model through a simple form instead of just running code cells.

<details>
<summary><strong>Concept Check: How Gradio Connects to Functions</strong></summary>

**Q: How does Gradio know to pass the text from `gr.Textbox` into the `prompt` variable in the `ask_model` function?**<br>
**A:** Gradio maps the list of `inputs` directly to the arguments of the function specified in `fn`, in order. Because `gr.Textbox` is the only input defined, whatever the user types there is passed as the first (and only) argument (`prompt`) to the `ask_model` function. Whatever `ask_model` returns is then sent to the `outputs` component.
</details>

## Send Multimodal Prompts

Gemini models can understand prompts containing multiple types of input, such as images and text together.

### Multimodal Generation (Image + Text)

```python
import requests # To download the image
import pathlib # To handle file paths
from PIL import Image # To work with the image object

# Download an image
IMG_URL = "https://storage.googleapis.com/generativeai-downloads/data/jetpack.png"
img_bytes = requests.get(IMG_URL).content
img_path = pathlib.Path('jetpack.png')
img_path.write_bytes(img_bytes)

# Open the image using PIL
image = Image.open(img_path)
image.thumbnail([512,512]) # Resize for display convenience

# Display the image in the notebook (optional)
from IPython.display import display
display(image)

# Send image and text prompt together
response = client.models.generate_content(
    model=MODEL_ID, # Ensure model supports multimodal, e.g., gemini-2.5-flash
    contents=[
        image, # Pass the PIL Image object directly
        "Write a short and engaging blog post based on this picture." # Text part
    ]
)

# Display the text response
Markdown(response.text)
```

-   **Explanation:** This demonstrates sending both an image and text in a single prompt.
    -   The code first downloads an image from a URL and saves it locally.
    -   It opens the image using the Python Imaging Library (PIL).
    -   The key part is the `contents` argument in `generate_content`. It's now a *list* containing multiple parts: the `image` object (PIL format is supported directly by the SDK) and the text prompt string.
    -   The model processes both inputs to generate the response (in this case, a blog post about the image).

### Multimodal Generation with Gradio Interface

```python
import gradio as gr
# Other necessary imports (requests, pathlib, PIL.Image) assumed from previous cell

def generate_blog(image_input, prompt):
    # The 'image_input' from Gradio is already a PIL Image object if type="pil"
    if image_input is None:
        return "Please upload an image."

    # No need to save/reload if Gradio provides PIL object directly
    pil_image = image_input
    pil_image.thumbnail([512, 512]) # Optional resize for consistency

    # Call Gemini with the PIL image and text prompt
    try:
        response = client.models.generate_content(
            model=MODEL_ID, # Ensure model supports multimodal
            contents=[
                pil_image,
                prompt
            ]
        )
        return response.text
    except Exception as e:
        return f"Error processing request: {e}"


# Gradio UI for multimodal input
gr.Interface(
    fn=generate_blog,
    inputs=[
        gr.Image(type="pil", label="Upload an image"), # Image input component
        gr.Textbox(lines=2, placeholder="e.g., Write a blog post about this...", label="Prompt") # Text input
    ],
    outputs=gr.Markdown(label="Generated Blog Post"), # Text output
    title="AI Blog Generator from Image",
    description="Upload an image and let the Gemini model write a short blog post for you!"
).launch()
```

-   **Explanation:** This wraps the multimodal functionality in a Gradio interface.
    -   `generate_blog(image_input, prompt)`: This function now takes two arguments: `image_input` (from the Gradio image component) and `prompt` (from the textbox).
    -   `gr.Image(type="pil", ...)`: This Gradio input component allows users to upload an image. Setting `type="pil"` ensures that the `image_input` argument passed to our function is already a PIL Image object, simplifying the code.
    -   The rest of the function calls `generate_content` with the image and text, returning the generated text to be displayed in the `gr.Markdown` output component.
    -   *(Note on Scope):* While this example successfully uses Gradio for *image input*, recall the earlier point: reliably displaying *generated* images or audio from the model within Gradio *output* components can be complex and is considered outside the core scope of the required lab exercises. We focus on text/Markdown output for simplicity.

<details>
<summary><strong>Deep Dive: Gradio Image Types</strong></summary>

**Q: What happens if we don't include `type="pil"` in `gr.Image()`?**<br>
**A:** By default, Gradio passes images to your function as a NumPy array (a grid of numbers representing pixel values). While useful for OpenCV or strict machine learning matrix operations, the `google-genai` SDK is designed to easily accept `PIL.Image` objects directly. Using `type="pil"` tells Gradio to convert the user's uploaded image into a PIL object automatically, saving you the trouble of converting a NumPy array back into an image format that Gemini understands.
</details>

## Configure Model Parameters

API calls can include parameters to control the generation process.

### Generation with Custom Configuration

```python
# Make sure 'types' is imported: from google.genai import types

response = client.models.generate_content(
    model=MODEL_ID,
    contents="Tell me how the internet works, but pretend I'm a puppy who only understands squeaky toys.",
    config=types.GenerateContentConfig(
        temperature=0.4,       # Controls randomness (lower = more deterministic)
        candidate_count=1,     # Number of response candidates to generate
        seed=5,                # For reproducible results (if possible with model)
        max_output_tokens=100, # Maximum length of the response
    )
)

print(response.text)
```

*   **Explanation:** This demonstrates how to influence the model's output beyond just the prompt.
    *   The `config` argument takes a `GenerateContentConfig` object (from `google.genai.types`).
    *   Inside `GenerateContentConfig`, various parameters can be set:
        *   `temperature`: Controls creativity vs. focus. Lower values (e.g., 0.2) make output more predictable; higher values (e.g., 0.9) make it more random/creative.
        *   `top_p`, `top_k`: Alternative methods to control randomness by limiting the pool of tokens the model considers at each step.
        *   `max_output_tokens`: Limits response length.
        *   `stop_sequences`: Causes the model to stop generating if it produces one of these strings.
        *   `seed`: Allows for potentially reproducible outputs, though not guaranteed across all models/versions.
        *   `presence_penalty`, `frequency_penalty`: Help control repetitiveness.
    *   Experimenting with these parameters is key to tuning the model's behavior for specific needs.

### Configuration Control with Gradio Interface

```python
import gradio as gr
# Assume 'client', 'MODEL_ID', 'types' are available

def generate_response(prompt, temperature, top_p, top_k, seed, max_tokens, stop_seq, presence_penalty, frequency_penalty):
    # Prepare stop sequences list
    stop_sequences = [stop_seq] if stop_seq else None # Handle empty input

    # Create the configuration object from Gradio inputs
    config = types.GenerateContentConfig(
        temperature=float(temperature),
        top_p=float(top_p),
        top_k=int(top_k),
        candidate_count=1,
        seed=int(seed) if seed is not None else None, # Handle potential None input
        max_output_tokens=int(max_tokens),
        stop_sequences=stop_sequences,
        presence_penalty=float(presence_penalty),
        frequency_penalty=float(frequency_penalty),
    )

    # Call the model
    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt,
            config=config
        )
        return response.text
    except Exception as e:
         return f"Error processing request: {e}"

# Gradio Interface with sliders and number inputs for parameters
gr.Interface(
    fn=generate_response,
    inputs=[
        gr.Textbox(label="Prompt", lines=3, placeholder="e.g., Explain quantum physics to a cat..."),
        gr.Slider(0.0, 1.0, value=0.4, step=0.05, label="Temperature"),
        gr.Slider(0.0, 1.0, value=0.95, step=0.01, label="Top-p"),
        gr.Slider(1, 100, value=20, step=1, label="Top-k"),
        gr.Number(value=5, label="Seed", precision=0), # Use precision=0 for integer
        gr.Number(value=100, label="Max Output Tokens", precision=0),
        gr.Textbox(label="Stop Sequence (optional)", placeholder="e.g., STOP!"),
        gr.Slider(0.0, 1.0, value=0.0, step=0.1, label="Presence Penalty"),
        gr.Slider(0.0, 1.0, value=0.0, step=0.1, label="Frequency Penalty")
    ],
    outputs=gr.Markdown(label="Model Response"),
    title="Gemini Prompt with Custom Config",
    description="Customize generation settings and interact with the Gemini model."
).launch()
```

*   **Explanation:** This Gradio app allows interactive experimentation with the generation parameters.
    *   The `generate_response` function now takes the prompt and all the configuration parameters as arguments. These will come from the corresponding Gradio input components.
    *   Inside the function, it constructs the `GenerateContentConfig` object using the values passed from the UI. Note the type conversions (e.g., `float()`, `int()`) as Gradio inputs might be strings or floats that need to match the types expected by `GenerateContentConfig`.
    *   The `gr.Interface` uses various input components like `gr.Slider` and `gr.Number` to provide intuitive controls for the numerical parameters.
    * More about[top-p](https://rentry.co/samplers#top-p)
    * More about[top-k](https://rentry.co/samplers#top-k)

<details>
<summary><strong>Code Hint: Handling Variable Types in Gradio</strong></summary>

Notice in the code above that we explicitly cast variables, such as `float(temperature)` or `int(top_k)`. Gradio inputs generally return numerical data if configured as Sliders or Numbers, but explicitly casting them in your function is a defensive programming best practice. It ensures the API receives exactly the data type it expects, preventing runtime crashes.
</details>

## Configure Safety Filters

The API includes safety filters to block potentially harmful content. These can be adjusted.

```python
# Assume 'client', 'MODEL_ID', 'types' are available

prompt = """
    Write a list of 2 disrespectful things that I might say to  friend after stubbing my toe in the dark.
"""

# Define safety settings configuration
# Example: Block only high-probability dangerous content
safety_settings =[
    types.SafetySetting(
        category="HARM_CATEGORY_DANGEROUS_CONTENT",
        threshold="BLOCK_ONLY_HIGH",
    ),
    # Can add settings for other categories like HARASSMENT, HATE_SPEECH, SEXUALLY_EXPLICIT
]

# Call generate_content with safety_settings in the config
# Note: Safety settings are part of GenerateContentConfig
try:
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=prompt,
        config=types.GenerateContentConfig(
            safety_settings=safety_settings,
            # Can combine with other config parameters like temperature if needed
        ),
        # Alternative: safety_settings can sometimes be passed as a direct argument too
        # safety_settings=safety_settings
    )
    #Markdown(response.text)
    print(response.text)    
except Exception as e:
    # Responses might be blocked entirely if they violate stricter settings.
    # Check response.prompt_feedback for safety ratings/blocks
    print(f"An error or block occurred: {e}")
    # if hasattr(response, 'prompt_feedback'): print(response.prompt_feedback)
```

*   **Explanation:** This code demonstrates how to customize the API's built-in safety mechanisms.
    *   `safety_settings` is a list of `SafetySetting` objects. Each object specifies a `category` (e.g., `HARM_CATEGORY_DANGEROUS_CONTENT`) and a `threshold` (e.g., `BLOCK_NONE`, `BLOCK_LOW_AND_ABOVE`, `BLOCK_MEDIUM_AND_ABOVE`, `BLOCK_ONLY_HIGH`).
    *   These settings are passed within the `GenerateContentConfig` object (or sometimes directly as an argument) to the `generate_content` call.
    *   Adjusting these thresholds changes the likelihood that the API will block prompts or responses it deems potentially harmful according to its classifiers. It's important to configure these appropriately for the application's use case and target audience. If a response is blocked due to safety settings, the API might return an error or an empty response; detailed feedback is often available in `response.prompt_feedback`.

## Start a Multi-turn Chat

The SDK supports conversational interactions where context is maintained across turns.

### Basic Chat Interaction

```python
# Assume 'client', 'MODEL_ID', 'types' are available

# Optional: Define system instructions for the chat persona/behavior
system_instruction="""
  You are an expert software developer and a helpful coding assistant.
  You are able to generate high-quality code in any programming language.
"""

# Configure chat parameters (optional, can include temperature, etc.)
chat_config = types.GenerateContentConfig(
    system_instruction=system_instruction,
    temperature=0.5,
    # other config parameters can go here
)

# Start a new chat session
chat = client.chats.create(
    model=MODEL_ID,
    config=chat_config,
    # History can be pre-filled here if needed: history=[...]
)

# Send the first user message
response = chat.send_message("Write a function that checks if a year is a leap year.")
Markdown(response.text) # Display first response
```

```python
# Send a follow-up message; the chat object maintains history
response = chat.send_message("Okay, write a unit test of the generated function.")
Markdown(response.text) # Display second response
```


*   **Explanation:** This code sets up and conducts a multi-turn conversation.
    *   `system_instruction`: An optional initial instruction defining the AI's persona or core task for the entire chat session.
    *   `chat_config`: A `GenerateContentConfig` can be applied to the chat session, including the system instruction and generation parameters like temperature.
    *   `client.chats.create()`: Initializes a new chat session. It takes the model ID and optional configuration. You can also provide an initial `history` list here to start from a previous conversation.
    *   `chat.send_message()`: Sends a user message to the chat session. The SDK automatically manages the conversation history (previous user messages and model responses) and includes it in subsequent calls to the API, allowing the model to respond contextually.
    *   Each call to `send_message` returns the model's response for that turn.

### Chat Interaction with Gradio Interface

```python
import gradio as gr
# Assume 'client', 'MODEL_ID', 'types' are available

# Note: This Gradio example starts a *new* chat session for *each* interaction.
# For a persistent chat UI, you'd need to manage the 'chat' object state across calls,
# typically using gr.State or external storage, which adds complexity.
# This simplified version demonstrates passing system instructions and a single turn.

def chat_with_assistant(system_instruction, user_prompt, temperature):
    # Define chat config with system instruction and temperature for this turn
    chat_config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=float(temperature),
    )

    # Create a *new* chat session for this interaction
    # (No history is carried over from previous interactions in this simple UI)
    try:
        chat = client.chats.create(
            model=MODEL_ID,
            config=chat_config,
        )
        # Send the user's message
        response = chat.send_message(user_prompt)
        return response.text
    except Exception as e:
        return f"Error processing request: {e}"

# Gradio Interface
gr.Interface(
    fn=chat_with_assistant,
    inputs=[
        gr.Textbox(label="System Instruction", lines=3, value="You are an expert software developer and a helpful coding assistant."),
        gr.Textbox(label="Your Message", lines=3, placeholder="e.g., Write a function that checks if a year is a leap year."),
        gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Temperature")
    ],
    outputs=gr.Markdown(label="Assistant Response"),
    title="Chat with Gemini (Custom System Instruction)",
    description="Define how the assistant should behave, then send a prompt to the Gemini model. (Note: Each interaction starts a new chat)."
).launch()
```

*   **Explanation:** This Gradio app provides an interface for interacting with the chat functionality, allowing users to set the system instruction.
    *   The `chat_with_assistant` function takes the system instruction, user prompt, and temperature from the UI.
    *   **Important Limitation:** As noted in the comments and description, this simple Gradio implementation creates a *new chat session* every time the user submits a prompt. It does not maintain conversation history between interactions in the UI. A true chatbot UI in Gradio would require state management (`gr.State`) to keep track of the `chat` object and its history across multiple turns. This example focuses only on demonstrating the passing of system instructions and single-turn interaction via Gradio.

<details>
<summary><strong>Test Your Knowledge: Gradio State and Chat History</strong></summary>

**Q: Why doesn't the model remember my previous question in the Gradio example above, even though we are using `client.chats.create()`?**<br>
**A:** Gradio interfaces are "stateless" by default. Every time you click the submit button, Gradio executes the `chat_with_assistant` function from top to bottom entirely fresh. It creates a brand-new `chat` object, sends your single message, returns the text, and then that `chat` object is deleted from memory. To build a continuous chatbot in Gradio, you would need to use `gr.State` or `gr.ChatInterface` to save the history between button clicks.
</details>

-----

## Generate JSON (Structured Output)

Gemini can be instructed to generate responses formatted as JSON, adhering to a specific schema. This is extremely useful for integrating LLM output into applications.

### Basic JSON Generation (Pydantic Schema)

```python
from pydantic import BaseModel # Import Pydantic
# Assume 'client', 'MODEL_ID', 'types' are available

# Define the desired structure using a Pydantic model
class Recipe(BaseModel):
    recipe_name: str
    recipe_description: str
    recipe_ingredients: list[str]

# Make the API call, specifying JSON output and the schema
response = client.models.generate_content(
    model=MODEL_ID,
    contents="Provide a popular cookie recipe and its ingredients.",
    config=types.GenerateContentConfig(
        response_mime_type="application/json", # Request JSON output
        response_schema=Recipe, # Provide the Pydantic model as the schema
    ),
)

# The response.text should now contain a JSON string matching the Recipe schema
# Use Markdown to display it nicely, potentially with JSON formatting
Markdown(f"```json\n{response.text}\n```")
# print(response.text) # Raw JSON string
```

*   **Explanation:** This code forces the model to output JSON conforming to the `Recipe` structure.
    *   `from pydantic import BaseModel`: Imports the necessary class from Pydantic.
    *   `class Recipe(BaseModel): ...`: Defines a Pydantic model. This acts as the schema, specifying the expected fields (`recipe_name`, `recipe_description`, `recipe_ingredients`) and their types (`str`, `str`, `list[str]`).
    *   `GenerateContentConfig`:
        *   `response_mime_type="application/json"`: This tells the model to generate JSON.
        *   `response_schema=Recipe`: This provides the Pydantic class as the schema definition. The model will attempt to structure its output accordingly.
    *   The `response.text` will contain the generated JSON string (or an error if it fails). Using Markdown with ```json ... ``` helps render it clearly.

### JSON Generation with Gradio Interface

```python
import gradio as gr
from pydantic import BaseModel
import json # To parse the JSON string for potentially nicer formatting
# Assume 'client', 'MODEL_ID', 'types' are available

# Define Pydantic model for recipe (same as before)
class Recipe(BaseModel):
    recipe_name: str
    recipe_description: str
    recipe_ingredients: list[str]

# Gradio-compatible function
def get_recipe(prompt):
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=Recipe, # Use the Pydantic model
    )

    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt,
            config=config
        )

        # Try to parse and format the JSON for better display in Markdown
        try:
            recipe_data = json.loads(response.text)
            formatted = f"### {recipe_data.get('recipe_name', 'N/A')}\n\n" \
                        f"**Description:** {recipe_data.get('recipe_description', 'N/A')}\n\n" \
                        f"**Ingredients:**\n" + "\n".join(f"- {item}" for item in recipe_data.get('recipe_ingredients',[]))
            return formatted
        except Exception as parse_error:
            # If parsing fails, return the raw text with a warning
            return f"⚠️ Failed to parse JSON response: {parse_error}\n\n**Raw Output:**\n```json\n{response.text}\n```"

    except Exception as api_error:
        return f"API Error: {api_error}"


# Build Gradio app
gr.Interface(
    fn=get_recipe,
    inputs=gr.Textbox(label="Prompt", lines=2, placeholder="e.g., Provide a popular cookie recipe"),
    outputs=gr.Markdown(label="Generated Recipe"), # Display formatted recipe as Markdown
    title="Recipe Generator (Structured JSON)",
    description="Ask for a recipe. The model returns a JSON object matched to a Pydantic schema, which is then formatted for display."
).launch()
```

*   **Explanation:** This Gradio interface allows users to request structured data (a recipe).
    *   The `get_recipe` function takes the user's prompt.
    *   It configures the API call to expect JSON output conforming to the `Recipe` schema.
    *   After receiving the `response.text` (which should be a JSON string), it attempts to parse this JSON using `json.loads()`.
    *   If parsing is successful, it extracts the data and formats it into a human-readable Markdown string for display in the `gr.Markdown` output component.
    *   Error handling is included for both API call failures and JSON parsing failures.

<details>
<summary><strong>Deep Dive: JSON Parsing in Python</strong></summary>

**Q: What exactly does `json.loads(response.text)` do in the code above?**<br>
**A:** When the Gemini API returns a JSON response, it is delivered as one giant string of text (e.g., `"{ 'recipe_name': 'Cookies' }"`). You cannot programmatically access the data inside a string using python dictionary logic. `json.loads()` (Load String) takes that string and converts it into a native Python dictionary. This allows you to extract specific values later using `.get('recipe_name')`. The inner `try/except` block is a safety net: if the model hallucinates and returns broken JSON, `json.loads` will crash, but the `except` block catches it and displays the raw text instead of breaking the app.
</details>

### Additional JSON / Pydantic Examples

To further illustrate the power of structured output, consider these scenarios:

**1. Extracting Contact Information:**

```python
# Assume necessary imports: BaseModel, Field, Optional, client, types, MODEL_ID
from pydantic import Field
from typing import Optional

class ContactInfo(BaseModel):
    name: Optional[str] = Field(None, description="The full name of the person")
    email: Optional[str] = Field(None, description="The email address")
    phone: Optional[str] = Field(None, description="The phone number, including area code if present")

def extract_contacts(text_block):
    prompt = f"Extract the primary contact details (name, email, phone) from the following text:\n\n{text_block}"
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=ContactInfo,
    )
    try:
        response = client.models.generate_content(model=MODEL_ID, contents=prompt, config=config)
        return response.text # Return raw JSON string
    except Exception as e:
        return f"Error: {e}"

# Example usage (outside Gradio)
text = "Reach out to John Smith (jsmith@example.com) or call 987-654-3210 for details."
json_output = extract_contacts(text)
print(json_output)
# Expected: {"name": "John Smith", "email": "jsmith@example.com", "phone": "987-654-3210"}
```
*   **Use Case:** Parsing unstructured text like emails or meeting transcripts to extract key information into a usable format. Could be wrapped in a Gradio interface taking text input and outputting formatted contact details or the raw JSON.
<!-- 
**2. Summarizing Action Items (Raw JSON Schema):**

```python
# Assume necessary imports: json, client, types, MODEL_ID

# Define schema as a Python dictionary (representing JSON Schema)
action_item_schema = {
    "type": "object",
    "properties": {
        "summary": {"type": "string", "description": "Brief summary of the meeting source."},
        "action_items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "The specific action item"},
                    "assignee": {"type": "string", "description": "Who is responsible for the task"},
                    "due_date": {"type": "string", "description": "When the task is due (YYYY-MM-DD or relative term like 'EOW')"}
                },
                "required": ["task", "assignee"]
            }
        }
    },
     "required": ["action_items"]
}

def summarize_actions(meeting_notes):
    prompt = f"Extract action items from these meeting notes:\n\n{meeting_notes}\n\nProvide a brief summary and list all action items with assignee and due date (if mentioned)."
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=action_item_schema, # Pass the dictionary schema
    )
    try:
        response = client.models.generate_content(model=MODEL_ID, contents=prompt, config=config)
        return response.text # Return raw JSON string
    except Exception as e:
        return f"Error: {e}"

# Example usage (outside Gradio)
notes = "Project Alpha Sync:\n- Design team (Alice) to finalize mockups by Friday.\n- Bob needs to send client the report EOD.\n- Review budget next week (Contact: Carol)."
json_output = summarize_actions(notes)
print(json_output)
# Expected structure: {"summary": "...", "action_items":[{"task": "Finalize mockups", "assignee": "Alice/Design team", "due_date": "Friday"}, ...]}
```
*   **Use Case:** Processing meeting minutes or project updates to automatically generate task lists. Pydantic is generally recommended for complex schemas, but raw JSON schema dictionaries are also supported. -->

**2. Generating Product Descriptions:**

```python
# Assume necessary imports: BaseModel, Field, List, client, types, MODEL_ID
from pydantic import Field
from typing import List

class ProductDesc(BaseModel):
    product_name: str = Field(..., description="Catchy, short product name")
    tagline: str = Field(..., description="Memorable slogan (max 10 words)")
    key_features: List[str] = Field(..., min_items=3, max_items=5, description="Bulleted list of 3-5 main features")
    target_audience: str = Field(..., description="Who is this product primarily for?")

def generate_product_description(product_concept):
    prompt = f"Generate a structured product description based on this concept: {product_concept}"
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=ProductDesc,
    )
    try:
        response = client.models.generate_content(model=MODEL_ID, contents=prompt, config=config)
        return response.text # Return raw JSON string
    except Exception as e:
        return f"Error: {e}"

# Example usage (outside Gradio)
concept = "An AI assistant that automatically schedules meetings based on email threads."
json_output = generate_product_description(concept)
print(json_output)
# Expected: {"product_name": "SchedulAI", "tagline": "...", "key_features": ["...", "...", "..."], "target_audience": "..."}
```
*   **Use Case:** Quickly generating consistent, structured content for websites, catalogs, or marketing materials.

These examples demonstrate the versatility of JSON mode for various data extraction and generation tasks.

## Generate Images

Some Gemini models can generate images based on text prompts.

```python
# Required imports for image generation/display
from IPython.display import Image as IPImage, Markdown
# Assume 'client', 'types' are available
import base64 # For decoding image data if needed (inline_data)
import io # For handling byte streams for images
from PIL import Image as PILImage

# Select a model capable of image generation (often experimental or specific versions)
IMAGE_GEN_MODEL = "gemini-3.1-flash-lite-preview" # Update if needed

prompt = 'Create a 3d rendered image of a cat astronaut planting a flag on a cheese moon.'

try:
    # Configure the request to expect Text and Image modalities
    response = client.models.generate_content(
        model=IMAGE_GEN_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=['Text', 'Image'] # Specify expected output types
        )
    )

    # Process the response parts
    text_desc = ""
    generated_image = None
    for part in response.candidates[0].content.parts:
        if part.text is not None:
            text_desc += part.text + "\n"
            display(Markdown(part.text)) # Display text description
        elif part.inline_data is not None:
            # Handle image data (usually base64 encoded)
            mime_type = part.inline_data.mime_type
            if mime_type.startswith('image/'):
                image_data = part.inline_data.data
                # Use PIL to open image from bytes
                generated_image = PILImage.open(io.BytesIO(image_data))
                display(generated_image) # Display the generated image in the notebook

except Exception as e:
    print(f"An error occurred during image generation: {e}")
    # Check prompt feedback if available
    # if hasattr(response, 'prompt_feedback'): print(response.prompt_feedback)

```

*   **Explanation:** This section demonstrates text-to-image generation.
    *   A model capable of image generation must be selected (`IMAGE_GEN_MODEL`).
    *   `GenerateContentConfig`: The key here is `response_modalities=['Text', 'Image']`, indicating that the response might contain both text and image parts.
    *   Response Parsing: The response's `parts` list needs to be iterated. Text parts have a `text` attribute. Image parts often have `inline_data` containing the `mime_type` and the image `data` (frequently base64 encoded).
    *   The code checks the MIME type, decodes the data if necessary (implicitly handled by `PILImage.open(io.BytesIO(data))` if data is raw bytes), and uses PIL/IPython display functions to show the image.
    *   *(Note on Scope):* As mentioned before, while image generation works, displaying the `generated_image` reliably in a *Gradio output component* requires careful handling and is not part of the core lab requirement. The Gradio example provided in the original notebook attempts this but may face challenges.

### Image Generation with Gradio Interface (Conceptual / Demo Code)

The notebook includes Gradio code for image generation. We include it here for completeness, reiterating the scope note.

```python
# Imports from the image generation cell + Gradio
import gradio as gr
# ... other necessary imports: base64, io, PILImage, IPImage, Markdown, client, types ...

def generate_text_and_image(prompt):
    # Select appropriate model
    IMAGE_GEN_MODEL = "gemini-3.1-flash-lite-preview" # Update if needed
    config = types.GenerateContentConfig(
        response_modalities=['Text', 'Image']
    )
    text_output = ""
    image_output = None # Will hold the PIL image object for Gradio

    try:
        response = client.models.generate_content(
            model=IMAGE_GEN_MODEL,
            contents=prompt,
            config=config
        )

        for part in response.candidates[0].content.parts:
            if hasattr(part, "text") and part.text:
                text_output += part.text + "\n"
            elif hasattr(part, "inline_data") and part.inline_data:
                mime = part.inline_data.mime_type
                data = part.inline_data.data
                if mime.startswith('image'):
                    try:
                        # Decode if base64 - assumes raw bytes work directly with BytesIO here
                        image_data_bytes = data # Assume raw bytes if not explicitly base64
                        # If API returns base64 string: image_data_bytes = base64.b64decode(data)
                        image_output = PILImage.open(io.BytesIO(image_data_bytes))
                    except Exception as img_e:
                        print(f"Error processing image data: {img_e}")
                        image_output = None

        return text_output.strip(), image_output # Return text and PIL image object

    except Exception as api_e:
        print(f"API Error: {api_e}")
        return f"API Error: {api_e}", None


# Gradio interface
gr.Interface(
    fn=generate_text_and_image,
    inputs=gr.Textbox(label="Prompt", lines=2, placeholder="e.g., Create a 3D image of a flying cat..."),
    outputs=[
        gr.Markdown(label="Generated Description"),
        gr.Image(label="Generated Image", type="pil") # Output component for the PIL image
    ],
    title="Gemini: Text + Image Generator (Demo)",
    description="Send a prompt to Gemini and get back text and an AI-generated image. (Display in Gradio may depend on API/library versions)."
).launch()
```

*   **Explanation:** This Gradio interface attempts to display the generated image.
    *   The function `generate_text_and_image` calls the API requesting text and image.
    *   It parses the response, aiming to extract text into `text_output` and the generated image into `image_output` as a PIL Image object.
    *   The `gr.Interface` defines two outputs: `gr.Markdown` for the text and `gr.Image(type="pil")` for the image. Gradio attempts to render the returned PIL object.
    *   **Success is not guaranteed** and may depend on specific API response formats and library compatibility. This is provided as a demonstration from the notebook, not a required functional component for the lab.

## Generate Content Stream

For long responses, the API can "stream" the output, sending chunks as they are generated rather than waiting for the entire response.

### Basic Streaming

```python
# Assume 'client', 'MODEL_ID' are available

# Use generate_content_stream instead of generate_content
response_stream = client.models.generate_content_stream(
    model=MODEL_ID,
    contents="Tell me a story about a lonely robot who finds friendship in a most unexpected place."
    # Configuration (temperature etc.) can be passed via 'config=' argument here too
)

# Iterate through the stream chunks
print("--- Streaming Response ---")
for chunk in response_stream:
    if chunk.text: # Check if the chunk contains text
      print(chunk.text, end="") # Print chunk text without extra newlines
      # You might add a small delay or flush stdout if running in certain environments
      # import sys; sys.stdout.flush()
      # import time; time.sleep(0.1)
print("\n--- End of Stream ---")

# Note: The full response is not assembled automatically when streaming.
# You need to concatenate chunks yourself if the full text is needed afterwards.
# Accessing response_stream.text after iteration will likely fail or be empty.
```

*   **Explanation:** This code demonstrates receiving the response incrementally.
    *   `client.models.generate_content_stream()` is used instead of `generate_content()`. It returns an iterator immediately.
    *   The `for` loop iterates over the chunks as the model generates them.
    *   `chunk.text` accesses the text content of the current chunk.
    *   This provides a more responsive user experience for long generations, as text appears gradually. The full response needs to be manually assembled by concatenating the text from each chunk if required.

### Streaming with Gradio Interface

```python
import gradio as gr
# Assume 'client', 'MODEL_ID' are available

def stream_response_gradio(prompt):
    full_response = ""
    try:
        response_stream = client.models.generate_content_stream(
            model=MODEL_ID,
            contents=prompt
        )
        # Iterate and yield chunks for Gradio's streaming output
        for chunk in response_stream:
            if hasattr(chunk, "text") and chunk.text:
                full_response += chunk.text
                yield full_response # Yield the *cumulative* response so far
    except Exception as e:
        yield f"Error during streaming: {e}"

# Gradio interface for streaming
# Uses a generator function to update the output incrementally
gr.Interface(
    fn=stream_response_gradio, # Function is now a generator
    inputs=gr.Textbox(lines=2, label="Prompt", placeholder="e.g., Tell me a long story..."),
    outputs=gr.Textbox(lines=20, label="Streamed Output"), # Textbox updates as yielded
    title="Streaming Response Generator",
    description="Streams and displays the response from Gemini incrementally."
).launch()
```

*   **Explanation:** This Gradio interface displays the streamed response as it arrives.
    *   The function `stream_response_gradio` is now a *generator* function (it uses `yield`).
    *   It calls `generate_content_stream`.
    *   Inside the loop, it accumulates the response text in `full_response`.
    *   `yield full_response`: Instead of returning once at the end, it yields the current state of `full_response` after each chunk is received. Gradio's `gr.Textbox` output component automatically updates its content each time the function yields a value. This creates the effect of the text appearing incrementally in the UI.

<details>
<summary><strong>Concept Check: Return vs. Yield in Python/Gradio</strong></summary>

**Q: Why do we use `yield` instead of `return` in this streaming Gradio example?**<br>
**A:** A standard `return` statement immediately terminates a function and hands back a single, final output. `yield` turns a Python function into a *generator*. It hands back a value, pauses the function, and waits to be called again to generate the next value. Gradio is specifically designed to recognize generators. When it sees `yield`, it keeps the web interface connection open and visually updates the text box every single time a new `yield` is executed, creating the typewriter "streaming" effect.
</details>

## Upload Files (File API)

For larger files or files used repeatedly, the File API allows uploading them first and then referencing them in prompts. This is often necessary for multimodal inputs beyond small, directly included images.

### Overview

The process generally involves:
1.  Preparing the file (downloading or accessing locally).
2.  Uploading the file using `client.files.upload()`. This returns a `File` object.
3.  Waiting for the file state to become `ACTIVE` (especially important for video).
4.  Passing the `File` object (or its `uri`) in the `contents` list when calling `generate_content`.

<details>
<summary><strong>Test Your Knowledge: The File API vs. Direct Uploads</strong></summary>

**Q: Earlier in the Multimodal section, we sent an image directly to `generate_content()`. Why do we need a separate `client.files.upload()` API now?**<br>
**A:** Sending a file directly in `generate_content` (passing the raw bytes or PIL image) works fine for small, single images. However, the API has payload size limits. For massive text files, large PDFs, hour-long audio files, or video files, sending the raw data inside the prompt is impossible or highly inefficient. `client.files.upload` uploads the file to Google's servers *first*, giving you a temporary link (`URI`). You then send just that lightweight link in your prompt.
</details>

### Upload an Image File

```python
# Assume necessary imports: requests, pathlib, client, MODEL_ID, Markdown
import requests # To download the image
import pathlib # To handle file paths
from PIL import Image # To work with the image object

# 1. Prepare the file
IMG_URL = "https://storage.googleapis.com/generativeai-downloads/data/jetpack.png"
img_bytes = requests.get(IMG_URL).content
img_path = pathlib.Path('jetpack_uploaded.png') # Use a distinct name
img_path.write_bytes(img_bytes)

# 2. Upload the file using the API
print(f"Uploading file: {img_path}...")
file_upload = client.files.upload(file=img_path)
print(f"Completed upload: {file_upload.uri}, State: {file_upload.state}") # State is usually ACTIVE quickly for images

# 3. Use the uploaded file in a prompt
prompt = "Write a short technical description of the device shown in the image."
response = client.models.generate_content(
    model=MODEL_ID, # Use a multimodal model
    contents=[
        file_upload, # Pass the File object directly
        prompt,
    ]
)

Markdown(response.text)
```
*   **Explanation:** Uploads an image via the File API and then uses it in a prompt. The `file_upload` object returned by `client.files.upload` is passed directly in the `contents` list.

### Upload Text File

```python
# Assume necessary imports: requests, pathlib, client, MODEL_ID, Markdown

# 1. Prepare the file (large text file example)
TEXT_URL = "https://storage.googleapis.com/generativeai-downloads/data/a11.txt"
text_bytes = requests.get(TEXT_URL).content
text_path = pathlib.Path('a11_transcript.txt')
text_path.write_bytes(text_bytes)

# 2. Upload the file
print(f"Uploading file: {text_path}...")
file_upload = client.files.upload(file=text_path)
print(f"Completed upload: {file_upload.uri}, State: {file_upload.state}")

# 3. Use the uploaded file
prompt = "Summarize the key events mentioned in the first part of this transcript."
response = client.models.generate_content(
    model=MODEL_ID,
    contents=[
        file_upload,
        prompt,
    ]
)
Markdown(response.text)
```
*   **Explanation:** Similar process for a text file. Useful for providing large amounts of text context that might exceed standard prompt limits.

### Upload a PDF File

```python
# Assume necessary imports: requests, pathlib, client, MODEL_ID, Markdown

# 1. Prepare the file
PDF_URL = "https://storage.googleapis.com/generativeai-downloads/data/Smoothly%20editing%20material%20properties%20of%20objects%20with%20text-to-image%20models%20and%20synthetic%20data.pdf"
pdf_bytes = requests.get(PDF_URL).content
pdf_path = pathlib.Path('google_research_article.pdf')
pdf_path.write_bytes(pdf_bytes)

# 2. Upload the file
print(f"Uploading file: {pdf_path}...")
file_upload = client.files.upload(file=pdf_path)
print(f"Completed upload: {file_upload.uri}, State: {file_upload.state}")

# 3. Use the uploaded file
prompt = "List the main contributions of this research paper as bullet points."
response = client.models.generate_content(
    model=MODEL_ID, # Ensure model supports PDF input
    contents=[
        file_upload,
        prompt,
    ]
)
Markdown(response.text)
```
*   **Explanation:** Demonstrates uploading and analyzing a PDF document.

### Upload an Audio File

```python
# Assume necessary imports: requests, pathlib, client, MODEL_ID, Markdown

# 1. Prepare the file
AUDIO_URL = "https://storage.googleapis.com/generativeai-downloads/data/State_of_the_Union_Address_30_January_1961.mp3"
audio_bytes = requests.get(AUDIO_URL).content
audio_path = pathlib.Path('jfk_speech.mp3')
audio_path.write_bytes(audio_bytes)

# 2. Upload the file
print(f"Uploading file: {audio_path}...")
file_upload = client.files.upload(file=audio_path)
print(f"Completed upload: {file_upload.uri}, State: {file_upload.state}")
# Audio/Video might take longer to process, state might be PROCESSING initially

# Optional: Wait for processing (simple loop, better check might be needed)
import time
while file_upload.state == 'PROCESSING':
    print("Waiting for audio processing...")
    time.sleep(5)
    file_upload = client.files.get(name=file_upload.name) # Refresh file state
    print(f"Current state: {file_upload.state}")

if file_upload.state != 'ACTIVE':
    print(f"File processing failed or timed out. State: {file_upload.state}")
else:
    # 3. Use the uploaded file (only if ACTIVE)
    prompt = "What are the main themes discussed in this speech audio?"
    response = client.models.generate_content(
        model=MODEL_ID, # Ensure model supports audio input
        contents=[
            file_upload,
            prompt,
        ]
    )
    Markdown(response.text)

```
*   **Explanation:** Shows audio file upload. Includes a basic check for the file processing state, as audio/video often require server-side processing after upload before they can be used. The `client.files.get(name=...)` method is used to refresh the file's status.

### Upload a Video File

```python
# Assume necessary imports: pathlib, client, MODEL_ID, Markdown, time
# Requires wget to be installed in the environment
import os

# 1. Prepare the file (Download using wget)
VIDEO_URL = "https://download.blender.org/peach/bigbuckbunny_movies/BigBuckBunny_320x180.mp4"
video_file_name = "BigBuckBunny_320x180.mp4"
# Use os.system to run wget
os.system(f"wget -q -O {video_file_name} {VIDEO_URL}")
video_path = pathlib.Path(video_file_name)

if not video_path.is_file():
    print("Video download failed.")
else:
    # 2. Upload the file
    print(f"Uploading file: {video_path}...")
    video_file = client.files.upload(file=video_path)
    print(f"Initial upload status: {video_file.uri}, State: {video_file.state}")

    # 3. Wait for processing to complete
    while video_file.state == "PROCESSING":
        print(f'Waiting for video ({video_file.name}) to be processed...')
        time.sleep(10) # Check every 10 seconds
        try:
            video_file = client.files.get(name=video_file.name) # Refresh state
        except Exception as e:
            print(f"Error getting file status: {e}")
            break # Exit loop on error

    if video_file.state != "ACTIVE":
      print(f"Video processing failed or stopped. Final State: {video_file.state}")
    else:
      # 4. Use the uploaded file in a prompt (only if ACTIVE)
      print(f'Video processing complete: {video_file.uri}')
      prompt = "Describe the main events happening in this short video clip."
      response = client.models.generate_content(
          model=MODEL_ID, # Ensure model supports video input
          contents=[
              video_file,
              prompt,
          ]
      )
      Markdown(response.text)

    # Optional: Clean up downloaded file
    # os.remove(video_path)
```
*   **Explanation:** Handles video upload, including the crucial step of waiting for the file's state to become `ACTIVE` using a loop and `client.files.get()`. Video processing can take significantly longer than other file types.

### Process a YouTube Link

YouTube videos can be processed directly without uploading the video file, by providing the URL using a specific structure.

```python
# Assume 'client', 'MODEL_ID', 'types', 'Markdown' are available

youtube_url = 'https://www.youtube.com/watch?v=WsEQjeZoEng' # Google I/O 2024 example

# Construct the 'contents' using types.Part and types.FileData
video_prompt_content = types.Content(
    parts=[
        types.Part(text="Provide a concise summary of the key announcements in this video."),
        types.Part(
            file_data=types.FileData(
                mime_type="video/mp4", # Specify mime type (optional but good practice)
                file_uri=youtube_url   # Use file_uri for URLs
            )
        )
    ]
)

# Send the request
try:
    response = client.models.generate_content(
        model=MODEL_ID, # Ensure model supports video/YouTube input
        contents=video_prompt_content
    )
    Markdown(response.text)
except Exception as e:
    print(f"An error occurred processing YouTube URL: {e}")

```
*   **Explanation:** Demonstrates analyzing a YouTube video directly.
    *   Instead of uploading, the `contents` list includes a `types.Part` containing `types.FileData`.
    *   `file_uri=youtube_url` is used to specify the video source.
    *   Providing the `mime_type` (e.g., "video/mp4") is recommended.
    *   Note the limitations mentioned in the notebook: usually only one YouTube link per request, and it must be provided via `FileData`, not just embedded in the text prompt.

## Instruct Prompting Practice

Obtaining desired results from LLMs often requires careful prompt formulation. This is known as **Instruct Prompting**. Key elements include:

*   **Clear Task:** State precisely what the model should do.
*   **Context:** Provide necessary background information.
*   **Persona:** Define the role the model should adopt (e.g., "Act as..."). Use `system_instruction` in chat or include in the prompt.
*   **Format:** Specify the desired output structure (list, JSON, paragraph count, etc.). Crucial for JSON mode.
*   **Constraints:** Define what to include/exclude (tone, length, specific elements).

### Practice Exercises (For Lab/Self-Study)

Use the techniques learned (basic `generate_content`, chat, JSON mode) to attempt the following:

1.  **Persona and Constraints:**
    *   **Task:** Formulate a prompt requesting an explanation of nuclear fusion.
    *   **Constraints:** Target audience: high-school physics student. Tone: neutral science educator. Length: under 150 words. Factual accuracy is essential.
    *   *(Hint: Use `generate_content` or the chat interface with a system instruction.)*

2.  **Improving Specificity:**
    *   **Initial Prompt:** "Explain cloud computing."
    *   **Desired Output:** A brief comparison of IaaS, PaaS, SaaS, focusing on user management responsibilities, presented as three short paragraphs.
    *   **Task:** Rewrite the initial prompt to achieve the specific desired output and format.
    *   *(Hint: Clearly state the comparison goal and the required output structure in the prompt.)*

3.  **Structured Formatting (JSON):**
    *   **Task:** Formulate a prompt asking for the capitals of Finland, Sweden, and Norway.
    *   **Constraint:** The output *must* be a JSON object where keys are the country names (lowercase) and values are the capital cities.
    *   *(Hint: Use `generate_content` with `response_mime_type="application/json"` and clearly state the desired JSON structure in the prompt. You might optionally provide a Pydantic model or JSON schema.)*

## Conclusion

This session covered the fundamentals of interacting with the Gemini API using the Python SDK. We explored setup, sending text and multimodal prompts, configuring generation parameters and safety settings, managing chat conversations, generating structured JSON output, streaming responses, and utilizing the File API for various media types. The importance of effective instruct prompting was also highlighted.

The accompanying lab will provide hands-on practice with these concepts, applying them within Gradio interfaces and preparing you to integrate Gemini calls into your Phase 2 of the mini-project. 

<details>
<summary>In Brief: how to interact with the Gemini API using the client library</summary>


Here is a summary of how to interact with the Gemini API using the client library.

**Core Interaction Methods**

You have three main ways to generate content with the Gemini API client:

1.  **`client.models.generate_content`**:
    *   **Use Case:** For single-turn requests where you send a prompt and receive a complete response back at once.
    *   **Mechanism:** Sends the entire prompt, waits for the model to process, and returns the full response. Suitable for non-conversational tasks or simple questions.

2.  **`client.models.generate_content_stream`**:
    *   **Use Case:** For single-turn requests where you want to receive the response progressively as it's generated.
    *   **Mechanism:** Sends the prompt, and the API returns the response in chunks (streams) as they become available. Useful for long responses or displaying results incrementally to the user.

3.  **`client.chats.create`**:
    *   **Use Case:** For multi-turn conversations (like a chatbot).
    *   **Mechanism:** Manages the conversation history automatically. You send user messages, and it maintains the context for subsequent interactions within that chat session.

**Uploading Files**

*   **`client.files.upload`**:
    *   **Use Case:** To use local files (like images, audio, video) as part of your input prompt (multimodal input).
    *   **Mechanism:** Uploads the specified file to the API and returns a reference (e.g., a URI). You then include this reference within the `contents` of your generation request.

**General Usage Pattern (Example with `generate_content`)**

The most common pattern for a single request involves defining the model, the prompt (contents), and optional configuration:

```python
# Import necessary libraries (assuming already done)
# import google.generativeai as genai
# from pydantic import BaseModel, Field # If using response_schema
# import google.ai.generativelanguage as types # For config types

# Initialize the client (Requires API Key setup - not shown here)
# client = genai.GenerativeModel(...) or similar initialization

# Define the model ID you want to use
MODEL_ID = "gemini-1.5-flash" # Or another appropriate model

# Define your input prompt
prompt = "Describe the key features of Python."
# For multimodal input, 'contents' would include file references from client.files.upload

# Define optional configuration settings
config = types.GenerateContentConfig(
    # Controls randomness (0.0 = deterministic, higher = more creative)
    temperature=0.4,
    # Maximum number of tokens in the generated response
    max_output_tokens=200,
    # Safety filters (example below)
    safety_settings=[
        types.SafetySetting(
            category="HARM_CATEGORY_DANGEROUS_CONTENT",
            threshold="BLOCK_ONLY_HIGH",
        ),
        # Add more safety settings as needed
    ],
    # If expecting structured JSON output:
    # response_mime_type="application/json",
    # response_schema=YourPydanticModel # Define YourPydanticModel using BaseModel
)

# Make the API call
response = client.models.generate_content(
    model=MODEL_ID,
    contents=prompt,
    config=config
)

# Process the response (access response.text, response.parts, etc.)
print(response.text)
```

**Configuration (`config`) Properties Explained**

The `config` object fine-tunes the generation process:

*   `safety_settings`: Allows you to set thresholds for blocking potentially harmful content based on categories (e.g., Dangerous Content, Harassment). You define the category and the blocking level (e.g., `BLOCK_ONLY_HIGH`, `BLOCK_MEDIUM_AND_ABOVE`).
*   `response_mime_type`: Use `"application/json"` when you want the model to output structured JSON data.
*   `response_schema`: Provide a Pydantic `BaseModel` class here (like the `ContactInfo` example) when using `response_mime_type="application/json"`. This instructs the model to format its output according to the defined schema.
*   `temperature`: Controls the randomness of the output. Lower values (e.g., 0.2) make the output more focused and deterministic. Higher values (e.g., 0.9) make it more creative and diverse.
*   `max_output_tokens`: Sets a limit on the length of the generated response, measured in tokens (pieces of words).

This covers the primary methods for interacting with the Gemini API client for text generation and file handling. Remember to handle authentication and client initialization before making these calls.
</details>


----

<details>
<summary><strong>Deep Dive: Nuances of the Gemini API</strong></summary>

<br>

As you transition from learning the basics to building real applications, keep the following API nuances in mind.

### 1. Colab-Specific Magic (`@param`)

In the setup section, you saw this comment:
`MODEL_ID = "gemini-2.5-flash" # @param ["gemini-2.5-flash-lite", ...]`

This `# @param` syntax is a specific feature of **Google Colab**. It tells the Colab interface to generate a visual dropdown menu on the right side of the code cell. If you copy this code into a standard local Python script or Jupyter Notebook, the comment is ignored, and the dropdown will not appear. The code will still function correctly by defaulting to the string value provided.

### 2. Comprehensive Safety Filters

The safety settings introduced earlier use specific enumeration values to categorize and threshold content. When configuring `SafetySetting`, the available categories typically include:
*   `HARM_CATEGORY_HARASSMENT`
*   `HARM_CATEGORY_HATE_SPEECH`
*   `HARM_CATEGORY_SEXUALLY_EXPLICIT`
*   `HARM_CATEGORY_DANGEROUS_CONTENT`

The threshold determines how strictly the filter is applied. Available thresholds include:
*   `BLOCK_NONE`: Turns off the filter (use with caution).
*   `BLOCK_ONLY_HIGH`: Blocks content with a high probability of being harmful.
*   `BLOCK_MEDIUM_AND_ABOVE`: Blocks medium and high probability content.
*   `BLOCK_LOW_AND_ABOVE`: The strictest setting.

If your application returns empty text or raises an error on seemingly benign prompts, check if your safety filters are set too aggressively.

### 3. The Importance of Structured JSON

While we covered generating text and images, **JSON structured output** (using Pydantic schemas) is arguably the most critical feature for developers. 

When you build a software application, you cannot rely on an LLM to reliably return conversational text (e.g., *"Here is the data you requested: Name: Alice, Age: 30"*). Software requires predictable data structures to populate databases, trigger functions, or render UI elements. Using `response_mime_type="application/json"` combined with a strict schema bridges the gap between unpredictable AI text generation and deterministic software engineering.

**Concept Check: Error Handling**

**Q: What happens if the API rate limit is exceeded?**<br>
**A:** The API will throw an exception (typically a `429 Too Many Requests` error). In production code, you should wrap your `generate_content` calls in `try...except` blocks and implement "exponential backoff" (waiting a few seconds and retrying automatically) to ensure your application doesn't crash when the API is temporarily overwhelmed.

</details>


---
## Useful Links

-[https://ai.google.dev/gemini-api/docs/file-prompting-strategies](https://ai.google.dev/gemini-api/docs/file-prompting-strategies)
-[Gemini API: Getting started with Gemini 2.](https://colab.research.google.com/github/google-gemini/cookbook/blob/main/quickstarts/Get_started.ipynb)
- [Gemini API: Authentication Quickstart](https://colab.research.google.com/github/google-gemini/cookbook/blob/main/quickstarts/Authentication.ipynb)
- [Gemini Models](https://ai.google.dev/gemini-api/docs/models) 
- [Gemini QuickStart](https://ai.google.dev/gemini-api/docs/quickstart?lang=python) 
- [Free images](https://unsplash.com/images/stock/public-domain) 



<!-- 
-[How to train a new language model from scratch using Transformers and Tokenizers](https://huggingface.co/blog/how-to-train)  
-->