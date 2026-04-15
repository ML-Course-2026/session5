# Exploring the Gemini API

**Objective:** To collaboratively interact with a large language model (LLM) via an Application Programming Interface (API), we’ll use Google's models as our primary example. This approach enables us to leverage powerful, pre-trained models without managing the underlying infrastructure. We’ll explore the capabilities of the LLM API and demonstrate how to integrate its features into simple Gradio interfaces.

**Instructions for Breakout Rooms:**
*   Work together in your assigned group.
*   One member should share their screen with a Google Colab notebook environment ready.
*   Discuss each task, review the corresponding lecture material or demo notebook sections, and implement the code collaboratively.
*   The goal is understanding and experimentation, not just copying code. Discuss the results and any challenges encountered.
*   If you get stuck on a task, refer to the solution provided within the `<details>` tag. Try to solve it first before looking.


> [!NOTE]
> When using the free tier of the Gemini API, you may occasionally encounter errors indicating that the service is overwhelmed or that rate limits have been reached (such as 429 Too Many Requests or 503 Service Unavailable). If your code fails with an API-related error, it is often a temporary issue. Wait a few moments and try executing the cell again.

## Part 1: Setup and Initialization

This section ensures your environment is correctly configured to use the Gemini API. Follow these steps carefully.

### 1.1 Create an API key

If you haven't already, you need a Google API key to use Gemini.
You can [create](https://aistudio.google.com/app/apikey) your API key using Google AI Studio with a single click. Follow the instructions provided there.

**Important:** Treat your API key like a password. Do not share it publicly or commit it to version control systems like GitHub.

<details>
<summary><strong>Concept Check: Why are API Keys so sensitive?</strong></summary>

**Q: What is the worst that can happen if I accidentally upload my API key to GitHub?**<br>
**A:** Automated bots scrape public GitHub repositories 24/7 looking for exposed API keys. If they find yours, they can use your account to run massive, expensive computing jobs (like generating thousands of images or running automated bots). Because the key is tied to your account, you (or your organization) are financially responsible for the computing bill. This is why tools like Colab Secrets or `.env` files are mandatory best practices.
</details>

### 1.2 Add your key to Colab Secrets

Using Colab Secrets is the recommended way to handle your API key securely within Google Colab.

1.  Open your Google Colab notebook.
2.  Click on the **🔑 Secrets** tab in the left panel. (Refer to the image in the lecture notes if needed).
3.  Create a new secret. Enter the name `GOOGLE_API_KEY`.
4.  Paste the API key you created in step 1.1 into the `Value` input box.
5.  Ensure the **"Notebook access"** toggle button on the left is enabled (usually blue/on).

### 1.3 Install SDK and Initialize Client

Now, write and execute the Python code to install the necessary library and set up the API client using your stored secret key.

1.  **Install the SDK:** Run the following command in a code cell.
    ```python
    %pip install -q google-genai gradio
    ```
    *(Explanation: This installs the Google GenAI SDK and the Gradio library.)*

2.  **Import Libraries:** Run the following code in a *new* cell to import all the modules needed for this lab.
    ```python
    # Core Gemini and Colab libraries
    from google.colab import userdata
    from google import genai
    from google.genai import types # types is used for specific configurations later

    # Gradio for UI
    import gradio as gr

    # Utilities for display, file handling, JSON, etc.
    from IPython.display import Markdown, Image as IPImage
    import json
    from pydantic import BaseModel, Field # For structured JSON later
    from typing import Optional, List    # For type hinting later
    import requests                      # For downloading files
    import pathlib                       # For handling file paths
    from PIL import Image as PILImage    # For image manipulation
    import io                            # For handling byte streams
    import time                          # For delays (e.g., waiting for file processing)
    import base64                        # For potential image decoding

    print("Libraries imported successfully.")
    ```
    *(Explanation: This cell brings all the necessary Python tools into your notebook's memory so you can use their functions and classes.)*

### 1.4 Initialize API Client and Select Model

Retrieve your API key from Colab Secrets and use it to create the Gemini client object.

1.  **Retrieve Key and Initialize Client:** Run the following code in a *new* cell. This assumes your `GOOGLE_API_KEY` secret is correctly set up as per step 1.2.
    ```python
    # Retrieve the API key from Colab Secrets
    GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')

    # Initialize the client - This line connects to Google's service
    # If the key is invalid or missing, this line will likely cause an error.
    client = genai.Client(api_key=GOOGLE_API_KEY)

    print("Gemini Client initialized.")

    # Choose a model ID to use for subsequent requests
    MODEL_ID = "gemini-2.5-flash" # @param["gemini-2.5-flash-lite", "gemini-3.1-flash-lite-preview"] {"allow-input":true, isTemplate: true}    
    print(f"Using Model ID: {MODEL_ID}")
    ```
    *(Explanation: This code gets your secret key, uses it to create the `client` object which is your main tool for talking to Gemini, and sets a default `MODEL_ID` variable.)*

    **Important Note:** If the cell above fails, double-check that:
    *   You completed step 1.2 correctly (Secret name is `GOOGLE_API_KEY`, value is correct, Notebook access is ON).
    *   You have internet connectivity.
    *   The API key itself is valid.

<details>
<summary>More Robust Initialization Code (Optional)</summary>

For situations where you want to handle potential errors during initialization more gracefully (e.g., if the key is missing or invalid), you could use a `try...except` block like this:

```python
# Robust Initialization Example:
try:
    GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
    if not GOOGLE_API_KEY:
        # Raise an error specifically if the key wasn't found in secrets
        raise ValueError("API Key 'GOOGLE_API_KEY' not found in Colab Secrets. Please ensure it's set correctly and Notebook access is enabled.")

    # Initialize the client
    client = genai.Client(api_key=GOOGLE_API_KEY)
    print("Successfully initialized Gemini Client.")

    # Choose a model (only if client was initialized)
    MODEL_ID = "gemini-2.5-flash" # @param["gemini-2.5-flash-lite", "gemini-3.1-flash-lite-preview"] {"allow-input":true, isTemplate: true}    print(f"Using Model ID: {MODEL_ID}")

# Catch any exception during the process
except Exception as e:
    print(f"Error during initialization: {e}")
    print("\nPlease check:")
    print("- API key is correct in Google AI Studio / Cloud Console.")
    print("- Secret 'GOOGLE_API_KEY' exists in Colab Secrets with the correct value.")
    print("- 'Notebook access' is enabled for the secret.")
    print("- Internet connection is active.")
    # Prevent further code relying on 'client' from running if it failed
    # You might need to handle the absence of 'client' in later cells if using this block.
```

This robust version provides more specific feedback if something goes wrong, but the simpler version above is sufficient if you carefully follow the setup steps.

</details>

**Group Discussion:** Confirm that everyone understands the purpose of the API key, Colab Secrets, the SDK, and the client initialization. Ensure the client initializes successfully (you should see the "Gemini Client initialized" message) before proceeding to the next part of the lab. If there are errors, troubleshoot using the 'Important Note' points above.

## Part 2: Basic Text Generation

Explore sending simple text prompts and displaying responses.

### Task 2.1: Simple Text Prompt

1.  Write code to ask the Gemini model a factual question, such as "What is the capital of France?".
2.  Use the `client.models.generate_content` method with your `MODEL_ID` and the prompt.
3.  Print the `response.text`.

<details>
<summary>Solution Code</summary>

```python
# Assumes 'client' was initialized successfully in the previous step.
prompt = "What is the capital of Finland?"
response = client.models.generate_content(
    model=MODEL_ID,
    contents=prompt
)
print(f"Prompt: {prompt}")
print(f"Response:\n{response.text}")
```

</details>

<details>
<summary><strong>📡 Concept Check: Under the Hood of an API Call</strong></summary>

**Q: What is actually happening when you run `generate_content`? Does the model live on your computer?**<br>
**A:** No, the model lives on massive servers in Google's data centers. When you call this function, the Python SDK takes your text, packages it into an HTTP Request (specifically a JSON payload), and sends it over the internet to Google. Google's servers run the complex neural network calculations, generate the text, and send an HTTP Response back to your notebook. This is why an active internet connection is strictly required for this lab.
</details>

### Task 2.2: Text Prompt with Gradio

1.  Adapt the code from the lecture/demo notebook to create a simple Gradio interface.
2.  The interface should have one `gr.Textbox` for input and one `gr.Markdown` for output.
3.  The function called by Gradio should take the user's prompt, call `generate_content`, and return the `response.text`.
4.  Test the interface with a few different prompts.

<details>
<summary>Solution Code</summary>

```python
# Assumes 'client' was initialized successfully.
def ask_gemini_gradio(user_prompt):
    if not user_prompt:
        return "Please enter a prompt."
    # Note: If API call fails here, Gradio might show a generic error.
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=user_prompt
    )
    return response.text

# Create Gradio interface
text_qa_interface = gr.Interface(
    fn=ask_gemini_gradio,
    inputs=gr.Textbox(lines=3, placeholder="Enter your question here...", label="Your Prompt"),
    outputs=gr.Markdown(label="Gemini Response"),
    title="Simple Gemini Q&A",
    description="Enter a prompt and get a text response from the Gemini model."
)
text_qa_interface.launch()

```

</details>

**Group Discussion:** Discuss the difference between running the code directly and using the Gradio interface. How does Gradio simplify interaction for a non-programmer?

## Part 3: Multimodal Input (Image + Text)

Explore sending prompts that include both images and text.

### Task 3.1: Image Analysis Prompt

1.  Find the URL of an image online (e.g., a picture of a specific animal or object).
2.  Write code to:
    *   Download the image using `requests`.
    *   Open the image using `PIL.Image`.
    *   Send a prompt to Gemini containing *both* the PIL image object and a text question about the image (e.g., "What type of animal is this?", "Describe the main object in this image.").
    *   Use a model that supports multimodal input (e.g., `gemini-1.5-flash-latest`).
    *   Print the `response.text`.

<details>
<summary>Solution Code</summary>

```python
# Assumes 'client' was initialized successfully.
# Example Image URL (replace with your own if desired)
IMAGE_URL = "https://storage.googleapis.com/generativeai-downloads/cli/oak_tree_story/scene1.png" # Google Pixel phone

# Download and open image
img_bytes = requests.get(IMAGE_URL).content
pil_image = PILImage.open(io.BytesIO(img_bytes))

# Prepare prompt
text_prompt = "Describe the object shown in this image. What might it be used for?"

# Send multimodal request
response = client.models.generate_content(
    model=MODEL_ID, # Assumes MODEL_ID is multimodal capable
    contents=[pil_image, text_prompt] # List contains Image and Text
)

print(f"Text Prompt: {text_prompt}")
print(f"Response:\n{response.text}")

# Optionally display the image in Colab
# from IPython.display import display
# display(pil_image.resize((200,200)))

```

</details>

<details>
<summary><strong>Concept Check: How does an LLM "see" an image?</strong></summary>

**Q: A language model is built to predict text. How can it process an image?**<br>
**A:** Multimodal models like Gemini use a specialized vision encoder. Before the data reaches the core text generation model, the image is passed through a neural network (like a Convolutional Neural Network or Vision Transformer) that converts the pixels into mathematical vectors (embeddings) that occupy the same "mathematical space" as words. To the core model, the image simply looks like a very long string of highly descriptive, mathematically encoded "words."
</details>

### Task 3.2: Image Analysis with Gradio

1.  Adapt the multimodal Gradio example from the lecture/notebook.
2.  The interface should take an uploaded image (`gr.Image(type="pil")`) and a text prompt (`gr.Textbox`).
3.  The function should send both the image (as a PIL object) and the text prompt to Gemini.
4.  The output should be the text response displayed in `gr.Markdown`.
5.  Test by uploading an image and asking a question about it.

<details>
<summary>Solution Code</summary>

```python
# Assumes 'client' was initialized successfully.
def analyze_image_gradio(image_input, text_prompt):
    if image_input is None:
        return "Please upload an image."
    if not text_prompt:
        return "Please enter a text prompt."

    # image_input is already a PIL image due to type="pil"
    pil_image = image_input

    # Note: If API call fails here, Gradio might show a generic error.
    response = client.models.generate_content(
        model=MODEL_ID, # Assumes multimodal capable model
        contents=[pil_image, text_prompt]
    )
    return response.text

# Create Gradio interface
image_analysis_interface = gr.Interface(
    fn=analyze_image_gradio,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Textbox(lines=2, placeholder="Ask something about the image...", label="Text Prompt")
    ],
    outputs=gr.Markdown(label="Analysis Response"),
    title="Image + Text Analysis with Gemini",
    description="Upload an image and provide a text prompt for Gemini to analyze."
)
image_analysis_interface.launch()

```

</details>

**Group Discussion:** Discuss potential applications for multimodal models that can understand both images and text.

## Part 4: Configuring Generation Parameters

Explore how changing parameters like `temperature` affects the model's output.

### Task 4.1: Experiment with Temperature

1.  Choose a creative prompt (e.g., "Write a short poem about a rainy day in Helsinki.").
2.  Call `client.models.generate_content` with this prompt **twice**:
    *   First time, set `config=types.GenerateContentConfig(temperature=0.1)`.
    *   Second time, set `config=types.GenerateContentConfig(temperature=0.9)`.
3.  Print both responses and compare them.

<details>
<summary>Solution Code</summary>

```python
# Assumes 'client' was initialized successfully.
creative_prompt = "Write a short poem about a rainy day in Helsinki."
print(f"Prompt: {creative_prompt}\n")

# Low temperature (more focused)
config_low_temp = types.GenerateContentConfig(temperature=0.1)
response_low = client.models.generate_content(
    model=MODEL_ID,
    contents=creative_prompt,
    config=config_low_temp
)
print(f"--- Response (Temperature: 0.1) ---\n{response_low.text}\n")

# High temperature (more creative/random)
config_high_temp = types.GenerateContentConfig(temperature=0.9)
response_high = client.models.generate_content(
    model=MODEL_ID,
    contents=creative_prompt,
    config=config_high_temp
)
print(f"--- Response (Temperature: 0.9) ---\n{response_high.text}\n")

```

</details>

<details>
<summary><strong>Concept Check: The Math of Temperature</strong></summary>

**Q: Does increasing temperature make the model "smarter"?**<br>
**A:** No. An LLM works by calculating the probability of every possible next word. For example, after the phrase "The sky is", the model might calculate "blue" (90%), "dark" (5%), "falling" (1%). 
*   A **low temperature** (e.g., 0.1) makes the math "sharper"—the model almost exclusively picks the highest probability word (blue), leading to predictable, safe, and factual answers.
*   A **high temperature** (e.g., 0.9) "flattens" the probabilities, giving the lower-probability words a fair chance of being selected. This leads to more varied, surprising, and creative text, but increases the risk of hallucinations.
</details>

### Task 4.2: Parameter Control with Gradio

1.  Use the Gradio example from the lecture/notebook that allows controlling `temperature`, `max_output_tokens`, and potentially `top_k` or `top_p` via sliders/number inputs.
2.  Run the interface.
3.  Experiment with a single prompt but adjust the parameters using the UI controls. Observe how the output changes.

<details>
<summary>Solution Code</summary>

```python
# Assumes 'client' was initialized successfully.
# Ensure the function definition and gr.Interface call are present and executable.
def generate_response_configured(prompt, temperature, max_tokens): # Simplified version
    if not prompt:
        return "Please enter a prompt."

    config = types.GenerateContentConfig(
        temperature=float(temperature),
        max_output_tokens=int(max_tokens)
    )
    # Note: If API call fails here, Gradio might show a generic error.
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=prompt,
        config=config
    )
    return response.text

config_interface = gr.Interface(
    fn=generate_response_configured,
    inputs=[
        gr.Textbox(label="Prompt", lines=3),
        gr.Slider(0.0, 1.0, value=0.5, step=0.1, label="Temperature"),
        gr.Number(value=150, label="Max Output Tokens", precision=0)
    ],
    outputs=gr.Markdown(label="Model Response"),
    title="Gemini Prompt with Config Control",
    description="Experiment with temperature and max tokens."
)
config_interface.launch()

# Note: The full example in the lecture included more parameters (top_k, top_p, etc.)
# You can use that more complete version if preferred.
```

</details>

**Group Discussion:** Discuss when you might want a lower temperature versus a higher temperature for different types of tasks (e.g., factual summary vs. brainstorming).

## Part 5: Multi-turn Chat

Explore maintaining conversation context.

### Task 5.1: Simple Chat Sequence

1.  Referencing the chat example in the lecture/notebook:
    *   Create a chat session using `client.chats.create()`. You can optionally add a `system_instruction` (e.g., "You are a helpful assistant.").
    *   Send an initial message using `chat.send_message()` (e.g., "What are the main steps to bake bread?"). Print the response.
    *   Send a follow-up message that relies on the previous context (e.g., "What kind of flour is best for the first step?"). Print the response.

<details>
<summary>Solution Code</summary>

```python
# Assumes 'client' was initialized successfully.
# Start chat
chat = client.chats.create(
    model=MODEL_ID,
    config=types.GenerateContentConfig(
        system_instruction="You are a helpful baking assistant."
    )
)
print("Chat session started.\n")

# First message
msg1 = "What are the main steps to bake a simple loaf of bread?"
print(f"User: {msg1}")
response1 = chat.send_message(msg1)
print(f"Assistant:\n{response1.text}\n")

# Second message (contextual)
msg2 = "What kind of flour is generally recommended for the kneading step?"
print(f"User: {msg2}")
response2 = chat.send_message(msg2)
print(f"Assistant:\n{response2.text}\n")

```

</details>

<details>
<summary><strong>Concept Check: Does the Model have Memory?</strong></summary>

**Q: How does the model remember what we said in `msg1` when we ask `msg2`? Does it learn in real-time?**<br>
**A:** LLMs are stateless; they do not "learn" or permanently memorize your conversation. When you use `chat.send_message()`, the Google GenAI SDK silently takes your entire conversation history (msg1 + response1 + msg2) and sends the *entire transcript* back to the model in a single massive payload. The model reads the whole script from the top every single time you press enter. This is why conversations eventually hit a "context length" limit if they go on too long!
</details>

**Group Discussion:** How does the `chat` object help maintain context? Discuss the limitation of the simple Gradio chat example provided in the lecture (it starts a new chat each time).

## Part 6: Structured Output (JSON)

Force the model to output structured JSON data.

### Task 6.1: Generate JSON with Pydantic

1.  Define a Pydantic `BaseModel` for extracting information about a city: `name` (string), `country` (string), `population` (integer, optional).
2.  Write a prompt asking for information about a specific city (e.g., "Provide details for Helsinki: country and approximate population.").
3.  Call `generate_content` using `response_mime_type="application/json"` and providing your Pydantic model as the `response_schema`.
4.  Print the resulting `response.text` (which should be a JSON string).

<details>
<summary>Solution Code</summary>

```python
# Assumes 'client' was initialized successfully.
# Define Pydantic Model
class CityInfo(BaseModel):
    name: str = Field(description="Name of the city")
    country: str = Field(description="Country the city is in")
    population: Optional[int] = Field(None, description="Approximate population figure")

prompt = "Provide details for Helsinki: its country and approximate population."
print(f"Prompt: {prompt}\n")

config = types.GenerateContentConfig(
    response_mime_type="application/json",
    response_schema=CityInfo,
)
response = client.models.generate_content(
    model=MODEL_ID,
    contents=prompt,
    config=config
)

print(f"Raw JSON Response:\n{response.text}\n")

# Optional: Try parsing and accessing data
# Note: json.loads() will raise an error if response.text is not valid JSON
data = json.loads(response.text)
print(f"Parsed Data:\nName: {data.get('name')}\nCountry: {data.get('country')}\nPopulation: {data.get('population')}")

```

</details>

<details>
<summary><strong>Concept Check: Why use Pydantic?</strong></summary>

**Q: We can just ask the model to "Return JSON format" in the text prompt. Why go through the trouble of importing Pydantic and building a class?**<br>
**A:** Relying purely on natural language instructions for JSON is risky; the model might misspell keys, change data types (returning a string `"1000"` instead of an integer `1000`), or add conversational fluff before the JSON block. Passing a Pydantic schema forces the API at the system level to mathematically constrain its token generation strictly to your defined structure and data types, virtually guaranteeing a perfect parse for your application.
</details>

### Task 6.2: JSON Output with Gradio

1.  Create a Gradio interface where a user can input text (e.g., a short product review).
2.  Define a Pydantic model to extract sentiment (`positive`, `negative`, `neutral`) and a brief summary (string).
3.  The Gradio function should take the review text, call Gemini requesting JSON output matching the schema, and display the extracted sentiment and summary (or the raw JSON if parsing fails).

<details>
<summary>Solution Code</summary>

```python
# Assumes 'client' was initialized successfully.
# Define Pydantic model for sentiment analysis
class SentimentAnalysis(BaseModel):
    sentiment: str = Field(description="Overall sentiment: 'positive', 'negative', or 'neutral'")
    summary: str = Field(description="A brief one-sentence summary of the review")

def analyze_sentiment_gradio(review_text):
    if not review_text:
        return "Please enter review text."

    prompt = f"Analyze the sentiment of this product review and provide a brief summary:\n\n{review_text}"
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=SentimentAnalysis,
    )

    # Note: If API call fails here, Gradio might show a generic error.
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=prompt,
        config=config
    )
    # Note: If parsing fails here, Gradio might show raw JSON or error.
    # For cleaner display, parsing inside the function is better, but omitted here for simplicity.
    # data = json.loads(response.text)
    # formatted_output = f"**Sentiment:** {data.get('sentiment', 'N/A')}\n\n**Summary:** {data.get('summary', 'N/A')}"
    # return formatted_output
    return f"Raw JSON:\n```json\n{response.text}\n```" # Simple return for this version


# Create Gradio interface
sentiment_interface = gr.Interface(
    fn=analyze_sentiment_gradio,
    inputs=gr.Textbox(lines=5, placeholder="Enter product review here...", label="Product Review"),
    outputs=gr.Markdown(label="Analysis Result (JSON)"), # Display raw JSON as Markdown
    title="Sentiment Analysis (JSON Output)",
    description="Enter a review to get sentiment and summary extracted via JSON mode."
)
sentiment_interface.launch()

```

</details>

**Group Discussion:** Why is getting structured JSON output often more useful in applications than free-form text?

## Part 7: Streaming Output

Observe how responses can be received incrementally.

### Task 7.1: Stream a Story

1.  Use the `client.models.generate_content_stream()` method with a prompt asking for a short story (e.g., "Tell me a story about a curious squirrel exploring a city park.").
2.  Iterate through the response chunks and print `chunk.text` for each chunk as it arrives.

<details>
<summary>Solution Code</summary>

```python
# Assumes 'client' was initialized successfully.
prompt = "Tell me a short story about a curious squirrel exploring a city park for the first time."
print(f"Streaming story for prompt: {prompt}\n---")

response_stream = client.models.generate_content_stream(
    model=MODEL_ID,
    contents=prompt
)
full_story = ""
for chunk in response_stream:
    if chunk.text:
        print(chunk.text, end="")
        full_story += chunk.text # Assemble full story if needed later

print("\n---\nEnd of stream.")
# print(f"\nFull story assembled:\n{full_story}") # Optional: print assembled

```

</details>

<details>
<summary><strong>Concept Check: User Experience & Streaming</strong></summary>

**Q: Does using `generate_content_stream` make the model compute the answer faster?**<br>
**A:** No, the total time required to generate 500 words is exactly the same. However, streaming vastly improves *Perceived Latency*. Instead of a user staring at a loading spinner for 10 seconds before reading a massive wall of text, streaming allows them to begin reading the first sentence within 1 second while the rest is being calculated. 
</details>

### Task 7.2: Streaming with Gradio

1.  Implement the streaming Gradio example from the lecture/notebook.
2.  The function should be a generator using `yield` to update the output `gr.Textbox` incrementally.
3.  Test with a prompt that is likely to generate a longer response (like asking for a story or a detailed explanation). Observe the text appearing gradually in the output box.

<details>
<summary>Solution Code</summary>

```python
# Assumes 'client' was initialized successfully.
# Ensure the generator function and gr.Interface call are present and executable.
def stream_response_gradio_lab(prompt):
    if not prompt:
        yield "Please enter a prompt."
        return # Use return in generator context to stop iteration

    full_response = ""
    # Note: If API call fails here, Gradio might show an error or stop yielding.
    response_stream = client.models.generate_content_stream(
        model=MODEL_ID,
        contents=prompt
    )
    yield "" # Yield empty string initially to clear previous output
    for chunk in response_stream:
        if hasattr(chunk, "text") and chunk.text:
            full_response += chunk.text
            yield full_response # Yield cumulative response

streaming_interface = gr.Interface(
    fn=stream_response_gradio_lab,
    inputs=gr.Textbox(lines=2, label="Prompt", placeholder="e.g., Explain the water cycle in detail..."),
    outputs=gr.Textbox(lines=15, label="Streamed Output"),
    title="Streaming Response Generator",
    description="Watch the response appear incrementally."
)
streaming_interface.launch()

```

</details>

**Group Discussion:** When would streaming output be particularly beneficial in a user-facing application?

## Part 8: File API (Selected Examples)

Explore uploading files for the model to process.

### Task 8.1: Upload and Summarize a PDF

1.  Find a URL for a simple, publicly accessible PDF document online (e.g., a short research paper abstract, a simple brochure).
2.  Write code to:
    *   Download the PDF using `requests`.
    *   Upload the PDF using `client.files.upload()`.
    *   Send a prompt asking Gemini to summarize the main points of the PDF, passing the uploaded `File` object in the `contents`.
    *   Print the summary.

<details>
<summary>Solution Code</summary>

```python
# Assumes 'client' was initialized successfully.
# Example PDF URL (Replace if needed, ensure it's accessible)
# Using the one from the demo notebook:
PDF_URL = "https://storage.googleapis.com/generativeai-downloads/data/Smoothly%20editing%20material%20properties%20of%20objects%20with%20text-to-image%20models%20and%20synthetic%20data.pdf"
pdf_path = pathlib.Path("lab_uploaded_article.pdf")

# 1. Prepare file
print(f"Downloading PDF from {PDF_URL}...")
pdf_bytes = requests.get(PDF_URL).content
pdf_path.write_bytes(pdf_bytes)
print("Download complete.")

# 2. Upload file
print(f"Uploading {pdf_path}...")
file_upload = client.files.upload(file=pdf_path)
print(f"Upload complete. URI: {file_upload.uri}, State: {file_upload.state}")

# PDFs usually process quickly, but a check is good practice
# Note: This loop will run indefinitely if state never becomes ACTIVE/FAILED
if file_upload.state != 'ACTIVE':
     while file_upload.state == 'PROCESSING':
         print("Waiting for PDF processing...")
         time.sleep(3)
         file_upload = client.files.get(name=file_upload.name) # Refresh state
     if file_upload.state != 'ACTIVE':
         # This will raise an error if processing failed.
         raise ValueError(f"File processing failed. State: {file_upload.state}")

# 3. Use uploaded file
prompt = "Provide a brief summary of this document in 2-3 sentences."
print(f"\nSending prompt: {prompt}")

response = client.models.generate_content(
    model=MODEL_ID, # Ensure model supports PDF
    contents=[
        file_upload,
        prompt,
    ]
)
print(f"\nSummary Response:\n{response.text}")

# Optional: Clean up uploaded file reference from API storage and local disk
# client.files.delete(name=file_upload.name)
# pdf_path.unlink() # Delete local copy

```

</details>

<details>
<summary><strong>Concept Check: Why wait for PROCESSING?</strong></summary>

**Q: In the code above, we use a `while` loop with `time.sleep()`. Why doesn't the file just upload instantly?**<br>
**A:** When you upload an image, it usually is ready (`ACTIVE`) immediately. But when you upload large PDFs or videos, Google's servers must run preprocessing algorithms—such as Optical Character Recognition (OCR) to extract text, or frame-by-frame analysis for video extraction. This takes compute time. If you immediately request a summary while the state is `PROCESSING`, the API will return an error because the file isn't ready to be read by the LLM yet.
</details>

**Group Discussion:** Besides PDFs, what other file types were shown in the lecture that can be processed using the File API? Discuss why using the File API is preferred for larger files compared to embedding them directly in the prompt (if even possible).

## Part 9: Instruct Prompting Practice

Apply principles of effective prompting. Use `generate_content` for these tasks.

### Task 9.1: Role Play and Constraints

*   **Goal:** Get an explanation of photosynthesis suitable for a young child (approx. 6 years old), keeping it simple and under 50 words.
*   **Task:** Formulate a single prompt that instructs the model to act as a friendly teacher explaining photosynthesis to a 6-year-old, adhering to the length constraint and simplicity requirement. Execute the prompt and review the output.

<details>
<summary>Solution Prompt Idea</summary>

```python
# Assumes 'client' was initialized successfully.
prompt_task_9_1 = """
Act as a friendly teacher explaining to a 6-year-old child.
Explain photosynthesis very simply in under 50 words. Tell them how plants make their own food using sunlight, water, and air.
"""

response = client.models.generate_content(model=MODEL_ID, contents=prompt_task_9_1)
print(f"Prompt:\n{prompt_task_9_1}\n")
print(f"Response:\n{response.text}")

```

</details>

<details>
<summary><strong>Prompting Concept: Zero-Shot vs Few-Shot</strong></summary>

The prompt above is an example of **Zero-Shot Prompting**. You provided instructions, but you did not provide any examples of the correct format before asking the question. If you wanted the model to format answers in a highly specific, quirky way, you could use **Few-Shot Prompting** by providing 2 or 3 examples directly in your text prompt before asking your real question. This helps "tune" the model's behavior contextually.
</details>

### Task 9.2: Improving Specificity and Format

*   **Initial Vague Prompt:** "Tell me about laptops."
*   **Desired Output:** A comparison of typical battery life for standard Windows laptops versus MacBooks, presented as two bullet points.
*   **Task:** Rewrite the vague prompt into a specific instruct prompt to achieve the desired comparison and bullet-point format. Execute the prompt and review the output.

<details>
<summary>Solution Prompt Idea</summary>

```python
# Assumes 'client' was initialized successfully.
prompt_task_9_2 = """
Compare the typical battery life expectations for standard Windows laptops versus Apple MacBooks.
Provide the answer as exactly two bullet points:
- One bullet point summarizing typical battery life for Windows laptops.
- One bullet point summarizing typical battery life for MacBooks.
Be concise.
"""

response = client.models.generate_content(model=MODEL_ID, contents=prompt_task_9_2)
print(f"Prompt:\n{prompt_task_9_2}\n")
print(f"Response:\n{response.text}")

```

</details>

### Task 9.3: Structured Formatting (JSON without Pydantic)

*   **Goal:** Get the primary colors as a JSON list.
*   **Task:** Formulate a prompt asking for the list of primary colors (Red, Yellow, Blue). Instruct the model to provide the output *only* as a JSON array of strings. Use `response_mime_type="application/json"` but *do not* provide a Pydantic schema (let the model infer from the instruction). Execute and review.

<details>
<summary>Solution Prompt Idea</summary>

```python
# Assumes 'client' was initialized successfully.
prompt_task_9_3 = """
List the primary colors (Red, Yellow, Blue).
Output *only* a JSON array containing these three color names as strings.
"""

config = types.GenerateContentConfig(response_mime_type="application/json")
response = client.models.generate_content(
    model=MODEL_ID,
    contents=prompt_task_9_3,
    config=config
)
print(f"Prompt:\n{prompt_task_9_3}\n")
print(f"Raw JSON Response:\n{response.text}")

# Optional: Validate it's a list of 3 strings
# Note: json.loads() will raise an error if response.text is not valid JSON
data = json.loads(response.text)
if isinstance(data, list) and len(data) == 3 and all(isinstance(s, str) for s in data):
     print("\nValidation: Output appears to be a JSON list of 3 strings.")
else:
     print("\nValidation: Output format might not be the expected JSON list.")

```

</details>

**Group Discussion:** Discuss how providing clear instructions (role, format, constraints) in the prompt improves the quality and usability of the model's responses.

## Part 10: Wrap-up

*   Review the different functionalities explored: text, multimodal, configuration, chat, JSON, streaming, File API.
*   Discuss which features seem most relevant or useful for the group's mini-project ideas (connecting to the Part 1 mock UI).
*   Identify any remaining questions or areas of confusion.

## Note on Robust Code and Error Handling

Throughout this lab, for simplicity, we have omitted explicit error handling (using `try...except` blocks) around most of the API calls and data processing steps in the solution code.

**Why is error handling important?**
In real-world applications, many things can go wrong:
*   **Network issues:** The connection to the API server might fail.
*   **API errors:** The API key might be invalid, rate limits exceeded, or the API service might have temporary problems.
*   **Invalid input:** User input might be malformed or unexpected.
*   **Unexpected output:** The model might not return data in the expected format (e.g., invalid JSON when JSON was requested).
*   **Resource issues:** Problems reading/writing local files.

Without error handling, these issues typically cause the program to crash, providing a poor user experience.

**How `try...except` helps:**
The `try...except` structure allows you to anticipate potential errors and define how the program should respond instead of crashing.

**Example 1: Handling API Call Errors**
```python
# Instead of just:
# response = client.models.generate_content(...)

# Use try...except:
try:
    response = client.models.generate_content(model=MODEL_ID, contents=prompt)
    # Process response.text here
    print(response.text)
except Exception as e:
    # Code to execute if *any* error occurs during the 'try' block
    print(f"An error occurred calling the Gemini API: {e}")
    # You could return a default message or log the error
```

**Example 2: Handling JSON Parsing Errors**
```python
# Assume 'response_text' contains the JSON string from the API
try:
    data = json.loads(response_text)
    # Use the parsed 'data' dictionary here
    print(f"Successfully parsed data: {data}")
except json.JSONDecodeError as e:
    # Code to execute specifically if the text is not valid JSON
    print(f"Failed to parse JSON response: {e}")
    print(f"Raw response was: {response_text}")
except Exception as e:
    # Catch any other potential errors
    print(f"An unexpected error occurred processing the response: {e}")

```

While not required for completing this lab, incorporating `try...except` blocks is a standard practice for writing more reliable and user-friendly applications that interact with external services or process potentially unpredictable data.

----

<details>
<summary><strong>Capstone and Robustness</strong></summary>

To finish this lab, you will combine the concepts learned into a single, robust application.

### Task 11.1: Systematic Error Handling

When working with APIs, especially on free tiers, network timeouts and rate limits (`429 Too Many Requests`) are common. 

**Task:** Rewrite the basic text generation function from Task 2.2 to include a `try...except` block. This ensures that if the API fails, the Gradio app displays a friendly error message instead of crashing.

**Solution Code**

```python
def robust_ask_gemini(user_prompt):
    if not user_prompt:
        return "Please enter a prompt."
        
    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=user_prompt
        )
        return response.text
    except Exception as e:
        # Returns a user-friendly string instead of crashing the UI
        return f"⚠️ An error occurred while communicating with the API. Please wait a moment and try again.\n\nError details: {str(e)}"

# You can test this by temporarily providing an invalid MODEL_ID.
```

### Task 11.2: Multimodal to JSON Extraction

Instead of trying to output generated images to Gradio (which can be complex), let's use multimodal input to generate structured data. 

**Task:** 
1. Define a Pydantic model called `ImageAnalysis` containing a `summary` (string) and a `list_of_objects` (list of strings).
2. Create a function that takes a PIL Image as input.
3. Pass the image to Gemini with the prompt "List the main objects in this image," forcing the output to match your JSON Pydantic schema.

**Solution Code**

```python
class ImageAnalysis(BaseModel):
    summary: str = Field(description="A brief summary of the image scene.")
    list_of_objects: List[str] = Field(description="A list of distinct objects found in the image.")

def extract_objects_from_image(image_input):
    if image_input is None:
        return "No image provided."
        
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=ImageAnalysis
    )
    
    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=[image_input, "Analyze this image and list the objects."],
            config=config
        )
        return response.text # Returns the raw JSON string safely
    except Exception as e:
        return f'{{"error": "{str(e)}"}}'
```


### Task 11.3: Final Synthesis Capstone UI

**Task:** Build a comprehensive Gradio application that acts as a "Swiss Army Knife" for Gemini. 
Your Gradio interface should include:
*   **Inputs:** A Textbox (for the prompt), an Image upload component (optional input), and a Slider for Temperature.
*   **Output:** A Markdown component for the response.
*   **Logic:** Your function should check if an image is provided. If yes, perform a multimodal call. If no, perform a text-only call. The function must apply the chosen temperature and include a `try...except` block.

**Solution Code**

```python
def capstone_gemini_app(prompt, image_input, temp):
    if not prompt:
        return "Please enter a prompt."
        
    config = types.GenerateContentConfig(temperature=float(temp))
    
    # Determine if request is multimodal or text-only based on image presence
    request_contents =[image_input, prompt] if image_input is not None else prompt
    
    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=request_contents,
            config=config
        )
        return response.text
    except Exception as e:
        return f"**API Error:** {str(e)}"

capstone_ui = gr.Interface(
    fn=capstone_gemini_app,
    inputs=[
        gr.Textbox(label="Prompt", lines=3),
        gr.Image(type="pil", label="Optional Image Input"),
        gr.Slider(0.0, 1.0, value=0.5, step=0.1, label="Creativity (Temperature)")
    ],
    outputs=gr.Markdown(label="Response"),
    title="Gemini Capstone Application",
    description="Combine text, images, and parameter control in one robust interface."
)
capstone_ui.launch()
```

</details>


<!-- You should consider adding it when building your final project code. Here's a [demo](./activity1-v1.md). -->