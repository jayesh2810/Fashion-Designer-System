import streamlit as st
import os
import base64
import re
import tempfile
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the OpenAI client
from openai import OpenAI
client = OpenAI()

# Import Crew AI and related modules
from crewai import Agent, Task, Crew
from langchain.tools import Tool
from duckduckgo_search import DDGS

##############################################
# Background and CSS
##############################################

# Function to get base64 string of an image file
def get_base64_of_file(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Get the base64 string of your local background image (update path if needed)
background_image = get_base64_of_file("banner.jpg")

# Create CSS style to set the background image
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-image: url("data:image/jpeg;base64,{background_image}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

##############################################
# Utility Functions
##############################################

def encode_image(image_path: str) -> str:
    """Encodes an image to a base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def analyze_dressing_style(directory_path: str) -> list:
    """
    Analyzes the dressing style in all images in the specified directory.
    Returns a list of dressing styles for the images.
    """
    results = []
    try:
        for filename in os.listdir(directory_path):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image_path = os.path.join(directory_path, filename)
                base64_image = encode_image(image_path)
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": """
                                You are an expert fashion analyst specializing in identifying dressing styles based on images. 
                                Your task is to classify the dressing style in the given image accurately using only **two words**.
                                
                                Instructions:
                                1. **Analyze the outfit composition**, including clothing types, colors, formality, and overall aesthetic.
                                2. **Focus on established fashion categories** such as: 
                                   - Casual, Formal, Smart Casual, Business Casual, Streetwear, Athleisure, Chic, Bohemian, Vintage, Minimalist.
                                3. **Ignore background elements, facial expressions, and non-fashion details**.
                                4. **Respond in exactly two words**, ensuring they best describe the overall dressing style.
                                5. **Prioritize accuracy over vague responses**.
                            """
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Define the dressing style in the image"},
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}", "details": "low"}},
                            ],
                        }
                    ],
                    max_tokens=50,
                )
                results.append(response.choices[0].message.content)
        return results
    except Exception as e:
        st.error(f"Error during dressing style analysis: {str(e)}")
        return [f"Error: {str(e)}"]

def analyze_typical_dressing_style(dressing_styles: list) -> str:
    """
    Analyzes a list of dressing styles and deduces the user's typical dressing style and sense.
    """
    if not dressing_styles:
        return "No dressing styles provided."
    style_frequency = {}
    for style in dressing_styles:
        style_frequency[style] = style_frequency.get(style, 0) + 1
    most_frequent_style = max(style_frequency, key=style_frequency.get)
    summary = (
        f"Based on the analyzed images, the user's typical dressing style is {most_frequent_style}. "
        f"This suggests a preference for {most_frequent_style.lower()} outfits. "
        f"The user's dressing sense leans towards {most_frequent_style.lower()}. "
    )
    return summary

def extract_price(description: str) -> str:
    """
    Extracts the actual price from the product description using regex.
    """
    price_pattern = r'(?:₹|\$|€|£|INR\s*)?\d{1,3}(?:,\d{3})*(?:\.\d{2})?'
    match = re.search(price_pattern, description)
    if match:
        return match.group(0)
    return None

def search_with_duckduckgo(query: str, shopping_site: str, max_results: int = 5) -> list:
    """
    Searches DuckDuckGo for the given query on the preferred shopping site and returns a list of results with corrected prices.
    """
    query_with_site = f"{query} site:{shopping_site}.com"
    results = DDGS().text(query_with_site, max_results=max_results, region="in")
    for result in results:
        actual_price = extract_price(result['body'])
        if actual_price:
            result['price'] = actual_price
    return results

##############################################
# Define Tools and Agents
##############################################

# Create the search tool using DDGS
search_tool = Tool(
    name="DuckDuckGo Search",
    func=search_with_duckduckgo,
    description=(
        "Searches DuckDuckGo for items matching the query. "
        "Returns a list of results including titles, descriptions, and links."
    )
)

# Tool for analyzing dressing style
analyze_dressing_style_tool = Tool(
    name="Analyze Dressing Style",
    func=analyze_dressing_style,
    description="Analyzes the dressing style in all images in a directory and returns the results as a list."
)

# Fashion Analyst Agent: analyzes images to identify dressing styles
fashion_analyst = Agent(
    role="You are a fashion analyst",
    goal="You analyze images of outfits and identify their dressing styles.",
    backstory=(
        "You have years of experience in the fashion industry. "
        "Your expertise lies in identifying and classifying dressing styles based on visual cues."
    ),
    tools=[analyze_dressing_style_tool],
    allow_delegation=False,
    verbose=True,
    max_iter=3
)

# Task for the fashion analyst
task = Task(
    description="Analyze the dressing styles in all images in the directory {image_path}.",
    agent=fashion_analyst,
    expected_output="A well-optimized and correct description of the dressing styles in each image in markdown format. Also, the agent should provide the user’s typical dressing style and dressing sense",
)

# Wardrobe Specification Agent: collects user preferences and generates a wardrobe plan
wardrobe_specification_agent = Agent(
    name="Wardrobe Specification Agent",
    role="Collect user preferences and generate a detailed wardrobe specification which aligns well with the "
         "User's Typical Dressing Style and Dressing Sense identified by fashion_analyst.",
    goal="Collect event gender, type, budget (in INR), constraints and preferred shopping site from the user, and generate a wardrobe plan "
         "based on the dressing style identified by the fashion_analyst.",
    backstory=(
        "You are a fashion consultant with expertise in creating personalized wardrobe plans. "
        "Your task is to interactively gather user preferences and provide a detailed wardrobe specification tailored to their dressing style."
    ),
    tools=[],
    verbose=True,
    max_iter=3
)

wardrobe_specification_task = Task(
    description=(
        "Gender is {gender}, event type is {event_type}, budget is {budget} (in INR), constraints are {constraints} and preferred shopping site is {shopping_site} from the user. "
        "Generate a detailed wardrobe specification based on the dressing style identified by the fashion_analyst and the event type, budget (in INR), and constraints."
    ),
    agent=wardrobe_specification_agent,
    expected_output=(
        "A detailed wardrobe specification including outfit suggestions, accessories, footwear, "
        "budget breakdown, and additional tips tailored to the user's dressing style, gender, event type, and budget (in INR)."
    )
)

# Shopping Agent: searches online for complete wardrobe sets within the user’s budget
shopping_agent = Agent(
    name="Shopping Agent",
    role="Search major shopping sites for wardrobe items within the user's budget (in INR).",
    goal=(
        "Find at least 3 complete wardrobe sets (including shirts, pants, coats, shoes, ties, etc.) provided by wardrobe_specification_agent "
        "within the user's budget (in INR) and return the items with prices (in INR), shopping links, and direct purchase links."
    ),
    backstory=(
        "You are a shopping expert with extensive knowledge of major online shopping platforms. "
        "Your task is to search for wardrobe items that match the user's dressing style, gender, event type, "
        "and budget (in INR), and provide a list of complete sets with prices, links, and a summary of the whole set."
    ),
    tools=[search_tool],
    verbose=True,
    max_iter=8
)

shopping_task = Task(
    description=(
        "Search major shopping sites (e.g., Amazon, Myntra) for complete wardrobe sets that match "
        "the user's dressing style, gender, event type, and budget (in INR). If the user's gender is Male, "
        "then search for male items only; if Female then search for female items only. "
        "Each set should include all items such as shirts, pants, coats, shoes, ties, and hair accessories. "
        "Provide at least 3 choices, including item prices and legitimate shopping links. "
        "Ensure the total cost of each set is within the user's budget of {budget} (in INR) and is for user's gender {gender} only. "
        "Write all the sets to a local file Selection.md."
    ),
    agent=shopping_agent,
    context=[wardrobe_specification_task],
    expected_output=(
        "A list of at least 3 complete wardrobe sets, each including: "
        "- Item names and descriptions "
        "- Their individual prices (in INR) "
        "- Individual shopping links "
        "- Total cost of the set "
        "- A summary of the whole set "
        "All sets must be within the user's budget (in INR)."
    )
)

# Assemble the Crew with all agents and tasks
wardrobe_crew = Crew(
    agents=[fashion_analyst, wardrobe_specification_agent, shopping_agent],
    tasks=[task, wardrobe_specification_task, shopping_task]
)

##############################################
# Multipage App: Landing Page and Upload Page
##############################################

# Create a sidebar selectbox for page navigation
page = st.sidebar.selectbox("Select Page", ["Landing Page", "Upload Page"])

if page == "Landing Page":
    st.title("Welcome to the Wardrobe Specification Assistant!")
    st.write("""
        **About This Project:**
        
        This application uses advanced AI to help you discover your personal style and create a customized wardrobe plan. 
        By uploading images of your outfits, the app analyzes your dressing style using a cutting-edge AI model. 
        It then combines your preferences (such as event type, budget, and constraints) to generate personalized wardrobe recommendations.
        
        **How It Works:**
        1. **Image Analysis:** AI-powered analysis of your outfit images to determine your typical dressing style.
        2. **Personalized Recommendations:** Based on your style and preferences, the app suggests complete wardrobe sets.
        3. **Harnessing AI:** The project leverages state-of-the-art AI (powered by models like GPT-4) to deliver insights and recommendations that are tailored to your unique style.
        
        Navigate to the **Upload Page** using the sidebar to get started!
    """)
    # st.image("banner.jpg", use_column_width=True)
    
elif page == "Upload Page":
    st.title("Wardrobe Specification Assistant")
    st.write("Upload your outfit images and provide your preferences to generate a personalized wardrobe plan.")
    
    # File uploader for images
    uploaded_files = st.file_uploader("Upload JPG or PNG images", type=["jpg", "png"], accept_multiple_files=True)
    
    # Sidebar for user inputs (repeated here so that all controls are on the Upload Page)
    st.sidebar.header("User Preferences (Upload Page)")
    gender = st.sidebar.text_input("What is your gender?", value="Male")
    event_type = st.sidebar.text_input("Event type (e.g., Wedding, Party, Formal Meeting):", value="Party")
    budget = st.sidebar.text_input("Budget (in INR):", value="50000")
    constraints = st.sidebar.text_input("Constraints (e.g., Silk fabrics, Minimalist designs):", value="")
    shopping_site = st.sidebar.selectbox("Preferred Shopping Site", 
                                          ["Amazon", "Myntra", "Flipkart", "Ajio", "Nykaa", "Snapdeal", "Meesho", "Zara", "H&M", "Bewakoof"])
    
    user_inputs = {
        "gender": gender,
        "event_type": event_type,
        "budget": budget,
        "constraints": constraints,
        "shopping_site": shopping_site.lower()
    }
    
    if st.button("Run Analysis"):
        if not uploaded_files:
            st.error("Please upload at least one image.")
        else:
            # Save uploaded images to a temporary directory without printing file details.
            temp_dir = tempfile.mkdtemp()
            for uploaded_file in uploaded_files:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            image_path = temp_dir
            
            # --- Fashion Analyst Task ---
            st.write("Fashion Analyst is analyzing the uploaded images to identify your typical dressing style.")
            # st.image("fashion_agent.jpg", use_column_width=True)
            with st.spinner("Analyzing dressing styles..."):
                dressing_styles = analyze_dressing_style(image_path)
            st.subheader("Dressing Styles Identified:")
            for idx, style in enumerate(dressing_styles, 1):
                st.write(f"Image {idx}: {style}")
            st.write("Generating typical dressing style summary...")
            typical_style_summary = analyze_typical_dressing_style(dressing_styles)
            st.subheader("Typical Dressing Style Summary:")
            st.write(typical_style_summary)
            
            # --- Wardrobe Specification Task ---
            st.write("Wardrobe Specification Agent is collecting your preferences and generating a detailed wardrobe plan based on your typical dressing style.")
            # st.image("wardrobe_agent.jpg", use_column_width=True)
            
            # --- Shopping Agent Task ---
            st.write("Shopping Agent is searching major shopping sites for complete wardrobe sets within your budget.")
            # st.image("shopping_agent.jpg", use_column_width=True)
            
            st.write("Combining your inputs and generating the complete wardrobe plan...")
            inputs = {
                "image_path": image_path,
                **user_inputs
            }
            with st.spinner("Generating your wardrobe plan..."):
                result = wardrobe_crew.kickoff(inputs=inputs)
            
            st.header("Your Wardrobe Plan")
            st.markdown(result.raw)
            st.download_button("Download Wardrobe Plan (Markdown)", data=result.raw, file_name="wardrobe_sets.md")