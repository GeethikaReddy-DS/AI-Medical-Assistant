from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import googlemaps
import requests
import re

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Google Gemini AI Model Setup
genai.configure(api_key="AIzaSyAsUBNSVshmRiITNGfL-_sobCIjou0eN3g")
model = genai.GenerativeModel("gemini-1.5-pro")

# Google Maps API Setup
gmaps = googlemaps.Client(key="AIzaSyD1jWIPDmJ-m--MN5GEdU6JxKwdiz4N5GA")

# In-memory session storage (use database for production)
session_memory = {}

# List of greeting phrases
GREETINGS = [
    "hi", "hello", "hey", "greetings", "good morning", "good afternoon",
    "good evening", "what's up", "howdy", "yo"
]

ENDEXPRESSIONS = [
    "bye", "thankyou", "thanks", "thank you", "thank you so much", "tq"
]

medical_terms = {
    "symptoms": [
        "headache", "fever", "cough", "shortness of breath", "fatigue",
        "chest pain", "dizziness", "nausea", "vomiting", "diarrhea",
        "abdominal pain", "sore throat", "runny nose", "muscle pain"
    ],
    "conditions": [
        "diabetes", "hypertension", "asthma", "covid-19", "flu",
        "cold", "arthritis", "cancer", "migraine", "pneumonia",
        "tuberculosis", "depression", "anxiety", "eczema", "insomnia"
    ],
    "medications": [
        "paracetamol", "ibuprofen", "amoxicillin", "metformin",
        "insulin", "atorvastatin", "omeprazole", "aspirin",
        "azithromycin", "cetirizine"
    ],
    "bodyParts": [
        "heart", "lungs", "kidneys", "liver", "stomach",
        "brain", "intestines", "skin", "eyes", "throat"
    ],
    "diagnostics": [
        "blood test", "x-ray", "mri", "ct scan", "ecg",
        "urine test", "biopsy", "ultrasound"
    ],
    "firstAid": [
        "cpr", "burn treatment", "wound care", "choking first aid",
        "fracture support", "bleeding control"
    ],
    "specialists": [
        "cardiologist", "neurologist", "dermatologist", "gynecologist",
        "pediatrician", "orthopedic", "psychiatrist", "oncologist"
    ],
    "medicalProcedures": [
        "surgery", "chemotherapy", "radiation therapy", "dialysis",
        "vaccination", "endoscopy", "transplant", "physical therapy"
    ],
    "lifestyle": [
        "diet for diabetes", "exercise for weight loss", "yoga for anxiety",
        "healthy sleep habits", "stress management"
    ],
    "others": [
        "side effects", "allergy", "covid vaccine", "health insurance",
        "medical emergency", "nearest hospital", "home remedies"
    ]
}

# Flatten all terms into one list
all_medical_terms = [term.lower() for sublist in medical_terms.values() for term in sublist]


class ChatRequest(BaseModel):
    message: str
    age: int | None = None
    gender: str | None = None
    location: str | None = None
    session_id: str | None = None

class ResetRequest(BaseModel):
    session_id: str

def filter_disclaimers(response_text):
    """
    Removes disclaimers, warnings, and extra medical cautionary text.
    """
    disclaimer_patterns = [
        r"I am an AI.*?\.",  # Removes lines starting with "I am an AI..."
        r"I can't give medical advice.*?\.",  # Removes "I can't give medical advice"
        r"Consult a doctor.*?\.",  # Removes "Consult a doctor" warnings
        r"This is not a substitute.*?\.",  # Removes disclaimers about medical advice
        r"When to see a doctor:.*",  # Removes emergency warning lists
        r"Always consult your healthcare provider.*?\.",  # Removes additional disclaimers
        r"Remember to always.*?\.",  # Removes general cautionary statements
    ]
    
    for pattern in disclaimer_patterns:
        response_text = re.sub(pattern, "", response_text, flags=re.IGNORECASE)

    return response_text.strip()

def format_response(response_text):
    """
    Converts response into short, bullet-point format for better readability.
    """
    lines = [line.strip() for line in response_text.split("\n") if line.strip()]
    
    if len(lines) > 1:
        formatted_response = "\n".join([f"- {line}" for line in lines])
    else:
        formatted_response = response_text  # Keep single responses as-is

    return formatted_response

def find_nearby_hospitals(location):
    """
    Finds nearby hospitals using Google Maps API.
    """
    try:
        # Check if location is a city name and convert to coordinates
        geocode_result = gmaps.geocode(location)
        if geocode_result:
            coordinates = geocode_result[0]['geometry']['location']
            places_result = gmaps.places_nearby(location=coordinates, radius=3000, type='hospital')
            hospitals = []
            for place in places_result['results']:
                hospital_name = place.get('name', 'Unknown Hospital')  # Fallback if name is missing
                place_id = place.get('place_id')  # Get place_id safely
                if place_id != '' and place_id is not None:  # Ensure place_id is valid
                    google_maps_link = f"https://www.google.com/maps/place/?q=place_id:{place_id}"
                    hospitals.append(f"{hospital_name} -- {google_maps_link}")
                else:
                    hospitals.append(f"{hospital_name} -- Link not available")  # Fallback if place_id is missing
            return hospitals
        else:
            return ["Location not found."]
    except Exception as e:
        print(f"Error fetching nearby hospitals: {e}")
        return ["Error fetching nearby hospitals."]

@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Handles chatbot queries with memory retention and ensures dosage is provided.
    """
    user_message = request.message.strip().lower()
    session_id = request.session_id or "default"  # Use a session ID to track conversation

    if not user_message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    if session_id not in session_memory:
        session_memory[session_id] = {"messages": [], "illness": None, "context": None}
        # return session_memory

    if user_message in GREETINGS:
        return {"response": "Hello! How can I assist you with your medical concern today?"}

    if user_message in ENDEXPRESSIONS:
        return {"response": "You are welcome, You can start a new chat if you have any other medical concerns."}
    # # Check if any medical term is in the user input
    # if not any(term in user_message for term in all_medical_terms):
    #     return {"response": "Sorry, I can only assist you with medical concerns, please mention your medical concern."}

    # Check if the user is asking for medication or dosage details
    if "medication" in user_message or "dosage" in user_message or "prescription" in user_message or "medicine" in user_message:
        if not request.age or not request.gender:
            return {"response": "I need to know your age and gender to provide medication and dosage information. Please enter your age and gender."}
        
        # Use the previously mentioned illness if available
        illness = session_memory[session_id]["illness"]
        if not illness:
            return {"response": "Please specify the illness you need medication for."}
        
        # Frame a question to Gemini AI for medication details
        age = request.age
        gender = request.gender.lower()
        structured_prompt = (
            f"Provide medication details for {illness} for a {age}-year-old {gender}. "
            f"Include common medications and dosage information."
        )

        session_memory[session_id]["messages"].append(f"User: {user_message}")

        try:
            chat_history = "\n".join(session_memory[session_id]["messages"][-5:])  # Keep last 5 messages for context
            full_prompt = f"Previous conversation:\n{chat_history}\nUser: {structured_prompt}\nAI:"

            response = model.generate_content(full_prompt)
            ai_response = response.text.strip()

            cleaned_response = filter_disclaimers(ai_response)
            formatted_response = format_response(cleaned_response)

            session_memory[session_id]["messages"].append(f"AI: {formatted_response}")

            return {"response": formatted_response}

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # Check if the user is mentioning an illness
    if "suffering with" in user_message or "have" in user_message or "having" in user_message:
        illness = user_message.split("suffering with")[-1].strip() if "suffering with" in user_message else user_message.split("have")[-1].strip()
        session_memory[session_id]["illness"] = illness
        session_memory[session_id]["context"] = "illness"
        return {"response": f"Got it. You are suffering with {illness}. How can I assist you further?"}

    # Check if the user is asking for nearby hospitals
    if "hospital" in user_message and request.location or "hospitals" in user_message and request.location or "clinics" in user_message and request.location or  "clinic" in user_message and request.location:
        hospitals = find_nearby_hospitals(request.location)
        return {"response": f"Nearby hospitals are: {', '.join(hospitals)}"}
    
    # Check if any medical term is in the user input
    if not any(term in user_message for term in all_medical_terms):
        return {"response": "Sorry, I can only assist you with medical concerns, please mention your medical concern."}


    # Handle general information queries using Gemini AI
    structured_prompt = (
        f"Provide clear, point-wise information about: {user_message}. "
        f"Include precautions, remedies, and other relevant details."
    )

    session_memory[session_id]["messages"].append(f"User: {user_message}")

    try:
        chat_history = "\n".join(session_memory[session_id]["messages"][-5:])  # Keep last 5 messages for context
        full_prompt = f"Previous conversation:\n{chat_history}\nUser: {structured_prompt}\nAI:"

        response = model.generate_content(full_prompt)
        ai_response = response.text.strip()

        cleaned_response = filter_disclaimers(ai_response)
        formatted_response = format_response(cleaned_response)

        session_memory[session_id]["messages"].append(f"AI: {formatted_response}")

        # Update context based on the response
        if "remedies" in user_message or "instructions" in user_message:
            session_memory[session_id]["context"] = "remedies"

        return {"response": formatted_response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reset")
async def reset(request: ResetRequest):
    """
    Resets the session memory for a given session ID.
    """
    session_id = request.session_id
    if session_id in session_memory:
        del session_memory[session_id]
    return {"response": "Session reset successfully."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)