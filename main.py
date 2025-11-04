from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from sentence_transformers import SentenceTransformer, util

# Initialize FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load a pre-trained embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Dummy internship dataset (later replace with PM Internship portal data)
internships = [
    {"title": "AI Research Internship", "description": "Work on machine learning models and data-driven AI projects."},
    {"title": "Data Science Internship", "description": "Analyze datasets and build predictive models using Python and ML."},
    {"title": "Web Development Internship", "description": "Frontend and backend development with React and Django."},
    {"title": "Cybersecurity Internship", "description": "Research in ethical hacking, security tools, and network safety."},
    {"title": "IoT & Embedded Systems Internship", "description": "Work on IoT devices, embedded C, and hardware integration."}
]

# Encode all internship descriptions once
internship_embeddings = model.encode([i["description"] for i in internships], convert_to_tensor=True)

# Request schema
class RecommendRequest(BaseModel):
    skills: List[str]
    interests: List[str]

@app.post("/recommend")
def recommend(req: RecommendRequest):
    # Combine user input into a profile text
    profile_text = " ".join(req.skills + req.interests)
    profile_embedding = model.encode(profile_text, convert_to_tensor=True)

    # Compute similarity between user and internships
    similarities = util.cos_sim(profile_embedding, internship_embeddings)[0]

    # Rank internships by similarity score
    ranked_indices = similarities.argsort(descending=True)

    recommendations = []
    for idx in ranked_indices[:3]:  # Top 3 recommendations
        internship = internships[idx]
        score = float(similarities[idx])
        recommendations.append({
            "title": internship["title"],
            "description": internship["description"],
            "score": round(score, 3)  # similarity score
        })

    return {"recommendations": recommendations}
