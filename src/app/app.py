import uvicorn 
from fastapi import FastAPI, HTTPException 
from fastapi.responses import JSONResponse 
from pydantic import BaseModel, Field 
 
class QuestionRequest(BaseModel): 
    query: str = Field(..., description="The question to query.") 
    threshold_confidence: float = Field(..., gt=0, le=1, description="Confidence threshold between 0 and 1.") 
    is_soft_answer: bool = Field(..., description="Flag to enable soft answers.") 
    is_soft_discard: bool = Field(..., description="Flag to enable soft discard logic.") 
 
def create_app() -> FastAPI: 
    """ 
    Factory function to create and configure the FastAPI application. 
    """ 
    app = FastAPI(title="Question Answering API", version="1.0.0") 
    return app 
 
app = create_app() 
 
@app.post("/get_answer/", response_class=JSONResponse) 
async def get_answer(q_request: QuestionRequest): 
    """ 
    Endpoint to retrieve an augmented generated answer based on the question request. 
    """ 
    results = { 
        "answer": 'Some generated answer', 
        "retrieve_logs": 'Process logs for debugging' 
    } 
 
    if not q_request.query: 
        raise HTTPException(status_code=400, detail="Query cannot be empty.") 
    return JSONResponse(content=results) 
 
@app.get("/hello/", response_class=JSONResponse) 
async def say_hello(): 
    """ 
    Simple hello endpoint to verify that the API is running. 
    """ 
    return JSONResponse(content={"message": "Hello from FastAPI!"}) 
 
if __name__ == "__main__": 
    uvicorn.run( 
        "app:app", 
        host="127.0.0.1", 
        port=4830, 
        log_level="info", 
        # reload=True
    )