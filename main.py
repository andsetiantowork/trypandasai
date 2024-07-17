from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import pandas as pd
import os
import logging
import threading
from pandasai import SmartDataframe
import matplotlib.pyplot as plt
import uuid

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Path to the local file
file_path =  "D:\\Andi Work\\fastapi_project\\Transaction.csv"

# Set the PandasAI API key
os.environ['PANDASAI_API_KEY'] = "$2a$10$k9RaVcAHiatVipkFuc3gk.R.eyz9sNwRNiBqbDZ8X.T6OUGrg2DRm"

# Read the dataset
try:
    df = pd.read_csv(file_path)
    logger.info(f"File {file_path} read successfully")
except Exception as e:
    logger.error(f"An error occurred while reading the file: {e}")
    raise HTTPException(status_code=500, detail="Failed to read the dataset")

# Initialize SmartDataframe
try:
    sdf = SmartDataframe(df)
    logger.info("SmartDataframe initialized successfully")
except Exception as e:
    logger.error(f"An error occurred while initializing SmartDataframe: {e}")
    raise HTTPException(status_code=500, detail="Failed to initialize SmartDataframe")

class ChatRequest(BaseModel):
    question: str

def ask_question(question, response_holder):
    try:
        response_holder['response'] = sdf.chat(question)
    except Exception as e:
        response_holder['error'] = str(e)

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        logger.info("Received request for data analysis or visualization")

        question = request.question
        logger.info(f"Received question: {question}")

        if not question:
            logger.error("No question provided in the request")
            raise HTTPException(status_code=400, detail="No question provided")

        response_holder = {'response': None, 'error': None}

        if "plot" in question.lower() or "visualize" in question.lower():
            response = sdf.chat(question)
            file_name = f"{uuid.uuid4()}.png"
            plt.savefig(file_name)
            response_holder['response'] = file_name
            response_type = "file"
        else:
            thread = threading.Thread(target=ask_question, args=(question, response_holder))
            response_type = "text"

        if response_type == "text":
            thread.start()
            thread.join(timeout=30)  # Timeout set to 30 seconds

            if thread.is_alive():
                logger.error("Timeout occurred.")
                thread.join()  # Ensure thread cleanup
                raise HTTPException(status_code=500, detail="Timeout occurred while processing the question")
            elif response_holder['error']:
                logger.error(f"Error occurred while processing the question: {response_holder['error']}")
                raise HTTPException(status_code=500, detail=response_holder['error'])
            else:
                logger.info(f"Response created: {response_holder['response']}")
                return JSONResponse(content={"response": response_holder['response']}, status_code=200)
        else:
            return FileResponse(path=response_holder['response'], media_type='image/png', filename=file_name)

    except HTTPException as http_exc:
        logger.error(f"HTTP exception occurred: {http_exc.detail}")
        raise
    except Exception as e:
        logger.exception("An error occurred during processing")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
