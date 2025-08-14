from datetime import datetime

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import HTMLResponse, RedirectResponse
from uvicorn import run as app_run

from typing import Optional

from us_visa.constants import *
from us_visa.pipeline.prediction_pipeline import USvisaData, USvisaClassifier
from us_visa.pipeline.training_pipeline import TrainPipeline





# Initialize FastAPI app
app = FastAPI()
# Mount static files (CSS, JS, images)
app.mount("/static", StaticFiles(directory="static"), name="static")
# Setup Jinja2 templates for HTML rendering
templates = Jinja2Templates(directory='templates')
# Allow CORS from all origins (can be restricted in production)
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DataForm:
    
    """
    Helper class to parse and store form data for US Visa prediction.
    """
    
    def __init__(self, request: Request):
        self.request: Request = request
        self.continent: Optional[str] = None
        self.education_of_employee: Optional[str] = None
        self.has_job_experience: Optional[str] = None
        self.requires_job_training: Optional[str] = None
        self.no_of_employees: Optional[str] = None
        self.company_age: Optional[str] = None    # Need to change this predictor to yr_of_estab
        self.region_of_employment: Optional[str] = None
        self.prevailing_wage: Optional[str] = None
        self.unit_of_wage: Optional[str] = None
        self.full_time_position: Optional[str] = None
        

    async def get_usvisa_data(self):
        
        """
        Asynchronously extracts form data submitted by the user.
        """
        
        form = await self.request.form()
        self.continent = form.get("continent")
        self.education_of_employee = form.get("education_of_employee")
        self.has_job_experience = form.get("has_job_experience")
        self.requires_job_training = form.get("requires_job_training")
        self.no_of_employees = form.get("no_of_employees")
        self.company_age = form.get("company_age")    # need to change this predictor to yr_of_estab
        self.region_of_employment = form.get("region_of_employment")
        self.prevailing_wage = form.get("prevailing_wage")
        self.unit_of_wage = form.get("unit_of_wage")
        self.full_time_position = form.get("full_time_position")



@app.get("/", tags=["authentication"])
async def index(request: Request):

    """
    Renders the main HTML form for US Visa prediction.
    """
    
    return templates.TemplateResponse(
            "usvisa.html",{"request": request, "context": "Rendering"})


@app.get("/train")
async def trainRouteClient():
    
    """
    Endpoint to trigger the training pipeline.
    """
    
    try:
        train_pipeline = TrainPipeline()

        train_pipeline.run_pipeline()

        return Response("Training successful !!")

    except Exception as e:
        return Response(f"Error Occurred! {e}")


@app.post("/")
async def predictRouteClient(request: Request):
    
    """
    Handles form submission for visa prediction.
        - Reads form input
        - Converts to DataFrame
        - Loads model and predicts approval status
        - Renders result back to HTML template
    """
    
    try:
        # Parse form data
        form = DataForm(request)
        await form.get_usvisa_data()
        
        # Create input object for prediction
        usvisa_data = USvisaData(
                                continent= form.continent,
                                education_of_employee = form.education_of_employee,
                                has_job_experience = form.has_job_experience,
                                requires_job_training = form.requires_job_training,
                                no_of_employees= form.no_of_employees,
                                yr_of_estab= int(form.company_age) + datetime.today().year,    # need to change this to yr_of_estab
                                region_of_employment = form.region_of_employment,
                                prevailing_wage= form.prevailing_wage,
                                unit_of_wage= form.unit_of_wage,
                                full_time_position= form.full_time_position,
                                )
        
        # Convert to DataFrame for model input
        usvisa_df = usvisa_data.get_usvisa_input_data_frame()
        # Load classifier and make prediction
        model_predictor = USvisaClassifier()

        value = model_predictor.predict(dataframe=usvisa_df)[0]

        # Map prediction result to user-friendly message
        status = "Visa-approved" if value == 1 else "Visa Not-Approved"

        # Render the prediction result
        return templates.TemplateResponse(
            "usvisa.html",
            {"request": request, "context": status},
        )
        
    except Exception as e:
        return {"status": False, "error": f"{e}"}


if __name__ == "__main__":
    
    """
    Entry point for running the FastAPI app locally.
    """
    
    APP_HOST = "0.0.0.0"
    APP_PORT = 8080
    
    # link : http://localhost:8080/
    app_run(app, host=APP_HOST, port=APP_PORT)


