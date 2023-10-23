from pydantic import BaseModel, Field

class InputData(BaseModel):
    Sex: str = Field(str, "Sex")
    Length: float = Field(float, "Length")
    Diameter: float = Field(float, "Diameter")
    Height: float = Field(float, "Height")
    Whole_weight : float = Field(float, "Whole weight")
    Shucked_weight : float = Field(float, "Shucked weight")
    Viscera_weight : float = Field(float, "Viscera weight")
    Shell_weight : float = Field(float, "Shell weight")

class PredictionOut(BaseModel):
    Age: float
