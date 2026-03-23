from pydantic import BaseModel, EmailStr, Field


class RadiologistReportValidator(BaseModel):
    name: str = Field(..., min_length=2)
    phone: str = Field(..., pattern=r'^\+?\d{10,15}$')  # basic international format
    email: EmailStr
    comments: str = Field(..., min_length=5)
