from pydantic import BaseModel
from typing import List, Optional


class DetectionItem(BaseModel):
	date: str
	vehicle_type: str
	plate: Optional[str] = None
	thumbnail: str  # base64
	# Optional URL path to the saved image file (e.g. /analysis_results/....jpg)
	file_path: Optional[str] = None


class AnalyzeResponse(BaseModel):
	summary: str
	items: List[DetectionItem]


