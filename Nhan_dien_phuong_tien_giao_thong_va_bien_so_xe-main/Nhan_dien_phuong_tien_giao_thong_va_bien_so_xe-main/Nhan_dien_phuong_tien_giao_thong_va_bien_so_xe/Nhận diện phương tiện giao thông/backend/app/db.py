from datetime import datetime
from typing import List, Dict, Any
import os

from pymongo import MongoClient, ASCENDING, DESCENDING


class MongoStore:
	def __init__(self, uri: str | None = None, db_name: str | None = None) -> None:
		# Allow overriding via env vars
		uri = uri or os.getenv("MONGODB_URI", "mongodb://localhost:27017")
		db_name = db_name or os.getenv("MONGODB_DB", "traffic")
		self.client = MongoClient(uri, serverSelectionTimeoutMS=1000)
		self.db = self.client[db_name]
		self.col = self.db["detections"]
		# Ensure helpful indexes (best effort if server not available)
		try:
			self.col.create_index([("ts", DESCENDING)], name="ts_desc")
			self.col.create_index([("plate", ASCENDING)], name="plate_idx")
		except Exception:
			pass

	def insert_items(self, items: List[Dict[str, Any]]) -> None:
		if not items:
			return
		for it in items:
			it.setdefault("ts", datetime.utcnow())
		self.col.insert_many(items)

	def clear(self) -> None:
		self.col.delete_many({})


