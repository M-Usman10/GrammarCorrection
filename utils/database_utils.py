# utils/database_utils.py

import os
import datetime
from typing import Optional, Dict, List, Any, Union
from pymongo import MongoClient
import gridfs

# Example env usage:
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("MONGO_DBNAME", "my_multimodal_db")

class MongoDBClient:
    def __init__(self, uri: str = MONGO_URI, db_name: str = DB_NAME):
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.fs = gridfs.GridFS(self.db)

    def store_interaction(
        self,
        interaction_type: str,
        input_data: str,
        output_data: str,
        model_name: str,
        metadata: Optional[Dict] = None,
        activity_id: Optional[str] = None,
        step_index: int = 0
    ) -> str:
        """
        Store a text-based record of an interaction (text, speech, etc.).
        Now includes 'activity_id' and 'step_index' for navigation.
        """
        if metadata is None:
            metadata = {}
        record = {
            "interaction_type": interaction_type,
            "input_data": input_data,
            "output_data": output_data,
            "model_name": model_name,
            "created_at": datetime.datetime.utcnow(),
            "metadata": metadata,
            "activity_id": activity_id,
            "step_index": step_index
        }
        result = self.db["records"].insert_one(record)
        return str(result.inserted_id)

    def store_file(
        self,
        file_bytes: bytes,
        filename: str,
        content_type: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Store any binary file (audio, image, video, etc.) in GridFS.
        Returns the file's ObjectID as string.
        """
        if metadata is None:
            metadata = {}
        file_id = self.fs.put(
            file_bytes,
            filename=filename,
            contentType=content_type,
            metadata=metadata
        )
        return str(file_id)

    def get_file(self, file_id: str) -> bytes:
        """
        Retrieve file content from GridFS by string ID.
        """
        from bson import ObjectId
        try:
            oid = ObjectId(file_id)
        except Exception:
            raise ValueError(f"Invalid file ObjectId: {file_id}")

        try:
            grid_out = self.fs.get(oid)
            return grid_out.read()
        except Exception as e:
            raise ValueError(f"Error retrieving file with ID {file_id}: {e}")

    def get_records(
        self,
        interaction_type: Optional[str] = None,
        model_name: Optional[str] = None,
        start_date: Optional[datetime.datetime] = None,
        end_date: Optional[datetime.datetime] = None
    ) -> List[Dict[str, Any]]:
        query = {}
        if interaction_type:
            query["interaction_type"] = interaction_type
        if model_name:
            query["model_name"] = model_name
        if start_date and end_date:
            query["created_at"] = {"$gte": start_date, "$lte": end_date}
        elif start_date:
            query["created_at"] = {"$gte": start_date}
        elif end_date:
            query["created_at"] = {"$lte": end_date}

        results = self.db["records"].find(query).sort("created_at", -1)
        return list(results)

    def get_records_by_activity_id(self, activity_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all records for a given activity_id.
        """
        if not activity_id:
            return []
        results = self.db["records"].find({"activity_id": activity_id})
        return list(results)

    def get_records_paginated(
        self,
        interaction_type: Optional[str] = None,
        model_name: Optional[str] = None,
        start_date: Optional[datetime.datetime] = None,
        end_date: Optional[datetime.datetime] = None,
        page: int = 1,
        page_size: int = 50
    ) -> List[Dict[str, Any]]:
        query = {}
        if interaction_type:
            query["interaction_type"] = interaction_type
        if model_name:
            query["model_name"] = model_name
        if start_date and end_date:
            query["created_at"] = {"$gte": start_date, "$lte": end_date}
        elif start_date:
            query["created_at"] = {"$gte": start_date}
        elif end_date:
            query["created_at"] = {"$lte": end_date}

        skip_count = (page - 1) * page_size
        cursor = (self.db["records"]
                  .find(query)
                  .sort("created_at", -1)
                  .skip(skip_count)
                  .limit(page_size))
        return list(cursor)

    def count_records(
        self,
        interaction_type: Optional[str] = None,
        model_name: Optional[str] = None,
        start_date: Optional[datetime.datetime] = None,
        end_date: Optional[datetime.datetime] = None
    ) -> int:
        query = {}
        if interaction_type:
            query["interaction_type"] = interaction_type
        if model_name:
            query["model_name"] = model_name
        if start_date and end_date:
            query["created_at"] = {"$gte": start_date, "$lte": end_date}
        elif start_date:
            query["created_at"] = {"$gte": start_date}
        elif end_date:
            query["created_at"] = {"$lte": end_date}

        return self.db["records"].count_documents(query)

    def run_raw_query(self, raw_query: Union[Dict, str]) -> List[Dict[str, Any]]:
        import json
        if isinstance(raw_query, str) and raw_query.strip():
            try:
                query_dict = json.loads(raw_query)
            except Exception as e:
                raise ValueError(f"Could not parse raw_query as JSON: {e}")
        elif isinstance(raw_query, dict):
            query_dict = raw_query
        else:
            raise ValueError("raw_query must be a non-empty JSON string or a dictionary.")

        results = self.db["records"].find(query_dict).sort("created_at", -1)
        return list(results)

    def close(self):
        self.client.close()