# utils/database_utils.py

import os
import datetime
from typing import Optional, Dict
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
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Store a text-based record of an interaction (text, speech, etc.).
        Returns the inserted record's ID as string.
        """
        if metadata is None:
            metadata = {}
        record = {
            "interaction_type": interaction_type,
            "input_data": input_data,
            "output_data": output_data,
            "model_name": model_name,
            "created_at": datetime.datetime.utcnow(),
            "metadata": metadata
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
        Retrieve file content from GridFS by ID.
        Returns raw bytes.
        """
        grid_out = self.fs.get(file_id)
        return grid_out.read()

    def store_image_to_db(
        self,
        image_bytes: bytes,
        filename: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Optional convenience function to store images.
        """
        return self.store_file(
            file_bytes=image_bytes,
            filename=filename,
            content_type="image/png",
            metadata=metadata
        )

    def store_video_to_db(
        self,
        video_bytes: bytes,
        filename: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Optional convenience function to store videos.
        """
        return self.store_file(
            file_bytes=video_bytes,
            filename=filename,
            content_type="video/mp4",
            metadata=metadata
        )

    def close(self):
        """
        Close DB client if needed.
        """
        self.client.close()