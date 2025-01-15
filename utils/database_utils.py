import os
import datetime
from typing import Optional, Dict, List, Any, Union
from pymongo import MongoClient
import gridfs
import json
from bson import ObjectId

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("MONGO_DBNAME", "my_multimodal_db")

class MongoDBClient:
    def __init__(self, uri: str = MONGO_URI, db_name: str = DB_NAME):
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.fs = gridfs.GridFS(self.db)
        self.transactions_coll = self.db["transactions"]

    def create_new_transaction(self, username: str, yy: str, record_id: str, transaction_id: str):
        """
        new structure:
        {
          "umisource": {
            "<username>": {
              "<yy>": {
                "<yyddmmhHH>": {
                  "<unique_id>": {
                    "transaction id": transaction_id,
                    "chat_history": [],
                    "status": "open",
                    "created_at": ...
                  }
                }
              }
            }
          }
        }
        """
        unique_key = str(ObjectId())
        now = datetime.datetime.utcnow()
        doc = {
            "umisource": {
                username: {
                    yy: {
                        record_id: {
                            unique_key: {
                                "transaction id": transaction_id,
                                "chat_history": [],
                                "status": "open",
                                "created_at": now
                            }
                        }
                    }
                }
            }
        }
        result = self.transactions_coll.insert_one(doc)
        return result.inserted_id

    def get_transaction_by_oid(self, oid_str: str) -> Optional[Dict[str, Any]]:
        try:
            oid = ObjectId(oid_str)
        except:
            return None
        return self.transactions_coll.find_one({"_id": oid})

    def delete_transaction_by_oid(self, oid_str: str):
        try:
            oid = ObjectId(oid_str)
            self.transactions_coll.delete_one({"_id": oid})
        except:
            pass

    def update_transaction_status(self, oid_str: str, username: str, yy: str, record_id: str,
                                  unique_key: str, new_status: str):
        """
        doc["umisource"][username][yy][record_id][unique_key]["status"] = new_status
        """
        try:
            oid = ObjectId(oid_str)
            path = f"umisource.{username}.{yy}.{record_id}.{unique_key}.status"
            self.transactions_coll.update_one({"_id": oid}, {"$set": {path: new_status}})
        except:
            pass

    def update_transaction_subdict(self, oid_str: str, username: str, yy: str, record_id: str,
                                   unique_key: str, transaction_dict: Dict[str, Any]):
        """
        doc["umisource"][username][yy][record_id][unique_key] = transaction_dict
        """
        try:
            oid = ObjectId(oid_str)
            path = f"umisource.{username}.{yy}.{record_id}.{unique_key}"
            self.transactions_coll.update_one({"_id": oid}, {"$set": {path: transaction_dict}})
        except:
            pass

    def find_transactions_by_tid(self, t_id: str) -> List[Dict[str, Any]]:
        """
        We gather all docs, then walk doc["umisource"] => [username] => [yy] => [record_id]
        => [unique_id], check if "transaction id"== t_id
        """
        out=[]
        all_docs = self.transactions_coll.find()
        for doc in all_docs:
            try:
                umisource_obj = doc["umisource"]
                # go through each user
                for user_key in umisource_obj.keys():
                    user_obj = umisource_obj[user_key]
                    for yk in user_obj:
                        rec_map = user_obj[yk]
                        for rk in rec_map:
                            unique_map = rec_map[rk]
                            for uk, tdict in unique_map.items():
                                if tdict.get("transaction id")==t_id:
                                    out.append(doc)
                                    break
            except:
                pass
        return out

    def append_chat_message_stt(self, oid_str: str, content: str, file_id: str,
                                username: str):
        """
        /api/transcribe calls this. We'll find that single transaction, append user stt
        """
        doc = self.get_transaction_by_oid(oid_str)
        if not doc:
            return
        try:
            umisource_obj = doc["umisource"]
            user_obj = umisource_obj[username]
            # just pick the first year, record, unique?
            for yk in user_obj:
                rec_map = user_obj[yk]
                for rk in rec_map:
                    unique_map = rec_map[rk]
                    for uk, tdict in unique_map.items():
                        ch = tdict.get("chat_history",[])
                        msg = {
                            "role":"user",
                            "content":content,
                            "interaction_type":"speech-to-text",
                            "timestamp": datetime.datetime.utcnow(),
                            "audio_file_id": file_id
                        }
                        ch.append(msg)
                        tdict["chat_history"]= ch
                        self.update_transaction_subdict(oid_str, username, yk, rk, uk, tdict)
                        return
        except:
            pass

    # ---------------------------------------------------------------
    # PAGINATION / FILTERS
    # ---------------------------------------------------------------
    def get_transactions_paginated(
        self,
        year_val: Optional[str] = None,
        record_id: Optional[str] = None,
        transaction_id: Optional[str] = None,
        start_date: Optional[datetime.datetime] = None,
        end_date: Optional[datetime.datetime] = None,
        page: int = 1,
        page_size: int = 50
    ) -> List[Dict[str, Any]]:
        """
        We'll do naive approach: fetch all docs, filter in Python for these fields:
        - doc["umisource"][someUser][year_val][record_id] => uniqueMap => "transaction id"...
        """
        all_docs = self.transactions_coll.find()
        results=[]
        for doc in all_docs:
            if "umisource" not in doc:
                continue
            umobj = doc["umisource"]
            # we don't know user name keys => check them all
            matched=False
            for user_key in umobj:
                user_obj = umobj[user_key]  # => { "yy" : { rec_id : { uniqueKey: {...} } } }
                for yk in user_obj:
                    if year_val and yk!=year_val:
                        continue
                    rec_map = user_obj[yk]
                    for rk in rec_map:
                        if record_id and rk!=record_id:
                            continue
                        unique_map = rec_map[rk]
                        for uk, tdict in unique_map.items():
                            tid = tdict.get("transaction id","")
                            if transaction_id and tid!=transaction_id:
                                continue
                            created_at = tdict.get("created_at")
                            if not isinstance(created_at, datetime.datetime):
                                # no valid date => skip if date filters exist
                                if start_date or end_date:
                                    continue
                            else:
                                # check date
                                if start_date and created_at < start_date:
                                    continue
                                if end_date and created_at > end_date:
                                    continue
                            # if we get here => matched
                            matched=True
                            break
                        if matched:
                            break
                    if matched:
                        break
                if matched:
                    break
            if matched or (not year_val and not record_id and not transaction_id and not start_date and not end_date):
                results.append(doc)

        results.sort(key=lambda d: d["_id"], reverse=True)
        start_i = (page-1)*page_size
        end_i = start_i+page_size
        return results[start_i:end_i]

    def count_transactions(
        self,
        year_val: Optional[str] = None,
        record_id: Optional[str] = None,
        transaction_id: Optional[str] = None,
        start_date: Optional[datetime.datetime] = None,
        end_date: Optional[datetime.datetime] = None
    ) -> int:
        all_docs = self.transactions_coll.find()
        c=0
        for doc in all_docs:
            if "umisource" not in doc:
                continue
            umobj = doc["umisource"]
            matched=False
            for user_key in umobj:
                user_obj = umobj[user_key]
                for yk in user_obj:
                    if year_val and yk!=year_val:
                        continue
                    rec_map = user_obj[yk]
                    for rk in rec_map:
                        if record_id and rk!=record_id:
                            continue
                        unique_map= rec_map[rk]
                        for uk, tdict in unique_map.items():
                            tid = tdict.get("transaction id","")
                            if transaction_id and tid!=transaction_id:
                                continue
                            created_at= tdict.get("created_at")
                            if isinstance(created_at, datetime.datetime):
                                if start_date and created_at<start_date:
                                    continue
                                if end_date and created_at>end_date:
                                    continue
                            else:
                                # no valid date => skip if we have filters
                                if start_date or end_date:
                                    continue
                            matched=True
                            break
                        if matched:
                            break
                    if matched:
                        break
                if matched:
                    break
            if matched or (not year_val and not record_id and not transaction_id and not start_date and not end_date):
                c+=1
        return c

    def run_raw_query_transactions(self, raw_query: Union[str,Dict[str,Any]]) -> List[Dict[str,Any]]:
        """
        Perform a direct .find(query). Then sort by _id desc.
        If user wants to search "umisource.samuel.25.251205h09.<unique>.transaction id"
        they'd do bracket notation in the JSON.
        """
        if isinstance(raw_query,str) and raw_query.strip():
            try:
                qdict = json.loads(raw_query)
            except Exception as e:
                raise ValueError(f"Could not parse raw_query as JSON: {e}")
        elif isinstance(raw_query,dict):
            qdict = raw_query
        else:
            raise ValueError("raw_query must be non-empty JSON or dict.")

        return list(self.transactions_coll.find(qdict).sort("_id", -1))

    # File storage
    def store_file(
        self,
        file_bytes: bytes,
        filename: str,
        content_type: str,
        metadata: Optional[Dict[str,Any]]=None
    ) -> str:
        if metadata is None:
            metadata={}
        file_id = self.fs.put(file_bytes, filename=filename, contentType=content_type, metadata=metadata)
        return str(file_id)

    def get_file(self, file_id: str) -> bytes:
        try:
            oid = ObjectId(file_id)
            grid_out = self.fs.get(oid)
            return grid_out.read()
        except:
            raise ValueError(f"Cannot retrieve file with ID {file_id}")

    def close(self):
        self.client.close()
