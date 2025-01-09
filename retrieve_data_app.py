from flask import Flask, render_template, request
from utils.database_utils import MongoDBClient
import base64
import datetime
import math

app = Flask(__name__)
mongo_client = MongoDBClient()

@app.route('/', methods=['GET'])
def db_view():
    """
    Displays a page where we can filter the 'records' in MongoDB
    with pagination (50 per page).
    We fit (Interaction Type, Model Name, Date From, Date To, Activity ID)
    in the first row, and Raw Query (JSON) in the second row.
    """
    page_size = 50
    page = request.args.get('page', 1, type=int)

    filter_interaction_type = request.args.get('filter_interaction_type', '').strip() or None
    filter_model_name = request.args.get('filter_model_name', '').strip() or None
    date_from_str = request.args.get('filter_date_from', '').strip()
    date_to_str = request.args.get('filter_date_to', '').strip()
    filter_activity_id = request.args.get('filter_activity_id', '').strip() or None
    raw_query = request.args.get('raw_query', '').strip() or None

    # Attempt to parse date_from / date_to
    try:
        date_from_obj = datetime.datetime.fromisoformat(date_from_str) if date_from_str else None
    except ValueError:
        date_from_obj = None

    try:
        date_to_obj = datetime.datetime.fromisoformat(date_to_str) if date_to_str else None
    except ValueError:
        date_to_obj = None

    # If a raw query is supplied, we ignore the standard filters
    if raw_query:
        try:
            records = mongo_client.run_raw_query(raw_query)
            total_count = len(records)
            records = records[(page - 1) * page_size : page * page_size]
            total_pages = math.ceil(total_count / page_size) if total_count else 1
        except ValueError as ve:
            return render_template(
                'db_view.html',
                error_msg=f"Raw query error: {ve}",
                records=[],
                filter_interaction_type=filter_interaction_type or "",
                filter_model_name=filter_model_name or "",
                filter_date_from=date_from_str,
                filter_date_to=date_to_str,
                filter_activity_id=filter_activity_id or "",
                raw_query=raw_query,
                current_page=page,
                total_pages=1
            )
    else:
        # If filter_activity_id is given, handle that
        if filter_activity_id:
            all_records = mongo_client.get_records_by_activity_id(filter_activity_id)
            total_count = len(all_records)
            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size
            records = all_records[start_idx:end_idx]
            total_pages = math.ceil(total_count / page_size) if total_count else 1
        else:
            # Normal path: use other filters with pagination
            records = mongo_client.get_records_paginated(
                interaction_type=filter_interaction_type,
                model_name=filter_model_name,
                start_date=date_from_obj,
                end_date=date_to_obj,
                page=page,
                page_size=page_size
            )
            total_count = mongo_client.count_records(
                interaction_type=filter_interaction_type,
                model_name=filter_model_name,
                start_date=date_from_obj,
                end_date=date_to_obj
            )
            total_pages = math.ceil(total_count / page_size) if total_count else 1

    processed_records = []
    for r in records:
        rec_copy = dict(r)
        rec_copy["_id"] = str(r["_id"])

        audio_file_id = None
        if "file_id=" in rec_copy.get("input_data", ""):
            audio_file_id = rec_copy["input_data"].split("file_id=")[-1].strip()
        elif "file_id=" in rec_copy.get("output_data", ""):
            audio_file_id = rec_copy["output_data"].split("file_id=")[-1].strip()

        if audio_file_id:
            try:
                audio_bytes = mongo_client.get_file(audio_file_id)
                b64_audio = base64.b64encode(audio_bytes).decode("utf-8")
                rec_copy["audio_base64"] = b64_audio
                rec_copy["audio_file_id"] = audio_file_id
            except ValueError as e:
                rec_copy["audio_error"] = str(e)

        processed_records.append(rec_copy)

    return render_template(
        'db_view.html',
        records=processed_records,
        filter_interaction_type=filter_interaction_type or "",
        filter_model_name=filter_model_name or "",
        filter_date_from=date_from_str,
        filter_date_to=date_to_str,
        filter_activity_id=filter_activity_id or "",
        raw_query=raw_query or "",
        error_msg=None,
        current_page=page,
        total_pages=total_pages
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)