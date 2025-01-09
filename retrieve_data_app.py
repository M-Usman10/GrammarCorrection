from flask import Flask, render_template, request, jsonify, redirect, url_for
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
    """
    page_size = 50
    page = request.args.get('page', 1, type=int)

    filter_interaction_type = request.args.get('filter_interaction_type', '').strip() or None
    filter_model_name = request.args.get('filter_model_name', '').strip() or None
    date_from_str = request.args.get('filter_date_from', '').strip()
    date_to_str = request.args.get('filter_date_to', '').strip()
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

    # Decide how to retrieve records
    if raw_query:
        # If a raw query is supplied, we do not use the normal filters
        # If you want pagination for raw queries as well, you'd need a
        # run_raw_query_paginated(...) function. We'll just do a full run here:
        try:
            records = mongo_client.run_raw_query(raw_query)
            total_count = len(records)  # naive (no pagination example)
            # For real pagination, create run_raw_query_paginated(...) or
            # slice in Python. E.g. records[ (page-1)*page_size : page*page_size ]
            records = records[(page - 1) * page_size: page * page_size]
        except ValueError as ve:
            return render_template(
                'db_view.html',
                error_msg=f"Raw query error: {ve}",
                records=[],
                filter_interaction_type=filter_interaction_type or "",
                filter_model_name=filter_model_name or "",
                filter_date_from=date_from_str,
                filter_date_to=date_to_str,
                raw_query=raw_query,
                current_page=page,
                total_pages=1
            )
    else:
        # Normal path: use filters with pagination
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

    # Calculate total pages
    total_pages = math.ceil(total_count / page_size) if total_count else 1

    # Process audio references
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
        raw_query=raw_query or "",
        error_msg=None,
        current_page=page,
        total_pages=total_pages
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)