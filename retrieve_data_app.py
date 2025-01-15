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
    Show transactions from the 'transactions' collection using the new structure.
    We'll pass them to 'db_view.html' with audio playback.
    Filters: year, record_id, transaction_id, date_from, date_to, raw_query.
    """
    page_size = 50
    page = request.args.get('page', 1, type=int)

    filter_year = request.args.get('filter_year','').strip() or None
    filter_record_id = request.args.get('filter_record_id','').strip() or None
    filter_transaction_id = request.args.get('filter_transaction_id','').strip() or None
    date_from_str = request.args.get('filter_date_from','').strip()
    date_to_str = request.args.get('filter_date_to','').strip()
    raw_query = request.args.get('raw_query','').strip() or None

    date_from_obj = None
    date_to_obj = None
    try:
        if date_from_str:
            date_from_obj = datetime.datetime.fromisoformat(date_from_str)
    except:
        pass
    try:
        if date_to_str:
            date_to_obj = datetime.datetime.fromisoformat(date_to_str)
    except:
        pass

    if raw_query:
        try:
            all_docs = mongo_client.run_raw_query_transactions(raw_query)
            total_count = len(all_docs)
            start_i = (page - 1) * page_size
            end_i = start_i + page_size
            docs = all_docs[start_i:end_i]
            total_pages = math.ceil(total_count/page_size) if total_count else 1
        except ValueError as ve:
            return render_template(
                "db_view.html",
                error_msg=f"Raw query error: {ve}",
                transactions=[],
                filter_year=filter_year or "",
                filter_record_id=filter_record_id or "",
                filter_transaction_id=filter_transaction_id or "",
                filter_date_from=date_from_str,
                filter_date_to=date_to_str,
                raw_query=raw_query or "",
                current_page=page,
                total_pages=1
            )
    else:
        docs = mongo_client.get_transactions_paginated(
            year_val=filter_year,
            record_id=filter_record_id,
            transaction_id=filter_transaction_id,
            start_date=date_from_obj,
            end_date=date_to_obj,
            page=page,
            page_size=page_size
        )
        total_count = mongo_client.count_transactions(
            year_val=filter_year,
            record_id=filter_record_id,
            transaction_id=filter_transaction_id,
            start_date=date_from_obj,
            end_date=date_to_obj
        )
        total_pages = math.ceil(total_count / page_size) if total_count else 1

    # We may want to embed audio_base64 for each chat message that has an audio_file_id
    processed = []
    for d in docs:
        dcopy = dict(d)
        dcopy["_id"] = str(d["_id"])
        # Let's walk "umisource -> user -> year -> record -> unique -> chat_history"
        # and embed audio for any "audio_file_id"
        umisource_obj = dcopy.get("umisource", {})
        # we do a double/triple loop or a small recursive approach
        # For brevity, let's do a manual approach:
        for user_k, user_dict in umisource_obj.items():
            for year_k, year_dict in user_dict.items():
                for rec_k, rec_dict in year_dict.items():
                    for uk, tdict in rec_dict.items():
                        chat_hist = tdict.get("chat_history", [])
                        for msg in chat_hist:
                            audio_id = msg.get("audio_file_id")
                            if audio_id:
                                try:
                                    audio_bytes = mongo_client.get_file(audio_id)
                                    b64 = base64.b64encode(audio_bytes).decode("utf-8")
                                    msg["audio_base64"] = b64
                                except ValueError as e:
                                    msg["audio_error"] = str(e)
        processed.append(dcopy)

    return render_template(
        "db_view.html",
        transactions=processed,
        filter_year=filter_year or "",
        filter_record_id=filter_record_id or "",
        filter_transaction_id=filter_transaction_id or "",
        filter_date_from=date_from_str,
        filter_date_to=date_to_str,
        raw_query=raw_query or "",
        error_msg=None,
        current_page=page,
        total_pages=total_pages
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
