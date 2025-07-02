import os
import google_auth_httplib2
import google_auth_oauthlib
import googleapiclient.discovery
import googleapiclient.errors
import googleapiclient.http

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]


def authenticate_youtube():
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
    client_secrets_file = "/Users/rithikalias/personal/youtube-automation/client.json"

    flow = google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file(
        client_secrets_file, SCOPES)
    credentials = flow.run_local_server()

    youtube = googleapiclient.discovery.build(
        "youtube", "v3", credentials=credentials)

    return youtube


def upload_video(youtube,video_path,title,description,tags):
    request_body = {
        "snippet": {
            "categoryId": "22",
            "title": title,
            "description": description,
            "tags": tags
        },
        "status": {
            "privacyStatus": "public"
        }
    }

    # put the path of the video that you want to upload
    request = youtube.videos().insert(
        part="snippet,status",
        body=request_body,
        media_body=googleapiclient.http.MediaFileUpload(video_path, chunksize=-1, resumable=True)
    )

    response = None

    while response is None:
        status, response = request.next_chunk()
        print(response, status)
        if status:
            print(f"Upload {int(status.progress() * 100)}%")

        print(f"Video uploaded with ID: {response['id']}")


def video_upload(video_path,title,description,tags):
    youtube = authenticate_youtube()
    upload_video(youtube,video_path,title,description,tags)


