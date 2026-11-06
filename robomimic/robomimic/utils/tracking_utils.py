import os
def best_effort_notify(message: str):
    '''
    Send a notification using ntfy service, best effort basis.
    pip install python-ntfy to enable this, and set env vars NTFY_SERVER and NTFY_AUTH_TOKEN.
    '''
    try:
        from python_ntfy import NtfyClient
        client = NtfyClient(topic="robomimic", server=os.getenv("NTFY_SERVER"), auth=os.getenv("NTFY_AUTH_TOKEN"))
        client.send(message)
    except:
        pass