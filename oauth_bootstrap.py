# oauth_bootstrap.py
import os
from dotenv import load_dotenv
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import google.oauth2.credentials as oauth2

SCOPES = ["https://www.googleapis.com/auth/calendar"]

def main():
    load_dotenv(".env.local")
    secrets_path = os.getenv("GOOGLE_OAUTH_CLIENT_SECRETS", "./google_oauth_client_secret.json")
    token_path   = os.getenv("GOOGLE_OAUTH_TOKEN", "./google_token.json")

    flow = InstalledAppFlow.from_client_secrets_file(secrets_path, SCOPES)
    creds = flow.run_local_server(port=0, prompt="consent")
    with open(token_path, "w") as f:
        f.write(creds.to_json())
    print(f"Saved token to {token_path}")

if __name__ == "__main__":
    main()
