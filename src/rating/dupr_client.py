import base64
import logging
import requests


class DUPRClient:
    def __init__(self, api_url, client_key, client_secret, sandbox_mode=False):
        self.api_url = api_url.rstrip('/')
        self.client_key = client_key
        self.client_secret = client_secret
        self.sandbox = sandbox_mode
        self.token = None
        self.logger = logging.getLogger("DUPR_Client")

    def authenticate(self):
        """
        Exchanges client credentials for a stateless JWT token.
        """
        auth_str = f"{self.client_key}:{self.client_secret}"
        encoded_auth = base64.b64encode(auth_str.encode('utf-8')).decode('utf-8')
        headers = {
            "x-authorization": f"Basic {encoded_auth}",
            "Content-Type": "application/json"
        }

        endpoint = f"{self.api_url}/auth/v1.0/token"
        try:
            response = requests.post(endpoint, headers=headers, timeout=10)
            if response.status_code == 200:
                self.token = response.json().get("token")
                return True
            else:
                self.logger.error(f"Authentication failed: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            self.logger.error(f"Error during DUPR authentication: {str(e)}")
            return False

    def upload_match_result(self, payload):
        """
        Posts verified match results to DUPR.
        """
        if self.token is None:
            if not self.authenticate():
                raise ConnectionError("Authentication required to post match results.")

        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }

        endpoint = f"{self.api_url}/v1.0/match"
        try:
            response = requests.post(endpoint, json=payload, headers=headers, timeout=10)
            if response.status_code in [200, 201]:
                return {"status": "SUCCESS", "data": response.json()}
            elif response.status_code == 429:
                return {"status": "RATE_LIMITED"}
            else:
                return {"status": "FAILED", "error": response.text}
        except Exception as e:
            return {"status": "ERROR", "exception": str(e)}
