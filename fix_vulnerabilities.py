#!/usr/bin/env python3
"""
Script to fix security vulnerabilities across multiple Docker environments.
This script addresses USN-7768-1 (dpkg), USN-7760-1 (GNU C Library), 
USN-7743-1 (libxml2), USN-7761-1 (PAM), USN-7751-1 (SQLite), and USN-7700-1 (GCC).
"""

import os
import re
from pathlib import Path

# Define vulnerability fixes for different Ubuntu versions
UBUNTU_22_04_FIXES = """
# --- Security Vulnerability Fixes ---
# Fix dpkg vulnerability (USN-7768-1)
RUN apt-get update && \\
    apt-get install -y --only-upgrade \\
    dpkg-dev=1.21.1ubuntu2.6 \\
    dpkg=1.21.1ubuntu2.6 \\
    libdpkg-perl=1.21.1ubuntu2.6

# Fix GNU C Library vulnerability (USN-7760-1)
RUN apt-get install -y --only-upgrade \\
    libc-bin=2.35-0ubuntu3.11 \\
    libc6-dev=2.35-0ubuntu3.11 \\
    libc-dev-bin=2.35-0ubuntu3.11 \\
    locales=2.35-0ubuntu3.11 \\
    libc6=2.35-0ubuntu3.11

# Fix libxml2 vulnerability (USN-7743-1)
RUN apt-get install -y --only-upgrade \\
    libxml2=2.9.13+dfsg-1ubuntu0.9
"""

UBUNTU_24_04_FIXES = """
# --- Security Vulnerability Fixes ---
# Fix dpkg vulnerability (USN-7768-1) for Ubuntu 24.04
RUN apt-get update && \\
    apt-get install -y --only-upgrade \\
    dpkg-dev=1.22.6ubuntu6.5 \\
    dpkg=1.22.6ubuntu6.5 \\
    libdpkg-perl=1.22.6ubuntu6.5

# Fix GNU C Library vulnerability (USN-7760-1) for Ubuntu 24.04
RUN apt-get install -y --only-upgrade \\
    libc-bin=2.39-0ubuntu8.6 \\
    libc6-dev=2.39-0ubuntu8.6 \\
    libc-dev-bin=2.39-0ubuntu8.6 \\
    libc6=2.39-0ubuntu8.6

# Fix libxml2 vulnerability (USN-7743-1) for Ubuntu 24.04
RUN apt-get install -y --only-upgrade \\
    libxml2=2.9.14+dfsg-1.3ubuntu3.5

# Fix PAM vulnerability (USN-7761-1) for Ubuntu 24.04
RUN apt-get install -y --only-upgrade \\
    libpam-runtime=1.5.3-5ubuntu5.5 \\
    libpam0g=1.5.3-5ubuntu5.5 \\
    libpam-modules-bin=1.5.3-5ubuntu5.5 \\
    libpam-modules=1.5.3-5ubuntu5.5

# Fix SQLite vulnerability (USN-7751-1) for Ubuntu 24.04
RUN apt-get install -y --only-upgrade \\
    libsqlite3-0=3.45.1-1ubuntu2.5
"""

GCC_FIXES = """
# Fix GCC vulnerability (USN-7700-1)
RUN apt-get install -y --only-upgrade \\
    cpp-11=11.4.0-1ubuntu1~22.04.2 \\
    libtsan0=11.4.0-1ubuntu1~22.04.2 \\
    libstdc++-11-dev=11.4.0-1ubuntu1~22.04.2 \\
    gcc-11=11.4.0-1ubuntu1~22.04.2 \\
    libgcc-11-dev=11.4.0-1ubuntu1~22.04.2 \\
    g++-11=11.4.0-1ubuntu1~22.04.2 \\
    libasan6=11.4.0-1ubuntu1~22.04.2 \\
    gcc-11-base=11.4.0-1ubuntu1~22.04.2 \\
    libquadmath0=12.3.0-1ubuntu1~22.04.2 \\
    libstdc++6=12.3.0-1ubuntu1~22.04.2 \\
    libatomic1=12.3.0-1ubuntu1~22.04.2 \\
    liblsan0=12.3.0-1ubuntu1~22.04.2 \\
    libubsan1=12.3.0-1ubuntu1~22.04.2 \\
    libcc1-0=12.3.0-1ubuntu1~22.04.2 \\
    libgcc-s1=12.3.0-1ubuntu1~22.04.2 \\
    libgomp1=12.3.0-1ubuntu1~22.04.2 \\
    gcc-12-base=12.3.0-1ubuntu1~22.04.2 \\
    libitm1=12.3.0-1ubuntu1~22.04.2
"""

# Map of environment names to their expected Ubuntu versions
ENVIRONMENT_VERSIONS = {
    # Ubuntu 22.04 environments
    'acft-group-relative-policy-optimization': '22.04',
    'acft-hf-nlp-gpu': '22.04',
    'acft-medimageinsight-adapter-finetune': '22.04',
    'acft-medimageinsight-embedding': '22.04',
    'acft-medimageinsight-embedding-generator': '22.04',
    'acft-medimageparse-finetune': '22.04',
    'acft-mmdetection-image-gpu': '22.04',
    'acft-mmtracking-video-gpu': '22.04',
    'acft-multimodal-gpu': '22.04',
    'acft-transformers-image-gpu': '22.04',
    'acpt-automl-image-framework-selector-gpu': '22.04',
    'acpt-pytorch-2.2-cuda12.1': '22.04',
    'ai-ml-automl-dnn-forecasting-gpu': '22.04',
    'ai-ml-automl-dnn-gpu': '22.04',
    'ai-ml-automl-dnn-text-gpu': '22.04',
    'ai-ml-automl-dnn-text-gpu-ptca': '22.04',
    'ai-ml-automl-dnn-vision-gpu': '22.04',
    'ai-ml-automl-gpu': '22.04',
    'automl-dnn-vision-gpu': '22.04',
    'automl-gpu': '22.04',
    'tensorflow-2.16-cuda12': '22.04',
    
    # Ubuntu 24.04 environments
    'acft-hf-nlp-data-import': '24.04',
    'ai-ml-automl': '24.04',
    'ai-ml-automl-dnn': '24.04',
    'aoai-data-upload-finetune': '24.04',
    'lightgbm-3.3': '24.04',
    'sklearn-1.5': '24.04',
}

def find_dockerfiles():
    """Find all Dockerfile paths in the workspace."""
    base_path = Path("e:/azureml-assets")
    dockerfiles = []
    for dockerfile_path in base_path.rglob("**/context/Dockerfile"):
        if dockerfile_path.exists():
            dockerfiles.append(dockerfile_path)
    return dockerfiles

def detect_ubuntu_version(dockerfile_content):
    """Detect Ubuntu version from Dockerfile content."""
    if 'ubuntu2204' in dockerfile_content.lower() or 'ubuntu22.04' in dockerfile_content.lower():
        return '22.04'
    elif 'ubuntu2404' in dockerfile_content.lower() or 'ubuntu24.04' in dockerfile_content.lower():
        return '24.04'
    elif 'ubuntu18.04' in dockerfile_content.lower() or 'ubuntu1804' in dockerfile_content.lower():
        return '18.04'
    return None

def apply_security_fixes(dockerfile_path):
    """Apply security fixes to a Dockerfile."""
    with open(dockerfile_path, 'r') as f:
        content = f.read()
    
    # Skip if already has security fixes
    if "Security Vulnerability Fixes" in content:
        print(f"Skipping {dockerfile_path} - already has security fixes")
        return False
    
    # Skip test files
    if "/test/" in str(dockerfile_path):
        print(f"Skipping test file {dockerfile_path}")
        return False
    
    # Detect Ubuntu version
    ubuntu_version = detect_ubuntu_version(content)
    if not ubuntu_version:
        print(f"Could not detect Ubuntu version for {dockerfile_path}")
        return False
    
    # Choose appropriate fixes
    if ubuntu_version == '22.04':
        fixes = UBUNTU_22_04_FIXES
        # Add GCC fixes for tensorflow-2.16-cuda12
        if 'tensorflow-2.16-cuda12' in str(dockerfile_path):
            fixes += GCC_FIXES
    elif ubuntu_version == '24.04':
        fixes = UBUNTU_24_04_FIXES
    else:
        print(f"Unsupported Ubuntu version {ubuntu_version} for {dockerfile_path}")
        return False
    
    # Find insertion point - after FROM and USER statements
    lines = content.split('\n')
    insert_index = 0
    
    for i, line in enumerate(lines):
        if line.strip().startswith('FROM '):
            insert_index = i + 1
        elif line.strip().startswith('USER root'):
            insert_index = i + 1
            break
        elif line.strip().startswith('RUN apt-get update'):
            insert_index = i
            break
    
    # Insert security fixes
    lines.insert(insert_index, fixes)
    
    # Write back to file
    with open(dockerfile_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"Applied security fixes to {dockerfile_path}")
    return True

def main():
    """Main function to apply security fixes to all Dockerfiles."""
    dockerfiles = find_dockerfiles()
    
    # Filter to only environments mentioned in the vulnerability report
    vulnerable_envs = [
        'acft-group-relative-policy-optimization',
        'acft-hf-nlp-data-import', 
        'acft-hf-nlp-gpu',
        'acft-medimageinsight-adapter-finetune',
        'acft-medimageinsight-embedding',
        'acft-medimageinsight-embedding-generator',
        'acft-medimageparse-finetune',
        'acft-mmdetection-image-gpu',
        'acft-mmtracking-video-gpu', 
        'acft-multimodal-gpu',
        'acft-transformers-image-gpu',
        'acpt-automl-image-framework-selector-gpu',
        'acpt-pytorch-2.2-cuda12.1',
        'ai-ml-automl',
        'ai-ml-automl-dnn',
        'ai-ml-automl-dnn-forecasting-gpu',
        'ai-ml-automl-dnn-gpu',
        'ai-ml-automl-dnn-text-gpu',
        'ai-ml-automl-dnn-text-gpu-ptca',
        'ai-ml-automl-dnn-vision-gpu',
        'ai-ml-automl-gpu',
        'aoai-data-upload-finetune',
        'automl-dnn-vision-gpu',
        'automl-gpu',
        'lightgbm-3.3',
        'sklearn-1.5',
        'tensorflow-2.16-cuda12'
    ]
    
    updated_count = 0
    
    for dockerfile in dockerfiles:
        dockerfile_str = str(dockerfile)
        
        # Check if this Dockerfile is for a vulnerable environment
        is_vulnerable = False
        for env_name in vulnerable_envs:
            if env_name in dockerfile_str:
                is_vulnerable = True
                break
        
        if is_vulnerable:
            if apply_security_fixes(dockerfile):
                updated_count += 1
    
    print(f"\nApplied security fixes to {updated_count} Dockerfiles")

if __name__ == "__main__":
    main()