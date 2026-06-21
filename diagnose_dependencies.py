import sys
import os
import importlib.metadata
from typing import Dict, List

def get_pkg_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "Not Installed"

def run_diagnostics():
    print("=" * 60)
    print(" [AUDIT] AGENTIC RAG SYSTEM - DEPENDENCY AUDIT DIAGNOSTICS")
    print("=" * 60)
    
    # 1. System Info
    print(f"Python Version      : {sys.version.split()[0]}")
    print(f"Platform            : {sys.platform}")
    print(f"Working Directory   : {os.getcwd()}")
    print("-" * 60)

    # 2. Package Versions
    packages = [
        "protobuf",
        "chromadb",
        "transformers",
        "sentence-transformers",
        "torch",
        "langchain",
        "langchain-community",
        "langchain-google-genai",
        "google-genai",
        "googleapis-common-protos",
        "grpcio",
        "streamlit"
    ]
    
    versions: Dict[str, str] = {}
    for pkg in packages:
        ver = get_pkg_version(pkg)
        versions[pkg] = ver
        print(f"{pkg:<24}: {ver}")
        
    print("-" * 60)

    # 3. GPU Availability
    gpu_available = False
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        device_name = torch.cuda.get_device_name(0) if gpu_available else "N/A"
        print(f"PyTorch CUDA (GPU)  : {'Available [OK]' if gpu_available else 'Not Available'}")
        if gpu_available:
            print(f"GPU Device Name     : {device_name}")
    except ImportError:
        print("PyTorch CUDA (GPU)  : Import Failed")
        
    print("-" * 60)

    # 4. Conflict Checking and Import Trace
    print("Performing import audit and checking compatibility...")
    conflicts: List[str] = []
    
    # Test importing and loading protobuf
    try:
        import google.protobuf.descriptor  # noqa: F401
    except TypeError as e:
        if "Descriptors cannot be created directly" in str(e):
            conflicts.append(
                "[FAIL] Protobuf Conflict: 'Descriptors cannot be created directly' was raised during descriptor load. "
                "This indicates that protobuf>=4.0.0 was loaded, but some generated protobuf files (like in old google-api-core "
                "or transformers) are outdated."
            )
        else:
            conflicts.append(f"[FAIL] Protobuf error: {e}")
    except ImportError:
        conflicts.append("[FAIL] protobuf package is not installed.")
        
    # Check version combinations
    proto_ver = versions.get("protobuf", "Not Installed")
    chroma_ver = versions.get("chromadb", "Not Installed")
    if proto_ver != "Not Installed" and chroma_ver != "Not Installed":
        try:
            proto_major = int(proto_ver.split(".")[0])
            if proto_major < 4:
                conflicts.append(
                    f"[WARN] Incompatible combination: protobuf version is {proto_ver} but chromadb is {chroma_ver}. "
                    "Modern ChromaDB expects protobuf >= 4.0.0. Running on protobuf 3.x might cause errors."
                )
        except ValueError:
            pass

    # Test importing individual libraries and intercepting protobuf errors
    import_tests = [
        ("chromadb", "chromadb"),
        ("transformers", "transformers"),
        ("sentence-transformers", "sentence_transformers"),
        ("torch", "torch"),
        ("langchain-google-genai", "langchain_google_genai"),
        ("google-genai", "google.genai"),
        ("grpcio", "grpc")
    ]
    
    for pkg_name, module_name in import_tests:
        try:
            # Dynamically import
            if "." in module_name:
                parts = module_name.split(".")
                mod = __import__(parts[0])
                for part in parts[1:]:
                    mod = getattr(mod, part)
            else:
                __import__(module_name)
        except ImportError:
            # Only add to conflicts if it's supposed to be installed
            if versions.get(pkg_name, "Not Installed") != "Not Installed":
                conflicts.append(f"[FAIL] Could not import installed package '{pkg_name}' (module: '{module_name}')")
        except TypeError as e:
            if "Descriptors cannot be created directly" in str(e):
                conflicts.append(
                    f"[FAIL] TypeError during import of '{pkg_name}': {e}.\n"
                    "-> This confirms a protobuf compilation mismatch in its dependencies."
                )
            else:
                conflicts.append(f"[FAIL] TypeError during import of '{pkg_name}': {e}")
        except Exception as e:
            conflicts.append(f"[WARN] Exception during import of '{pkg_name}': {e}")

    # Python version check
    py_major, py_minor = sys.version_info[:2]
    if py_major == 3 and py_minor >= 13:
        if proto_ver.startswith("3."):
            conflicts.append(
                f"[WARN] Running Python {sys.version.split()[0]} with Protobuf 3.x is deprecated and unstable."
            )

    if not conflicts:
        print("[OK] No dependency conflicts detected! The environment is natively compatible.")
    else:
        print("Mismatches/Conflicts detected:")
        for idx, conflict in enumerate(conflicts):
            print(f" {idx + 1}. {conflict}")
            
    print("=" * 60)

if __name__ == "__main__":
    run_diagnostics()
