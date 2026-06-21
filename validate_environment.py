import sys
import importlib.metadata

def validate_env():
    """Validates the deployment environment, checking python version and core packages.
    Raises RuntimeError if critical incompatibilities are found.
    """
    errors = []
    
    # 1. Check Python Version
    py_major, py_minor = sys.version_info[:2]
    python_ver = f"{sys.version.split()[0]}"
    
    # Python 3.14+ completely crashes protobuf compilation (tp_new metaclass error).
    if py_major == 3 and py_minor >= 14:
        errors.append(
            f"Incompatible Python Version: {python_ver} detected.\n"
            f"Protobuf and Streamlit do not support Python 3.14+ due to tp_new C-API limitations.\n"
            f"Please ensure your deployment uses Python 3.11 by verifying runtime.txt is present."
        )

    # 2. Check Protobuf version and metaclass custom tp_new imports
    try:
        import google.protobuf
        proto_ver = importlib.metadata.version("protobuf")
        
        # Test loading descriptor directly to intercept tp_new errors
        try:
            import google.protobuf.descriptor  # noqa: F401
        except TypeError as e:
            if "Metaclasses with custom tp_new are not supported" in str(e) or "Descriptors cannot be created directly" in str(e):
                errors.append(
                    f"Protobuf Error: {e}\n"
                    f"This occurs when running protobuf on incompatible Python versions (like Python 3.14).\n"
                    f"Ensure you are running on Python 3.11 and install requirements.txt."
                )
    except ImportError:
        errors.append("protobuf package is not installed.")
        proto_ver = "None"

    # 3. Check Streamlit version
    try:
        streamlit_ver = importlib.metadata.version("streamlit")
    except importlib.metadata.PackageNotFoundError:
        errors.append("streamlit package is not installed.")
        streamlit_ver = "None"

    # 4. Check Transformers version
    try:
        trans_ver = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError:
        errors.append("transformers package is not installed.")
        trans_ver = "None"

    # 5. Check Sentence-Transformers version
    try:
        st_ver = importlib.metadata.version("sentence-transformers")
    except importlib.metadata.PackageNotFoundError:
        errors.append("sentence-transformers package is not installed.")
        st_ver = "None"

    # 6. Check Torch version
    try:
        torch_ver = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        errors.append("torch package is not installed.")
        torch_ver = "None"

    # 7. Check ChromaDB version
    try:
        chroma_ver = importlib.metadata.version("chromadb")
    except importlib.metadata.PackageNotFoundError:
        errors.append("chromadb package is not installed.")
        chroma_ver = "None"

    # Version mismatches rules (warnings)
    if proto_ver != "None" and chroma_ver != "None":
        try:
            proto_major = int(proto_ver.split(".")[0])
            if proto_major < 4:
                errors.append(
                    f"Version Conflict: chromadb is {chroma_ver} but protobuf is {proto_ver}.\n"
                    f"Modern ChromaDB requires protobuf>=4.25.0 to run. Please install requirements.txt."
                )
        except ValueError:
            pass

    # If errors, raise RuntimeError
    if errors:
        error_msg = "\n" + "=" * 60 + "\n[FATAL] ENVIRONMENTAL AUDIT FAILURE\n" + "=" * 60 + "\n"
        error_msg += "\n\n".join(errors)
        error_msg += "\n" + "=" * 60
        raise RuntimeError(error_msg)
        
    print("=" * 60)
    print("[OK] Environmental audit passed successfully!")
    print(f"Python      : {python_ver}")
    print(f"Protobuf    : {proto_ver}")
    print(f"Streamlit   : {streamlit_ver}")
    print(f"ChromaDB    : {chroma_ver}")
    print(f"Torch       : {torch_ver}")
    print(f"Transformers: {trans_ver}")
    print(f"Sentence-Transformers: {st_ver}")
    print("=" * 60)

if __name__ == "__main__":
    validate_env()
