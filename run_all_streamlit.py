import os
import sys
import importlib.util
try:
    import streamlit as st
except Exception:
    raise RuntimeError("Streamlit is required to run this launcher. Install with: pip install streamlit")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="RNN Demos Launcher", page_icon="🤖")
st.title("RNN RNN Demo Launcher")
st.write("Select one of the RNN demo apps below to run it inside this Streamlit session.")

options = {
    "Many-to-One (SMS Spam)": os.path.join(BASE_DIR, "Many-to-one", "app.py"),
    "Many-to-Many (NER)": os.path.join(BASE_DIR, "Many-to-Many", "app.py"),
    "One-to-One (Digit)": os.path.join(BASE_DIR, "one-to-one", "app.py"),
    "One-to-Many (Text Gen)": os.path.join(BASE_DIR, "one-to-many", "app.py"),
}

choice = st.sidebar.selectbox("Choose a demo to run", list(options.keys()))
selected_path = options[choice]

st.sidebar.write("Module file:")
st.sidebar.code(selected_path)

def load_module_from_path(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

try:
    module_name = f"rnn_demo_{abs(hash(selected_path))}"
    module = load_module_from_path(module_name, selected_path)
except Exception as e:
    st.error(f"Failed to load module: {e}")
else:
    if hasattr(module, "app"):
        module.app()
    else:
        st.error("The selected module does not expose an `app()` function.")
