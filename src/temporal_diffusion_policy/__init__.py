from dotenv import load_dotenv

__version__ = "0.1.0"

# Preserve the repository's existing dotenv behaviour without mutating sys.path
# when the project is imported from an installed wheel.
load_dotenv()
