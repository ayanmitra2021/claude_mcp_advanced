import os

# Must be set before client.py is imported so AsyncAnthropic() does not raise.
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-placeholder")
