# Installable client SDK (experimental)

Build with `python -m pip install build && python -m build`, then install the wheel from `dist/`. CI builds and installs the package and publishes an artifact. It is not published to PyPI.

```python
from apicalai_sdk import Client
client = Client("http://127.0.0.1:8080", api_key="your-local-client-key")
print(client.models())
print(client.infer("demo-sentiment-v1", "great service"))
```

For remote access use a trusted HTTPS origin. The client does not follow redirects, to avoid forwarding credentials. It does not provision a backend or provide a desktop installer.
